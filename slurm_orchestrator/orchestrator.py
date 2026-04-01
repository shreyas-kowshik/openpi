#!/usr/bin/env python3
"""
Slurm Job Orchestrator

Monitors slurm jobs on the preempt partition and resubmits them when they
fail, complete without being requeued, or fail to launch.

Usage:
    # Start the orchestrator (run inside tmux/screen so it persists):
    python slurm_orchestrator/orchestrator.py

    # Check current status of all tracked jobs:
    python slurm_orchestrator/orchestrator.py --status

    # Retire a job (stop resubmitting it):
    python slurm_orchestrator/orchestrator.py --retire book_caddy.slurm

    # Un-retire a job and clear its state:
    python slurm_orchestrator/orchestrator.py --reset book_caddy.slurm

    # Use a custom config:
    python slurm_orchestrator/orchestrator.py --config path/to/config.json
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orchestrator")

DEFAULT_CONFIG = "slurm_orchestrator/config_ep1.json"


@dataclass
class JobEntry:
    script: str
    job_id: Optional[str] = None
    submit_count: int = 0
    last_submit_time: Optional[str] = None
    last_status: Optional[str] = None
    retired: bool = False


def setup_logging(log_file: Optional[str] = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def load_state(state_path: str) -> dict[str, JobEntry]:
    if not os.path.exists(state_path):
        return {}
    with open(state_path) as f:
        data = json.load(f)
    return {script: JobEntry(**info) for script, info in data.items()}


def save_state(state_path: str, entries: dict[str, JobEntry]):
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({s: asdict(e) for s, e in entries.items()}, f, indent=2)
    os.replace(tmp_path, state_path)


def submit_job(script_path: str) -> Optional[str]:
    """Submit a slurm job via sbatch. Returns job ID or None on failure."""
    try:
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.error(
                "sbatch failed for %s (rc=%d): %s",
                script_path,
                result.returncode,
                result.stderr.strip(),
            )
            return None
        # sbatch output format: "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        logger.info("Submitted %s -> job %s", script_path, job_id)
        return job_id
    except subprocess.TimeoutExpired:
        logger.error("sbatch timed out for %s", script_path)
        return None
    except Exception as e:
        logger.error("Failed to submit %s: %s", script_path, e)
        return None


def get_queue_state(job_id: str) -> Optional[tuple[str, str]]:
    """Return (state, reason) for a job, or None if not in queue.

    state  = %T field, e.g. RUNNING, PENDING, REQUEUE_HOLD
    reason = %R field, e.g. "(launch failure limit exceeded requeued held)"
    """
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "--noheader", "--format=%T %R"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout.strip()
        if not output:
            return None
        parts = output.split(" ", 1)
        state = parts[0]
        reason = parts[1] if len(parts) > 1 else ""
        return state, reason
    except Exception as e:
        logger.warning("squeue check failed for job %s: %s", job_id, e)
        # Assume still active to avoid accidental double-submit
        return "UNKNOWN", ""


def is_launch_failed_held(queue_state: tuple[str, str]) -> bool:
    """Return True if the job is stuck in REQUEUE_HOLD due to launch failures."""
    state, reason = queue_state
    return state == "REQUEUE_HOLD" and "launch failure" in reason


def is_job_in_queue(job_id: str) -> bool:
    """Check if a job is in the slurm queue (PENDING, RUNNING, REQUEUED, etc.)."""
    state = get_queue_state(job_id)
    return state is not None


def cancel_job(job_id: str):
    """Cancel a slurm job via scancel."""
    try:
        subprocess.run(["scancel", job_id], timeout=15, check=True)
        logger.info("Cancelled job %s", job_id)
    except Exception as e:
        logger.warning("Failed to cancel job %s: %s", job_id, e)


def get_job_name_from_script(script_path: str) -> Optional[str]:
    """Extract the #SBATCH --job-name from a slurm script."""
    try:
        with open(script_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#SBATCH") and "--job-name=" in line:
                    return line.split("--job-name=")[1].strip()
    except Exception as e:
        logger.warning("Failed to parse job name from %s: %s", script_path, e)
    return None


def find_running_job_by_name(job_name: str) -> Optional[str]:
    """Check if a job with the given name is already in the queue. Returns job ID or None."""
    try:
        result = subprocess.run(
            ["squeue", "--me", "--name=" + job_name, "--noheader", "--format=%i"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        if lines:
            return lines[0]
    except Exception as e:
        logger.warning("squeue name check failed for '%s': %s", job_name, e)
    return None


def get_job_status(job_id: str) -> Optional[str]:
    """Get the status of a job from sacct (works for completed/failed jobs)."""
    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "--format=State",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        if not lines:
            return None
        # First line is the overall job state; subsequent lines are job steps
        return lines[0]
    except Exception as e:
        logger.warning("sacct check failed for job %s: %s", job_id, e)
        return None


class Orchestrator:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.base_dir = self.config.get("base_dir", os.getcwd())
        self.slurm_dir = os.path.join(self.base_dir, self.config["slurm_dir"])
        self.poll_interval = self.config.get("poll_interval_seconds", 120)
        self.max_resubmits = self.config.get("max_resubmits", 0)  # 0 = unlimited
        self.state_path = os.path.join(
            self.base_dir,
            self.config.get("state_file", "slurm_orchestrator/state.json"),
        )
        self.scripts = self.config["scripts"]
        self.entries: dict[str, JobEntry] = {}
        self._shutdown = False
        self._job_names: dict[str, Optional[str]] = {}

        # Pre-load job names from config or by parsing slurm scripts
        config_job_names = self.config.get("job_names", {})
        for script in self.scripts:
            if script in config_job_names:
                self._job_names[script] = config_job_names[script]
            else:
                script_path = os.path.join(self.slurm_dir, script)
                self._job_names[script] = get_job_name_from_script(script_path)

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info("Received signal %s, shutting down after current cycle...", signum)
        self._shutdown = True

    def initialize(self):
        """Load persisted state and create entries for any new scripts."""
        self.entries = load_state(self.state_path)
        for script in self.scripts:
            if script not in self.entries:
                self.entries[script] = JobEntry(script=script)
        save_state(self.state_path, self.entries)

    def _check_adopt_existing(self, entry: JobEntry) -> bool:
        """Check if a job with the same name is already running. If so, adopt it.

        Returns True if an existing job was adopted (caller should skip submission).
        """
        job_name = self._job_names.get(entry.script)
        if not job_name:
            return False
        existing_id = find_running_job_by_name(job_name)
        if existing_id and existing_id != entry.job_id:
            logger.info(
                "Found existing job %s for '%s' (%s) — adopting instead of submitting.",
                existing_id,
                job_name,
                entry.script,
            )
            entry.job_id = existing_id
            entry.last_status = "ADOPTED"
            return True
        return False

    def _should_submit(self, entry: JobEntry) -> bool:
        """Determine if a job needs (re)submission."""
        if entry.retired:
            return False
        if self.max_resubmits > 0 and entry.submit_count >= self.max_resubmits:
            logger.warning(
                "Max resubmits (%d) reached for %s — retiring.",
                self.max_resubmits,
                entry.script,
            )
            entry.retired = True
            return False
        # No job ID means never submitted or previous sbatch failed
        if entry.job_id is None:
            if self._check_adopt_existing(entry):
                return False
            return True
        # Job has an ID — check if it's still in the queue
        queue_state = get_queue_state(entry.job_id)
        if queue_state is not None:
            if is_launch_failed_held(queue_state):
                logger.warning(
                    "Job %s (%s) is held due to launch failures — cancelling and relaunching.",
                    entry.job_id,
                    entry.script,
                )
                cancel_job(entry.job_id)
                entry.job_id = None
                entry.last_status = "LAUNCH_FAILED_HELD"
                return True
            return False
        # Not in queue — it exited. Log the terminal status.
        status = get_job_status(entry.job_id)
        entry.last_status = status
        logger.info(
            "Job %s (%s) left the queue — sacct status: %s",
            entry.job_id,
            entry.script,
            status,
        )
        # Before resubmitting, check if someone already launched a replacement
        if self._check_adopt_existing(entry):
            return False
        return True

    def _submit(self, entry: JobEntry):
        """Submit (or resubmit) a job for the given entry."""
        script_path = os.path.join(self.slurm_dir, entry.script)
        if not os.path.exists(script_path):
            logger.error("Script not found: %s", script_path)
            return

        job_id = submit_job(script_path)
        if job_id is not None:
            entry.job_id = job_id
            entry.submit_count += 1
            entry.last_submit_time = datetime.now().isoformat()
            entry.last_status = "SUBMITTED"
        else:
            # sbatch failed — leave job_id as-is so next cycle retries
            entry.job_id = None
            entry.last_status = "SUBMIT_FAILED"

        save_state(self.state_path, self.entries)

    def poll(self):
        """Single poll cycle: check every tracked job and (re)submit as needed."""
        for script in self.scripts:
            if self._shutdown:
                break
            entry = self.entries[script]
            if self._should_submit(entry):
                action = "Submitting" if entry.submit_count == 0 else "Resubmitting"
                logger.info("%s %s (attempt #%d)", action, script, entry.submit_count + 1)
                self._submit(entry)
        save_state(self.state_path, self.entries)

    def print_status(self):
        """Log a status table for all tracked jobs."""
        logger.info("=" * 70)
        logger.info("%-45s %-10s %-8s %s", "SCRIPT", "JOB_ID", "STATUS", "SUBMITS")
        logger.info("-" * 70)
        for script in self.scripts:
            entry = self.entries[script]
            if entry.retired:
                status = "RETIRED"
            elif entry.job_id is None:
                status = "NONE"
            elif (queue_state := get_queue_state(entry.job_id)) is not None:
                status = "HELD" if is_launch_failed_held(queue_state) else "ACTIVE"
            else:
                status = entry.last_status or "EXITED"
            logger.info(
                "%-45s %-10s %-8s %d",
                script,
                entry.job_id or "-",
                status,
                entry.submit_count,
            )
        logger.info("=" * 70)

    def run(self):
        """Main loop: submit missing jobs, then poll periodically."""
        logger.info("Slurm Job Orchestrator starting")
        logger.info("Monitoring %d scripts from %s", len(self.scripts), self.slurm_dir)
        logger.info("Poll interval: %ds", self.poll_interval)
        if self.max_resubmits > 0:
            logger.info("Max resubmits per job: %d", self.max_resubmits)
        else:
            logger.info("Max resubmits: unlimited")

        self.initialize()

        # Initial submission
        self.poll()
        self.print_status()

        while not self._shutdown:
            time.sleep(self.poll_interval)
            if self._shutdown:
                break
            logger.info("--- Poll cycle ---")
            self.poll()
            self.print_status()

            # Exit if every job is retired
            if all(e.retired for e in self.entries.values()):
                logger.info("All jobs retired. Exiting.")
                break

        save_state(self.state_path, self.entries)
        logger.info("Orchestrator stopped.")


def main():
    parser = argparse.ArgumentParser(description="Slurm Job Orchestrator")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to config file (default: %(default)s)",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--status",
        action="store_true",
        help="Print current job status and exit.",
    )
    group.add_argument(
        "--retire",
        metavar="SCRIPT",
        help="Stop resubmitting a script (e.g. book_caddy.slurm).",
    )
    group.add_argument(
        "--reset",
        metavar="SCRIPT",
        help="Clear retired flag and job ID for a script.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("log_file"))

    orch = Orchestrator(args.config)
    orch.initialize()

    if args.status:
        orch.print_status()
        return

    if args.retire:
        if args.retire in orch.entries:
            orch.entries[args.retire].retired = True
            save_state(orch.state_path, orch.entries)
            logger.info("Retired: %s", args.retire)
        else:
            logger.error("Unknown script: %s", args.retire)
            sys.exit(1)
        return

    if args.reset:
        if args.reset in orch.entries:
            orch.entries[args.reset].retired = False
            orch.entries[args.reset].job_id = None
            orch.entries[args.reset].last_status = None
            save_state(orch.state_path, orch.entries)
            logger.info("Reset: %s", args.reset)
        else:
            logger.error("Unknown script: %s", args.reset)
            sys.exit(1)
        return

    orch.run()


if __name__ == "__main__":
    main()
