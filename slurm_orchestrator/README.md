# Slurm Job Orchestrator

A lightweight watchdog that monitors Slurm jobs submitted to the `preempt` partition and automatically resubmits them whenever they leave the queue — whether due to preemption, failure, timeout, cancellation, or clean completion that wasn't requeued.

Because every training script uses `--resume`, resubmitting a job is always safe: it picks up from the last checkpoint. If training is truly finished, the resubmitted job will simply exit quickly.

## Directory Structure

```
slurm_orchestrator/
├── config.json        # What to monitor and how (you edit this)
├── orchestrator.py    # The orchestrator itself
├── launch.sh          # Helper to start the orchestrator in a tmux session
├── state.json         # Auto-generated at runtime — tracks job IDs and submit counts
└── orchestrator.log   # Auto-generated at runtime — full log of all actions
```

## Prerequisites

- Python 3.10+ (uses `dict[str, ...]` type hints; only stdlib, no pip installs needed)
- Slurm CLI tools available: `sbatch`, `squeue`, `sacct`
- `tmux` (optional, only needed if you use `launch.sh`)

## Quick Start

All commands below assume you are in the project root (`/home/skowshik/vla/codebase/openpi`).

### 1. Edit the config

Open `slurm_orchestrator/config.json` and verify the settings match your setup:

```json
{
  "base_dir": "/home/skowshik/vla/codebase/openpi",
  "slurm_dir": "slurm/libero10/3_demos",
  "poll_interval_seconds": 120,
  "max_resubmits": 0,
  "log_file": "slurm_orchestrator/orchestrator.log",
  "state_file": "slurm_orchestrator/state.json",
  "scripts": [
    "alphabet_soup_cream_cheese_basket.slurm",
    "alphabet_soup_tomato_sauce_basket.slurm",
    "black_bowl_bottom_drawer.slurm",
    "book_caddy.slurm",
    "both_mokapots_stove.slurm",
    "cream_cheese_butter_basket.slurm",
    "stove_mokapot.slurm",
    "white_mug_chocolate_pudding.slurm",
    "white_mug_plates.slurm",
    "yellow_white_mug_microwave.slurm"
  ]
}
```

See [Configuration Reference](#configuration-reference) below for what each field does.

### 2. Launch the orchestrator

**Option A — tmux (recommended, persists after SSH disconnect):**

```bash
bash slurm_orchestrator/launch.sh
```

This starts the orchestrator in a detached tmux session named `slurm_orch`. You can safely disconnect from SSH and it will keep running.

To start and immediately see the output:

```bash
bash slurm_orchestrator/launch.sh --attach
```

**Option B — run directly in your terminal:**

```bash
python slurm_orchestrator/orchestrator.py
```

> Note: if you run it directly, the orchestrator will stop when you close the terminal or disconnect from SSH.

### 3. What happens on launch

1. The orchestrator reads `config.json` and loads any existing state from `state.json`.
2. For each script in the `scripts` list, it checks whether there is already an active job in the Slurm queue (`squeue`).
3. Any script without an active job gets submitted via `sbatch`.
4. It then enters the poll loop — every `poll_interval_seconds` (default: 120s) it rechecks all jobs and resubmits any that have left the queue.

## Day-to-Day Usage

### Check job status

Print a table of all tracked jobs without starting the monitor loop:

```bash
python slurm_orchestrator/orchestrator.py --status
```

Example output:

```
SCRIPT                                        JOB_ID     STATUS   SUBMITS
----------------------------------------------------------------------
book_caddy.slurm                              1234567    ACTIVE   2
both_mokapots_stove.slurm                     1234568    ACTIVE   1
stove_mokapot.slurm                           -          SUBMIT_FAILED 3
white_mug_plates.slurm                        1234570    RETIRED  5
```

Status meanings:
| Status | Meaning |
|---|---|
| `ACTIVE` | Job is currently in the Slurm queue (PENDING, RUNNING, or REQUEUED) |
| `SUBMITTED` | Job was just submitted; hasn't been checked yet |
| `SUBMIT_FAILED` | The last `sbatch` call failed; will retry on next poll cycle |
| `RETIRED` | Job will no longer be resubmitted (manually retired or hit max_resubmits) |
| `COMPLETED` / `FAILED` / `CANCELLED` / `TIMEOUT` / `NODE_FAIL` / `PREEMPTED` | Terminal Slurm state from `sacct`; job will be resubmitted on the next poll cycle |
| `NONE` | No job ID recorded yet |

### Retire a job (stop resubmitting)

When a training run is truly finished and you don't want it resubmitted anymore:

```bash
python slurm_orchestrator/orchestrator.py --retire book_caddy.slurm
```

The orchestrator (if running) will skip this script on all future poll cycles.

### Un-retire / reset a job

To start resubmitting a previously retired script again:

```bash
python slurm_orchestrator/orchestrator.py --reset book_caddy.slurm
```

This clears the retired flag and the stored job ID. The next poll cycle will submit it fresh.

### View live logs

The orchestrator logs to both stdout and `slurm_orchestrator/orchestrator.log`. To tail the log file:

```bash
tail -f slurm_orchestrator/orchestrator.log
```

### Re-attach to the tmux session

If you launched with `launch.sh` and disconnected:

```bash
tmux attach -t slurm_orch
```

To detach again without stopping it: press `Ctrl+b` then `d`.

### Stop the orchestrator

**If running in tmux:**

```bash
tmux kill-session -t slurm_orch
```

**If running directly in a terminal:**

Press `Ctrl+c`. It will finish the current poll cycle, save state, and exit gracefully.

### Restart the orchestrator

Just launch it again. It reads `state.json` on startup, verifies which jobs are still active, and only resubmits the ones that are missing from the queue. Submit counts and history are preserved across restarts.

```bash
bash slurm_orchestrator/launch.sh
```

> `launch.sh` automatically kills any existing `slurm_orch` tmux session before starting a new one, so you don't need to manually stop it first.

## Configuration Reference

All fields in `config.json`:

| Field | Type | Default | Description |
|---|---|---|---|
| `base_dir` | string | current working directory | Absolute path to the project root. All relative paths (`slurm_dir`, `state_file`, `log_file`) are resolved relative to this. |
| `slurm_dir` | string | *required* | Path to the directory containing the `.slurm` scripts (relative to `base_dir`). |
| `scripts` | list of strings | *required* | Filenames of the `.slurm` scripts to monitor (must exist inside `slurm_dir`). |
| `poll_interval_seconds` | int | `120` | How often (in seconds) the orchestrator checks job statuses. |
| `max_resubmits` | int | `0` | Maximum number of times a single script can be resubmitted. `0` means unlimited. When the limit is hit, the job is automatically retired. |
| `log_file` | string | none | Path to the log file (relative to `base_dir`). If omitted, logs only go to stdout. |
| `state_file` | string | `slurm_orchestrator/state.json` | Path to the state persistence file (relative to `base_dir`). |

## How It Decides to Resubmit

On each poll cycle, for every non-retired script, the orchestrator runs through this logic:

```
1. Is the script retired?
   YES → skip
   NO  → continue

2. Has it exceeded max_resubmits?
   YES → retire it, skip
   NO  → continue

3. Does it have a job ID?
   NO  → submit it (first time, or previous sbatch failed)
   YES → continue

4. Is the job ID still in `squeue`? (PENDING / RUNNING / REQUEUED)
   YES → job is alive, do nothing
   NO  → job left the queue, resubmit it
```

When a job leaves the queue, the orchestrator queries `sacct` to log the terminal state (COMPLETED, FAILED, CANCELLED, TIMEOUT, NODE_FAIL, PREEMPTED, etc.) before resubmitting. This is purely informational — the resubmission happens regardless of the terminal state.

## Using With a Different Set of Jobs

To monitor a different set of slurm scripts, create a new config file:

```bash
cp slurm_orchestrator/config.json slurm_orchestrator/config_5demos.json
```

Edit `config_5demos.json` to point `slurm_dir` and `scripts` to the new set of jobs. Then launch with:

```bash
python slurm_orchestrator/orchestrator.py --config slurm_orchestrator/config_5demos.json
```

> Tip: use a different `state_file` in each config so they don't overwrite each other.

## State File (`state.json`)

This file is auto-generated and auto-updated. You should not need to edit it manually, but here is what it looks like for reference:

```json
{
  "book_caddy.slurm": {
    "script": "book_caddy.slurm",
    "job_id": "1234567",
    "submit_count": 3,
    "last_submit_time": "2026-03-09T14:23:01.456789",
    "last_status": "SUBMITTED",
    "retired": false
  }
}
```

If you ever need a completely fresh start, just delete the state file:

```bash
rm slurm_orchestrator/state.json
```

## Troubleshooting

**"sbatch failed" errors in the log:**
The orchestrator will keep retrying on each poll cycle. Common causes: Slurm is down, partition is paused, or the `.slurm` script path is wrong. Check `slurm_dir` in your config.

**Jobs keep getting resubmitted immediately:**
This is expected if the job fails quickly (e.g., a bug in the training script). Check the Slurm output/error logs at the path specified in your `.slurm` files (`--output` / `--error` directives) to diagnose the training failure.

**Orchestrator itself was killed unexpectedly:**
Just restart it. It reads `state.json`, checks which jobs are still alive in `squeue`, and only resubmits the missing ones. No jobs will be duplicated.

**Want to change the poll interval without restarting:**
Edit `poll_interval_seconds` in `config.json`. The change takes effect after the current sleep cycle completes (at most `old_interval` seconds). However, since the config is read at startup, you'll need to restart the orchestrator for this change to take effect.

**A job is stuck in RETIRED but you want it running again:**

```bash
python slurm_orchestrator/orchestrator.py --reset book_caddy.slurm
```
