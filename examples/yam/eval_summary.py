"""End-of-run aggregator subscriber for BC-eval rollouts.

Reads ``SuccessLabelSubscriber.latest`` at the end of each episode, keeps a
running tally, and on ``finalize`` writes ``summary_<run_tag>_<stamp>.json``
into the rollout output directory and prints a banner.

Counters:
    n_episodes  : labeled "success" or "failure" (the success_rate denominator)
    n_success   : labeled "success"
    n_failure   : labeled "failure"
    n_aborted   : operator pressed 2 — discarded, not counted toward success rate
    n_retried   : operator pressed r — discarded; the same episode index re-ran

The running banner printed after each episode shows ``k / n_episodes`` so the
operator can read the rolling rate without leaving the terminal.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override

from success_label_subscriber import SuccessLabelSubscriber


class EvalSummarySubscriber(_subscriber.Subscriber):
    """Aggregates per-episode labels; ``finalize`` writes the summary JSON."""

    def __init__(
        self,
        label_source: SuccessLabelSubscriber,
        output_dir: str | Path,
        run_tag: str = "rollout",
        prompt: str | None = None,
    ) -> None:
        self._label = label_source
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._run_tag = run_tag
        self._prompt = prompt
        self._stamp = time.strftime("%Y%m%d_%H%M%S")
        self._n_success = 0
        self._n_failure = 0
        self._n_aborted = 0
        self._n_retried = 0
        self._start_wall_time = time.time()

    @override
    def on_episode_start(self) -> None:
        pass

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        pass

    @override
    def on_episode_end(self) -> None:
        label = self._label.latest
        if label is None:
            logging.warning(
                "EvalSummary: episode ended with no label from SuccessLabelSubscriber "
                "(check subscriber registration order)"
            )
            return
        if label.status == "success":
            self._n_success += 1
        elif label.status == "failure":
            self._n_failure += 1
        elif label.status == "aborted":
            self._n_aborted += 1
        elif label.status == "retry":
            self._n_retried += 1
        else:
            logging.warning("EvalSummary: unknown label status %r", label.status)
            return
        n_counted = self._n_success + self._n_failure
        sr = self._n_success / max(n_counted, 1)
        logging.info(
            "EvalSummary: episode %d=%s | running %d/%d (%.1f%%) abort=%d retry=%d",
            label.episode_index, label.status,
            self._n_success, max(n_counted, 1), 100.0 * sr,
            self._n_aborted, self._n_retried,
        )

    def finalize(self) -> Path:
        n_counted = self._n_success + self._n_failure
        sr = self._n_success / max(n_counted, 1)
        duration_s = time.time() - self._start_wall_time
        out = {
            "run_tag": self._run_tag,
            "stamp": self._stamp,
            "prompt": self._prompt,
            "n_episodes": n_counted,
            "n_success": self._n_success,
            "n_failure": self._n_failure,
            "n_aborted": self._n_aborted,
            "n_retried": self._n_retried,
            "success_rate": sr,
            "duration_s": duration_s,
        }
        path = self._output_dir / f"summary_{self._run_tag}_{self._stamp}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(
            f"\n=== Eval summary === {self._n_success}/{n_counted} success "
            f"({100.0 * sr:.1f}%), abort={self._n_aborted}, retry={self._n_retried}, "
            f"duration={duration_s:.1f}s\nWrote {path}\n"
        )
        logging.info("EvalSummary: wrote %s", path)
        return path
