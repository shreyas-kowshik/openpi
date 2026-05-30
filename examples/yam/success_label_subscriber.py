"""Subscriber that lets the operator terminate AND label a rollout at any point.

The operator can press a label key either:
  - MID-rollout (during ``on_step``): the subscriber sets a flag that
    ``YamEnvironment.is_episode_complete`` reads, so the Runtime exits the
    episode loop on the next iteration. The end-of-rollout prompt is skipped
    and the captured key becomes the label.
  - END of rollout (during ``on_episode_end``): the subscriber blocks on
    stdin for up to ``timeout_sec`` seconds before defaulting to ``failure``.

Keys:

    1 -> success    (count toward N, save HDF5 with is_success=True)
    0 -> failure    (count toward N, save HDF5 with is_success=False)
    2 -> aborted    (discard rollout; do NOT count toward N; advance to next ep)
    r -> retry      (discard rollout; do NOT count; re-run the SAME episode idx)

When stdin is not a TTY (headless / pipe), mid-rollout polling is a no-op
and the end-of-rollout label defaults to ``failure`` immediately so the eval
client doesn't hang.

The latest label is exposed on ``self.latest`` (an ``EpisodeLabel`` dataclass)
so downstream subscribers (``RolloutSaverSubscriber`` for HDF5 attrs,
``EvalSummarySubscriber`` for aggregation) and the runner loop can branch on
it after ``on_episode_end`` returns.

The episode index advances on ``success`` / ``failure`` / ``aborted`` and is
held back on ``retry`` (see ``reset_for_retry``). The runner is responsible
for invoking ``reset_for_retry`` and re-running ``Runtime._run_episode``.
"""

from __future__ import annotations

import logging
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass

from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


VALID_STATUSES = ("success", "failure", "aborted", "retry")


@dataclass
class EpisodeLabel:
    status: str
    is_success: bool
    episode_index: int
    labeled_at_ns: int


def _stdin_is_tty() -> bool:
    try:
        return sys.stdin.isatty()
    except (ValueError, AttributeError):
        return False


_MID_ROLLOUT_KEY_MAP = {
    "1": "success",
    "0": "failure",
    "2": "aborted",
    "r": "retry",
    "R": "retry",
}


class SuccessLabelSubscriber(_subscriber.Subscriber):
    """Mid-rollout polling + end-of-rollout prompting for 1/0/2/r labels.

    During each step (``on_step``) the subscriber polls stdin non-blockingly
    for a label key. If one is pressed:
      - ``self._pending_status`` records the label
      - ``self.terminate_episode`` flips to True; ``YamEnvironment.is_episode_complete``
        reads this flag and ends the Runtime's per-episode loop on the next iter
      - ``on_episode_end`` then uses the pending status directly, skipping the
        end-of-rollout prompt

    Stdin is put in cbreak mode for the duration of the episode
    (``on_episode_start`` -> ``on_episode_end``) so the per-step polling
    sees single-byte reads. Between episodes, stdin returns to normal so the
    operator can type freely while resetting the scene.
    """

    def __init__(self, timeout_sec: float = 300.0) -> None:
        self._timeout = float(timeout_sec)
        self._episode_index = -1
        self.latest: EpisodeLabel | None = None
        # Mid-rollout state. Reset on each on_episode_start.
        self._pending_status: str | None = None
        self.terminate_episode: bool = False
        self._stdin_old_settings = None

    @override
    def on_episode_start(self) -> None:
        self._episode_index += 1
        self.latest = None
        self._pending_status = None
        self.terminate_episode = False
        if _stdin_is_tty():
            self._stdin_old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        else:
            self._stdin_old_settings = None

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        # Already terminating — don't double-poll; the runtime will exit on
        # the next is_episode_complete check.
        if self._pending_status is not None:
            return
        if not _stdin_is_tty():
            return
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return
        ch = sys.stdin.read(1)
        status = _MID_ROLLOUT_KEY_MAP.get(ch)
        if status is None:
            return
        self._pending_status = status
        self.terminate_episode = True
        logging.info(
            "SuccessLabel: mid-rollout key '%s' -> %s; ending episode %d",
            ch, status, self._episode_index,
        )

    @override
    def on_episode_end(self) -> None:
        # Restore stdin BEFORE prompting so the prompt path is the original
        # _prompt() implementation (which manages its own cbreak window).
        if self._stdin_old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._stdin_old_settings)
            except Exception:  # noqa: BLE001
                logging.exception("SuccessLabel: failed to restore stdin settings")
            self._stdin_old_settings = None

        if self._pending_status is not None:
            status = self._pending_status
        else:
            status = self._prompt()

        self.latest = EpisodeLabel(
            status=status,
            is_success=(status == "success"),
            episode_index=self._episode_index,
            labeled_at_ns=time.time_ns(),
        )
        logging.info("SuccessLabel: episode %d -> %s", self._episode_index, status)

    def reset_for_retry(self) -> None:
        """Roll the episode index back by one so the next ``on_episode_start``
        re-uses the same index. Used by the runner after a ``retry`` label.
        """
        self._episode_index -= 1

    def _prompt(self) -> str:
        if not _stdin_is_tty():
            logging.warning(
                "SuccessLabel: stdin is not a TTY; defaulting episode %d to failure",
                self._episode_index,
            )
            return "failure"
        print(
            "\nEpisode %d finished. Label: (1) Success  (0) Failure  (2) Abort  (r) Retry"
            "  [waits up to %ds, default failure]:"
            % (self._episode_index, int(self._timeout))
        )
        deadline = time.time() + self._timeout
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while time.time() < deadline:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue
                ch = sys.stdin.read(1)
                if ch == "1":
                    return "success"
                if ch == "0":
                    return "failure"
                if ch == "2":
                    return "aborted"
                if ch in ("r", "R"):
                    return "retry"
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        logging.warning(
            "SuccessLabel: no label received within %.1fs; defaulting episode %d to failure",
            self._timeout, self._episode_index,
        )
        return "failure"
