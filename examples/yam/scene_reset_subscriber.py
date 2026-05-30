"""Subscriber that pauses at the start of each episode for a scene reset.

Counts down from ``reset_sec`` seconds before the agent starts stepping the
env, so the operator has a window to put objects back in place after the
robot has homed + driven to its init-trajectory pose. Any keystroke during
the countdown ends the wait immediately.

The subscriber's ``on_episode_start`` runs AFTER ``YamEnvironment.reset()``
returns (per ``Runtime._run_episode``), so the robot is already at its
init-state target by the time the countdown begins.
"""

from __future__ import annotations

import logging
import select
import sys
import termios
import time
import tty

from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


def _stdin_is_tty() -> bool:
    try:
        return sys.stdin.isatty()
    except (ValueError, AttributeError):
        return False


class SceneResetSubscriber(_subscriber.Subscriber):
    """Pre-step countdown with optional key-press early-exit."""

    def __init__(self, reset_sec: float) -> None:
        self._reset_sec = float(reset_sec)

    @override
    def on_episode_start(self) -> None:
        if self._reset_sec <= 0:
            return
        deadline = time.time() + self._reset_sec
        ended_by_key = False
        # cbreak so we can poll for any keystroke without blocking.
        if _stdin_is_tty():
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        else:
            old_settings = None
        try:
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                if _stdin_is_tty():
                    ready, _, _ = select.select([sys.stdin], [], [], 0)
                    if ready:
                        try:
                            sys.stdin.read(1)
                        except Exception:  # noqa: BLE001
                            pass
                        ended_by_key = True
                        break
                print(
                    f"\r[scene reset] {remaining:5.1f}s remaining "
                    f"(press any key to start now)        ",
                    end="", flush=True,
                )
                time.sleep(min(0.2, remaining))
        finally:
            if old_settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except Exception:  # noqa: BLE001
                    logging.exception("SceneReset: failed to restore stdin settings")
        msg = "skipped (key)" if ended_by_key else "done"
        print(f"\r[scene reset] {msg}.                                                  ")

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        pass

    @override
    def on_episode_end(self) -> None:
        pass
