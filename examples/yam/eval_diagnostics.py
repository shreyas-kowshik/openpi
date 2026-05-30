"""Lightweight YAM eval diagnostics subscriber.

Augments the eval rollout pipeline with the signals the residual training plan
calls for (``actor_version``, ``base_action``, ``a_exec_norm``,
``policy_timing``, action deltas) without touching the existing
``RolloutSaverSubscriber`` HDF5 schema.

Two granularities:

* per-query: every time the policy server is hit (detected deterministically
  from ``step_idx % action_horizon == 0`` — matches ``ActionChunkBroker``'s
  re-query schedule and is robust to Python id reuse). We record
  ``actor_version``, inference latency, the executed-window slices of
  ``base_action`` and ``a_exec_norm``, and the max-abs divergence between
  the two on the executed window (with argmax timestep/joint so a jerk root
  cause can be pinpointed without reloading the rollout).
* per-step: running max of ``|action_t - action_{t-1}|`` in raw transformed
  action space (the same units the robot executes), so a spike in jerk
  shows up in the summary.

At ``on_episode_end`` we print a compact summary and, when a
``RolloutSaverSubscriber`` is supplied, drop a JSON summary plus an NPZ with
the per-query ``base_action`` / ``a_exec_norm`` arrays next to ``episode.hdf5``.
The JSON keeps the human-readable scalars (and per-query argmax); the NPZ
keeps the arrays so a downstream notebook can plot per-joint trajectories.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


def _argmax_abs_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    """Max ``|a - b|`` over the overlap, with argmax (row, col).

    Returns ``(max_abs, argmax_t, argmax_dim)`` where ``argmax_t`` indexes the
    chunk row (0 .. action_horizon - 1) and ``argmax_dim`` indexes the
    14-d action axis. On an empty overlap returns ``(0.0, -1, -1)``.
    """
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return 0.0, -1, -1
    diff = np.abs(a[:n] - b[:n])
    flat = int(np.argmax(diff))
    t, d = divmod(flat, diff.shape[1])
    return float(diff[t, d]), int(t), int(d)


class EvalDiagnosticsSubscriber(_subscriber.Subscriber):
    """Print + dump per-query diagnostics for residual YAM evals.

    Args:
        action_horizon: steps the broker actually executes per query. Used
            to slice ``a_exec_norm`` / ``base_action`` to the executed
            window before computing ``max_abs_norm_delta_from_base``, and
            to detect query boundaries deterministically.
        saver: optional ``RolloutSaverSubscriber``. When provided, a
            ``diagnostics.json`` (+ ``diagnostics.npz`` with the per-query
            arrays) is written into the saver's per-episode directory at
            ``on_episode_end``. Without a saver, the summary prints to the
            logger only and arrays are dropped on the floor.
        keep_arrays: whether to retain per-query ``base_action`` /
            ``a_exec_norm`` slices in memory. Default True. Set False if
            the memory footprint matters (each query slice is ~1.7 KB at
            action_horizon=30, action_dim=14).
    """

    def __init__(
        self,
        action_horizon: int,
        saver: Any | None = None,
        *,
        keep_arrays: bool = True,
    ) -> None:
        self._action_horizon = int(action_horizon)
        if self._action_horizon <= 0:
            raise ValueError(f"action_horizon must be > 0, got {action_horizon!r}")
        self._saver = saver
        self._keep_arrays = bool(keep_arrays)

        self._per_query: list[dict] = []
        self._per_query_base: list[np.ndarray] = []
        self._per_query_a_exec: list[np.ndarray] = []

        self._step_idx: int = 0
        self._prev_action: np.ndarray | None = None
        self._max_abs_raw_action_delta: float = 0.0
        self._actor_versions_seen: list[int] = []
        # Saver clears its ``_episode_dir`` at the end of its own
        # ``on_episode_end``. Whether the runtime calls us before or after
        # the saver, on_step always runs while ``_episode_dir`` is live, so
        # we cache it then and consume the cache at episode end.
        self._cached_episode_dir: Path | None = None

    @override
    def on_episode_start(self) -> None:
        self._per_query = []
        self._per_query_base = []
        self._per_query_a_exec = []
        self._step_idx = 0
        self._prev_action = None
        self._max_abs_raw_action_delta = 0.0
        self._actor_versions_seen = []
        self._cached_episode_dir = None

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        # Refresh the saver path cache early — the saver populates
        # ``_episode_dir`` in its ``on_episode_start`` and clears it in
        # ``on_episode_end``; during on_step it is always live.
        if self._cached_episode_dir is None and self._saver is not None:
            ep_dir = getattr(self._saver, "_episode_dir", None)
            if ep_dir is not None:
                self._cached_episode_dir = Path(ep_dir)

        # --- Per-query detection -------------------------------------------
        # ActionChunkBroker re-queries exactly every ``action_horizon``
        # steps (broker.infer's _cur_step counter resets when it hits the
        # horizon). Using the deterministic schedule avoids relying on
        # Python id() — ids can be reused after a chunk is released, so an
        # id-based detector can silently miss a query. Step 0 is the first
        # query of the episode.
        if self._step_idx % self._action_horizon == 0:
            self._record_query(action)

        # --- Per-step running max -----------------------------------------
        raw = action.get("actions")
        if raw is not None:
            raw_arr = np.asarray(raw)
            if self._prev_action is not None and raw_arr.shape == self._prev_action.shape:
                self._max_abs_raw_action_delta = max(
                    self._max_abs_raw_action_delta,
                    float(np.abs(raw_arr - self._prev_action).max()),
                )
            self._prev_action = raw_arr.copy()

        self._step_idx += 1

    def _record_query(self, action: dict) -> None:
        base_action = action.get("base_action")
        if base_action is None:
            # Server doesn't include diagnostic fields (e.g. running against
            # a plain pi0.5 policy instead of serve_yam_residual). Skip the
            # per-query record; per-step running max still applies.
            return
        base_action = np.asarray(base_action, dtype=np.float32)
        a_exec = action.get("a_exec_norm")
        if a_exec is not None:
            a_exec = np.asarray(a_exec, dtype=np.float32)

        # Slice both chunks to the executed window. The server may pad
        # ``a_exec_norm`` to chunk_len; comparing the full chunk would
        # include rows that never run on the robot.
        H = self._action_horizon
        base_slice = base_action[:H]
        a_exec_slice = a_exec[:H] if a_exec is not None else None

        if a_exec_slice is not None:
            delta, t_argmax, d_argmax = _argmax_abs_diff(a_exec_slice, base_slice)
        else:
            delta, t_argmax, d_argmax = float("nan"), -1, -1

        actor_version = action.get("actor_version", -1)
        try:
            actor_version = int(actor_version)
        except (TypeError, ValueError):
            actor_version = -1
        self._actor_versions_seen.append(actor_version)

        timing = action.get("policy_timing") or {}
        infer_ms = timing.get("infer_ms") if isinstance(timing, dict) else None
        try:
            infer_ms = float(infer_ms) if infer_ms is not None else None
        except (TypeError, ValueError):
            infer_ms = None

        record = {
            "step": int(self._step_idx),
            "actor_version": actor_version,
            "infer_ms": infer_ms,
            "max_abs_norm_delta_from_base": float(delta),
            "argmax_delta_t": int(t_argmax),
            "argmax_delta_joint": int(d_argmax),
            "base_action_chunk_len": int(base_action.shape[0]),
            "a_exec_norm_chunk_len": int(a_exec.shape[0]) if a_exec is not None else 0,
        }
        self._per_query.append(record)
        if self._keep_arrays:
            self._per_query_base.append(np.ascontiguousarray(base_slice))
            self._per_query_a_exec.append(
                np.ascontiguousarray(a_exec_slice)
                if a_exec_slice is not None
                else np.full_like(base_slice, np.nan)
            )

    @override
    def on_episode_end(self) -> None:
        n_queries = len(self._per_query)
        if n_queries == 0:
            logging.info("EvalDiagnostics: no queries observed; nothing to summarize.")
            return

        infer_ms_vals = [q["infer_ms"] for q in self._per_query if q["infer_ms"] is not None]
        delta_vals = [
            q["max_abs_norm_delta_from_base"]
            for q in self._per_query
            if not np.isnan(q["max_abs_norm_delta_from_base"])
        ]
        versions = self._actor_versions_seen
        unique_versions = sorted(set(versions))

        # Cross-query argmax: which query had the worst |a_exec - base|, and
        # which (t, joint) within it. Without this the per-query argmax is
        # only useful by drilling into the JSON.
        worst_query_idx = -1
        worst_t = -1
        worst_joint = -1
        if delta_vals:
            worst_query_idx = int(
                max(
                    range(n_queries),
                    key=lambda i: (
                        self._per_query[i]["max_abs_norm_delta_from_base"]
                        if not np.isnan(self._per_query[i]["max_abs_norm_delta_from_base"])
                        else -1.0
                    ),
                )
            )
            worst_t = self._per_query[worst_query_idx]["argmax_delta_t"]
            worst_joint = self._per_query[worst_query_idx]["argmax_delta_joint"]

        summary = {
            "num_steps": int(self._step_idx),
            "num_queries": int(n_queries),
            "action_horizon": int(self._action_horizon),
            "actor_version_first": int(versions[0]) if versions else -1,
            "actor_version_last": int(versions[-1]) if versions else -1,
            "actor_version_unique": unique_versions,
            "actor_version_changed_mid_rollout": bool(len(unique_versions) > 1),
            "max_abs_norm_delta_from_base": (
                float(max(delta_vals)) if delta_vals else float("nan")
            ),
            "max_abs_norm_delta_query_idx": worst_query_idx,
            "max_abs_norm_delta_argmax_t": int(worst_t),
            "max_abs_norm_delta_argmax_joint": int(worst_joint),
            "mean_abs_norm_delta_from_base": (
                float(statistics.fmean(delta_vals)) if delta_vals else float("nan")
            ),
            "max_abs_raw_action_delta": float(self._max_abs_raw_action_delta),
            "infer_ms_mean": (
                float(statistics.fmean(infer_ms_vals)) if infer_ms_vals else None
            ),
            "infer_ms_median": (
                float(statistics.median(infer_ms_vals)) if infer_ms_vals else None
            ),
            "infer_ms_max": (
                float(max(infer_ms_vals)) if infer_ms_vals else None
            ),
        }

        logging.info(
            "EvalDiagnostics: steps=%d queries=%d actor_version first/last=%d/%d "
            "(unique=%s, changed=%s) | max|a_exec_norm-base|=%.3f at (q=%d, t=%d, joint=%d) "
            "mean=%.3f | max|raw_action_delta|=%.3f | infer_ms mean=%s median=%s max=%s",
            summary["num_steps"], summary["num_queries"],
            summary["actor_version_first"], summary["actor_version_last"],
            summary["actor_version_unique"], summary["actor_version_changed_mid_rollout"],
            summary["max_abs_norm_delta_from_base"],
            summary["max_abs_norm_delta_query_idx"],
            summary["max_abs_norm_delta_argmax_t"],
            summary["max_abs_norm_delta_argmax_joint"],
            summary["mean_abs_norm_delta_from_base"],
            summary["max_abs_raw_action_delta"],
            _fmt_opt(summary["infer_ms_mean"]),
            _fmt_opt(summary["infer_ms_median"]),
            _fmt_opt(summary["infer_ms_max"]),
        )

        # Quick interpretation hints. These mirror the "Expected Outcome"
        # bullets in the YAM alignment plan so the operator sees the right
        # diagnosis without re-opening the plan.
        if summary["actor_version_first"] == 0:
            logging.warning(
                "EvalDiagnostics: actor_version==0 on first query — the server is "
                "evaluating a random (untrained) residual actor."
            )
        if summary["actor_version_changed_mid_rollout"]:
            logging.warning(
                "EvalDiagnostics: actor_version changed mid-rollout — server is "
                "live-updating during eval. Freeze the checkpoint for eval if you "
                "want a reproducible per-rollout policy."
            )
        if delta_vals and max(delta_vals) > 1.0:
            logging.warning(
                "EvalDiagnostics: max|a_exec_norm - base|=%.3f exceeds 1.0 — the "
                "residual actor is dominating base in normalized action space; "
                "consider tightening the trust-region cap or alpha.",
                max(delta_vals),
            )

        episode_dir = self._resolve_episode_dir()
        if episode_dir is not None:
            try:
                self._dump_artifacts(episode_dir, summary)
            except Exception:  # noqa: BLE001
                logging.exception("EvalDiagnostics: failed to write diagnostics artifacts")

    def _resolve_episode_dir(self) -> Path | None:
        if self._cached_episode_dir is not None:
            return self._cached_episode_dir
        # Fallback: the saver may not have cleared its dir yet if we run
        # before it in subscriber order. Read it live as a last resort.
        if self._saver is None:
            return None
        ep_dir = getattr(self._saver, "_episode_dir", None)
        return Path(ep_dir) if ep_dir is not None else None

    def _dump_artifacts(self, episode_dir: Path, summary: dict) -> None:
        # The saver removes its per-episode dir on aborted/retry rollouts.
        # If that already happened (cached path points to a deleted dir),
        # skip the write rather than resurrecting the directory.
        if not episode_dir.exists():
            logging.info(
                "EvalDiagnostics: episode dir %s has been removed (likely aborted/retry); "
                "skipping diagnostics artifacts",
                episode_dir,
            )
            return
        json_out = episode_dir / "diagnostics.json"
        payload = {"summary": summary, "per_query": self._per_query}
        with json_out.open("w") as f:
            json.dump(payload, f, indent=2)
        logging.info("EvalDiagnostics: wrote %s", json_out)

        if self._keep_arrays and self._per_query_base:
            base_stack = np.stack(self._per_query_base, axis=0)        # (Q, H, A)
            a_exec_stack = np.stack(self._per_query_a_exec, axis=0)    # (Q, H, A)
            query_steps = np.asarray(
                [q["step"] for q in self._per_query], dtype=np.int64,
            )
            actor_versions = np.asarray(self._actor_versions_seen, dtype=np.int64)
            npz_out = episode_dir / "diagnostics.npz"
            np.savez_compressed(
                npz_out,
                base_action=base_stack,
                a_exec_norm=a_exec_stack,
                query_steps=query_steps,
                actor_versions=actor_versions,
            )
            logging.info("EvalDiagnostics: wrote %s", npz_out)


def _fmt_opt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"
