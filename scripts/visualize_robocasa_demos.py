"""Visualize RoboCasa demo episodes as .mp4 videos.

Loads a training config, builds the dataset with scene/object filtering,
and writes one .mp4 per episode showing base and wrist camera views
side-by-side with action overlay.

Usage:
    # Visualize first 3 episodes from a config
    python scripts/visualize_robocasa_demos.py \
        --config-name pi05_robocasa_single_task_lora_fresh_debug_v1 \
        --output-dir vis_demos/ \
        --num-episodes 3

    # Visualize a specific episode
    python scripts/visualize_robocasa_demos.py \
        --config-name pi05_robocasa_single_task_lora_fresh_debug_v1 \
        --output-dir vis_demos/ \
        --episode-idx 0
"""

import argparse
import json
import logging
import pathlib
import subprocess
import tempfile

import cv2
import numpy as np

import openpi.groot_utils.groot_openpi_dataset as groot_ds
import openpi.training.config as _config


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(formatter)


def get_enriched_data_dirs(config: _config.TrainConfig) -> list[dict]:
    """Extract enriched data_dirs from the training config (with scene filtering params)."""
    data_factory = config.data
    data_dirs = data_factory.data_dirs
    if data_dirs is None:
        raise ValueError("Config does not have data_dirs (not a RoboCasa config?)")

    enriched = []
    for d in data_dirs:
        d_copy = dict(d)
        if getattr(data_factory, "layout_and_style_ids", None) is not None:
            d_copy["layout_and_style_ids"] = data_factory.layout_and_style_ids
        if getattr(data_factory, "num_demos", None) is not None:
            d_copy["num_demos"] = data_factory.num_demos
        if getattr(data_factory, "obj_category", None) is not None:
            d_copy["obj_category"] = data_factory.obj_category
        if getattr(data_factory, "fixture_refs", None) is not None:
            d_copy["fixture_refs"] = data_factory.fixture_refs
        if getattr(data_factory, "match_episode_id", None) is not None:
            d_copy["match_episode_id"] = data_factory.match_episode_id
        enriched.append(d_copy)
    return enriched


def render_action_text(frame: np.ndarray, actions: np.ndarray, timestep: int, total: int) -> np.ndarray:
    """Overlay action values and timestep info on the frame."""
    h, w = frame.shape[:2]
    # Semi-transparent overlay bar at bottom
    overlay = frame.copy()
    bar_h = 60
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Timestep counter
    cv2.putText(frame, f"t={timestep}/{total}", (8, h - bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Action values (first 7 most important dims: ee_pos(3), ee_rot(3), gripper(1))
    act_strs = [f"{actions[i]:.3f}" for i in range(min(7, len(actions)))]
    act_line = "act: " + " ".join(act_strs)
    cv2.putText(frame, act_line, (8, h - bar_h + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # Dim labels
    labels = "      ee_x   ee_y   ee_z   r_x    r_y    r_z    grip"
    cv2.putText(frame, labels, (8, h - bar_h + 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)

    return frame


def render_prompt(frame: np.ndarray, prompt: str) -> np.ndarray:
    """Overlay the task prompt at the top of the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.putText(frame, prompt, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def write_episode_video(
    dataset: groot_ds.GrootOpenpiSingleDataset,
    traj_idx: int,
    output_path: pathlib.Path,
    fps: int = 20,
):
    """Write a single episode to an H.264 mp4 file with base+wrist views side by side.

    Uses ffmpeg to encode raw frames into a widely-compatible H.264 mp4.
    """
    episode_id = dataset.trajectory_ids[traj_idx]
    episode_length = dataset.trajectory_lengths[traj_idx]

    logging.info(f"Writing episode {episode_id} ({episode_length} frames) -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Probe first frame to get dimensions
    raw0 = dataset.get_step_data(episode_id, 0)
    base0 = np.asarray(raw0["video.robot0_agentview_left"])
    if base0.ndim == 4:
        base0 = base0[0]
    h, w = base0.shape[0], base0.shape[1] * 2  # side-by-side

    # Ensure even dimensions (H.264 requirement)
    h = h if h % 2 == 0 else h + 1
    w = w if w % 2 == 0 else w + 1

    # Launch ffmpeg process: read raw RGB frames from stdin, write H.264 mp4
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",
        str(output_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for t in range(episode_length):
        raw = dataset.get_step_data(episode_id, t)

        # Extract images: raw video data is (1, H, W, 3) from delta_indices=[0]
        base_img = np.asarray(raw["video.robot0_agentview_left"])
        wrist_img = np.asarray(raw["video.robot0_eye_in_hand"])
        if base_img.ndim == 4:
            base_img = base_img[0]
        if wrist_img.ndim == 4:
            wrist_img = wrist_img[0]

        # Ensure uint8
        if np.issubdtype(base_img.dtype, np.floating):
            base_img = (base_img * 255).astype(np.uint8)
            wrist_img = (wrist_img * 255).astype(np.uint8)

        # Extract actions (first timestep of chunk)
        actions_raw = np.concatenate([
            np.asarray(raw["action.end_effector_position"]),
            np.asarray(raw["action.end_effector_rotation"]),
            np.asarray(raw["action.gripper_close"]),
            np.asarray(raw["action.base_motion"]),
            np.asarray(raw["action.control_mode"]),
        ], axis=-1)
        if actions_raw.ndim == 2:
            actions_raw = actions_raw[0]

        # Extract prompt
        prompt = str(raw.get("annotation.human.task_description", [""])[0])

        # Resize wrist to match base height if needed
        if base_img.shape[0] != wrist_img.shape[0]:
            wrist_img = cv2.resize(wrist_img, (base_img.shape[1], base_img.shape[0]))

        # Concatenate side by side
        combined = np.concatenate([base_img, wrist_img], axis=1)

        # Add overlays (these work in RGB via cv2 drawing)
        combined = render_prompt(combined, prompt)
        combined = render_action_text(combined, actions_raw, t, episode_length)

        # Add camera labels
        cv2.putText(combined, "base", (base_img.shape[1] // 2 - 15, 48),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(combined, "wrist", (base_img.shape[1] + wrist_img.shape[1] // 2 - 18, 48),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Pad to even dimensions if needed
        if combined.shape[0] != h or combined.shape[1] != w:
            padded = np.zeros((h, w, 3), dtype=np.uint8)
            padded[:combined.shape[0], :combined.shape[1]] = combined
            combined = padded

        # Write raw RGB frame to ffmpeg stdin
        proc.stdin.write(combined.tobytes())

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        logging.error(f"ffmpeg failed (rc={proc.returncode}): {stderr[-500:]}")
    else:
        file_size = output_path.stat().st_size / 1024
        logging.info(f"  Saved {output_path} ({episode_length} frames, {episode_length / fps:.1f}s, {file_size:.0f}KB)")


def main():
    parser = argparse.ArgumentParser(description="Visualize RoboCasa demo episodes as .mp4 videos")
    parser.add_argument("--config-name", required=True, help="Training config name")
    parser.add_argument("--output-dir", default="vis_demos", help="Output directory for .mp4 files")
    parser.add_argument("--num-episodes", type=int, default=3, help="Number of episodes to visualize")
    parser.add_argument("--episode-idx", type=int, default=None,
                        help="Specific episode index within the filtered set (0-based). Overrides --num-episodes.")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS (default 20, matches RoboCasa control freq)")
    args = parser.parse_args()

    init_logging()

    config = _config.get_config(args.config_name)
    enriched_data_dirs = get_enriched_data_dirs(config)
    action_horizon = config.model.action_horizon

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dir_idx, ds_meta in enumerate(enriched_data_dirs):
        ds_path = pathlib.Path(ds_meta["path"])
        logging.info(f"Loading dataset from {ds_path}")

        dataset = groot_ds.GrootOpenpiSingleDataset(
            dataset_meta=ds_meta,
            action_horizon=action_horizon,
        )

        num_episodes = len(dataset.trajectory_ids)
        logging.info(f"Dataset has {num_episodes} episodes, {len(dataset)} total samples")

        # Log episode metadata
        for i in range(min(5, num_episodes)):
            ep_id = dataset.trajectory_ids[i]
            ep_len = dataset.trajectory_lengths[i]
            meta = groot_ds.get_ep_meta_for_episode(ds_path, ep_id)
            obj_cat = meta.get("object_cfgs", [{}])[0].get("info", {}).get("cat", "?")
            logging.info(
                f"  Episode {ep_id}: {ep_len} frames, "
                f"layout={meta.get('layout_id')}, style={meta.get('style_id')}, obj={obj_cat}"
            )
        if num_episodes > 5:
            logging.info(f"  ... and {num_episodes - 5} more episodes")

        # Determine which episodes to visualize
        if args.episode_idx is not None:
            if args.episode_idx >= num_episodes:
                raise ValueError(f"--episode-idx {args.episode_idx} out of range (dataset has {num_episodes} episodes)")
            traj_indices = [args.episode_idx]
        else:
            traj_indices = list(range(min(args.num_episodes, num_episodes)))

        for traj_idx in traj_indices:
            ep_id = dataset.trajectory_ids[traj_idx]
            meta = groot_ds.get_ep_meta_for_episode(ds_path, ep_id)
            obj_cat = meta.get("object_cfgs", [{}])[0].get("info", {}).get("cat", "unknown")
            layout_id = meta.get("layout_id", "?")
            style_id = meta.get("style_id", "?")

            filename = f"ep{ep_id:04d}_l{layout_id}_s{style_id}_{obj_cat}.mp4"
            if len(enriched_data_dirs) > 1:
                filename = f"ds{dir_idx}_{filename}"

            write_episode_video(dataset, traj_idx, output_dir / filename, fps=args.fps)

    logging.info(f"Done. Videos saved to {output_dir}/")


if __name__ == "__main__":
    main()
