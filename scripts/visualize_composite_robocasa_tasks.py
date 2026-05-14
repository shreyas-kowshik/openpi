"""Visualize one demo per RoboCasa composite task as .mp4.

Dumps side-by-side (base + wrist camera) .mp4 videos for
selected composite tasks.

Usage:
    python scripts/visualize_composite_robocasa_tasks.py \
        --output-dir robocasa_vis
"""

import argparse
import logging
import pathlib
import subprocess

import cv2
import numpy as np

import openpi.groot_utils.groot_openpi_dataset as groot_ds

# Composite tasks selected for online RL (multi-step with pick-and-place, horizon <= 1200)
COMPOSITE_TASKS = [
    "CuttingToolSelection",   # 800
    "ScrubCuttingBoard",      # 800
    "RinseSinkBasin",         # 900
    "KettleBoiling",          # 1000
    "CategorizeCondiments",   # 1100
    "WashLettuce",            # 1100
    "PrepareCoffee",          # 1200
    "LoadDishwasher",         # 1200
    "PanTransfer",            # 1200
]

DATASET_BASES = [
    pathlib.Path("/data/hf_cache/datasets/robocasa/v1.0"),
    pathlib.Path("/home/skowshik/vla/codebase/openpi_robocasa/robocasa/datasets/v1.0"),
]
SPLITS = ["target", "pretrain"]
CATEGORIES = ["composite", "atomic"]


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


def find_dataset_path(task_name: str) -> pathlib.Path | None:
    """Find the first available dataset path for a task across base dirs, splits, and categories."""
    for base in DATASET_BASES:
        for split in SPLITS:
            for category in CATEGORIES:
                task_dir = base / split / category / task_name
                if not task_dir.exists():
                    continue
                for date_dir in sorted(task_dir.iterdir()):
                    lerobot_dir = date_dir / "lerobot"
                    if lerobot_dir.exists() and (lerobot_dir / "meta" / "episodes.jsonl").exists():
                        return lerobot_dir
    return None


def write_episode_video(
    dataset: groot_ds.GrootOpenpiSingleDataset,
    traj_idx: int,
    output_path: pathlib.Path,
    fps: int = 20,
):
    """Write a single episode to an H.264 mp4 file with base+wrist views side by side."""
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

    # Launch ffmpeg process
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

        base_img = np.asarray(raw["video.robot0_agentview_left"])
        wrist_img = np.asarray(raw["video.robot0_eye_in_hand"])
        if base_img.ndim == 4:
            base_img = base_img[0]
        if wrist_img.ndim == 4:
            wrist_img = wrist_img[0]

        if np.issubdtype(base_img.dtype, np.floating):
            base_img = (base_img * 255).astype(np.uint8)
            wrist_img = (wrist_img * 255).astype(np.uint8)

        prompt = str(raw.get("annotation.human.task_description", [""])[0])

        # Resize wrist to match base height if needed
        if base_img.shape[0] != wrist_img.shape[0]:
            wrist_img = cv2.resize(wrist_img, (base_img.shape[1], base_img.shape[0]))

        combined = np.concatenate([base_img, wrist_img], axis=1)

        # Add prompt overlay
        overlay = combined.copy()
        cv2.rectangle(overlay, (0, 0), (combined.shape[1], 28), (0, 0, 0), -1)
        combined = cv2.addWeighted(overlay, 0.6, combined, 0.4, 0)
        cv2.putText(combined, prompt, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Add timestep overlay
        overlay = combined.copy()
        bar_h = 40
        cv2.rectangle(overlay, (0, combined.shape[0] - bar_h), (combined.shape[1], combined.shape[0]), (0, 0, 0), -1)
        combined = cv2.addWeighted(overlay, 0.6, combined, 0.4, 0)
        cv2.putText(combined, f"t={t}/{episode_length}", (8, combined.shape[0] - bar_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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

        proc.stdin.write(combined.tobytes())

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        logging.error(f"ffmpeg failed (rc={proc.returncode}): {stderr[-500:]}")
        return False
    else:
        file_size = output_path.stat().st_size / 1024
        logging.info(f"  Saved {output_path} ({episode_length} frames, {episode_length / fps:.1f}s, {file_size:.0f}KB)")
        return True


def main():
    parser = argparse.ArgumentParser(description="Visualize one demo per RoboCasa composite task")
    parser.add_argument("--output-dir", default="robocasa_vis",
                        help="Output directory for .mp4 files")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")
    parser.add_argument("--action-horizon", type=int, default=50, help="Action horizon for dataset loading")
    args = parser.parse_args()

    init_logging()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    found = []
    missing = []

    for task_name in COMPOSITE_TASKS:
        ds_path = find_dataset_path(task_name)
        if ds_path is None:
            missing.append(task_name)
            logging.warning(f"MISSING: {task_name} - no dataset found")
            continue

        output_file = output_dir / f"{task_name}.mp4"
        if output_file.exists():
            logging.info(f"SKIP: {task_name} - already exists at {output_file}")
            found.append(task_name)
            continue

        logging.info(f"Processing: {task_name} from {ds_path}")

        try:
            dataset_meta = {
                "path": str(ds_path),
                "filter_key": None,
            }
            dataset = groot_ds.GrootOpenpiSingleDataset(
                dataset_meta=dataset_meta,
                action_horizon=args.action_horizon,
            )

            num_episodes = len(dataset.trajectory_ids)
            logging.info(f"  Dataset has {num_episodes} episodes")

            if num_episodes == 0:
                logging.warning(f"  No episodes in dataset for {task_name}")
                missing.append(task_name)
                continue

            # Get metadata for first episode
            ep_id = dataset.trajectory_ids[0]
            meta = groot_ds.get_ep_meta_for_episode(ds_path, ep_id)
            obj_cat = meta.get("object_cfgs", [{}])[0].get("info", {}).get("cat", "?")
            layout_id = meta.get("layout_id", "?")
            style_id = meta.get("style_id", "?")
            logging.info(f"  Episode {ep_id}: layout={layout_id}, style={style_id}, obj={obj_cat}")

            success = write_episode_video(dataset, 0, output_file, fps=args.fps)
            if success:
                found.append(task_name)
            else:
                missing.append(task_name)

        except Exception as e:
            logging.error(f"  Failed to process {task_name}: {e}")
            import traceback
            traceback.print_exc()
            missing.append(task_name)

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(found)}/{len(COMPOSITE_TASKS)} tasks visualized")
    print("=" * 60)
    if found:
        print(f"\nVisualized ({len(found)}):")
        for t in found:
            print(f"  {t}.mp4")
    if missing:
        print(f"\nMissing data ({len(missing)}):")
        for t in missing:
            print(f"  {t}")


if __name__ == "__main__":
    main()
