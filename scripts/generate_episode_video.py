"""Generate .mp4 for a specific episode.

Usage:
    python scripts/generate_episode_video.py \
        --dataset /path/to/lerobot \
        --episode 53 \
        --output robocasa_demos_vis/ep53_no_occlusion.mp4
"""

import argparse
import pathlib
import subprocess

import cv2
import numpy as np

import openpi.groot_utils.groot_openpi_dataset as groot_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    ds_path = pathlib.Path(args.dataset)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = groot_ds.GrootOpenpiSingleDataset(
        dataset_meta={"path": str(ds_path), "filter_key": None},
        action_horizon=50,
    )

    ep_id = args.episode
    traj_idx = list(dataset.trajectory_ids).index(ep_id)
    ep_len = dataset.trajectory_lengths[traj_idx]
    meta = groot_ds.get_ep_meta_for_episode(ds_path, ep_id)
    obj_cat = meta.get("object_cfgs", [{}])[0].get("info", {}).get("cat", "?")
    layout_id = meta.get("layout_id", "?")
    style_id = meta.get("style_id", "?")

    raw0 = dataset.get_step_data(ep_id, 0)
    prompt = str(raw0.get("annotation.human.task_description", [""])[0])

    print(f"Episode {ep_id}: length={ep_len}, layout={layout_id}, style={style_id}, obj={obj_cat}")
    print(f"Prompt: {prompt}")
    print(f"Output: {output_path}")

    # Probe first frame
    base0 = np.asarray(raw0["video.robot0_agentview_left"])
    if base0.ndim == 4:
        base0 = base0[0]
    h, w = base0.shape[0], base0.shape[1] * 2  # side-by-side
    h = h if h % 2 == 0 else h + 1
    w = w if w % 2 == 0 else w + 1

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(args.fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",
        str(output_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for t in range(ep_len):
        raw = dataset.get_step_data(ep_id, t)

        base_img = np.asarray(raw["video.robot0_agentview_left"])
        wrist_img = np.asarray(raw["video.robot0_eye_in_hand"])
        if base_img.ndim == 4:
            base_img = base_img[0]
        if wrist_img.ndim == 4:
            wrist_img = wrist_img[0]

        if np.issubdtype(base_img.dtype, np.floating):
            base_img = (base_img * 255).astype(np.uint8)
            wrist_img = (wrist_img * 255).astype(np.uint8)

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
        cv2.putText(combined, f"t={t}/{ep_len}  ep={ep_id} layout={layout_id} style={style_id} obj={obj_cat}",
                    (8, combined.shape[0] - bar_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Camera labels
        cv2.putText(combined, "base", (base_img.shape[1] // 2 - 15, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(combined, "wrist", (base_img.shape[1] + wrist_img.shape[1] // 2 - 18, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        if combined.shape[0] != h or combined.shape[1] != w:
            padded = np.zeros((h, w, 3), dtype=np.uint8)
            padded[:combined.shape[0], :combined.shape[1]] = combined
            combined = padded

        proc.stdin.write(combined.tobytes())

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        print(f"ffmpeg failed (rc={proc.returncode}): {stderr[-500:]}")
    else:
        file_size = output_path.stat().st_size / 1024
        print(f"Saved {output_path} ({ep_len} frames, {ep_len / args.fps:.1f}s, {file_size:.0f}KB)")


if __name__ == "__main__":
    main()
