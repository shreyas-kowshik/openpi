"""Dump .mp4 videos from a LIBERO-style .hdf5 file.

Usage:
    python scripts/dump_hdf5_video.py \
        --hdf5 /data/group_data/maxlab/common_datasets/skowshik/expert_hdf5/libero10_task0_clear_clutter_ep1.hdf5 \
        --out_dir ./videos \
        --fps 20 \
        --flip
"""

import argparse
import json
import os

import h5py
import imageio
import numpy as np


def dump_videos(hdf5_path: str, out_dir: str, fps: int, flip: bool, demo_idx: int | None):
    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(hdf5_path))[0]

    with h5py.File(hdf5_path, "r") as f:
        # Print language instruction if available.
        raw_info = f["data"].attrs.get("problem_info", "")
        try:
            info = json.loads(raw_info)
            lang = info.get("language_instruction", "")
            if isinstance(lang, list):
                lang = "".join(lang)
            print(f"Task: {lang}")
        except Exception:
            pass

        demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo")])
        if demo_idx is not None:
            demo_keys = [demo_keys[demo_idx]]

        print(f"Found {len(demo_keys)} demo(s) in {hdf5_path}")

        for demo_key in demo_keys:
            demo = f["data"][demo_key]
            agentview = demo["obs/agentview_rgb"][:]  # (T, H, W, 3)
            wrist = demo["obs/eye_in_hand_rgb"][:]    # (T, H, W, 3)

            if flip:
                agentview = agentview[:, ::-1, ::-1]
                wrist = wrist[:, ::-1, ::-1]

            T = agentview.shape[0]

            # Side-by-side: agentview | wrist
            frames = np.concatenate([agentview, wrist], axis=2)  # (T, H, 2W, 3)

            out_path = os.path.join(out_dir, f"{basename}_{demo_key}.mp4")
            writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
            for t in range(T):
                writer.append_data(frames[t])
            writer.close()
            print(f"  Saved {out_path} ({T} frames, {T/fps:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True, help="Path to .hdf5 file")
    parser.add_argument("--out_dir", default="./videos", help="Output directory")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--flip", action="store_true", help="Flip images (robosuite stores them flipped)")
    parser.add_argument("--demo_idx", type=int, default=None, help="Only dump this demo index")
    args = parser.parse_args()
    dump_videos(args.hdf5, args.out_dir, args.fps, args.flip, args.demo_idx)
