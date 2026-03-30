"""Extract frame montages from PickPlaceCounterToStove episodes for visual inspection.

For each episode, saves a montage of sampled frames (base camera) as a PNG,
plus prints the task description and episode metadata.
"""

import json
import pathlib
import sys

import cv2
import numpy as np

import openpi.groot_utils.groot_openpi_dataset as groot_ds

DS_PATH = pathlib.Path("/home/skowshik/vla/codebase/openpi_robocasa/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToStove/20250818/lerobot")
OUT_DIR = pathlib.Path("/home/skowshik/vla/codebase/openpi/robocasa_demos_vis/inspect")

NUM_SAMPLE_FRAMES = 12  # frames per montage
START_EPISODE = 10
MAX_EPISODES = 20  # inspect up to this many


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = groot_ds.GrootOpenpiSingleDataset(
        dataset_meta={"path": str(DS_PATH), "filter_key": None},
        action_horizon=50,
    )

    num_eps = min(START_EPISODE + MAX_EPISODES, len(dataset.trajectory_ids))
    print(f"Dataset has {len(dataset.trajectory_ids)} episodes, inspecting {START_EPISODE} to {num_eps-1}")

    for traj_idx in range(START_EPISODE, num_eps):
        ep_id = dataset.trajectory_ids[traj_idx]
        ep_len = dataset.trajectory_lengths[traj_idx]
        meta = groot_ds.get_ep_meta_for_episode(DS_PATH, ep_id)
        obj_cat = meta.get("object_cfgs", [{}])[0].get("info", {}).get("cat", "?")
        layout_id = meta.get("layout_id", "?")
        style_id = meta.get("style_id", "?")

        # Get task description from first frame
        raw0 = dataset.get_step_data(ep_id, 0)
        prompt = str(raw0.get("annotation.human.task_description", [""])[0])

        print(f"\n--- Episode {ep_id} (traj_idx={traj_idx}) ---")
        print(f"  Length: {ep_len} frames")
        print(f"  Layout: {layout_id}, Style: {style_id}")
        print(f"  Object: {obj_cat}")
        print(f"  Prompt: {prompt}")

        # Sample frames evenly across the trajectory
        indices = np.linspace(0, ep_len - 1, NUM_SAMPLE_FRAMES, dtype=int)
        frames = []
        for t in indices:
            raw = dataset.get_step_data(ep_id, int(t))
            base_img = np.asarray(raw["video.robot0_agentview_left"])
            if base_img.ndim == 4:
                base_img = base_img[0]
            if np.issubdtype(base_img.dtype, np.floating):
                base_img = (base_img * 255).astype(np.uint8)
            # Add timestep label
            labeled = base_img.copy()
            cv2.putText(labeled, f"t={t}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            frames.append(labeled)

        # Create 3-row montage (4 frames per row)
        row1 = np.concatenate(frames[:4], axis=1)
        row2 = np.concatenate(frames[4:8], axis=1)
        row3 = np.concatenate(frames[8:12], axis=1)
        montage = np.concatenate([row1, row2, row3], axis=0)

        # Save as PNG (convert RGB to BGR for cv2)
        out_path = OUT_DIR / f"ep{ep_id:04d}_l{layout_id}_s{style_id}_{obj_cat}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        print(f"  Montage saved: {out_path}")


if __name__ == "__main__":
    main()
