"""Dense frame sampling for specific episodes to check object visibility."""

import json
import pathlib
import sys

import cv2
import numpy as np

import openpi.groot_utils.groot_openpi_dataset as groot_ds

DS_PATH = pathlib.Path("/home/skowshik/vla/codebase/openpi_robocasa/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToStove/20250818/lerobot")
OUT_DIR = pathlib.Path("/home/skowshik/vla/codebase/openpi/robocasa_demos_vis/inspect_dense")

# Episodes to inspect densely
TARGET_TRAJ_INDICES = [10, 24, 27]
NUM_FRAMES = 20  # sample 20 frames -> 5x4 grid


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = groot_ds.GrootOpenpiSingleDataset(
        dataset_meta={"path": str(DS_PATH), "filter_key": None},
        action_horizon=50,
    )

    for traj_idx in TARGET_TRAJ_INDICES:
        ep_id = dataset.trajectory_ids[traj_idx]
        ep_len = dataset.trajectory_lengths[traj_idx]
        meta = groot_ds.get_ep_meta_for_episode(DS_PATH, ep_id)
        obj_cat = meta.get("object_cfgs", [{}])[0].get("info", {}).get("cat", "?")
        layout_id = meta.get("layout_id", "?")
        style_id = meta.get("style_id", "?")

        raw0 = dataset.get_step_data(ep_id, 0)
        prompt = str(raw0.get("annotation.human.task_description", [""])[0])

        print(f"\n--- Episode {ep_id} (traj_idx={traj_idx}) ---")
        print(f"  Length: {ep_len}, Layout: {layout_id}, Style: {style_id}, Object: {obj_cat}")
        print(f"  Prompt: {prompt}")

        indices = np.linspace(0, ep_len - 1, NUM_FRAMES, dtype=int)
        frames = []
        for t in indices:
            raw = dataset.get_step_data(ep_id, int(t))
            img = np.asarray(raw["video.robot0_agentview_left"])
            if img.ndim == 4:
                img = img[0]
            if np.issubdtype(img.dtype, np.floating):
                img = (img * 255).astype(np.uint8)
            labeled = img.copy()
            cv2.putText(labeled, f"t={t}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            frames.append(labeled)

        # 5 rows x 4 cols
        rows = []
        for r in range(5):
            row_frames = frames[r*4:(r+1)*4]
            rows.append(np.concatenate(row_frames, axis=1))
        montage = np.concatenate(rows, axis=0)

        out_path = OUT_DIR / f"ep{ep_id:04d}_l{layout_id}_s{style_id}_{obj_cat}_dense.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
