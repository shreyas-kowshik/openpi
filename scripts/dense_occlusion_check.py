"""Dense occlusion check: generate high-density montage for specific episodes.

Usage:
    python scripts/dense_occlusion_check.py \
        --dataset /path/to/lerobot \
        --output-dir robocasa_demos_vis/dense_check \
        --episodes 34 53 66 94
"""

import argparse
import pathlib

import cv2
import numpy as np

import openpi.groot_utils.groot_openpi_dataset as groot_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--frames", type=int, default=20, help="Frames per montage")
    args = parser.parse_args()

    ds_path = pathlib.Path(args.dataset)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = groot_ds.GrootOpenpiSingleDataset(
        dataset_meta={"path": str(ds_path), "filter_key": None},
        action_horizon=50,
    )

    cols = 5
    rows = (args.frames + cols - 1) // cols

    for ep_id in args.episodes:
        # Find traj_idx for this episode
        traj_idx = list(dataset.trajectory_ids).index(ep_id)
        ep_len = dataset.trajectory_lengths[traj_idx]
        meta = groot_ds.get_ep_meta_for_episode(ds_path, ep_id)
        obj_cat = meta.get("object_cfgs", [{}])[0].get("info", {}).get("cat", "?")
        layout_id = meta.get("layout_id", "?")
        style_id = meta.get("style_id", "?")

        raw0 = dataset.get_step_data(ep_id, 0)
        prompt = str(raw0.get("annotation.human.task_description", [""])[0])

        print(f"--- Episode {ep_id} ---")
        print(f"  Length: {ep_len}, Layout: {layout_id}, Style: {style_id}, Obj: {obj_cat}")
        print(f"  Prompt: {prompt}")

        indices = np.linspace(0, ep_len - 1, args.frames, dtype=int)
        frame_list = []
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
            frame_list.append(labeled)

        while len(frame_list) < rows * cols:
            frame_list.append(np.zeros_like(frame_list[0]))

        row_imgs = []
        for r in range(rows):
            row_imgs.append(np.concatenate(frame_list[r*cols:(r+1)*cols], axis=1))
        montage = np.concatenate(row_imgs, axis=0)

        out_path = out_dir / f"ep{ep_id:04d}_l{layout_id}_s{style_id}_{obj_cat}_dense.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        print(f"  Montage: {out_path}\n")


if __name__ == "__main__":
    main()
