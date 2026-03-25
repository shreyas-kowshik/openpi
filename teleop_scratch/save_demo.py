"""Save a collected trajectory as HDF5 (LiberoProHDF5 format) and MP4 video.

HDF5 layout (matching LiberoProHDF5Dataset expectations):
    data/
        attrs: problem_info  (JSON string with language_instruction)
        demo_0/
            obs/agentview_rgb       (T, 256, 256, 3)  uint8
            obs/eye_in_hand_rgb     (T, 256, 256, 3)  uint8
            obs/ee_states           (T, 6)            float32
            obs/gripper_states      (T, 2)            float32
            actions                 (T, 7)            float32

Note: Images are stored in the "raw" (flipped) convention — i.e. the 180°-rotated
versions are stored, matching how LIBERO-PRO HDF5 files store them. The
LiberoProHDF5Dataset then flips them back when loading.

Usage:
    from teleop_scratch.save_demo import save_demo
    save_demo(trajectory, demo_idx=0, task_description="put both moka pots on the stove")
"""

import json
import os
import pathlib

import h5py
import imageio
import numpy as np


def save_demo(
    trajectory: dict,
    demo_idx: int = 0,
    task_description: str = "put both moka pots on the stove",
    output_dir: str = "dummy_task_data",
    hdf5_filename: str | None = None,
):
    """Save a single demonstration trajectory to HDF5 and MP4.

    Args:
        trajectory: dict from run_episode containing:
            images_base:     list of (H, W, 3) uint8 arrays
            images_wrist:    list of (H, W, 3) uint8 arrays
            ee_states:       list of (6,) float32 arrays
            gripper_states:  list of (2,) float32 arrays
            actions:         list of (7,) float32 arrays
        demo_idx: index for naming (demo_0, demo_1, ...)
        task_description: language instruction string
        output_dir: root directory for output files
        hdf5_filename: optional custom filename; defaults to demo_{idx}.hdf5

    Returns:
        (hdf5_path, video_path): paths to saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)

    # Convert lists to arrays
    images_base = np.stack(trajectory["images_base"])      # (T+1, H, W, 3)
    images_wrist = np.stack(trajectory["images_wrist"])     # (T+1, H, W, 3)
    ee_states = np.stack(trajectory["ee_states"])           # (T+1, 6)
    gripper_states = np.stack(trajectory["gripper_states"]) # (T+1, 2)
    actions = np.stack(trajectory["actions"])               # (T, 7)

    # Align lengths: obs has T+1 entries (initial + after each step),
    # actions has T entries. We store T timesteps with obs[0:T] and actions[0:T].
    T = actions.shape[0]
    images_base = images_base[:T]
    images_wrist = images_wrist[:T]
    ee_states = ee_states[:T]
    gripper_states = gripper_states[:T]

    print(f"[save_demo] Saving demo_{demo_idx}: {T} timesteps")
    print(f"  images_base:    {images_base.shape}")
    print(f"  images_wrist:   {images_wrist.shape}")
    print(f"  ee_states:      {ee_states.shape}")
    print(f"  gripper_states: {gripper_states.shape}")
    print(f"  actions:        {actions.shape}")

    # --- Save HDF5 ---
    if hdf5_filename is None:
        hdf5_filename = f"demo_{demo_idx}.hdf5"
    hdf5_path = os.path.join(output_dir, hdf5_filename)

    with h5py.File(hdf5_path, "w") as f:
        # Root-level attributes
        data_grp = f.create_group("data")
        problem_info = json.dumps({"language_instruction": task_description})
        data_grp.attrs["problem_info"] = problem_info

        # Demo group
        demo_key = f"demo_{demo_idx}"
        demo_grp = data_grp.create_group(demo_key)

        # Observations
        obs_grp = demo_grp.create_group("obs")
        obs_grp.create_dataset("agentview_rgb", data=images_base, dtype=np.uint8)
        obs_grp.create_dataset("eye_in_hand_rgb", data=images_wrist, dtype=np.uint8)
        obs_grp.create_dataset("ee_states", data=ee_states, dtype=np.float32)
        obs_grp.create_dataset("gripper_states", data=gripper_states, dtype=np.float32)

        # Actions
        demo_grp.create_dataset("actions", data=actions, dtype=np.float32)

    print(f"  HDF5 saved: {hdf5_path}")

    # --- Save MP4 video ---
    video_path = os.path.join(output_dir, "videos", f"demo_{demo_idx}.mp4")
    _render_video(images_base, images_wrist, video_path, fps=10)
    print(f"  Video saved: {video_path}")

    return hdf5_path, video_path


def save_demos_combined(
    trajectories: list[dict],
    task_description: str = "put both moka pots on the stove",
    output_dir: str = "dummy_task_data",
    hdf5_filename: str = "demos_combined.hdf5",
):
    """Save multiple demo trajectories into a single HDF5 file.

    This is useful for creating a single file that LiberoProHDF5Dataset can load
    with multiple episodes.

    Args:
        trajectories: list of trajectory dicts from run_episode
        task_description: language instruction string
        output_dir: root directory
        hdf5_filename: output filename

    Returns:
        hdf5_path: path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
    hdf5_path = os.path.join(output_dir, hdf5_filename)

    with h5py.File(hdf5_path, "w") as f:
        data_grp = f.create_group("data")
        problem_info = json.dumps({"language_instruction": task_description})
        data_grp.attrs["problem_info"] = problem_info

        for demo_idx, traj in enumerate(trajectories):
            images_base = np.stack(traj["images_base"])
            images_wrist = np.stack(traj["images_wrist"])
            ee_states = np.stack(traj["ee_states"])
            gripper_states = np.stack(traj["gripper_states"])
            actions = np.stack(traj["actions"])

            T = actions.shape[0]
            images_base = images_base[:T]
            images_wrist = images_wrist[:T]
            ee_states = ee_states[:T]
            gripper_states = gripper_states[:T]

            demo_key = f"demo_{demo_idx}"
            demo_grp = data_grp.create_group(demo_key)

            obs_grp = demo_grp.create_group("obs")
            obs_grp.create_dataset("agentview_rgb", data=images_base, dtype=np.uint8)
            obs_grp.create_dataset("eye_in_hand_rgb", data=images_wrist, dtype=np.uint8)
            obs_grp.create_dataset("ee_states", data=ee_states, dtype=np.float32)
            obs_grp.create_dataset("gripper_states", data=gripper_states, dtype=np.float32)
            demo_grp.create_dataset("actions", data=actions, dtype=np.float32)

            print(f"[save_demos_combined] demo_{demo_idx}: {T} timesteps")

            # Also save individual video
            video_path = os.path.join(output_dir, "videos", f"demo_{demo_idx}.mp4")
            _render_video(images_base, images_wrist, video_path, fps=10)

    print(f"[save_demos_combined] Saved {len(trajectories)} demos to {hdf5_path}")
    return hdf5_path


def _render_video(images_base, images_wrist, video_path, fps=10):
    """Render side-by-side base + wrist video."""
    frames = []
    for i in range(len(images_base)):
        # Side-by-side: base on left, wrist on right
        frame = np.concatenate([images_base[i], images_wrist[i]], axis=1)
        frames.append(frame)

    imageio.mimwrite(video_path, frames, fps=fps)


def verify_hdf5(hdf5_path: str):
    """Print the structure and shapes of an HDF5 file for verification."""
    print(f"\n=== Verifying {hdf5_path} ===")
    with h5py.File(hdf5_path, "r") as f:
        problem_info = f["data"].attrs.get("problem_info", "")
        print(f"problem_info: {problem_info}")

        for demo_key in sorted(f["data"].keys()):
            demo = f["data"][demo_key]
            print(f"\n  {demo_key}:")
            print(f"    actions:             {demo['actions'].shape} {demo['actions'].dtype}")
            print(f"    obs/agentview_rgb:   {demo['obs/agentview_rgb'].shape} {demo['obs/agentview_rgb'].dtype}")
            print(f"    obs/eye_in_hand_rgb: {demo['obs/eye_in_hand_rgb'].shape} {demo['obs/eye_in_hand_rgb'].dtype}")
            print(f"    obs/ee_states:       {demo['obs/ee_states'].shape} {demo['obs/ee_states'].dtype}")
            print(f"    obs/gripper_states:  {demo['obs/gripper_states'].shape} {demo['obs/gripper_states'].dtype}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        verify_hdf5(sys.argv[1])
    else:
        print("Usage: python -m teleop_scratch.save_demo <path_to_hdf5>")
        print("  Verifies the structure of an HDF5 demo file.")
