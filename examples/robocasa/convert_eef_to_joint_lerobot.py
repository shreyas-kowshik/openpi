"""Convert an existing RoboCasa EEF-control LeRobot dataset to joint-control format.

For each episode, this script:
1. Reads the existing parquet data (16D state, 12D EEF actions)
2. Reads states.npz to extract arm joint positions from the MuJoCo state
3. Cross-validates gripper_qpos from parquet against states.npz to locate the
   gripper qpos pair, then derives the 7 contiguous Panda arm qpos indices
   immediately before it (MuJoCo reorders joints during compilation, so the
   XML source order is not reliable)
4. Stores absolute joint target positions as actions (NOT deltas)
5. Writes a new LeRobot dataset with 23D state and 13D actions

The delta conversion (actions -= state) is handled at training time by the
DeltaActions transform in the openpi pipeline, not here.

New state layout (23D):
  [0:7]   arm joint positions (7)
  [7:10]  EEF position relative to base (3)
  [10:14] EEF rotation relative to base, quaternion (4)
  [14:17] base position (3)
  [17:21] base rotation, quaternion (4)
  [21:23] gripper qpos (2)

New action layout (13D):
  [0:7]   arm joint target positions (absolute) (7)
  [7]     gripper close (1)
  [8:11]  base motion (3)  — vx, vy, vyaw
  [11]    torso delta (1)
  [12]    control mode (1)

Usage:
    cd /home/skowshik/vla/codebase/openpi
    source .venv/bin/activate
    python examples/robocasa/convert_eef_to_joint_lerobot.py \
        --src_dataset /data/hf_cache/datasets/robocasa/v1.0/pretrain/composite/PrepareCoffee/20250716/lerobot \
        --dst_dataset /data/hf_cache/datasets/robocasa/v1.0/pretrain/composite/PrepareCoffee/20250716/lerobot_joint
"""

import argparse
import gzip
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow as pa

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


# ---- Joint index discovery via cross-validation ----
#
# MuJoCo's compiled model reorders joints relative to the source XML, so we
# cannot reliably derive per-joint qpos indices from the XML alone. Instead we
# find the gripper finger indices by cross-matching known gripper_qpos values
# from the parquet against the qpos prefix of states.npz, then derive the 7
# arm joint indices as the contiguous block immediately before the gripper.
#
# The total qpos dimension (nq) IS reliably computable from the XML because
# it depends only on the set of joints and their types, not on their compiled
# ordering. We use nq to restrict the search to the qpos portion of the state
# vector, avoiding false matches in qvel or other state segments.


def _compute_nq_from_xml(model_xml_bytes: bytes) -> int:
    """Compute total qpos dimension from a MuJoCo model XML.

    This counts joint types and sums their qpos sizes. The total nq is the same
    regardless of how MuJoCo reorders joints during compilation — only the
    per-joint ordering within qpos changes, not the total size.
    """
    root = ET.fromstring(model_xml_bytes.decode())
    nq = 0
    for joint in root.iter("joint"):
        jtype = joint.get("type", "hinge")
        nq += 7 if jtype == "free" else (4 if jtype == "ball" else 1)
    return nq


def find_arm_qpos_indices(
    states: np.ndarray,
    gripper_qpos: np.ndarray,
    nq: int,
) -> list[int]:
    """Find the 7 arm joint qpos indices by anchoring on the gripper.

    The PandaOmron robot has arm joints (7) immediately before gripper joints (2)
    in the compiled MuJoCo qpos vector. We locate the gripper by matching known
    values from the parquet against the qpos prefix of the state vector, then
    derive arm indices = [gripper_start - 7, ..., gripper_start - 1].

    The search is restricted to the first `nq` columns (the qpos portion) to
    avoid false matches in qvel or other state segments. It collects all
    candidate matches and requires exactly one.

    Args:
        states: (T, state_dim) full MuJoCo state from states.npz.
        gripper_qpos: (T, 2) known gripper finger positions from parquet.
        nq: Total qpos dimension (from XML joint counting).

    Returns:
        List of 7 qpos indices for the arm joints.

    Raises:
        ValueError: If zero or multiple gripper matches are found.
    """
    T = len(gripper_qpos)
    if states.shape[0] < T:
        raise ValueError(
            f"states.npz has {states.shape[0]} rows but parquet has {T} frames"
        )
    if nq > states.shape[1]:
        raise ValueError(
            f"nq={nq} exceeds state vector width {states.shape[1]}"
        )

    qpos = states[:T, :nq]
    candidates = []
    for i in range(qpos.shape[1] - 1):
        if (
            np.allclose(qpos[:, i], gripper_qpos[:, 0], rtol=1e-6, atol=1e-8)
            and np.allclose(qpos[:, i + 1], gripper_qpos[:, 1], rtol=1e-6, atol=1e-8)
        ):
            candidates.append(i)

    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one gripper qpos match in qpos[:{nq}], "
            f"found {len(candidates)} candidates at indices {candidates}"
        )

    gripper_start = candidates[0]
    if gripper_start < 7:
        raise ValueError(
            f"Gripper found at qpos[{gripper_start}], but need 7 arm joints "
            f"before it (requires gripper_start >= 7)."
        )

    return list(range(gripper_start - 7, gripper_start))


def load_episode_extras(extras_dir: Path, episode_idx: int) -> tuple[np.ndarray, int]:
    """Load states.npz and compute nq from model.xml.gz for one episode.

    Returns:
        Tuple of (states array, nq dimension).
    """
    ep_dir = extras_dir / f"episode_{episode_idx:06d}"
    states_path = ep_dir / "states.npz"
    xml_path = ep_dir / "model.xml.gz"
    if not states_path.exists():
        raise FileNotFoundError(f"states.npz not found at {states_path}")
    if not xml_path.exists():
        raise FileNotFoundError(f"model.xml.gz not found at {xml_path}")

    states = np.load(states_path)["states"]
    with gzip.open(xml_path, "rb") as f:
        nq = _compute_nq_from_xml(f.read())
    return states, nq


def _slice_modality(arr: np.ndarray, modality: dict, group: str, key: str) -> np.ndarray:
    """Slice a state or action array using modality.json spec."""
    spec = modality[group][key]
    return arr[:, spec["start"]:spec["end"]]


def convert_episode(
    parquet_path: Path,
    extras_dir: Path,
    episode_idx: int,
    src_modality: dict,
) -> dict:
    """Convert one episode from EEF to joint control format.

    Returns dict with:
      - state: (T, 23) joint-control state
      - actions: (T, 13) joint-control actions
      - (other columns passed through from parquet)
    """
    # Load parquet
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    T = len(df)

    # Load MuJoCo states and qpos dimension
    states_mujoco, nq = load_episode_extras(extras_dir, episode_idx)

    # Extract existing state from parquet using source modality.json
    obs_state = np.array(df["observation.state"].tolist())    # (T, 16)
    eef_pos_rel = _slice_modality(obs_state, src_modality, "state", "end_effector_position_relative")
    eef_rot_rel = _slice_modality(obs_state, src_modality, "state", "end_effector_rotation_relative")
    base_pos = _slice_modality(obs_state, src_modality, "state", "base_position")
    base_rot = _slice_modality(obs_state, src_modality, "state", "base_rotation")
    gripper_qpos = _slice_modality(obs_state, src_modality, "state", "gripper_qpos")

    # Find arm joint indices by cross-validating gripper values against the qpos
    # prefix of states.npz. The search is restricted to qpos[:nq] to avoid
    # false matches in qvel or other state segments.
    arm_indices = find_arm_qpos_indices(states_mujoco, gripper_qpos, nq)
    arm_qpos = states_mujoco[:T, arm_indices]  # (T, 7)

    # Build new 23D state
    new_state = np.concatenate([
        arm_qpos,       # 7
        eef_pos_rel,    # 3
        eef_rot_rel,    # 4
        base_pos,       # 3
        base_rot,       # 4
        gripper_qpos,   # 2
    ], axis=1)  # (T, 23)

    # Extract existing EEF actions from parquet using source modality.json
    old_actions = np.array(df["action"].tolist())  # (T, 12)
    gripper_action = _slice_modality(old_actions, src_modality, "action", "gripper_close")
    base_motion_full = _slice_modality(old_actions, src_modality, "action", "base_motion")
    control_mode = _slice_modality(old_actions, src_modality, "action", "control_mode")

    # base_motion = [vx, vy, vyaw, torso_delta] (4D in source)
    base_motion_3d = base_motion_full[:, 0:3]  # (T, 3) — vx, vy, vyaw
    torso_action = base_motion_full[:, 3:4]    # (T, 1)

    # Absolute joint target positions: target[t] = joint_pos[t+1]
    # For the last timestep, repeat the last joint position (hold position)
    arm_targets = np.concatenate([arm_qpos[1:], arm_qpos[-1:]], axis=0)  # (T, 7)

    # Build new 13D actions (absolute joint targets, NOT deltas)
    # The DeltaActions transform in the training pipeline will convert
    # actions[0:7] -= state[0:7] to get deltas at training time.
    new_actions = np.concatenate([
        arm_targets,      # 7 — absolute joint positions for next timestep
        gripper_action,   # 1
        base_motion_3d,   # 3
        torso_action,     # 1
        control_mode,     # 1
    ], axis=1)  # (T, 13)

    return {
        "state": new_state,
        "actions": new_actions,
        "df": df,
    }


def _episode_parquet(root: Path, info: dict, ep_idx: int) -> Path:
    """Resolve the parquet path for an episode using info.json chunking metadata."""
    chunks_size = info.get("chunks_size", 1000)
    episode_chunk = ep_idx // chunks_size
    rel = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    ).format(episode_chunk=episode_chunk, episode_index=ep_idx)
    return root / rel


def main():
    parser = argparse.ArgumentParser(description="Convert EEF RoboCasa dataset to joint control format")
    parser.add_argument("--src_dataset", type=str, required=True, help="Path to source LeRobot dataset")
    parser.add_argument("--dst_dataset", type=str, required=True, help="Path to destination LeRobot dataset")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if it exists")
    args = parser.parse_args()

    src = Path(args.src_dataset)
    dst = Path(args.dst_dataset)
    extras_dir = src / "extras"
    meta_dir = src / "meta"

    if dst.exists():
        if args.overwrite:
            shutil.rmtree(dst)
        else:
            raise FileExistsError(f"Destination {dst} already exists. Use --overwrite to replace.")

    # Read source info and modality config
    with open(meta_dir / "info.json") as f:
        info = json.load(f)
    with open(meta_dir / "modality.json") as f:
        src_modality = json.load(f)

    total_episodes = info["total_episodes"]
    print(f"Source dataset: {src}")
    print(f"Total episodes: {total_episodes}")
    print(f"Destination: {dst}")

    # Create destination directory structure
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(exist_ok=True)

    # Copy videos directory (symlink to save space)
    src_videos = src / "videos"
    dst_videos = dst / "videos"
    if src_videos.exists():
        if dst_videos.exists():
            dst_videos.unlink() if dst_videos.is_symlink() else shutil.rmtree(dst_videos)
        os.symlink(src_videos.resolve(), dst_videos)
        print(f"Symlinked videos: {dst_videos} -> {src_videos.resolve()}")

    # Copy extras directory (symlink)
    src_extras = src / "extras"
    dst_extras = dst / "extras"
    if src_extras.exists():
        if dst_extras.exists():
            dst_extras.unlink() if dst_extras.is_symlink() else shutil.rmtree(dst_extras)
        os.symlink(src_extras.resolve(), dst_extras)
        print(f"Symlinked extras: {dst_extras} -> {src_extras.resolve()}")

    # Process each episode
    total_frames = 0
    all_states = []
    all_actions = []

    for ep_idx in tqdm(range(total_episodes), desc="Converting episodes"):
        parquet_path = _episode_parquet(src, info, ep_idx)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Source parquet missing: {parquet_path}")

        result = convert_episode(parquet_path, extras_dir, ep_idx, src_modality)

        new_state = result["state"]
        new_actions = result["actions"]
        df = result["df"]
        T = len(df)
        total_frames += T

        # Collect stats data
        all_states.append(new_state)
        all_actions.append(new_actions)

        # Update the parquet with new state and actions
        # Build new columns
        new_columns = {}
        for col in df.columns:
            if col == "observation.state":
                new_columns[col] = [row.tolist() for row in new_state]
            elif col == "action":
                new_columns[col] = [row.tolist() for row in new_actions]
            else:
                new_columns[col] = df[col].tolist()

        new_table = pa.table(new_columns)
        dst_parquet = _episode_parquet(dst, info, ep_idx)
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_table, dst_parquet)

    # Update info.json with new dimensions
    new_info = info.copy()
    new_info["features"]["observation.state"]["shape"] = [23]
    new_info["features"]["action"]["shape"] = [13]
    new_info["total_frames"] = total_frames

    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=4)

    # Copy other meta files
    for fname in ["episodes.jsonl", "tasks.jsonl"]:
        src_file = meta_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, dst / "meta" / fname)

    # Update modality.json for joint control
    new_modality = {
        "state": {
            "joint_position": {"original_key": "observation.state", "start": 0, "end": 7},
            "end_effector_position_relative": {"original_key": "observation.state", "start": 7, "end": 10},
            "end_effector_rotation_relative": {"original_key": "observation.state", "start": 10, "end": 14},
            "base_position": {"original_key": "observation.state", "start": 14, "end": 17},
            "base_rotation": {"original_key": "observation.state", "start": 17, "end": 21},
            "gripper_qpos": {"original_key": "observation.state", "start": 21, "end": 23},
        },
        "action": {
            "joint_position_target": {"original_key": "action", "start": 0, "end": 7},
            "gripper_close": {"original_key": "action", "start": 7, "end": 8},
            "base_motion": {"original_key": "action", "start": 8, "end": 11},
            "torso_delta": {"original_key": "action", "start": 11, "end": 12},
            "control_mode": {"original_key": "action", "start": 12, "end": 13},
        },
        "video": {
            "robot0_eye_in_hand": {"original_key": "observation.images.robot0_eye_in_hand"},
            "robot0_agentview_left": {"original_key": "observation.images.robot0_agentview_left"},
            "robot0_agentview_right": {"original_key": "observation.images.robot0_agentview_right"},
        },
        "annotation": {
            "human.task_description": {"original_key": "annotation.human.task_description"},
        },
    }

    with open(dst / "meta" / "modality.json", "w") as f:
        json.dump(new_modality, f, indent=4)

    # Compute and save normalization stats
    all_states_arr = np.concatenate(all_states, axis=0)  # (N, 23)
    all_actions_arr = np.concatenate(all_actions, axis=0)  # (N, 13)

    stats = {
        "observation.state": {
            "mean": all_states_arr.mean(axis=0).tolist(),
            "std": all_states_arr.std(axis=0).tolist(),
            "min": all_states_arr.min(axis=0).tolist(),
            "max": all_states_arr.max(axis=0).tolist(),
            "q01": np.percentile(all_states_arr, 1, axis=0).tolist(),
            "q99": np.percentile(all_states_arr, 99, axis=0).tolist(),
        },
        "action": {
            "mean": all_actions_arr.mean(axis=0).tolist(),
            "std": all_actions_arr.std(axis=0).tolist(),
            "min": all_actions_arr.min(axis=0).tolist(),
            "max": all_actions_arr.max(axis=0).tolist(),
            "q01": np.percentile(all_actions_arr, 1, axis=0).tolist(),
            "q99": np.percentile(all_actions_arr, 99, axis=0).tolist(),
        },
    }

    with open(dst / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # Print summary
    print(f"\nConversion complete!")
    print(f"  Episodes: {total_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  State dim: 16 -> 23")
    print(f"  Action dim: 12 -> 13")
    print(f"  Output: {dst}")
    print(f"\nAbsolute joint target stats (actions[0:7]):")
    joint_targets = all_actions_arr[:, :7]
    print(f"  mean: {joint_targets.mean(axis=0)}")
    print(f"  std:  {joint_targets.std(axis=0)}")
    print(f"  min:  {joint_targets.min(axis=0)}")
    print(f"  max:  {joint_targets.max(axis=0)}")
    print(f"\nJoint position state stats (state[0:7]):")
    joint_state = all_states_arr[:, :7]
    print(f"  mean: {joint_state.mean(axis=0)}")
    print(f"  std:  {joint_state.std(axis=0)}")


if __name__ == "__main__":
    main()
