"""Convert YAM BC-eval rollouts (HDF5 + per-camera mp4) into a LeRobot dataset.

Sibling of ``convert_yam_combined_to_lerobot.py``, consuming the layout that
``openpi/examples/yam/eval_saver.py`` writes:

    <input_dir>/
        <run_tag>_<stamp>_ep0000_<μs>/
            episode.hdf5    (actions (N,14), state (N,14), attrs: prompt,
                             is_success, status, episode_index, ...)
            top.mp4, left_wrist.mp4, right_wrist.mp4
        <run_tag>_<stamp>_ep0001_<μs>/
            ...

Differences from the combined converter:

  - HDF5 ``state`` is flat ``(N, 14)`` (not nested under
    ``robot/{left,right}/{joint,gripper}_pos``).
  - Per-episode ``task`` comes from ``f.attrs['prompt']`` (the prompt the
    eval client sent to the server).
  - Per-episode ``is_success`` is encoded as an extra int64 column
    (broadcast across the episode's frames) so the trainer's loader can
    route successful eval rollouts into the success buffer with the
    success reward/mask convention, and failures into the main buffer
    only with the failure convention.
  - Episodes whose ``attrs['status']`` is not ``success`` / ``failure``
    (i.e. aborted / retry) are skipped — they have no usable trajectory.

Example:

    uv run examples/yam/convert_yam_eval_rollouts_to_lerobot.py \\
        --input-dir /path/to/eval/rollouts/pi05_yam_pickplace_a_lora/trial_1_evals \\
        --repo-id local/yam_eval_pickplace_a_trial1

Then on the trainer side:

    EVAL_ROLLOUT_REPO_ID=local/yam_eval_pickplace_a_trial1 \\
    bash examples/scripts/run_real_world_yam.sh/online_rl_runner.sh
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil

import av
import h5py
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import torch
import tqdm
import tyro


IMAGE_SIZE = 224
CAMERAS = ["top", "left_wrist", "right_wrist"]
FPS = 60


def decode_video_frames(video_path: Path, target_size: int = IMAGE_SIZE) -> np.ndarray:
    """Decode mp4 video and resize to target_size x target_size. (N,H,W,3) uint8."""
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        if img.shape[0] != target_size or img.shape[1] != target_size:
            img = np.array(Image.fromarray(img).resize((target_size, target_size), Image.BICUBIC))
        frames.append(img)
    container.close()
    return np.stack(frames)


def load_episode(ep_dir: Path) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, dict]:
    """Load one eval-rollout episode: flat state + actions + 3 mp4s + attrs."""
    with h5py.File(ep_dir / "episode.hdf5", "r") as f:
        state = np.asarray(f["state"][:], dtype=np.float32)
        actions = np.asarray(f["actions"][:], dtype=np.float32)
        attrs = {k: f.attrs[k] for k in f.attrs.keys()}
    if state.shape[1] != 14:
        raise ValueError(f"Expected (N, 14) state in {ep_dir.name}, got {state.shape}")
    if actions.shape[1] != 14:
        raise ValueError(f"Expected (N, 14) actions in {ep_dir.name}, got {actions.shape}")

    with ThreadPoolExecutor(max_workers=len(CAMERAS)) as pool:
        futures = {cam: pool.submit(decode_video_frames, ep_dir / f"{cam}.mp4") for cam in CAMERAS}
        imgs = {cam: fut.result() for cam, fut in futures.items()}

    return imgs, torch.from_numpy(state), torch.from_numpy(actions), attrs


def collect_episode_dirs(input_dir: Path) -> list[Path]:
    """Single-level dir walk: every subdir of input_dir is an episode."""
    return sorted([d for d in input_dir.iterdir() if d.is_dir() or d.is_symlink()])


def create_empty_dataset(repo_id: str, image_size: int = IMAGE_SIZE) -> LeRobotDataset:
    motors = [
        "left_joint_0", "left_joint_1", "left_joint_2",
        "left_joint_3", "left_joint_4", "left_joint_5", "left_gripper",
        "right_joint_0", "right_joint_1", "right_joint_2",
        "right_joint_3", "right_joint_4", "right_joint_5", "right_gripper",
    ]
    features: dict[str, dict] = {
        "observation.state": {"dtype": "float32", "shape": (len(motors),), "names": [motors]},
        "action": {"dtype": "float32", "shape": (len(motors),), "names": [motors]},
        # Per-frame int64 columns, broadcast across the episode.
        "orig_traj_id_6": {"dtype": "int64", "shape": (1,), "names": ["orig_traj_id_6"]},
        "is_success": {"dtype": "int64", "shape": (1,), "names": ["is_success"]},
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (3, image_size, image_size),
            "names": ["channels", "height", "width"],
        }

    target = HF_LEROBOT_HOME / repo_id
    if target.exists():
        print(f"Removing existing dataset at {target}")
        shutil.rmtree(target)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type="yam",
        features=features,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=10,
        image_writer_threads=5,
    )


def convert(
    input_dir: Path,
    repo_id: str = "local/yam_eval_rollouts",
    success_only: bool = False,
    fallback_prompt: str = "",
    num_preload_workers: int = 4,
) -> None:
    """Convert eval rollouts to LeRobot. See module docstring.

    Args:
        input_dir: Path to a dir containing one subdir per eval episode
            (the layout ``yam_eval_client.sh`` writes under
            ``eval/rollouts/<POLICY_CONFIG>/...``).
        repo_id: LeRobot dataset repo id; lives under ``HF_LEROBOT_HOME``.
        success_only: Drop failure-labeled episodes; keep only successes.
        fallback_prompt: Per-episode ``task`` to use when an episode's
            HDF5 has no ``prompt`` attr (older eval rollouts). Empty
            string skips the episode instead.
        num_preload_workers: Background episode preload concurrency.
    """
    input_dir = Path(input_dir)
    episode_dirs = collect_episode_dirs(input_dir)
    print(f"Found {len(episode_dirs)} candidate episode dirs under {input_dir}")

    dataset = create_empty_dataset(repo_id)

    n_kept = 0
    n_skipped_status = 0
    n_skipped_no_prompt = 0

    with ThreadPoolExecutor(max_workers=num_preload_workers) as pool:
        futures = [pool.submit(load_episode, ep) for ep in episode_dirs]

        for ep_dir, future in tqdm.tqdm(
            zip(episode_dirs, futures), total=len(futures), desc="Converting"
        ):
            imgs, state, action, attrs = future.result()
            status = str(attrs.get("status", ""))
            is_success = bool(attrs.get("is_success", False))

            if status not in ("success", "failure"):
                print(f"  skip {ep_dir.name} (status={status!r}, not success/failure)")
                n_skipped_status += 1
                continue
            if success_only and not is_success:
                n_skipped_status += 1
                continue

            prompt = str(attrs.get("prompt", "") or fallback_prompt)
            if not prompt:
                print(
                    f"  skip {ep_dir.name} (no prompt in attrs and no --fallback-prompt provided)"
                )
                n_skipped_no_prompt += 1
                continue

            num_frames = state.shape[0]
            for cam, frames in imgs.items():
                if frames.shape[0] != num_frames:
                    raise ValueError(
                        f"Frame count mismatch in {ep_dir.name}: {cam} has {frames.shape[0]}, "
                        f"expected {num_frames}"
                    )

            # orig_traj_id_6: use episode_index from attrs when present, else
            # a sequential counter. Padded to int64 so the loader's existing
            # ``--demo_filter_orig_traj_id_6_*`` knobs still work on this
            # dataset if the operator wants to subset.
            ep_idx = int(attrs.get("episode_index", n_kept))
            traj_id_arr = np.array([ep_idx], dtype=np.int64)
            is_success_arr = np.array([int(is_success)], dtype=np.int64)

            for i in range(num_frames):
                frame: dict[str, np.ndarray | torch.Tensor] = {
                    "observation.state": state[i],
                    "action": action[i],
                    "orig_traj_id_6": traj_id_arr,
                    "is_success": is_success_arr,
                }
                for cam in CAMERAS:
                    frame[f"observation.images.{cam}"] = imgs[cam][i]
                dataset.add_frame(frame, task=prompt)
            dataset.save_episode()
            n_kept += 1

    dataset.stop_image_writer()
    print(f"\nDataset saved to {HF_LEROBOT_HOME / repo_id}")
    print(f"  kept:                          {n_kept}")
    print(f"  skipped (status not s/f or filter): {n_skipped_status}")
    print(f"  skipped (no prompt):           {n_skipped_no_prompt}")


if __name__ == "__main__":
    tyro.cli(convert)
