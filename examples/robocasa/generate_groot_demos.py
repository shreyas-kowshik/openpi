"""Generate WashLettuce rollout demos using a GR00T N1.5 checkpoint.

This script:
1. Creates the WashLettuce env with specific layout/style/fixture config
2. Loads the GR00T N1.5 policy directly (no wrapper needed)
3. Runs N rollout episodes
4. Saves each episode as MP4 video and collects all into a single HDF5 file
5. Writes a stats.json with success rate and episode statistics

Usage:
    conda activate groot
    export MUJOCO_GL=egl
    cd /home/skowshik/vla/codebase/Isaac-GR00T
    python /home/skowshik/vla/codebase/openpi/examples/robocasa/generate_groot_demos.py \
        --model_path /data/hf_cache/models/robocasa365_checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000 \
        --output_dir /data/hf_cache/datasets/wash_lettuce_demos \
        --n_episodes 20
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GR00T rollout demos")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/data/hf_cache/datasets/wash_lettuce_demos")
    parser.add_argument("--env_name", type=str, default="WashLettuce")
    parser.add_argument("--isaac_groot_path", type=str, default="/home/skowshik/vla/codebase/Isaac-GR00T")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--max_episode_steps", type=int, default=1500)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layout_id", type=int, default=None,
                        help="Single layout ID (use with --style_id). Overridden by --official_layouts.")
    parser.add_argument("--style_id", type=int, default=None,
                        help="Single style ID (use with --layout_id). Overridden by --official_layouts.")
    parser.add_argument("--official_layouts", action="store_true", default=True,
                        help="Use official eval layout_and_style_ids: [[1,1],[2,2],[4,4],[6,9],[7,10]] (default)")
    parser.add_argument("--obj_instance_split", type=str, default="target",
                        help="Object instance split. 'target' matches official eval 'B' split.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--denoising_steps", type=int, default=4)
    parser.add_argument("--debug", action="store_true", help="Print debug info on first step")
    parser.add_argument("--no_fixed_init", action="store_true",
                        help="Don't fix the initial state — env.reset() randomizes each episode")
    parser.add_argument("--random_layout", action="store_true",
                        help="Use random layout/style instead of fixed ones")
    return parser.parse_args()


def load_key_converter(isaac_groot_path):
    """Import PandaOmronKeyConverter from Isaac-GR00T's robocasa fork."""
    robots_init = os.path.join(
        isaac_groot_path,
        "external_dependencies", "robocasa", "robocasa", "models", "robots",
        "__init__.py",
    )
    spec = importlib.util.spec_from_file_location("groot_robocasa_robots", robots_init)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PandaOmronKeyConverter, mod.gather_robot_observations


def create_robocasa_data_config():
    """Create a data config matching the RoboCasa365 checkpoint metadata.

    The checkpoint metadata uses video keys:
      robot0_agentview_left, robot0_agentview_right, robot0_eye_in_hand
    So the data config must use: video.robot0_agentview_left, etc.
    """
    from gr00t.data.dataset import ModalityConfig
    from gr00t.data.transform.base import ComposedModalityTransform
    from gr00t.data.transform.concat import ConcatTransform
    from gr00t.data.transform.state_action import (
        StateActionToTensor,
        StateActionTransform,
    )
    from gr00t.data.transform.video import (
        VideoResize,
        VideoToNumpy,
        VideoToTensor,
    )
    from gr00t.model.transforms import GR00TTransform

    video_keys = [
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(16))

    # Modality configs
    modality_config = {
        "video": ModalityConfig(delta_indices=observation_indices, modality_keys=video_keys),
        "state": ModalityConfig(delta_indices=observation_indices, modality_keys=state_keys),
        "action": ModalityConfig(delta_indices=action_indices, modality_keys=action_keys),
        "language": ModalityConfig(delta_indices=observation_indices, modality_keys=language_keys),
    }

    # Transforms (eval mode - no augmentation)
    state_normalization_modes = {
        "state.end_effector_position_relative": "min_max",
        "state.end_effector_rotation_relative": "min_max",
        "state.gripper_qpos": "min_max",
        "state.base_position": "min_max",
        "state.base_rotation": "min_max",
    }
    state_target_rotations = {
        "state.end_effector_rotation_relative": "rotation_6d",
        "state.base_rotation": "rotation_6d",
    }
    action_normalization_modes = {
        "action.end_effector_position": "min_max",
        "action.end_effector_rotation": "min_max",
        "action.gripper_close": "binary",
        "action.base_motion": "min_max",
        "action.control_mode": "binary",
    }

    transforms = [
        VideoToTensor(apply_to=video_keys),
        VideoResize(apply_to=video_keys, height=224, width=224, interpolation="linear"),
        VideoToNumpy(apply_to=video_keys),
        StateActionToTensor(apply_to=state_keys),
        StateActionTransform(
            apply_to=state_keys,
            normalization_modes=state_normalization_modes,
            target_rotations=state_target_rotations,
        ),
        StateActionToTensor(apply_to=action_keys),
        StateActionTransform(
            apply_to=action_keys,
            normalization_modes=action_normalization_modes,
        ),
        ConcatTransform(
            video_concat_order=video_keys,
            state_concat_order=state_keys,
            action_concat_order=action_keys,
        ),
        GR00TTransform(
            state_horizon=len(observation_indices),
            action_horizon=len(action_indices),
            max_state_dim=64,
            max_action_dim=32,
        ),
    ]

    modality_transform = ComposedModalityTransform(transforms=transforms)

    return modality_config, modality_transform


def create_env(env_name, layout_and_style_ids=None, obj_instance_split="B", seed=None):
    """Create a RoboCasa env with specific layout/style config.

    Args:
        env_name: RoboCasa environment name (e.g. "ArrangeTea", "WashLettuce")
        layout_and_style_ids: List of [layout_id, style_id] pairs.
            Official eval uses [[1,1],[2,2],[4,4],[6,9],[7,10]].
            If None, env randomizes layout/style each episode.
        obj_instance_split: Object instance split. Official eval uses "B".
        seed: Random seed.
    """
    import robocasa  # noqa: F401
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    controller_configs = load_composite_controller_config(
        controller=None, robot="PandaOmron",
    )

    env_kwargs = dict(
        env_name=env_name,
        robots="PandaOmron",
        controller_configs=controller_configs,
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        camera_widths=[512, 512, 512],
        camera_heights=[512, 512, 512],
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        translucent_robot=False,
        layout_and_style_ids=layout_and_style_ids,
        layout_ids=None,
        style_ids=None,
        obj_instance_split=obj_instance_split,
        generative_textures=None,
        randomize_cameras=False,
    )

    env = robosuite.make(**env_kwargs)
    return env, env_kwargs


def pad_and_resize(img, target_size=(256, 256)):
    """Pad image to square and resize to 256x256."""
    h, w, _ = img.shape
    if h != w:
        dim = max(h, w)
        y_off = (dim - h) // 2
        x_off = (dim - w) // 2
        img = np.pad(img, ((y_off, dim - h - y_off), (x_off, dim - w - x_off), (0, 0)))
    if (img.shape[0], img.shape[1]) != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def build_policy_obs(raw_obs, env, key_converter):
    """Convert raw robosuite obs to GR00T N1.5 policy input format.

    Returns obs in NON-BATCHED format (matching gymnasium wrapper output):
      video.robot0_agentview_left: (256, 256, 3) uint8
      video.robot0_agentview_right: (256, 256, 3) uint8
      video.robot0_eye_in_hand: (256, 256, 3) uint8
      state.*: (D,) float32
      annotation.human.action.task_description: str

    The policy's get_action() handles adding batch/temporal dimensions internally.
    """
    # Flip images (robocasa renders upside-down)
    for k in list(raw_obs.keys()):
        if k.endswith("_image"):
            raw_obs[k] = np.ascontiguousarray(raw_obs[k][::-1, :, :])

    # Map state observations via key converter
    mapped_state = key_converter.map_obs(raw_obs)

    obs = {}

    # State keys: map from hand.*/body.* to state.* — 1-d arrays (D,)
    for k, v in mapped_state.items():
        if k.startswith("hand.") or k.startswith("body."):
            state_key = "state." + k[5:]
            obs[state_key] = np.array(v, dtype=np.float32)  # (D,)

    # Video keys: 3-d arrays (H, W, C)
    camera_map = {
        "robot0_agentview_left": "video.robot0_agentview_left",
        "robot0_agentview_right": "video.robot0_agentview_right",
        "robot0_eye_in_hand": "video.robot0_eye_in_hand",
    }
    for cam_name, video_key in camera_map.items():
        img = pad_and_resize(raw_obs[cam_name + "_image"])
        obs[video_key] = img  # (H, W, C)

    # Language as string
    ep_meta = env.get_ep_meta()
    obs["annotation.human.action.task_description"] = ep_meta.get("lang", "complete the task")

    return obs


def run_episode(env, policy, key_converter, max_steps, n_action_steps,
                 init_state=None, debug=False):
    """Run a single rollout and collect trajectory data.

    If init_state is provided, restore that sim state instead of randomizing.
    """
    from robosuite.controllers.composite.composite_controller import HybridMobileBase

    raw_obs = env.reset()
    if init_state is not None:
        env.sim.set_state(init_state)
        env.sim.forward()
        raw_obs = env._get_observations()
    obs = build_policy_obs(raw_obs, env, key_converter)
    ep_meta = env.get_ep_meta()

    obs_frames, raw_obs_list, action_list = [], [], []
    reward_list, done_list, states_list = [], [], []
    success = False
    total_steps = 0

    while total_steps < max_steps and not success:
        states_list.append(env.sim.get_state().flatten())

        # Snapshot raw obs for HDF5 (just copy image and state arrays)
        snap = {}
        for k, v in raw_obs.items():
            if isinstance(v, np.ndarray):
                snap[k] = np.copy(v)
        for k in list(snap.keys()):
            if k.endswith("_image"):
                snap[k] = np.ascontiguousarray(snap[k][::-1, :, :])
        raw_obs_list.append(snap)

        # Video frame for recording (concatenate all camera views at 256x256)
        frames = []
        for cam in ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]:
            key = f"video.{cam}"
            if key in obs:
                frames.append(obs[key])  # already (H, W, C) — no batch/time dims
        if frames:
            obs_frames.append(np.concatenate(frames, axis=1))

        # Get action from policy (non-batched input)
        action_dict = policy.get_action(obs)

        if debug and total_steps == 0:
            print("\n=== DEBUG: First timestep ===")
            print(f"Language: {obs.get('annotation.human.action.task_description', 'N/A')}")
            print("State keys:")
            for k, v in sorted(obs.items()):
                if k.startswith("state."):
                    print(f"  {k}: shape={np.array(v).shape}, values={np.array(v)}")
            print("Video keys:")
            for k, v in sorted(obs.items()):
                if k.startswith("video."):
                    print(f"  {k}: shape={np.array(v).shape}, dtype={np.array(v).dtype}, "
                          f"min={np.array(v).min()}, max={np.array(v).max()}")
            print("\nAction dict from policy:")
            for k, v in sorted(action_dict.items()):
                if isinstance(v, np.ndarray):
                    print(f"  {k}: shape={v.shape}, step0={v[0]}")
                else:
                    print(f"  {k}: {v}")
            print("=== END DEBUG ===\n")

        # action_dict keys: action.end_effector_position, action.end_effector_rotation,
        #   action.gripper_close, action.base_motion, action.control_mode
        # Shapes: (action_horizon, D) for each key (non-batched output)

        # Execute n_action_steps from the action horizon
        for ai in range(n_action_steps):
            step_action = {}
            for k, v in action_dict.items():
                if not k.startswith("action."):
                    continue
                if isinstance(v, np.ndarray):
                    idx = min(ai, v.shape[0] - 1)
                    step_action[k] = v[idx]
                else:
                    step_action[k] = v

            # Unmap from GR00T action format to robosuite format
            unmap = key_converter.unmap_action(step_action)

            # Build robosuite action vector
            env_action = []
            for robot in env.robots:
                cc = robot.composite_controller
                pf = robot.robot_model.naming_prefix
                action_vec = np.zeros(cc.action_limits[0].shape)
                for part_name in cc.part_controllers:
                    si, ei = cc._action_split_indexes[part_name]
                    key = f"{pf}{part_name}"
                    if key in unmap:
                        act = unmap[key]
                        if isinstance(act, (float, np.floating)):
                            act = np.array([act])
                        action_vec[si:ei] = act
                if isinstance(cc, HybridMobileBase):
                    bm_key = f"{pf}base_mode"
                    if bm_key in unmap:
                        action_vec[-1] = unmap[bm_key]
                env_action.append(action_vec)
            env_action = np.concatenate(env_action)

            if debug and total_steps == 0 and ai == 0:
                print(f"\n=== DEBUG: First env action ===")
                print(f"Step action: {step_action}")
                print(f"Unmapped: {unmap}")
                print(f"Env action vector: {env_action}")
                print(f"=== END DEBUG ===\n")

            action_list.append(env_action.copy())
            raw_obs, reward, done, _ = env.step(env_action)
            obs = build_policy_obs(raw_obs, env, key_converter)

            reward_list.append(reward)
            done_list.append(done)
            total_steps += 1

            if reward > 0:
                success = True
                break
            if total_steps >= max_steps:
                break

    return {
        "obs_frames": obs_frames,
        "raw_obs_list": raw_obs_list,
        "action_list": action_list,
        "reward_list": reward_list,
        "done_list": done_list,
        "states_list": states_list,
        "success": success,
        "ep_meta": ep_meta,
    }


def save_video(frames, filepath, fps=20):
    """Save frames as H.264 MP4."""
    if not frames:
        return
    import av

    h, w = frames[0].shape[:2]
    h -= h % 2
    w -= w % 2

    container = av.open(str(filepath), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "profile:v": "high"}

    for fd in frames:
        frame = av.VideoFrame.from_ndarray(fd[:h, :w], format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def save_hdf5(all_episodes, filepath, env_kwargs):
    """Save all episodes into a single HDF5 file."""
    with h5py.File(filepath, "w") as f:
        grp_data = f.create_group("data")
        grp_data.attrs["env_args"] = json.dumps(env_kwargs, default=str)
        grp_data.attrs["total"] = len(all_episodes)

        for ep_idx, ep in enumerate(all_episodes):
            grp = grp_data.create_group(f"demo_{ep_idx}")
            T = len(ep["action_list"])

            grp.create_dataset("actions", data=np.array(ep["action_list"], dtype=np.float64))
            grp.create_dataset("rewards", data=np.array(ep["reward_list"], dtype=np.float64))
            grp.create_dataset("dones", data=np.array(ep["done_list"], dtype=bool))

            if ep["states_list"]:
                grp.create_dataset("states", data=np.array(ep["states_list"], dtype=np.float64))

            obs_grp = grp.create_group("obs")
            n_raw = len(ep["raw_obs_list"])
            if n_raw > 0:
                sample = ep["raw_obs_list"][0]
                image_keys = [k for k in sample if k.endswith("_image")]
                non_image_keys = [
                    k for k in sample
                    if not k.endswith("_image")
                    and isinstance(sample[k], np.ndarray)
                ]
                for key in image_keys:
                    imgs = np.array([ep["raw_obs_list"][i][key] for i in range(n_raw)], dtype=np.uint8)
                    obs_grp.create_dataset(key, data=imgs, compression="gzip", compression_opts=4)
                for key in non_image_keys:
                    try:
                        vals = np.array([ep["raw_obs_list"][i][key] for i in range(n_raw)], dtype=np.float64)
                        obs_grp.create_dataset(key, data=vals)
                    except (ValueError, KeyError):
                        pass

            grp.attrs["num_samples"] = T
            grp.attrs["success"] = ep["success"]
            if ep["ep_meta"]:
                grp.attrs["ep_meta"] = json.dumps(ep["ep_meta"], default=str)


def main():
    import torch
    # Disable cuDNN — workaround for CUDNN_STATUS_NOT_INITIALIZED on this system
    torch.backends.cudnn.enabled = False

    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Determine layout_and_style_ids
    OFFICIAL_LAYOUTS = [[1, 1], [2, 2], [4, 4], [6, 9], [7, 10]]
    if args.random_layout:
        layout_and_style_ids = None
    elif args.layout_id is not None and args.style_id is not None:
        layout_and_style_ids = [[args.layout_id, args.style_id]]
    elif args.official_layouts:
        layout_and_style_ids = OFFICIAL_LAYOUTS
    else:
        layout_and_style_ids = None

    print(f"Output directory: {output_dir}")
    print(f"Model path: {args.model_path}")
    print(f"Env: {args.env_name}")
    print(f"Layout/style IDs: {layout_and_style_ids}")
    print(f"Obj instance split: {args.obj_instance_split}")
    print(f"Episodes: {args.n_episodes}, Max steps: {args.max_episode_steps}")
    print(f"Action steps: {args.n_action_steps}")

    # Create environment
    print(f"\nCreating {args.env_name} environment...")
    env, env_kwargs = create_env(
        args.env_name,
        layout_and_style_ids=layout_and_style_ids,
        obj_instance_split=args.obj_instance_split,
        seed=args.seed,
    )
    print("Environment created.")

    # Load key converter from Isaac-GR00T's robocasa fork
    PandaOmronKeyConverter, _ = load_key_converter(args.isaac_groot_path)

    # Load GR00T N1.5 policy with custom data config
    print("\nLoading GR00T N1.5 policy...")
    from gr00t.model.policy import Gr00tPolicy

    modality_config, modality_transform = create_robocasa_data_config()

    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag="new_embodiment",
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=args.denoising_steps,
        device=args.device,
    )
    print("Policy loaded.")

    # Optionally capture initial state for fixed-init rollouts
    init_state = None
    if not args.no_fixed_init:
        import copy
        print("\nCapturing initial state (single reset)...")
        env.reset()
        init_state = copy.deepcopy(env.sim.get_state())
        print("Initial state captured — will be restored for every rollout.")
    else:
        print("\nUsing randomized init state for each episode.")

    # Run episodes
    all_episodes = []
    successes = []

    for ep_idx in range(args.n_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep_idx + 1}/{args.n_episodes}")
        print(f"{'='*60}")

        t0 = time.time()
        ep_data = run_episode(
            env=env,
            policy=policy,
            key_converter=PandaOmronKeyConverter,
            max_steps=args.max_episode_steps,
            n_action_steps=args.n_action_steps,
            init_state=init_state,
            debug=args.debug and ep_idx == 0,
        )
        elapsed = time.time() - t0

        success = ep_data["success"]
        successes.append(success)
        n_steps = len(ep_data["action_list"])

        status = "SUCCESS" if success else "FAILURE"
        print(f"  {status} | Steps: {n_steps} | Time: {elapsed:.1f}s")
        print(f"  Running SR: {sum(successes)}/{len(successes)} = {np.mean(successes)*100:.1f}%")

        video_name = f"rollout_{ep_idx:02d}_{'success' if success else 'failure'}.mp4"
        save_video(ep_data["obs_frames"], video_dir / video_name)
        print(f"  Video: {video_dir / video_name}")

        all_episodes.append(ep_data)

    # Save outputs
    hdf5_path = output_dir / "demo.hdf5"
    print(f"\nSaving HDF5...")
    save_hdf5(all_episodes, hdf5_path, env_kwargs)
    print(f"HDF5: {hdf5_path}")

    stats = {
        "n_episodes": args.n_episodes,
        "success_rate": float(np.mean(successes)),
        "n_successes": int(sum(successes)),
        "n_failures": int(args.n_episodes - sum(successes)),
        "episode_results": [
            {"episode_idx": i, "success": bool(successes[i]),
             "n_steps": len(all_episodes[i]["action_list"])}
            for i in range(args.n_episodes)
        ],
        "config": vars(args),
    }
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Success rate: {stats['n_successes']}/{stats['n_episodes']} = {stats['success_rate']*100:.1f}%")
    print(f"Videos: {video_dir}")
    print(f"HDF5: {hdf5_path}")
    print(f"Stats: {stats_path}")

    env.close()


if __name__ == "__main__":
    main()
