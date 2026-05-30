"""Evaluation script for RoboCasa tasks with joint position control.

This is a variant of main.py that uses JOINT_POSITION control for the arm
instead of OSC_POSE. The key differences:
  - Environment created with JOINT_POSITION controller for the arm
  - State includes arm joint positions (23D instead of 16D)
  - Actions are 13D: [joint_targets(7), gripper(1), base(3), torso(1), mode(1)]
  - The AbsoluteActions transform in the policy server converts predicted deltas
    back to absolute joint targets; the JOINT_POSITION controller in absolute
    mode tracks those targets directly.

Usage:
    python examples/robocasa/main_joint.py \
        --env_name PrepareCoffee \
        --layout_and_style_ids '[(25,29)]' \
        --log_dir /path/to/logs \
        --host 0.0.0.0 --port 8000
"""

import collections
import dataclasses
import logging
import pathlib
import imageio
from datetime import datetime
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import json
import os
import robocasa
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter
import gymnasium as gym


# ---------------------------------------------------------------------------
# Joint-aware key converter
# ---------------------------------------------------------------------------
# Subclasses PandaOmronKeyConverter to:
#   1. Include robot0_joint_pos in observations (map_obs)
#   2. Map joint-control action dicts to robosuite format (unmap_action)
# This lets us go through the gym wrapper's standard step() path instead of
# bypassing it with raw env.step() calls.
# ---------------------------------------------------------------------------

class PandaOmronJointKeyConverter(PandaOmronKeyConverter):
    """Key converter for joint-control evaluation."""

    @classmethod
    def map_obs(cls, input_obs):
        out = super().map_obs(input_obs)
        out["body.joint_position"] = input_obs["robot0_joint_pos"]
        return out

    @classmethod
    def deduce_action_space(cls, env):
        from gymnasium import spaces
        action = {
            "hand.gripper_close": np.zeros(1),
            "body.joint_position_target": np.zeros(7),
            "body.base_motion": np.zeros(3),
            "body.torso_delta": np.zeros(1),
            "body.control_mode": np.zeros(1),
        }
        action_space = spaces.Dict()
        for k, v in action.items():
            action_space["action." + k[5:]] = spaces.Box(
                low=-10, high=10, shape=(len(v),), dtype=np.float32
            )
        return action_space

    @classmethod
    def unmap_action(cls, input_action):
        return {
            "robot0_right_gripper": (
                -1.0 if input_action["action.gripper_close"] < 0.5 else 1.0
            ),
            "robot0_right": input_action["action.joint_position_target"],
            "robot0_base": input_action["action.base_motion"],
            "robot0_torso": input_action["action.torso_delta"],
            "robot0_base_mode": (
                -1.0 if input_action["action.control_mode"] < 0.5 else 1.0
            ),
        }


def _episode_video(root: pathlib.Path, info: dict, ep_idx: int, video_key: str) -> pathlib.Path:
    """Resolve a video path using info.json chunking metadata."""
    chunks_size = info.get("chunks_size", 1000)
    episode_chunk = ep_idx // chunks_size
    rel = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    ).format(episode_chunk=episode_chunk, episode_index=ep_idx, video_key=video_key)
    return root / rel


def convert_joint_action(action):
    """Convert 13D joint-control action array to dict format.

    The model outputs absolute joint targets (AbsoluteActions adds current state
    back to the predicted deltas). These targets are sent directly to the
    JOINT_POSITION controller in absolute mode.
    """
    return {
        "action.joint_position_target": action[0:7],  # 7D absolute joint targets
        "action.gripper_close": action[7:8],           # 1D
        "action.base_motion": action[8:11],            # 3D (vx, vy, vyaw)
        "action.torso_delta": action[11:12],           # 1D
        "action.control_mode": action[12:13],          # 1D
    }


# ---------------------------------------------------------------------------
# Controller config
# ---------------------------------------------------------------------------

def get_joint_controller_config():
    """Load default PandaOmron config and switch arm to JOINT_POSITION."""
    from robosuite.controllers import load_composite_controller_config

    config = load_composite_controller_config(controller=None, robot="PandaOmron")

    # Replace OSC_POSE with JOINT_POSITION for the arm.
    # The config dict structure may be {"body_parts": {"right": ...}} or
    # {"body_parts": {"arms": {"right": ...}}} depending on robosuite version.
    body_parts = config["body_parts"]
    if "right" in body_parts:
        arm_cfg = body_parts["right"]
    else:
        arm_cfg = body_parts["arms"]["right"]

    arm_cfg["type"] = "JOINT_POSITION"
    # Use absolute mode: the policy outputs absolute joint targets, which the
    # controller tracks directly. This avoids the input scaling that delta mode
    # applies (mapping [-1,1] -> [-output_max, output_max]).
    arm_cfg["input_type"] = "absolute"
    arm_cfg["kp"] = 50
    arm_cfg["damping_ratio"] = 1
    arm_cfg["impedance_mode"] = "fixed"
    # Set output limits appropriate for 7-DOF joint-position control.
    # In absolute mode these are not used for scaling, but keeping them
    # consistent avoids shape mismatches if the controller inspects them.
    arm_cfg["output_max"] = [0.05] * 7
    arm_cfg["output_min"] = [-0.05] * 7
    # Remove OSC-specific keys that don't apply to JOINT_POSITION
    for key in ["uncouple_pos_ori", "input_ref_frame",
                "position_limits", "orientation_limits"]:
        arm_cfg.pop(key, None)

    return config


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    split: str = "pretrain"
    num_trials: int = 50
    task_set: list = None
    env_name: str | None = None
    layout_and_style_ids: list[tuple[int, int]] | None = None

    eval_init_mode: str | None = None
    dataset_path: str | None = None
    eval_pool_episode_ids: list[int] | None = None
    eval_pool_fixture_refs: dict[str, str] | None = None
    eval_pool_object_categories: list[str] | None = None
    eval_keep_robot_pose: bool = False
    eval_robot_pose_noise: float = 0.0
    eval_object_pose_noise: float = 0.0
    eval_object_ori_noise: float = 0.0
    skip_gripper_far_check: bool = False

    log_dir: str = None
    seed: int = 7


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def eval_main(args: Args) -> None:
    np.random.seed(args.seed)

    if args.env_name is None and not args.task_set:
        raise ValueError("Provide either --env-name or --task-set.")
    if args.log_dir is None:
        raise ValueError("--log-dir is required.")

    if args.skip_gripper_far_check:
        import robocasa.utils.object_utils as OU
        OU.gripper_obj_far = lambda *args, **kwargs: True
        logging.info("Patched gripper_obj_far to always return True")

    if args.env_name is not None:
        all_env_names = [args.env_name]
    else:
        all_env_names = []
        for task in args.task_set:
            all_env_names.extend(TASK_SET_REGISTRY[task])

    for env_name in all_env_names:
        eval_env(env_name, args)


def eval_env(env_name, args):
    split = args.split
    assert split in ["pretrain", "target"]
    task_horizon = get_task_horizon(env_name)
    horizon = int(task_horizon * 1.5)

    now_formatted = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_path = f"{args.log_dir}/evals/{split}/{env_name}/{now_formatted}"

    for root, dirs, files in os.walk(os.path.dirname(log_path)):
        if "stats.json" in files:
            print(f"{env_name}/{split}, stats path exists, skipping.")
            return

    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Build env kwargs (do NOT include controller_configs — see monkey-patch below)
    env_kwargs = {"seed": args.seed}
    if args.layout_and_style_ids is not None:
        env_kwargs["split"] = None
        env_kwargs["layout_and_style_ids"] = args.layout_and_style_ids
    else:
        env_kwargs["split"] = split

    # Construct eval reset controller if configured
    if args.eval_init_mode is not None:
        import sys
        dsrl_root = os.environ.get("DSRL_PI0_ROOT", os.getcwd())
        if dsrl_root not in sys.path:
            sys.path.insert(0, dsrl_root)
        from examples.robocasa_eval_reset import RoboCasaEvalResetController
        if args.dataset_path is None:
            raise ValueError("--dataset-path is required when using --eval-init-mode")
        eval_controller = RoboCasaEvalResetController(
            dataset_path=pathlib.Path(args.dataset_path),
            eval_init_mode=args.eval_init_mode,
            layout_and_style_ids=args.layout_and_style_ids,
            eval_pool_episode_ids=args.eval_pool_episode_ids,
            eval_pool_fixture_refs=args.eval_pool_fixture_refs,
            eval_pool_object_categories=args.eval_pool_object_categories,
            keep_robot_pose=args.eval_keep_robot_pose,
            robot_pose_noise=args.eval_robot_pose_noise,
            object_pose_noise=args.eval_object_pose_noise,
            object_ori_noise=args.eval_object_ori_noise,
            rng_seed=args.seed,
        )
        env_kwargs["eval_reset_controller"] = eval_controller

    # Monkey-patch load_composite_controller_config so that create_env() picks
    # up the joint-control config instead of the default OSC config.
    # This avoids the "duplicate controller_configs kwarg" error that occurs
    # when passing controller_configs through env_kwargs.
    import robocasa.utils.env_utils as _env_utils
    joint_cfg = get_joint_controller_config()
    _orig_load_controller = _env_utils.load_composite_controller_config
    _env_utils.load_composite_controller_config = lambda *a, **kw: joint_cfg
    try:
        # disable_env_checker avoids PassiveEnvChecker warnings when we
        # rebuild the observation/action spaces after swapping the converter.
        env = gym.make(f"robocasa/{env_name}", disable_env_checker=True, **env_kwargs)
    finally:
        _env_utils.load_composite_controller_config = _orig_load_controller

    # gym.make() wraps RoboCasaGymEnv in OrderEnforcing (and possibly
    # PassiveEnvChecker). Use env.unwrapped to reach the real RoboCasaGymEnv.
    gym_env = env.unwrapped          # RoboCasaGymEnv
    raw_env = gym_env.env            # raw robosuite env (e.g. PrepareCoffee)

    # Swap the key converter to our joint-aware version and rebuild the
    # observation/action spaces so they include state.joint_position and
    # the joint-control action keys.
    gym_env.key_converter = PandaOmronJointKeyConverter
    gym_env._create_obs_and_action_space()

    # Verify the arm controller is JOINT_POSITION
    robot = raw_env.robots[0]
    arm_ctrl = robot.composite_controller.part_controllers["right"]
    logging.info(f"Arm controller type: {type(arm_ctrl).__name__}, control_dim={arm_ctrl.control_dim}")
    assert arm_ctrl.control_dim == 7, f"Expected 7D arm control for JOINT_POSITION, got {arm_ctrl.control_dim}"

    # Load dataset info once for oracle video path resolution
    ds_info = None
    if args.dataset_path is not None:
        ds_info_path = pathlib.Path(args.dataset_path) / "meta" / "info.json"
        if ds_info_path.exists():
            with open(ds_info_path) as f:
                ds_info = json.load(f)

    total_episodes, total_successes = 0, 0
    task_episodes, task_successes = 0, 0

    for episode_idx in tqdm.tqdm(range(args.num_trials)):
        obs, info = env.reset()
        task_lang = obs["annotation.human.task_description"]
        action_plan = collections.deque()

        t = 0
        replay_images = []

        logging.info(f"Starting episode {task_episodes+1}...")
        while t < horizon:
            # Preprocess images
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(
                    np.ascontiguousarray(obs["video.robot0_agentview_left"]),
                    args.resize_size, args.resize_size
                )
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(
                    np.ascontiguousarray(obs["video.robot0_eye_in_hand"]),
                    args.resize_size, args.resize_size
                )
            )
            right_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(
                    np.ascontiguousarray(obs["video.robot0_agentview_right"]),
                    args.resize_size, args.resize_size
                )
            )

            if not action_plan:
                # Assemble 23D state for joint-control policy.
                # Joint positions come from the key converter's map_obs which
                # now includes "state.joint_position".
                state = np.concatenate((
                    obs["state.joint_position"],                   # 7D arm joints
                    obs["state.end_effector_position_relative"],   # 3D
                    obs["state.end_effector_rotation_relative"],   # 4D
                    obs["state.base_position"],                    # 3D
                    obs["state.base_rotation"],                    # 4D
                    obs["state.gripper_qpos"],                     # 2D
                ), axis=0)  # 23D

                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/image_right": right_img,
                    "observation/state": state,
                    "prompt": task_lang,
                }

                action_chunk = client.infer(element)["actions"]
                assert len(action_chunk) >= args.replan_steps
                action_plan.extend(action_chunk[:args.replan_steps])

            action = action_plan.popleft()
            action = convert_joint_action(action)

            # Step through the gym wrapper. PandaOmronJointKeyConverter.unmap_action()
            # maps our action dict to the robosuite composite controller format.
            obs, reward, done, truncated, info = env.step(action)
            done = info["success"]

            replay_img = env.render()
            replay_img = image_tools.convert_to_uint8(np.ascontiguousarray(replay_img))
            if t % 2 == 0 or t == horizon - 1 or done:
                replay_images.append(replay_img)
            if done:
                task_successes += 1
                total_successes += 1
                break
            t += 1

        task_episodes += 1
        total_episodes += 1

        suffix = "success" if done else "failure"
        imageio.mimwrite(
            pathlib.Path(log_path) / f"rollout_{episode_idx}_{suffix}.mp4",
            [np.asarray(x) for x in replay_images],
            fps=20,
        )

        if args.eval_init_mode is not None and "eval_reset_controller" in env_kwargs:
            ctrl = env_kwargs["eval_reset_controller"]
            reset_info = ctrl.last_reset_info
            if reset_info is not None and ds_info is not None:
                import shutil
                ep_id = reset_info["episode_id"]
                ds_path = pathlib.Path(args.dataset_path)
                oracle_src = _episode_video(
                    ds_path, ds_info, ep_id,
                    "observation.images.robot0_agentview_left",
                )
                if oracle_src.exists():
                    oracle_dst = pathlib.Path(log_path) / f"oracle_{episode_idx}_ep{ep_id}.mp4"
                    shutil.copy2(oracle_src, oracle_dst)

        logging.info(f"Success: {done}")
        logging.info(f"# episodes: {total_episodes}, successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

    logging.info(f"[{env_name}] Total success rate: {float(total_successes) / float(total_episodes)}")
    with open(os.path.join(log_path, "stats.json"), "w") as f:
        json.dump({"num_episodes": total_episodes, "success_rate": float(total_successes) / float(total_episodes)}, f, indent=4)

    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_main)
