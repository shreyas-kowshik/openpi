"""Run a single teleoperation episode from a list of action chunks.

Usage as a module:
    from teleop_scratch.run_episode import run_episode
    traj = run_episode(env, obs, action_chunks, ...)

Usage as a script (with a built-in example / calibration):
    python -m teleop_scratch.run_episode
"""

import os
import sys

import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from teleop_scratch.env_helper import get_state, render_obs, render_obs_raw


def run_episode(
    env,
    obs,
    action_chunks: list,
    output_dir: str = "teleop_scratch/renders",
    checkpoint_interval: int = 20,
    max_steps: int = 1020,
    render_every: int = 1,
    verbose: bool = True,
):
    """Execute action chunks in the environment and record trajectory.

    Args:
        env: LIBERO OffScreenRenderEnv (already reset)
        obs: initial observation dict (from reset_episode)
        action_chunks: list of tuples:
            (action_7d, num_repeats)
                action_7d: list/array of 7 floats [dx, dy, dz, droll, dpitch, dyaw, gripper]
                num_repeats: int — how many times to repeat this action
        output_dir: where to save checkpoint images
        checkpoint_interval: save images every N steps
        max_steps: maximum total steps before stopping
        render_every: record images every N steps for trajectory (1 = every step)
        verbose: print state at checkpoints

    Returns:
        dict with:
            images_base:  list of (H, W, 3) uint8 arrays (rotated)
            images_wrist: list of (H, W, 3) uint8 arrays (rotated)
            ee_states:    list of (6,) float32 arrays (pos + axisangle)
            gripper_states: list of (2,) float32 arrays
            actions:      list of (7,) float32 arrays
            success:      bool
            total_steps:  int
            checkpoint_paths: list of (base_path, wrist_path) saved at checkpoints
    """
    os.makedirs(output_dir, exist_ok=True)

    trajectory = {
        "images_base": [],
        "images_wrist": [],
        "ee_states": [],
        "gripper_states": [],
        "actions": [],
        "success": False,
        "total_steps": 0,
        "checkpoint_paths": [],
    }

    step_count = 0
    done = False

    # Record initial observation
    state = get_state(obs)
    base_img, wrist_img = render_obs_raw(obs)
    trajectory["images_base"].append(base_img)
    trajectory["images_wrist"].append(wrist_img)
    trajectory["ee_states"].append(
        np.concatenate([state["eef_pos"], state["eef_axisangle"]]).astype(np.float32)
    )
    trajectory["gripper_states"].append(state["gripper"].astype(np.float32))

    # Save initial checkpoint
    paths = render_obs(obs, step_idx=0, output_dir=output_dir)
    trajectory["checkpoint_paths"].append(paths)
    if verbose:
        print(f"[step 0] EEF={state['eef_pos']}  gripper={state['gripper']}")

    for chunk_idx, (action_7d, num_repeats) in enumerate(action_chunks):
        action = np.array(action_7d, dtype=np.float64)
        assert action.shape == (7,), f"Action must be 7D, got {action.shape}"

        for rep in range(num_repeats):
            if step_count >= max_steps:
                if verbose:
                    print(f"[step {step_count}] Reached max_steps={max_steps}, stopping.")
                break

            obs, reward, done, info = env.step(action.tolist())
            step_count += 1

            # Record trajectory data
            state = get_state(obs)
            trajectory["actions"].append(action.astype(np.float32).copy())

            if step_count % render_every == 0 or done:
                base_img, wrist_img = render_obs_raw(obs)
                trajectory["images_base"].append(base_img)
                trajectory["images_wrist"].append(wrist_img)
                trajectory["ee_states"].append(
                    np.concatenate([state["eef_pos"], state["eef_axisangle"]]).astype(np.float32)
                )
                trajectory["gripper_states"].append(state["gripper"].astype(np.float32))

            # Checkpoint: save images + print state
            if step_count % checkpoint_interval == 0 or done:
                paths = render_obs(obs, step_idx=step_count, output_dir=output_dir)
                trajectory["checkpoint_paths"].append(paths)
                if verbose:
                    print(
                        f"[step {step_count}] chunk={chunk_idx} rep={rep}/{num_repeats} "
                        f"EEF={state['eef_pos']}  gripper={state['gripper']}  "
                        f"done={done}"
                    )

            if done:
                trajectory["success"] = True
                if verbose:
                    print(f"[step {step_count}] SUCCESS! Task completed.")
                break

        if done or step_count >= max_steps:
            break

    # Final success check (env.check_success may differ from done flag)
    if not done:
        success = env.check_success()
        trajectory["success"] = success
        if verbose:
            print(f"\n[episode end] steps={step_count}  done={done}  check_success={success}")
    else:
        if verbose:
            print(f"\n[episode end] steps={step_count}  SUCCESS")

    trajectory["total_steps"] = step_count
    return trajectory


def run_calibration(env, obs, output_dir="teleop_scratch/renders/calibration"):
    """Run a short calibration to understand action-to-movement scale.

    Applies known deltas in x, y, z directions and reports the resulting EEF movement.
    """
    state_before = get_state(obs)
    print(f"\n=== CALIBRATION ===")
    print(f"Initial EEF pos: {state_before['eef_pos']}")

    # Test each axis
    for axis_name, action in [
        ("X (+0.05)", [0.05, 0, 0, 0, 0, 0, -1]),
        ("Y (+0.05)", [0, 0.05, 0, 0, 0, 0, -1]),
        ("Z (+0.05)", [0, 0, 0.05, 0, 0, 0, -1]),
    ]:
        # Apply 10 steps of this action
        for i in range(10):
            obs, _, _, _ = env.step(action)
        state_after = get_state(obs)
        delta = state_after["eef_pos"] - state_before["eef_pos"]
        print(f"  {axis_name} × 10 steps → delta_pos = {delta}  (total move = {np.linalg.norm(delta):.4f})")

        # Save render
        render_obs(obs, step_idx=999, output_dir=output_dir)

        # NOTE: cumulative - state_before is NOT reset between axes
        state_before = get_state(obs)


if __name__ == "__main__":
    from teleop_scratch.env_helper import create_env, reset_episode

    print("Creating environment...")
    env, task_desc, init_states = create_env(use_swap=True)

    print("Resetting episode...")
    obs = reset_episode(env, init_states, idx=0)

    print("Running calibration...")
    run_calibration(env, obs)

    print("\nDone. Closing env.")
    env.close()
