"""Environment helper for VLM-style teleoperation of LIBERO tasks.

Handles BDDL file reading, swap perturbation, env creation, reset,
rendering camera images, and proprioceptive state extraction.
"""

import math
import os
import pathlib
import tempfile

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
# Use LIBERO-PRO from dsrl_pi0 (has robosuite installed in dsrl_pi0 conda env)
_LIBERO_PRO_ROOT = pathlib.Path("/home/skowshik/vla/codebase/dsrl_pi0/LIBERO-PRO")

_BDDL_DIR = _LIBERO_PRO_ROOT / "libero" / "libero" / "bddl_files"
_INIT_DIR = _LIBERO_PRO_ROOT / "libero" / "libero" / "init_files"

_TASK_SUITE = "libero_10"
_TASK_SUITE_SWAP = "libero_10_temp"  # Pre-generated swapped BDDL + init states
_TASK_NAME = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
_BDDL_FILE = _BDDL_DIR / _TASK_SUITE / f"{_TASK_NAME}.bddl"
_BDDL_FILE_SWAP = _BDDL_DIR / _TASK_SUITE_SWAP / f"{_TASK_NAME}.bddl"
_INIT_FILE = _INIT_DIR / _TASK_SUITE / f"{_TASK_NAME}.pruned_init"
_INIT_FILE_SWAP = _INIT_DIR / _TASK_SUITE_SWAP / f"{_TASK_NAME}.pruned_init"

_RESOLUTION = 256
_SEED = 7

# ---------------------------------------------------------------------------
# Swap perturbation
# ---------------------------------------------------------------------------
# For KITCHEN_SCENE8 `use_swap`: swap the init region of moka_pot_1
# (right pot) with that of flat_stove_1.  This means the stove starts where
# moka_pot_1 used to be, and moka_pot_1 starts where the stove was.
#
# Original BDDL :init section:
#   (On flat_stove_1 kitchen_table_flat_stove_init_region)
#   (On moka_pot_1   kitchen_table_moka_pot_right_init_region)
#   (On moka_pot_2   kitchen_table_moka_pot_left_init_region)
#
# After swap:
#   (On flat_stove_1 kitchen_table_moka_pot_right_init_region)
#   (On moka_pot_1   kitchen_table_flat_stove_init_region)
#   (On moka_pot_2   kitchen_table_moka_pot_left_init_region)

_SWAP_PAIRS = [
    # (object_a, region_a_suffix, object_b, region_b_suffix)
    # We swap the region assignments of these two objects.
    ("flat_stove_1", "flat_stove_init_region", "moka_pot_1", "moka_pot_right_init_region"),
]


def _apply_swap_to_bddl(bddl_text: str) -> str:
    """Swap the (On <obj> <region>) lines in the :init block."""
    for obj_a, region_a, obj_b, region_b in _SWAP_PAIRS:
        # We swap by replacing the region each object is assigned to.
        # First replace with a placeholder to avoid double-swap.
        placeholder = "__SWAP_PLACEHOLDER__"
        # Pattern: (On <obj_a> kitchen_table_<region_a>)
        token_a = f"(On {obj_a} kitchen_table_{region_a})"
        token_b = f"(On {obj_b} kitchen_table_{region_b})"

        bddl_text = bddl_text.replace(token_a, placeholder)
        bddl_text = bddl_text.replace(token_b, f"(On {obj_b} kitchen_table_{region_a})")
        bddl_text = bddl_text.replace(placeholder, f"(On {obj_a} kitchen_table_{region_b})")

    return bddl_text


def _write_perturbed_bddl(use_swap: bool = True) -> str:
    """Read the original BDDL, optionally apply swap, write to a temp file."""
    bddl_text = _BDDL_FILE.read_text()
    if use_swap:
        bddl_text = _apply_swap_to_bddl(bddl_text)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".bddl", prefix="teleop_", delete=False
    )
    tmp.write(bddl_text)
    tmp.flush()
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def create_env(use_swap: bool = True, seed: int = _SEED):
    """Create the LIBERO OffScreenRenderEnv for the task.

    When use_swap=True, uses pre-generated swapped BDDL and init states
    from the libero_10_temp directory (moka_pot_2 ↔ flat_stove_1 swap).

    Returns:
        env: OffScreenRenderEnv instance
        task_description: str — the language instruction
        init_states: list of numpy arrays — initial MuJoCo states
    """
    from libero.libero.envs import OffScreenRenderEnv

    if use_swap:
        bddl_path = str(_BDDL_FILE_SWAP)
        init_path = _INIT_FILE_SWAP
    else:
        bddl_path = str(_BDDL_FILE)
        init_path = _INIT_FILE

    print(f"[env_helper] BDDL file: {bddl_path}")
    print(f"[env_helper] Swap perturbation: {use_swap}")

    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": _RESOLUTION,
        "camera_widths": _RESOLUTION,
        "horizon": 2000,  # extended for two-pot task
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)

    task_description = "put both moka pots on the stove"

    # Load initial states matching the BDDL layout
    init_states = torch.load(init_path, weights_only=False)
    print(f"[env_helper] Loaded {len(init_states)} init states from {init_path}")

    return env, task_description, init_states


# ---------------------------------------------------------------------------
# Episode reset
# ---------------------------------------------------------------------------

_DUMMY_ACTION = [0.0] * 6 + [-1.0]
_NUM_STEPS_WAIT = 10


def reset_episode(env, init_states, idx: int = 0, num_steps_wait: int = _NUM_STEPS_WAIT):
    """Reset the environment to a specific initial state.

    Waits `num_steps_wait` dummy steps for objects to settle.

    Returns:
        obs: dict — the observation after settling
    """
    env.reset()
    obs = env.set_init_state(init_states[idx])

    # Wait for objects to settle
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(_DUMMY_ACTION)

    return obs


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_obs(obs, step_idx: int, output_dir: str = "teleop_scratch/renders"):
    """Save base + wrist camera images as PNG files.

    Images are rotated 180° to match the training convention (the raw
    simulator images come flipped).

    Returns:
        (base_path, wrist_path): tuple of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    # Rotate 180° (flip both axes) to match training preprocessing
    base_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    base_path = os.path.join(output_dir, f"step_{step_idx:04d}_base.png")
    wrist_path = os.path.join(output_dir, f"step_{step_idx:04d}_wrist.png")

    Image.fromarray(base_img).save(base_path)
    Image.fromarray(wrist_img).save(wrist_path)

    return base_path, wrist_path


def render_obs_raw(obs):
    """Return rotated base + wrist images as numpy arrays (uint8).

    Does not save to disk.
    """
    base_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return base_img, wrist_img


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------

def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation.

    Copied from robosuite transform_utils.
    """
    q = quat.copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0

    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (q[:3] * 2.0 * math.acos(q[3])) / den


def get_state(obs):
    """Extract proprioceptive state from observation dict.

    Returns:
        dict with keys:
            eef_pos:        (3,) end-effector position
            eef_quat:       (4,) end-effector quaternion
            eef_axisangle:  (3,) end-effector axis-angle
            gripper:        (2,) gripper joint positions
            state_vector:   (8,) = concat(eef_pos, eef_axisangle, gripper)
    """
    eef_pos = obs["robot0_eef_pos"]
    eef_quat = obs["robot0_eef_quat"]
    eef_aa = _quat2axisangle(eef_quat)
    gripper = obs["robot0_gripper_qpos"]

    return {
        "eef_pos": eef_pos,
        "eef_quat": eef_quat,
        "eef_axisangle": eef_aa,
        "gripper": gripper,
        "state_vector": np.concatenate([eef_pos, eef_aa, gripper]),
    }


# ---------------------------------------------------------------------------
# Convenience: render initial scene
# ---------------------------------------------------------------------------

def render_initial_scene(use_swap: bool = True, init_idx: int = 0, output_dir: str = "teleop_scratch/renders"):
    """Create env, reset, and render the initial scene.

    Returns:
        env, task_description, init_states, obs, state_dict, (base_path, wrist_path)
    """
    env, task_desc, init_states = create_env(use_swap=use_swap)
    obs = reset_episode(env, init_states, idx=init_idx)
    state = get_state(obs)
    paths = render_obs(obs, step_idx=0, output_dir=output_dir)

    print(f"\n[env_helper] Task: {task_desc}")
    print(f"[env_helper] EEF pos:  {state['eef_pos']}")
    print(f"[env_helper] EEF quat: {state['eef_quat']}")
    print(f"[env_helper] EEF aa:   {state['eef_axisangle']}")
    print(f"[env_helper] Gripper:  {state['gripper']}")
    print(f"[env_helper] Saved images: {paths}")

    return env, task_desc, init_states, obs, state, paths


if __name__ == "__main__":
    env, task_desc, init_states, obs, state, paths = render_initial_scene(use_swap=True)
    print("\nDone. Close env.")
    env.close()
