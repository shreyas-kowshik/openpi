# Plan: Joint Control for RoboCasa PrepareCoffee Task

> **NOTE — This document is the original planning document. Several details were
> revised during implementation. The authoritative current state is:**
>
> - Dataset actions store **absolute next-step joint targets** as `action.joint_position_target`.
> - `DeltaActions` converts only action dims 0:7 to deltas during training.
> - `AbsoluteActions` converts predicted dims 0:7 back to absolute joint targets during inference.
> - Eval uses a `JOINT_POSITION` controller in `input_type="absolute"` mode.
> - The controller config is injected via monkey-patching `load_composite_controller_config`,
>   **not** by passing `controller_configs` through `gym.make(...)`.
> - The arm joint qpos indices are discovered by cross-validating gripper_qpos from the
>   parquet against `states.npz`, not by parsing the model XML joint order.
>
> See `CLAUDE_ROBOCASA_JOINT_CODE_DIFF.md` for the implemented code diff.

This document outlines the plan to switch the `pi05_robocasa_single_task_lora_vision_fullft_action_prepare_coffee_l25_s29` config from end-effector (EEF) control to joint position control.

---

## Current Architecture (EEF Control)

### Action Space (12D)
```
Dims 0-2:  EEF position delta (dx, dy, dz)          — 3D
Dims 3-5:  EEF rotation delta (axis-angle)           — 3D
Dim  6:    Gripper close command                      — 1D
Dims 7-10: Base motion (vx, vy, vyaw, torso)         — 4D
Dim  11:   Control mode flag (base active/inactive)   — 1D
```

### State Space (16D)
```
Dims 0-2:   EEF position relative to base            — 3D
Dims 3-6:   EEF rotation relative to base (quat)     — 4D
Dims 7-9:   Base position                             — 3D
Dims 10-13: Base rotation (quat)                      — 4D
Dims 14-15: Gripper qpos (finger joints)              — 2D
```

### Controller Stack
The environment uses `HYBRID_MOBILE_BASE` composite controller (from `default_pandaomron.json`):
- **Arm:** `OSC_POSE` — takes 6D EEF commands, internally converts to 7-DOF joint torques via Operational Space Control
- **Gripper:** `GRIP` — binary open/close
- **Torso:** `JOINT_POSITION` — 1D joint position
- **Base:** `JOINT_VELOCITY` — 3D velocity

### Data Pipeline
```
Groot dataset (LeRobot) → GrootOpenpiSingleDataset (reorder to 12D actions, 16D state)
  → RobocasaInputs (pad to 32D) → Pi-0.5 model → RobocasaOutputs (slice to 12D)
  → convert_action() → PandaOmronKeyConverter.unmap_action() → composite controller → env.step()
```

---

## Target Architecture (Joint Control)

### New Action Space (13D)
```
Dims 0-6:  Arm absolute next-step joint targets (7 Panda joints), stored as action.joint_position_target.
           Training converts these to deltas with DeltaActions before normalization.
           Inference converts predicted deltas back to absolute targets with AbsoluteActions.  — 7D
Dim  7:    Gripper close command                        — 1D
Dims 8-10: Base motion (vx, vy, vyaw)                  — 3D
Dim  11:   Torso delta                                  — 1D
Dim  12:   Control mode flag (base active/inactive)     — 1D
```

**Key change:** The arm goes from 6D (EEF pose delta) to 7D (absolute joint targets), net +1 dimension.

### New State Space (23D)
```
Dims 0-6:   Arm joint positions (7 Panda joints)      — 7D
Dims 7-9:   EEF position relative to base             — 3D  (kept for policy context)
Dims 10-13: EEF rotation relative to base (quat)      — 4D  (kept for policy context)
Dims 14-16: Base position                              — 3D
Dims 17-20: Base rotation (quat)                       — 4D
Dims 21-22: Gripper qpos (finger joints)               — 2D
```

**Key change:** Joint positions added to state (+7D). EEF relative state is retained because it provides useful task-space context for the policy.

### New Controller Config
Replace `OSC_POSE` with `JOINT_POSITION` for the arm. Do not pass
`controller_configs` through `gym.make(...)` — `create_env` injects that kwarg
internally. Instead, monkey-patch `robocasa.utils.env_utils.load_composite_controller_config`
before `gym.make(...)`, then restore it immediately (see `main_joint.py`).

```json
{
    "type": "HYBRID_MOBILE_BASE",
    "body_parts": {
        "arms": {
            "right": {
                "type": "JOINT_POSITION",
                "input_type": "absolute",
                "kp": 50,
                "damping_ratio": 1,
                "impedance_mode": "fixed",
                "output_max": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                "output_min": [-0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05],
                "gripper": { "type": "GRIP" }
            }
        },
        "torso": {
            "type": "JOINT_POSITION",
            "interpolation": "null",
            "kp": 2000
        },
        "base": {
            "type": "JOINT_VELOCITY",
            "interpolation": "null"
        }
    }
}
```

**Action split after this change:**
```
right (JOINT_POSITION, delta): indices 0-6   (7 DOF — was 6 with OSC_POSE)
right_gripper (GRIP):          index  7      (1 DOF)
torso (JOINT_POSITION):        index  8      (1 DOF)
base (JOINT_VELOCITY):         indices 9-11  (3 DOF)
base_mode:                     index  12     (1 DOF)
Total: 13 DOF (was 12 with OSC_POSE)
```

---

## Implementation Plan

### Phase 1: Generate New Training Data with Joint Actions

The existing dataset at `/data/hf_cache/datasets/robocasa/v1.0/pretrain/composite/PrepareCoffee/20250716/lerobot` uses EEF actions. We need a new dataset that stores joint-space actions and joint-position state.

**Approach A (Preferred): Re-collect demos with joint control enabled**

The raw HDF5 demos from robocasa contain the simulator states (`states` array with full MuJoCo qpos). We can replay these demos in a joint-control environment and record the resulting joint actions.

**Approach B: Convert existing EEF demos to joint actions via inverse kinematics**

Since the raw HDF5 demos store full simulator states (including joint positions at every timestep), we can compute joint-space deltas directly from consecutive qpos values without re-running the simulation.

#### Step 1.1: Create a joint-control demo conversion script

**New file:** `examples/robocasa/convert_eef_to_joint_lerobot.py`

This script:
1. Loads the original HDF5 demos (or the Groot/LeRobot dataset with sim states)
2. For each timestep, extracts the 7-DOF arm joint positions from the full sim state
3. Computes delta joint actions: `delta_joint[t] = joint_pos[t+1] - joint_pos[t]`
4. Extracts gripper, base, torso actions from the original action vector
5. Writes a new LeRobot dataset with:
   - **actions** (13D): `[joint_position_target(7), gripper(1), base_motion(3), torso(1), control_mode(1)]`
   - **state** (23D): `[joint_pos(7), eef_pos_rel(3), eef_rot_rel(4), base_pos(3), base_rot(4), gripper_qpos(2)]`

**Key concern:** The original actions were generated under OSC control. The joint deltas computed from consecutive states reflect the OSC controller's internal IK solution, NOT the policy's intended commands. However, since we're doing behavior cloning on the resulting state trajectories, this is valid — we're cloning the state-space trajectory, not the controller-space commands.

**Alternative:** If the raw HDF5 files don't store per-timestep joint positions in observations (need to verify — the `robot0_joint_pos` key should exist in raw robosuite observations), we can reconstruct them from the `states` array (full MuJoCo state vector) which IS stored. The MuJoCo state includes all qpos values, and we know the joint indices from `robot._ref_joint_pos_indexes`.

#### Step 1.2: Verify raw data availability

Before implementing, verify what's in the raw HDF5:

```python
import h5py
# Check if joint positions exist in the raw demo HDF5
# Path: /data/hf_cache/datasets/robocasa/v1.0/pretrain/composite/PrepareCoffee/20250716/
# or the original HDF5 before LeRobot conversion
with h5py.File("path/to/demo.hdf5", "r") as f:
    demo = f["data"]["demo_0"]
    # Check obs keys
    print(list(demo["obs"].keys()))
    # Expect: robot0_joint_pos, robot0_joint_vel, robot0_base_to_eef_pos, etc.
    # Check states array
    if "states" in demo:
        print(f"States shape: {demo['states'].shape}")  # (T, mujoco_state_dim)
```

**If `robot0_joint_pos` exists:** Use it directly as the joint position state.

**If only `states` exists:** Extract joint positions from the flattened MuJoCo state vector. This requires knowing the qpos indices for the Panda arm joints, which can be determined by creating the environment and querying `robot._ref_joint_pos_indexes`.

#### Step 1.3: Create the new LeRobot dataset

**New file:** `examples/robocasa/convert_eef_to_joint_lerobot.py`

Similar to the existing `convert_robocasa_to_lerobot.py` but with:
- State features: `(23,)` instead of `(16,)`
- Action features: `(13,)` instead of `(12,)`
- State composition: `[joint_pos(7), eef_pos_rel(3), eef_rot_rel(4), base_pos(3), base_rot(4), gripper_qpos(2)]`
- Action composition: `[joint_position_target(7), gripper(1), base_motion(3), torso(1), control_mode(1)]`

**Output path:** `/data/hf_cache/datasets/robocasa/v1.0/pretrain/composite/PrepareCoffee/20250716/lerobot_joint/`

---

### Phase 2: Update the Groot/LeRobot Dataset Loader

**File:** `src/openpi/groot_utils/groot_openpi_dataset.py`

#### Step 2.1: Update `GrootOpenpiSingleDataset.__getitem__()`

The `__getitem__` method needs to handle two modes: EEF (existing) and joint control (new).

Option A (recommended): Create a new dataset class `GrootOpenpiJointDataset` that overrides `__getitem__`:

```python
class GrootOpenpiJointDataset(GrootOpenpiSingleDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)  # Gets raw modality keys

        # New state ordering (23D):
        state = np.concatenate([
            item["state.joint_position"],                    # 7D
            item["state.end_effector_position_relative"],    # 3D
            item["state.end_effector_rotation_relative"],    # 4D
            item["state.base_position"],                     # 3D
            item["state.base_rotation"],                     # 4D
            item["state.gripper_qpos"],                      # 2D
        ], axis=1)

        # New action ordering (13D):
        actions = np.concatenate([
            item["action.joint_position_target"],     # 7D
            item["action.gripper_close"],            # 1D
            item["action.base_motion"],              # 3D (exclude torso from here)
            item["action.torso_delta"],              # 1D
            item["action.control_mode"],             # 1D
        ], axis=1)

        return {
            "observation/image": item["video.robot0_agentview_left"][0],
            "observation/wrist_image": item["video.robot0_eye_in_hand"][0],
            "observation/image_right": item["video.robot0_agentview_right"][0],
            "observation/state": state[0],
            "actions": actions,
            "prompt": item["annotation.human.task_description"][0],
        }
```

Option B (simpler, if creating a non-Groot LeRobot dataset): If we bypass the Groot format entirely and create a standard LeRobot dataset (like the `convert_robocasa_to_lerobot.py` script does), we can use a simpler data loading path. The dataset would store pre-concatenated `state` (23D) and `actions` (13D) directly, and we load it with the existing LeRobot loader rather than going through Groot.

**Recommendation:** Option B is simpler and avoids having to define new Groot modality configs. The `convert_eef_to_joint_lerobot.py` script would write the data in the same format as the existing `convert_robocasa_to_lerobot.py`, just with different dimensions.

#### Step 2.2: Update normalization stats reordering

**File:** `src/openpi/groot_utils/groot_openpi_dataset.py`, function `_load_norm_stats_from_groot_dataset()`

If using Option B (standalone LeRobot dataset), normalization stats will be computed directly from the data by the `compute_norm_stats.py` script, and no Groot-specific reordering is needed.

If using Option A (Groot format), new index mappings are needed for the 23D state and 13D action vectors.

---

### Phase 3: Update Policy Transforms

**File:** `src/openpi/policies/robocasa_policy.py`

#### Step 3.1: Create new input/output transforms

```python
@dataclasses.dataclass(frozen=True)
class RobocasaJointInputs(transforms.DataTransformFn):
    action_dim: int  # model's internal action dim (32)
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # State is 23D, pad to action_dim (32)
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        right_image = _parse_image(data["observation/image_right"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            # Actions are 13D, pad to action_dim (32)
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaJointOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 13 actions (joint control action_dim)
        return {"actions": np.asarray(data["actions"][:, :13])}
```

#### Step 3.2: Update `make_robocasa_example()` (for testing)

```python
def make_robocasa_joint_example() -> dict:
    return {
        "observation/state": np.random.rand(23),  # 23D instead of 16D
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }
```

---

### Phase 4: Update Training Configuration

**File:** `src/openpi/training/config.py`

#### Step 4.1: Create new training config

```python
TrainConfig(
    name="pi05_robocasa_joint_control_prepare_coffee_l25_s29",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        paligemma_variant="gemma_2b_lora",
        discrete_state_input=False,
    ),
    data=LeRobotRobocasaDataConfig(  # or a new LeRobotRobocasaJointDataConfig
        assets=AssetsConfig(
            assets_dir=None,
            asset_id="robocasa_joint",  # new asset id for joint control stats
        ),
        data_dirs=[{
            "path": "/data/hf_cache/datasets/robocasa/v1.0/pretrain/composite/PrepareCoffee/20250716/lerobot_joint",
            "filter_key": None,
        }],
        layout_and_style_ids=[(25, 29)],
        fixture_refs={"coffee_machine": "coffee_machine_main_group", "cab": "cab_4_main_group"},
        num_demos=1,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    checkpoint_base_dir="/data/hf_cache/models/pi05_robocasa_exps/",
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
    ).get_freeze_filter(),
    action_dim=13,  # 13 instead of 12
    ema_decay=None,
    num_train_steps=100_000,
    batch_size=64,
    save_interval=4000,
    keep_period=1000,
    num_workers=4,
    log_interval=100,
    action_l1_loss_interval=1000,
),
```

#### Step 4.2: Create (or extend) the data config class

If using the existing `LeRobotRobocasaDataConfig` class, update its `create()` method to accept a `use_joint_control` flag (or create a new `LeRobotRobocasaJointDataConfig`):

- The `create()` method must use `RobocasaJointInputs` and `RobocasaJointOutputs` instead of the EEF versions
- The normalization stats computation must handle the new 23D state / 13D action dimensions
- The action slicing in the output transform must return 13D instead of 12D

---

### Phase 5: Update Evaluation Script

**File:** `examples/robocasa/main.py`

This is the most critical change — the evaluation script must:

1. Create the environment with a **JOINT_POSITION** controller for the arm
2. Assemble the **23D state** vector (including joint positions)
3. Convert the **13D model output** to the correct joint-control action format

#### Step 5.1: Create new environment with joint controller

Do not pass `controller_configs` through `gym.make(...)` — `create_env` injects
that kwarg internally and passing it again causes a duplicate-kwarg `TypeError`.
Instead, monkey-patch `load_composite_controller_config` before env creation:

```python
import robocasa.utils.env_utils as _env_utils

joint_cfg = get_joint_controller_config()  # see main_joint.py
_orig = _env_utils.load_composite_controller_config
_env_utils.load_composite_controller_config = lambda *a, **kw: joint_cfg
try:
    env = gym.make(f"robocasa/{env_name}", disable_env_checker=True, **env_kwargs)
finally:
    _env_utils.load_composite_controller_config = _orig

# Swap key converter and rebuild spaces on the unwrapped env
gym_env = env.unwrapped
gym_env.key_converter = PandaOmronJointKeyConverter
gym_env._create_obs_and_action_space()
```

Where `get_joint_controller_config()` sets `input_type="absolute"` and 7D output limits:

```python
arm_cfg["type"] = "JOINT_POSITION"
arm_cfg["input_type"] = "absolute"
arm_cfg["kp"] = 50
arm_cfg["output_max"] = [0.05] * 7
arm_cfg["output_min"] = [-0.05] * 7
```

#### Step 5.2: Update state assembly in eval loop

```python
# Current (EEF):
state = np.concatenate((
    obs["state.end_effector_position_relative"],  # 3D
    obs["state.end_effector_rotation_relative"],  # 4D
    obs["state.base_position"],                   # 3D
    obs["state.base_rotation"],                   # 4D
    obs["state.gripper_qpos"],                    # 2D
), axis=0)  # 16D

# New (Joint):
# PandaOmronJointKeyConverter.map_obs() maps robot0_joint_pos -> state.joint_position
state = np.concatenate((
    obs["state.joint_position"],                  # 7D  — NEW (mapped from robot0_joint_pos)
    obs["state.end_effector_position_relative"],  # 3D
    obs["state.end_effector_rotation_relative"],  # 4D
    obs["state.base_position"],                   # 3D
    obs["state.base_rotation"],                   # 4D
    obs["state.gripper_qpos"],                    # 2D
), axis=0)  # 23D
```

**Note:** The raw robosuite observation key is `robot0_joint_pos`. The `PandaOmronJointKeyConverter` (defined in `main_joint.py`) maps it to `state.joint_position` via `map_obs()`. The converter is swapped onto `env.unwrapped` after env creation.

#### Step 5.3: Update action conversion

Replace `convert_action()` with a new `convert_joint_action()`. Torso is a
separate key (`action.torso_delta`), not packed into `action.base_motion`:

```python
def convert_joint_action(action):
    """Convert 13D joint-control action to env format."""
    return {
        "action.joint_position_target": action[0:7],  # 7D absolute joint targets
        "action.gripper_close": action[7:8],           # 1D
        "action.base_motion": action[8:11],            # 3D (vx, vy, vyaw)
        "action.torso_delta": action[11:12],           # 1D
        "action.control_mode": action[12:13],          # 1D
    }
```

#### Step 5.4: Joint key converter `unmap_action()`

`PandaOmronJointKeyConverter.unmap_action()` maps the action dict to robosuite
composite controller format. `robot0_right` receives absolute joint targets
(the controller uses `input_type="absolute"`):

```python
@classmethod
def unmap_action(cls, input_action):
    return {
        "robot0_right_gripper": (
            -1.0 if input_action["action.gripper_close"] < 0.5 else 1.0
        ),
        "robot0_right": input_action["action.joint_position_target"],  # 7D absolute
        "robot0_base": input_action["action.base_motion"],             # 3D
        "robot0_torso": input_action["action.torso_delta"],            # 1D
        "robot0_base_mode": (
            -1.0 if input_action["action.control_mode"] < 0.5 else 1.0
        ),
    }
```

#### Step 5.5: Environment creation (monkey-patch approach)

Do not pass `controller_configs` through `gym.make(...)` — `create_env()`
injects that kwarg internally. Monkey-patch `load_composite_controller_config`
before env creation, then restore it. After creation, swap the key converter
on `env.unwrapped` and rebuild spaces:

```python
import robocasa.utils.env_utils as _env_utils

joint_cfg = get_joint_controller_config()
_orig = _env_utils.load_composite_controller_config
_env_utils.load_composite_controller_config = lambda *a, **kw: joint_cfg
try:
    env = gym.make(f"robocasa/{env_name}", disable_env_checker=True, **env_kwargs)
finally:
    _env_utils.load_composite_controller_config = _orig

gym_env = env.unwrapped
gym_env.key_converter = PandaOmronJointKeyConverter
gym_env._create_obs_and_action_space()
```

---

### Phase 6: Joint Key Converter (no robocasa modifications needed)

Instead of modifying the robocasa package, the eval script defines a local
`PandaOmronJointKeyConverter` (in `main_joint.py`) that subclasses
`PandaOmronKeyConverter`:

- `map_obs()` adds `"body.joint_position": input_obs["robot0_joint_pos"]`
- `unmap_action()` maps `action.joint_position_target` to `robot0_right`,
  `action.base_motion` to `robot0_base`, `action.torso_delta` to `robot0_torso`
- `deduce_action_space()` declares the joint-control action keys

After env creation, the converter is swapped on `env.unwrapped` and observation/action
spaces are rebuilt with `gym_env._create_obs_and_action_space()`. The controller
config is injected via monkey-patching (see Step 5.5).

---

## Summary of Files (Actual Implementation)

| # | File | Change |
|---|------|--------|
| 1 | **`examples/robocasa/convert_eef_to_joint_lerobot.py`** | **NEW** — Convert EEF dataset to joint-control LeRobot dataset |
| 2 | **`examples/robocasa/main_joint.py`** | **NEW** — Joint-control evaluation script |
| 3 | `src/openpi/policies/robocasa_policy.py` | Add `RobocasaJointInputs`, `RobocasaJointOutputs` classes |
| 4 | `src/openpi/training/config.py` | Add `LeRobotRobocasaJointDataConfig` and `pi05_robocasa_joint_control_*` config |
| 5 | `src/openpi/training/data_loader.py` | Route `joint_control=True` to `GrootOpenpiJointDataset` |
| 6 | `src/openpi/groot_utils/groot_openpi_dataset.py` | Add `GrootOpenpiJointDataset` class |

---

## Execution Order

1. **Verify raw data** — Check if `robot0_joint_pos` or `states` array exists in the PrepareCoffee HDF5 demos
2. **Create demo conversion script** — Convert EEF demos to joint-control format
3. **Generate joint-control LeRobot dataset** — Write to disk with 23D state, 13D actions
4. **Update policy transforms** — Add `RobocasaJointInputs`/`Outputs`
5. **Update training config** — New config with `action_dim=13`
6. **Compute normalization stats** — Run `compute_norm_stats.py` on the new dataset
7. **Train** — Run training with the new config
8. **Update eval script** — Controller config, state assembly, action conversion
9. **Evaluate** — Run evaluation with joint-control environment

---

## Risks and Considerations

### 1. Demo Quality Under Action Space Change
The original demos were collected under OSC control. Computing joint deltas from consecutive simulator states gives us the "ground truth" joint trajectory, but the deltas may have different noise characteristics than what a JOINT_POSITION controller would produce when executing those deltas. Specifically:
- OSC internally applies torque-level control with compliance; JOINT_POSITION uses PD control
- The same joint trajectory may not be achievable with JOINT_POSITION if the PD gains and action scaling don't match

**Mitigation:** Tune the JOINT_POSITION controller's `kp`, `output_max`/`output_min` parameters to match the trajectory dynamics. Alternatively, re-collect demos directly under joint control using teleoperation.

### 2. Action Scaling and Normalization
Joint position deltas per timestep (at 20Hz) for the Panda are typically small (order of 0.01-0.05 radians). The JOINT_POSITION controller's `output_max`/`output_min` must be set appropriately:
- `output_max = 0.05` means the controller clips commanded deltas to ±0.05 rad/step
- The model's output range (after denormalization) must match

**Mitigation:** Inspect the actual joint delta statistics from the converted demos and set `output_max`/`output_min` accordingly.

### 3. Panda Joint Limits
The Panda arm has joint limits. The JOINT_POSITION controller respects these, but if the policy predicts actions that would exceed limits, the controller will clip. This is generally fine but may cause the policy to "stall" near limits.

### 4. State Space Size Increase
Going from 16D to 23D state increases the input size. Since the Pi-0.5 model pads everything to 32D anyway, the 23D state still fits within the existing architecture. No model architecture changes needed.

### 5. Absolute Joint Position Control (Implemented)
The implementation uses absolute joint position targets (`input_type: "absolute"`). The dataset stores absolute next-step joint positions as `action.joint_position_target`. The `DeltaActions` transform converts dims 0:7 to deltas during training. At inference, `AbsoluteActions` converts predicted deltas back to absolute targets, which the `JOINT_POSITION` controller tracks directly in absolute mode.

---

## Quick Validation Checklist

Before full training:

- [ ] Verify `robot0_joint_pos` is available in raw env observations (7D for Panda arm)
- [ ] Verify joint deltas from consecutive states match expected magnitudes (~0.01-0.05 rad/step)
- [ ] Verify JOINT_POSITION controller works with PandaOmron in a standalone test
- [ ] Verify the composite controller action vector length changes from 12 to 13
- [ ] Verify the gym wrapper correctly maps joint-control actions
- [ ] Run a single training step to verify the data pipeline doesn't crash
- [ ] Run a single eval step to verify the action execution loop works
