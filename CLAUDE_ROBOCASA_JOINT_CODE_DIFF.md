# RoboCasa Joint Control: Thought Process, Plan & Code Changes

> **NOTE — Parts of this document were written during early implementation and
> may be stale. Key corrections since initial writing:**
>
> - Actions are stored as **absolute joint targets** (`action.joint_position_target`),
>   not deltas. `DeltaActions`/`AbsoluteActions` handle conversion at train/inference time.
> - Eval uses `input_type="absolute"` for the `JOINT_POSITION` controller.
> - The controller config is injected by monkey-patching `load_composite_controller_config`,
>   not by passing `controller_configs` through `gym.make(...)`.
> - Eval uses `env.unwrapped` / `PandaOmronJointKeyConverter` and goes through the
>   gym wrapper's `env.step()` — it does not bypass the wrapper.
> - Arm joint qpos indices are found by cross-validating gripper values, not by
>   parsing the model XML (MuJoCo reorders joints during compilation).

This document captures the complete reasoning, investigation, design decisions, and code diff for converting the `pi05_robocasa_single_task_lora_vision_fullft_action_prepare_coffee_l25_s29` config from end-effector (EEF/OSC) control to joint position control.

---

## 1. Investigation & Discovery

### 1.1 Understanding the Current EEF Architecture

The existing pipeline uses **Operational Space Control (OSC_POSE)** for the Panda arm:

- **Action space (12D):** `[eef_pos_delta(3), eef_rot_delta(3), gripper(1), base_motion(4), control_mode(1)]`
- **State space (16D):** `[eef_pos_rel(3), eef_rot_rel(4), base_pos(3), base_rot(4), gripper_qpos(2)]`
- The robosuite composite controller internally converts 6D EEF commands to 7-DOF joint torques via OSC

The data flows through:
```
Groot dataset (LeRobot) → GrootOpenpiSingleDataset (reorder to 12D actions, 16D state)
  → RobocasaInputs (pad to 32D) → Pi-0.5 model → RobocasaOutputs (slice to 12D)
  → convert_action() → PandaOmronKeyConverter.unmap_action() → composite controller → env.step()
```

### 1.2 Identifying the Controller Architecture

From `/home/skowshik/vla/codebase/openpi/.venv/lib/python3.11/site-packages/robosuite/controllers/config/robots/default_pandaomron.json`:

```json
{
    "type": "HYBRID_MOBILE_BASE",
    "body_parts": {
        "arms": { "right": { "type": "OSC_POSE", ... } },
        "torso": { "type": "JOINT_POSITION", "kp": 2000 },
        "base": { "type": "JOINT_VELOCITY" }
    }
}
```

The composite controller splits actions as:
```
right (OSC_POSE):      indices 0-5   (6 DOF)
right_gripper (GRIP):  index   6     (1 DOF)
base (JOINT_VELOCITY): indices 7-9   (3 DOF)
torso (JOINT_POSITION):index   10    (1 DOF)
base_mode:             index   11    (1 DOF)
Total: 12 DOF
```

With JOINT_POSITION for arm, `right` becomes 7D → total becomes **13 DOF**.

### 1.3 Verifying Raw Data Availability

**Key question:** Do the PrepareCoffee demos contain joint position information?

**Finding 1:** The LeRobot dataset at `/data/hf_cache/datasets/robocasa/v1.0/pretrain/composite/PrepareCoffee/20250716/lerobot` stores only 16D EEF state and 12D EEF actions in parquet files. No joint positions.

**Finding 2:** Each episode has an `extras/episode_NNNNNN/` directory containing:
- `states.npz` — Full MuJoCo state vector (233D = qpos + qvel) for every timestep
- `model.xml.gz` — The exact MuJoCo model XML used to record that episode
- `ep_meta.json` — Scene metadata

**Finding 3:** MuJoCo reorders joints during compilation, so the XML source order is not reliable for qpos indexing. The converter cross-validates `gripper_qpos` from the parquet against `states.npz` to locate the gripper qpos pair, then derives the 7 arm indices as the contiguous block immediately before it. For the PrepareCoffee episodes, the observed compiled layout is:

```
torso:   qpos[0]     (1 joint: mobilebase0_joint_torso_height)
arm:     qpos[1:8]   (7 joints: robot0_joint1..7)
gripper: qpos[8:10]  (2 joints: gripper0_right_finger_joint1/2)
base:    qpos[10:13] (3 joints: mobilebase0_joint_mobile_forward/side/yaw)
```

Verified consistent across episodes 0, 5, and 10.

**Finding 4:** The robosuite environment exposes `robot0_joint_pos` (7D) in its raw observations, confirmed by creating an environment and checking:
```python
obs = env._get_observations(force_update=True)
obs["robot0_joint_pos"]  # shape=(7,), Panda arm joint positions
```

### 1.4 The `extra_delta_transform` Mechanism

The user suggested using `DeltaActions`/`AbsoluteActions` from the existing openpi transform pipeline. Investigation revealed:

**`DeltaActions` (transforms.py:218-237):** Converts absolute actions to deltas:
```python
actions[..., :dims] -= state[..., :dims]  # where mask=True
```

**`AbsoluteActions` (transforms.py:240-259):** Inverse operation:
```python
actions[..., :dims] += state[..., :dims]  # where mask=True
```

This is already used by DROID's `JOINT_POSITION` action space (config.py:654-660):
```python
if self.action_space == DroidActionSpace.JOINT_POSITION:
    delta_action_mask = make_bool_mask(7, -1)
    data_transforms = data_transforms.push(
        inputs=[DeltaActions(delta_action_mask)],
        outputs=[AbsoluteActions(delta_action_mask)],
    )
```

### 1.5 Design Decision: Absolute Targets + DeltaActions

Instead of storing pre-computed joint deltas in the dataset, we store **absolute joint target positions** (the joint positions at the next timestep). The `DeltaActions` transform computes deltas on-the-fly during training by subtracting the current state.

**Advantages:**
- Leverages the existing, well-tested `DeltaActions`/`AbsoluteActions` infrastructure
- At inference, `AbsoluteActions` converts predicted deltas back to absolute targets
- No information loss from pre-computing deltas
- Consistent with how DROID handles joint position control

**The mask:** `make_bool_mask(7, -1)` = `[True]*7 + [False]` — only the first 7 dims (arm joints) get delta treatment. Dims 7-12 (gripper, base velocity, torso, control_mode) are already in delta/velocity form and pass through unchanged.

---

## 2. Architecture: EEF vs Joint Control

### State Space

| Dim | EEF (16D) | Joint (23D) |
|-----|-----------|-------------|
| 0-6 | eef_pos_rel(3) + eef_rot_rel(4) | **arm joint positions (7)** |
| 7-9 | base_pos(3) | eef_pos_rel(3) |
| 10-13 | base_rot(4) | eef_rot_rel(4) |
| 14-16 | gripper_qpos(2) | base_pos(3) |
| 17-20 | — | base_rot(4) |
| 21-22 | — | gripper_qpos(2) |

### Action Space

| Dim | EEF (12D) | Joint (13D) |
|-----|-----------|-------------|
| 0-6 | eef_pos_delta(3) + eef_rot_delta(3) | **arm joint targets (7, absolute)** |
| 7 | gripper(1) | gripper(1) |
| 8-10 | base_motion(3) | base_motion(3) |
| 11 | torso(1) | torso(1) |
| 12 | control_mode(1) | control_mode(1) |

### Controller Config Change

```
Before: arms.right.type = "OSC_POSE"    → 6D EEF action → internal IK → joint torques
After:  arms.right.type = "JOINT_POSITION" → 7D joint deltas → PD control → joint torques
```

### Data Pipeline

```
Training:
  Joint dataset (23D state, 13D absolute actions)
    → GrootOpenpiJointDataset.__getitem__()
    → RobocasaJointInputs (pad to 32D)
    → DeltaActions(mask=[T,T,T,T,T,T,T,F]) → actions[0:7] -= state[0:7]
    → Normalize → Pi-0.5 model → Denormalize
    → AbsoluteActions → actions[0:7] += state[0:7]
    → RobocasaJointOutputs (slice to 13D)

Inference:
  Model predicts deltas → AbsoluteActions adds state → absolute joint targets
    → main_joint.py: delta = target - current_joint_pos
    → Send deltas to JOINT_POSITION controller → env.step()
```

---

## 3. Files Changed

### 3.1 New Files

#### `examples/robocasa/convert_eef_to_joint_lerobot.py`

Converts an existing EEF-control LeRobot dataset to joint-control format.

**What it does:**
1. For each episode, reads the parquet (16D state, 12D actions) and `extras/states.npz` (233D MuJoCo state)
2. Cross-validates `gripper_qpos` from parquet against `states.npz` to locate the gripper qpos pair, then derives the 7 contiguous arm qpos indices immediately before it
3. Extracts arm joint positions from states.npz at those indices
4. Computes absolute joint targets: `target[t] = joint_pos[t+1]` (last timestep holds position)
5. Builds new 23D state and 13D action arrays
6. Writes new parquet files, modality.json, info.json, stats.json
7. Symlinks videos/ and extras/ to save disk space

```diff
+examples/robocasa/convert_eef_to_joint_lerobot.py  (new file, ~300 lines)
```

Key functions:
- `find_arm_qpos_indices()` — Cross-validates gripper values to find arm qpos indices
- `convert_episode()` — Converts one episode's state/actions
- `main()` — Orchestrates the conversion, writes output dataset

#### `examples/robocasa/main_joint.py`

Evaluation script for joint-control policies.

**What it does differently from `main.py`:**
1. Creates the env with a `JOINT_POSITION` arm controller via a temporary
   monkey-patch of `robocasa.utils.env_utils.load_composite_controller_config`.
2. Installs `PandaOmronJointKeyConverter` on `env.unwrapped` and rebuilds spaces.
3. Reads joint state from mapped observations as `obs["state.joint_position"]`.
4. Assembles 23D state:
   `[joint_pos(7), eef_pos_rel(3), eef_rot_rel(4), base_pos(3), base_rot(4), gripper_qpos(2)]`.
5. Receives 13D absolute joint-target actions from the policy server after `AbsoluteActions`.
6. Converts the 13D array to `action.joint_position_target`, `action.gripper_close`,
   `action.base_motion`, `action.torso_delta`, and `action.control_mode`.
7. Steps through the standard gym wrapper with `env.step(action)`.

```diff
+examples/robocasa/main_joint.py  (new file)
```

Key functions:
- `get_joint_controller_config()` — Loads PandaOmron config, replaces OSC_POSE with JOINT_POSITION (absolute mode)
- `convert_joint_action()` — Packs 13D array into action dict keys
- `PandaOmronJointKeyConverter.map_obs()` — Adds `robot0_joint_pos` to observations
- `PandaOmronJointKeyConverter.unmap_action()` — Maps action dict to robosuite composite controller format

---

### 3.2 Modified Files

#### `src/openpi/policies/robocasa_policy.py`

Added joint-control policy transforms.

```diff
 @dataclasses.dataclass(frozen=True)
 class RobocasaOutputs(transforms.DataTransformFn):
     def __call__(self, data: dict) -> dict:
         # Only return the first 12 actions (RoboCasa action_dim).
         return {"actions": np.asarray(data["actions"][:, :12])}
+
+
+# ---- Joint control variants ----
+
+JOINT_STATE_DIM = 23  # 7 arm + 3 eef_pos + 4 eef_rot + 3 base_pos + 4 base_rot + 2 gripper
+JOINT_ACTION_DIM = 13  # 7 arm_target + 1 gripper + 3 base + 1 torso + 1 control_mode
+
+
+def make_robocasa_joint_example() -> dict:
+    """Creates a random input example for the RoboCasa joint-control policy."""
+    return {
+        "observation/state": np.random.rand(JOINT_STATE_DIM),
+        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
+        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
+        "observation/image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
+        "prompt": "do something",
+    }
+
+
+@dataclasses.dataclass(frozen=True)
+class RobocasaJointInputs(transforms.DataTransformFn):
+    """Input transform for RoboCasa joint-control policy (23D state, 13D actions).
+
+    Actions are stored as absolute joint targets in the dataset. The DeltaActions
+    transform (applied separately via extra_delta_transform) converts them to deltas
+    at training time by subtracting state[0:7] from actions[0:7].
+    """
+    action_dim: int
+    model_type: _model.ModelType = _model.ModelType.PI0
+
+    def __call__(self, data: dict) -> dict:
+        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)
+        base_image = _parse_image(data["observation/image"])
+        wrist_image = _parse_image(data["observation/wrist_image"])
+        right_image = _parse_image(data["observation/image_right"])
+        inputs = {
+            "state": state,
+            "image": {
+                "base_0_rgb": base_image,
+                "left_wrist_0_rgb": wrist_image,
+                "right_wrist_0_rgb": right_image,
+            },
+            "image_mask": {
+                "base_0_rgb": np.True_,
+                "left_wrist_0_rgb": np.True_,
+                "right_wrist_0_rgb": np.True_,
+            },
+        }
+        if "actions" in data:
+            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
+            inputs["actions"] = actions
+        if "prompt" in data:
+            inputs["prompt"] = data["prompt"]
+        return inputs
+
+
+@dataclasses.dataclass(frozen=True)
+class RobocasaJointOutputs(transforms.DataTransformFn):
+    def __call__(self, data: dict) -> dict:
+        # Return the first 13 actions (joint-control action_dim).
+        return {"actions": np.asarray(data["actions"][:, :JOINT_ACTION_DIM])}
```

#### `src/openpi/groot_utils/groot_openpi_dataset.py`

Added `GrootOpenpiJointDataset` class after `GrootOpenpiSingleDataset`.

```diff
+class GrootOpenpiJointDataset(GrootOpenpiSingleDataset):
+    """Groot dataset variant for joint-control actions (23D state, 13D actions).
+
+    State: [joint_pos(7), eef_pos_rel(3), eef_rot_rel(4), base_pos(3), base_rot(4), gripper_qpos(2)]
+    Actions: [joint_target(7), gripper(1), base_motion(3), torso(1), control_mode(1)]
+    """
+
+    def __getitem__(self, index: SupportsIndex) -> dict:
+        # Call the grandparent (LeRobotSingleDataset) __getitem__ to get raw modality keys
+        item = LeRobotSingleDataset.__getitem__(self, index)
+
+        state = np.concatenate([
+            item["state.joint_position"],
+            item["state.end_effector_position_relative"],
+            item["state.end_effector_rotation_relative"],
+            item["state.base_position"],
+            item["state.base_rotation"],
+            item["state.gripper_qpos"],
+        ], axis=1)
+        actions = np.concatenate([
+            item["action.joint_position_target"],
+            item["action.gripper_close"],
+            item["action.base_motion"],
+            item["action.torso_delta"],
+            item["action.control_mode"],
+        ], axis=1)
+
+        return {
+            "observation/image": item["video.robot0_agentview_left"][0],
+            "observation/wrist_image": item["video.robot0_eye_in_hand"][0],
+            "observation/image_right": item["video.robot0_agentview_right"][0],
+            "observation/state": state[0],
+            "actions": actions,
+            "prompt": item["annotation.human.task_description"][0],
+        }
```

The key name `action.joint_position_target` is intentionally absolute: the stored values are next-step absolute joint targets. `DeltaActions` converts only dims 0:7 to deltas inside the training transform pipeline; the dataset field itself should not be named or interpreted as a delta.

#### `src/openpi/training/config.py`

Three changes:

**1. Added `joint_control` flag to `DataConfig`:**

```diff
     # Groot/RoboCasa dataset directories
     data_dirs: list[dict] | None = None
     # Sampling weights for multi-dataset training.
     dataset_weights: list[float] | None = None
+    # If True, use the joint-control Groot dataset loader (GrootOpenpiJointDataset).
+    joint_control: bool = False
```

**2. Added `LeRobotRobocasaJointDataConfig` class (after `LeRobotRobocasaDataConfig`):**

```diff
+@dataclasses.dataclass(frozen=True)
+class LeRobotRobocasaJointDataConfig(LeRobotRobocasaDataConfig):
+    """Config for RoboCasa joint-control training."""
+
+    @override
+    def create(self, assets_dirs, model_config):
+        repack_transform = _transforms.Group()
+        data_transforms = _transforms.Group(
+            inputs=[robocasa_policy.RobocasaJointInputs(action_dim=model_config.action_dim, ...)],
+            outputs=[robocasa_policy.RobocasaJointOutputs()],
+        )
+        # Delta transform: actions[0:7] -= state[0:7] at training time
+        delta_action_mask = _transforms.make_bool_mask(7, -1)
+        data_transforms = data_transforms.push(
+            inputs=[_transforms.DeltaActions(delta_action_mask)],
+            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
+        )
+        # ... (same enrichment logic as parent) ...
+        return dataclasses.replace(base, ..., joint_control=True)
```

**3. Added training config `pi05_robocasa_joint_control_prepare_coffee_l25_s29`:**

```diff
+    TrainConfig(
+        name="pi05_robocasa_joint_control_prepare_coffee_l25_s29",
+        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, paligemma_variant="gemma_2b_lora"),
+        data=LeRobotRobocasaJointDataConfig(
+            assets=AssetsConfig(assets_dir=None, asset_id="robocasa_joint"),
+            data_dirs=[{
+                "path": ".../PrepareCoffee/20250716/lerobot_joint",
+                "filter_key": None,
+            }],
+            layout_and_style_ids=[(25, 29)],
+            fixture_refs={"coffee_machine": "coffee_machine_main_group", "cab": "cab_4_main_group"},
+            num_demos=1,
+        ),
+        action_dim=13,  # 7 arm + 1 gripper + 3 base + 1 torso + 1 mode
+        ...
+    ),
```

#### `src/openpi/training/data_loader.py`

Routes joint-control datasets to `GrootOpenpiJointDataset`.

```diff
     # Groot/RoboCasa dataset loading
     if getattr(data_config, "data_dirs", None):
         data_dirs = data_config.data_dirs
+        use_joint = getattr(data_config, "joint_control", False)
         if len(data_dirs) == 1:
-            return _groot_openpi_dataset.GrootOpenpiSingleDataset(
+            dataset_cls = _groot_openpi_dataset.GrootOpenpiJointDataset if use_joint else _groot_openpi_dataset.GrootOpenpiSingleDataset
+            return dataset_cls(
                 dataset_meta=data_dirs[0],
                 action_horizon=action_horizon,
             )
```

---

## 4. Generated Data

The conversion script produced:

```
/data/hf_cache/datasets/robocasa/v1.0/pretrain/composite/PrepareCoffee/20250716/lerobot_joint/
├── data/chunk-000/episode_000000.parquet .. episode_000101.parquet
├── meta/
│   ├── info.json        (updated: state shape [23], action shape [13])
│   ├── modality.json    (new modality keys for joint control)
│   ├── stats.json       (normalization stats with q01/q99)
│   ├── episodes.jsonl   (copied from source)
│   └── tasks.jsonl      (copied from source)
├── videos -> symlink to source/videos
└── extras -> symlink to source/extras
```

**`modality.json` for joint control:**
```json
{
    "state": {
        "joint_position":                    {"original_key": "observation.state", "start": 0,  "end": 7},
        "end_effector_position_relative":    {"original_key": "observation.state", "start": 7,  "end": 10},
        "end_effector_rotation_relative":    {"original_key": "observation.state", "start": 10, "end": 14},
        "base_position":                     {"original_key": "observation.state", "start": 14, "end": 17},
        "base_rotation":                     {"original_key": "observation.state", "start": 17, "end": 21},
        "gripper_qpos":                      {"original_key": "observation.state", "start": 21, "end": 23}
    },
    "action": {
        "joint_position_target":  {"original_key": "action", "start": 0,  "end": 7},
        "gripper_close":         {"original_key": "action", "start": 7,  "end": 8},
        "base_motion":           {"original_key": "action", "start": 8,  "end": 11},
        "torso_delta":           {"original_key": "action", "start": 11, "end": 12},
        "control_mode":          {"original_key": "action", "start": 12, "end": 13}
    },
    "video": { ... (same as source) },
    "annotation": { ... (same as source) }
}
```

**Dataset statistics:**
- 102 episodes, 81,835 total frames
- Joint target actions mean/std closely matches joint position state mean/std (as expected — targets are next-timestep positions)

---

## 5. How to Use

### Train

```bash
cd /home/skowshik/vla/codebase/openpi
source .venv/bin/activate

# Step 1: Compute normalization stats
python scripts/compute_norm_stats.py pi05_robocasa_joint_control_prepare_coffee_l25_s29

# Step 2: Train
python scripts/train.py pi05_robocasa_joint_control_prepare_coffee_l25_s29 \
    --exp-name joint_coffee_l25_s29
```

### Evaluate

```bash
# Step 1: Start the policy server (in one terminal)
python scripts/serve_policy.py pi05_robocasa_joint_control_prepare_coffee_l25_s29 \
    --checkpoint-path /data/hf_cache/models/pi05_robocasa_exps/joint_coffee_l25_s29/checkpoints/STEP

# Step 2: Run evaluation (in another terminal)
python examples/robocasa/main_joint.py \
    --env_name PrepareCoffee \
    --layout_and_style_ids '[(25,29)]' \
    --log_dir /data/hf_cache/models/pi05_robocasa_exps/joint_coffee_l25_s29 \
    --host 0.0.0.0 --port 8000 \
    --num_trials 10
```

### Re-convert data (if needed)

```bash
python examples/robocasa/convert_eef_to_joint_lerobot.py \
    --src_dataset /data/hf_cache/datasets/robocasa/v1.0/pretrain/composite/PrepareCoffee/20250716/lerobot \
    --dst_dataset /data/hf_cache/datasets/robocasa/v1.0/pretrain/composite/PrepareCoffee/20250716/lerobot_joint \
    --overwrite
```

---

## 6. Key Observations from Data Exploration

### MuJoCo State Layout (observed by gripper-qpos cross-validation)

For PrepareCoffee episodes, the qpos layout in `states.npz` is:
```
Index 0:     mobilebase0_joint_torso_height
Index 1-7:   robot0_joint1..7  (Panda arm)
Index 8-9:   gripper0_right_finger_joint1/2
Index 10-12: mobilebase0_joint_mobile_forward/side/yaw
Index 13+:   Scene-specific joints (objects, fixtures, etc.)
```

This is different from a fresh env (where base=[0,1,2], torso=[3], arm=[4,10]) because MuJoCo reorders joints during compilation. The listed layout is observed from converted PrepareCoffee episodes, not assumed from XML source order. The conversion script handles this by cross-validating parquet `gripper_qpos` against each episode's `states.npz`, locating the gripper qpos pair, and deriving the 7 contiguous Panda arm qpos indices immediately before it.

### Joint Delta Statistics

From the converted dataset:
```
Joint delta magnitudes (per step at 20Hz):
  mean_abs: ~0.00005 to 0.007 rad/step  (varies by joint)
  max:      ~0.04 to 0.07 rad/step
  std:      ~0.001 to 0.01 rad/step
```

These are small relative movements, consistent with the Panda arm's typical operating range during manipulation tasks.

### Consistency Check

Verified that `gripper_qpos` from parquet matches `states.npz[12:14]` across all timesteps (the compiled MuJoCo model places gripper at indices 12-13, not 8-9 as the XML source order would suggest).

---

## 7. No Changes to External Repositories

All changes are confined to the `openpi` repository. No modifications were made to:
- `openpi_robocasa/robocasa/` (the robocasa package)
- `robosuite` (the simulation framework)
- Any other external dependency

The eval script (`main_joint.py`) works around the gym wrapper's EEF-specific action handling by:
1. Injecting a custom controller config via env kwargs
2. Building the composite controller action vector manually
3. Stepping the underlying robosuite env directly (bypassing the gym wrapper's `step()` method)
