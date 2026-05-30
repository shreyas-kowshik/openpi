# RoboCasa Tasks Reference

This document covers available robocasa tasks, focusing on pick-and-place atomic tasks for sanity checking and simple training.

---

## Task Taxonomy

Robocasa tasks are divided into:

- **Atomic tasks**: Single-step manipulation (pick, place, open, close, etc.). ~60+ tasks.
- **Composite tasks**: Multi-step sequences (e.g., "prepare coffee" = multiple atomic steps). ~100+ tasks.

For sanity checking with minimal complexity, use **atomic pick-and-place tasks**.

---

## All Atomic Pick-and-Place Tasks (18 tasks)

These are the `PickPlace*` tasks implemented under `robocasa/environments/kitchen/atomic/kitchen_pick_place.py`.

### Recommended for Sanity Checking (Target Split Available)

These 5 tasks have **pre-collected target-split evaluation datasets** (500 human demos each) available for download:

| Task Name | Description | Target Split | Notes |
|-----------|-------------|:------------:|-------|
| `PickPlaceCounterToCabinet` | Pick object from counter, place in open cabinet | ✅ | **Best starting point**: most data, most tested in codebase, used as example in training config |
| `PickPlaceCounterToStove` | Pick object from counter, place on stove burner | ✅ | Stove is a wide flat surface, relatively easy |
| `PickPlaceDrawerToCounter` | Pick object from open drawer, place on counter | ✅ | Requires reaching into drawer |
| `PickPlaceSinkToCounter` | Pick object from sink basin, place on counter | ✅ | Object starts in sink (lower surface) |
| `PickPlaceToasterToCounter` | Pick object from toaster slot, place on counter | ✅ | Toaster slot is a narrow target for placing |

---

### All 18 PickPlace Tasks (Pretrain Split Available for All)

| # | Task Name | Horizon | Source | Description |
|---|-----------|---------|--------|-------------|
| 1 | `PickPlaceCabinetToCounter` | 300 | pretrain | Pick from inside cabinet (open door), place on counter |
| 2 | `PickPlaceCounterToBlender` | 500 | pretrain | Pick from counter, place in blender jug |
| 3 | `PickPlaceCounterToCabinet` | 500 | pretrain+target | Pick from counter, place in cabinet |
| 4 | `PickPlaceCounterToDrawer` | 500 | pretrain | Pick from counter, place in open drawer |
| 5 | `PickPlaceCounterToMicrowave` | 700 | pretrain | Pick from counter, place in microwave |
| 6 | `PickPlaceCounterToOven` | 500 | pretrain | Pick from counter, place in oven rack |
| 7 | `PickPlaceCounterToSink` | 500 | pretrain | Pick from counter, place in sink basin |
| 8 | `PickPlaceCounterToStandMixer` | 500 | pretrain | Pick from counter, place in stand mixer bowl |
| 9 | `PickPlaceCounterToStove` | 500 | pretrain+target | Pick from counter, place on stove |
| 10 | `PickPlaceCounterToToasterOven` | 500 | pretrain | Pick from counter, place in toaster oven |
| 11 | `PickPlaceDrawerToCounter` | 500 | pretrain+target | Pick from open drawer, place on counter |
| 12 | `PickPlaceFridgeDrawerToShelf` | 500 | pretrain | Pick from fridge bottom drawer, place on fridge shelf |
| 13 | `PickPlaceFridgeShelfToDrawer` | 500 | pretrain | Pick from fridge shelf, place in fridge bottom drawer |
| 14 | `PickPlaceMicrowaveToCounter` | 500 | pretrain | Pick from microwave interior, place on counter |
| 15 | `PickPlaceSinkToCounter` | 500 | pretrain+target | Pick from sink, place on counter |
| 16 | `PickPlaceStoveToCounter` | 500 | pretrain | Pick from stove, place on counter |
| 17 | `PickPlaceToasterOvenToCounter` | 500 | pretrain | Pick from toaster oven, place on counter |
| 18 | `PickPlaceToasterToCounter` | 500 | pretrain+target | Pick from toaster slot, place on counter |

**Horizon** = max episode length in timesteps at 20Hz control frequency.

---

## Ranking: Simplest to Most Complex

For sanity checking, rank by difficulty (simplest first):

### Tier 1 — Easiest (Flat surface to flat surface)
1. **`PickPlaceCounterToCabinet`** — Counter → open cabinet. Cabinet door is pre-opened by env. Target is a clearly defined recessed space. This is the most tested task in the codebase and should be your first choice.
2. **`PickPlaceCounterToStove`** — Counter → stove. Stove is a wide, flat, clearly visible surface.
3. **`PickPlaceSinkToCounter`** / **`PickPlaceStoveToCounter`** — From sink/stove back to counter. Object starts elevated, dropping to counter.

### Tier 2 — Moderate (Enclosed target or occluded source)
4. **`PickPlaceDrawerToCounter`** / **`PickPlaceCabinetToCounter`** — Source is inside an enclosure (open drawer/cabinet).
5. **`PickPlaceCounterToDrawer`** — Placing into a drawer slot (requires more precision).
6. **`PickPlaceCounterToMicrowave`** / **`PickPlaceMicrowaveToCounter`** — Microwave door is pre-opened; narrower opening.

### Tier 3 — Harder (Precision placement, unusual geometry)
7. **`PickPlaceCounterToSink`** — Sink is below counter level; reaching down.
8. **`PickPlaceCounterToBlender`** / **`PickPlaceCounterToStandMixer`** — Small, oddly-shaped target.
9. **`PickPlaceFridgeDrawerToShelf`** / **`PickPlaceFridgeShelfToDrawer`** — Inside fridge, two distinct regions.

---

## Controlling Randomness

Robocasa environments have three main sources of variability:

### 1. Kitchen Layout (`layout_id`)
Controls the floor plan of the kitchen (counter arrangement, appliance positions).
- Layout IDs: 1–10
- Pretrain split randomly samples from all layouts (layout_ids=-2)
- Fix a specific layout by passing `--layout 1` in `collect_demos.py`

### 2. Kitchen Style (`style_id`)
Controls textures/aesthetics (cabinet color, counter material, etc.). Does NOT change geometry.
- Style IDs: 1–10
- Fix with `--style 1` in `collect_demos.py`

### 3. Object Instance Split (`obj_instance_split`)
Controls which physical object instances are used:
- `"pretrain"`: Standard object pool
- `"target"`: Held-out object instances (harder; for evaluation generalization)
- `None`: All instances

### 4. Clutter Mode (`clutter_mode`)
Controls whether distracting objects are placed in the scene:
- `clutter_mode=0`: No clutter (easiest)
- `clutter_mode=1`: Some distractors added (used in target evaluation split)

### Layout IDs — Special Values (for `collect_demos.py`)

When collecting demos manually:

| Value | Meaning |
|-------|---------|
| `-1` | All layouts |
| `-2` | Simple layouts (no islands/wall stacks) — default for "pretrain" split |
| `-3` | Layouts with islands/wall stacks |
| `-4` | Layouts with dining areas |
| `1..10` | Specific numbered layout (deterministic) |

### Using `layout_and_style_ids` in Training Config

The training config's `layout_and_style_ids` parameter restricts which episodes are loaded:

```python
# Single fixed scene (lowest variance, easiest learning):
layout_and_style_ids = [(1, 1)]

# A few scenes for some generalization:
layout_and_style_ids = [(1, 1), (2, 2), (3, 3)]

# All target evaluation scenes (10 scenes):
layout_and_style_ids = [(i, i) for i in range(1, 11)]

# No restriction (all scenes in dataset):
layout_and_style_ids = None
```

**For sanity checking (5–50 demos)**, strongly recommend `layout_and_style_ids=[(1, 1)]` to minimize variability.

---

## Recommended Configuration for Sanity Checking

### 5 Demos (Minimum Viable Test)

Goal: Just verify the training pipeline runs end-to-end.

```python
# In src/openpi/training/config.py, update pi05_robocasa_single_task_lora:
data=LeRobotRobocasaDataConfig(
    assets=AssetsConfig(assets_dir=None, asset_id="robocasa"),
    data_dirs=[{
        "path": "DATASET_BASE_PATH/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot",
        "filter_key": None,
    }],
    layout_and_style_ids=[(1, 1)],   # Single scene
    num_demos=5,                      # Only 5 demos
),
num_train_steps=200,    # Override to just 200 steps
batch_size=4,           # Small batch to fit GPU
```

```bash
python scripts/train.py pi05_robocasa_single_task_lora \
    --exp-name smoke_test \
    --num-train-steps 200 \
    --batch-size 4 \
    --overwrite
```

### 40–50 Demos (Sanity Check)

Goal: Verify the model can overfit to a small dataset (loss should decrease significantly).

```python
data=LeRobotRobocasaDataConfig(
    assets=AssetsConfig(assets_dir=None, asset_id="robocasa"),
    data_dirs=[{
        "path": "DATASET_BASE_PATH/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot",
        "filter_key": None,
    }],
    layout_and_style_ids=[(1, 1)],   # Single scene
    num_demos=40,                    # ~40 demos
),
num_train_steps=10_000,
batch_size=32,
```

```bash
python scripts/train.py pi05_robocasa_single_task_lora \
    --exp-name sanity_40demos \
    --num-train-steps 10000 \
    --batch-size 32 \
    --overwrite
```

---

## Collecting Your Own Demos (if needed)

The following shows how to collect for the simplest task with minimal randomness.

### PickPlaceCounterToCabinet — Minimal Randomness Setup

```bash
cd /home/skowshik/vla/codebase/openpi_robocasa/robocasa

python robocasa/scripts/collect_demos.py \
    --environment PickPlaceCounterToCabinet \
    --robots PandaOmron \
    --directory /path/to/my_demos \
    --split target \
    --layout 1 \
    --style 1 \
    --device keyboard
```

**Arguments for reducing randomness:**
- `--split target`: Uses target evaluation scenes (fixed layout/style sets)
- `--layout 1`: Forces kitchen layout 1
- `--style 1`: Forces kitchen style 1 (aesthetics only)
- Omit `--generative_textures`: No procedural texture generation

**Note**: `collect_demos.py` requires a **display** (not headless). For server usage, you need X forwarding or a virtual display (`Xvfb`).

---

## Dataset Download Commands (All PickPlace Tasks with Target Split)

These 5 tasks have target-split data available:

```bash
cd /home/skowshik/vla/codebase/openpi_robocasa/robocasa

# All 5 tasks with target split (human demos, ~500 each)
python robocasa/scripts/download_datasets.py \
    --tasks PickPlaceCounterToCabinet PickPlaceCounterToStove \
            PickPlaceDrawerToCounter PickPlaceSinkToCounter PickPlaceToasterToCounter \
    --split target \
    --source human

# Single task for quick start:
python robocasa/scripts/download_datasets.py \
    --tasks PickPlaceCounterToCabinet \
    --split target \
    --source human
```

After download, data lands at:
```
DATASET_BASE_PATH/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot/
DATASET_BASE_PATH/v1.0/target/atomic/PickPlaceCounterToStove/20250818/lerobot/
DATASET_BASE_PATH/v1.0/target/atomic/PickPlaceDrawerToCounter/20250820/lerobot/
DATASET_BASE_PATH/v1.0/target/atomic/PickPlaceSinkToCounter/20250813/lerobot/
DATASET_BASE_PATH/v1.0/target/atomic/PickPlaceToasterToCounter/20250817/lerobot/
```

---

## Episode Layout/Style Distribution (Target Split)

The target split for PickPlaceCounterToCabinet has 500 human demos distributed across 10 (layout, style) pairs:

- Expected ~50 demos per (layout_id, style_id) pair
- Pairs are: (1,1), (2,2), ..., (10,10) — layout and style indices are matched
- Using `layout_and_style_ids=[(1,1)]` gives ~50 demos available
- Using `num_demos=5` on top would take just the first 5 from that scene

---

## Environment Class Reference

All pick-and-place tasks inherit from `PickPlace` which inherits from `Kitchen`. The base `PickPlace` class accepts:

```python
robosuite.make(
    env_name="PickPlaceCounterToCabinet",
    robots="PandaOmron",              # Mobile manipulator
    has_renderer=False,               # Headless
    has_offscreen_renderer=True,      # For image observations
    use_camera_obs=True,
    camera_names=["robot0_agentview_left", "robot0_eye_in_hand"],
    camera_height=256,
    camera_width=256,
    control_freq=20,                  # 20Hz control
    layout_ids=1,                     # Specific layout
    style_ids=1,                      # Specific style
    obj_instance_split="target",      # or "pretrain"
)
```

**obj_groups parameter**: Can restrict which objects are pickable. Default is `"all"` which samples from all object categories. For easiest policy (most predictable object appearance), consider restricting:
```python
# These are example obj_groups strings; refer to robocasa docs for full list
obj_groups="fruit"   # Only fruit objects
obj_groups="can"     # Only cans
```
This is set per-task in the class constructor, not typically in the download/training pipeline.

---

# RoboCasa Code Changes & Integration Details

This section documents every code change made to integrate RoboCasa simulation data into the OpenPI training and evaluation pipeline.

---

## 1. Architecture Overview

The RoboCasa integration touches five layers of the codebase:

```
┌─────────────────────────────────────────────────┐
│  Training Configs (config.py)                   │  LeRobotRobocasaDataConfig
├─────────────────────────────────────────────────┤
│  Data Loading (data_loader.py)                  │  Routes to Groot datasets
├─────────────────────────────────────────────────┤
│  Dataset Classes (groot_openpi_dataset.py)      │  GrootOpenpiSingleDataset / Multi
├─────────────────────────────────────────────────┤
│  Policy Transforms (robocasa_policy.py)         │  RobocasaInputs / RobocasaOutputs
├─────────────────────────────────────────────────┤
│  Evaluation (examples/robocasa/main.py)         │  Gym env + WebSocket policy client
└─────────────────────────────────────────────────┘
```

Data flows:
1. **HDF5 (raw robocasa)** → `convert_robocasa_to_lerobot.py` → **LeRobot format on disk**
2. **LeRobot on disk** → `GrootOpenpiSingleDataset` (scene-filtered) → `RobocasaInputs` transform → **Pi-0.5 model**
3. **Trained checkpoint** → `openpi_client` WebSocket server → `examples/robocasa/main.py` → **RoboCasa Gym env**

---

## 2. Data Conversion: HDF5 to LeRobot

**File:** `examples/robocasa/convert_robocasa_to_lerobot.py`

Converts raw RoboCasa HDF5 demo files (stored at paths like `/mnt/amlfs-01/shared/robocasa_benchmark/...`) into the LeRobot dataset format. Each HDF5 file contains multiple demos under `data/demo_0`, `data/demo_1`, etc.

### What it extracts per timestep

| LeRobot field | HDF5 source | Shape |
|---|---|---|
| `image` | `obs/robot0_agentview_left_image` | (128, 128, 3) |
| `wrist_image` | `obs/robot0_eye_in_hand_image` | (128, 128, 3) |
| `state` | Concatenation of 5 obs keys (see below) | (16,) |
| `actions` | `actions` | (12,) |
| `task` | `ep_meta["lang"]` from HDF5 attributes | string |

**State composition (16 dimensions):**

| Dims | Source key | Meaning |
|---|---|---|
| 0-2 | `robot0_base_to_eef_pos` | End-effector position relative to base |
| 3-6 | `robot0_base_to_eef_quat` | End-effector rotation quaternion relative to base |
| 7-9 | `robot0_base_pos` | Mobile base XY position + heading |
| 10-13 | `robot0_base_quat` | Mobile base rotation quaternion |
| 14-15 | `robot0_gripper_qpos` | Left/right finger joint positions |

**Action composition (12 dimensions):**

Raw actions from HDF5 are stored directly. The 12 dims correspond to:
- Dims 0-2: End-effector position delta
- Dims 3-5: End-effector rotation delta (axis-angle)
- Dim 6: Gripper close command
- Dims 7-10: Base motion (x, y, yaw velocity + something)
- Dim 11: Control mode flag

### Output format

The script writes to `LEROBOT_HOME/<REPO_NAME>/` using the `LeRobotDataset` API. Each demo becomes one LeRobot episode. The script supports converting multiple HDF5 files (multiple data collection runs) into a single dataset.

**Commented-out configs** at the top show the history of datasets converted:
- `robocasa/atomic_5tasks_mg` — 5 machine-generated atomic tasks
- `robocasa/atomic_human` — All 45 atomic human demo tasks
- `robocasa/PnPCounterToCabinet_mg_18k` — Large MG dataset for one task

---

## 3. Dataset Classes: Groot-OpenPI Bridge

**File:** `src/openpi/groot_utils/groot_openpi_dataset.py`

This module bridges the Groot dataset format (used by the RoboCasa benchmark) with OpenPI's training pipeline. It imports from the `robocasa` package:

```python
from robocasa.utils.groot_utils.groot_dataset import (
    LeRobotSingleDataset, LeRobotMixtureDataset,
    LE_ROBOT_MODALITY_FILENAME, ModalityConfig, LE_ROBOT_EPISODE_FILENAME
)
from robocasa.utils.groot_utils.embodiment_tags import EmbodimentTag
```

If `robocasa` is not installed, a warning is logged but the import doesn't crash (allowing the rest of OpenPI to work for non-RoboCasa configs).

### 3.1 Scene Filtering — `get_scene_filtered_demos()`

The core filtering function. Given a dataset path, it loads `meta/episodes.jsonl` and each episode's `extras/episode_NNNNNN/ep_meta.json`, then applies filters in order:

1. **Episode ID allowlist** (`episode_ids`): Fast pre-check, skips metadata loading for non-matching IDs
2. **Layout/style pair** (`layout_and_style_ids`): Checks `(meta["layout_id"], meta["style_id"])` against the allowed set
3. **Fixture refs** (`fixture_refs`): Subset match — every key/value in the requested dict must appear in the episode's `fixture_refs`
4. **Object categories** (`object_categories`): Checks the primary object's category from `object_cfgs[0].info.cat`
5. **num_demos truncation**: Takes the first N after all other filters

Raises `ValueError` if no episodes survive filtering (prevents silent empty training).

### 3.2 `GrootOpenpiSingleDataset`

Extends `LeRobotSingleDataset` (from robocasa's Groot utilities). Constructor:

1. Reads modality configuration from the dataset's `modality.json` file
2. Sets up `ModalityConfig` for four modality types:
   - **video**: Both cameras at index 0 only (current frame)
   - **state**: All state keys except `state.dummy_tensor`
   - **action**: All action keys with `delta_indices = range(0, action_horizon)` (future action chunk)
   - **language**: Task description at index 0
3. Applies scene filtering via `get_scene_filtered_demos()` if `layout_and_style_ids` is provided
4. Passes either `filter_key` or `subset_demos` to the parent (not both)

**`__getitem__` override** — the critical data transform:

The Groot dataset stores state and action as separate modality keys. This method reassembles them into the OpenPI format:

```python
# State reordering: Groot → OpenPI
state = concat([
    item["state.end_effector_position_relative"],   # 3 dims
    item["state.end_effector_rotation_relative"],    # 4 dims
    item["state.base_position"],                     # 3 dims
    item["state.base_rotation"],                     # 4 dims
    item["state.gripper_qpos"],                      # 2 dims
])  # Total: 16 dims, shape (1, 16) → take [0] for (16,)

# Action reordering: Groot → OpenPI
actions = concat([
    item["action.end_effector_position"],            # 3 dims
    item["action.end_effector_rotation"],             # 3 dims
    item["action.gripper_close"],                     # 1 dim
    item["action.base_motion"],                       # 4 dims
    item["action.control_mode"],                      # 1 dim
])  # Total: 12 dims, shape (action_horizon, 12)
```

Returns dict with keys: `observation/image`, `observation/wrist_image`, `observation/state`, `actions`, `prompt`.

**Camera mapping:**
- `video.robot0_agentview_left` → `observation/image` (base camera, third-person view)
- `video.robot0_eye_in_hand` → `observation/wrist_image` (wrist-mounted camera)

### 3.3 `GrootOpenpiMultiDataset`

Extends `LeRobotMixtureDataset` for multi-task training with weighted dataset sampling.

**Weight calculation:** Uses power-law weighting with alpha=0.4:
```python
weight_i = len(dataset_i) ^ 0.4
```
This upweights smaller datasets relative to larger ones (prevents large datasets from dominating). Weights are normalized so the first dataset has weight 1.0 (Groot requirement).

**`sample_step` override:** The parent class's sampling is overridden to use fully random sampling:
- Randomly picks a dataset (weighted by `dataset_sampling_weights`)
- Randomly picks a trajectory within that dataset
- Randomly picks a timestep within that trajectory

This differs from the parent's deterministic sampling (which uses seed-based hashing). The override effectively ignores the `index` parameter.

### 3.4 Normalization Statistics

Two functions handle norm stats loading from Groot's `meta/stats.json`:

**`_load_norm_stats_from_groot_dataset()`**

Groot stores statistics in its own ordering. This function reorders them to match OpenPI's expected order:

```
Groot state order:      [base_pos(3), base_rot(4), ee_pos(3), ee_rot(4), gripper(2)]
  indices:               0,1,2       3,4,5,6      7,8,9      10,11,12,13 14,15
OpenPI state order:     [ee_pos(3), ee_rot(4), base_pos(3), base_rot(4), gripper(2)]
  reorder indices:       7,8,9      10,11,12,13  0,1,2       3,4,5,6     14,15

Groot action order:     [base_motion(4), control_mode(1), ee_pos(3), ee_rot(3), gripper(1)]
  indices:               0,1,2,3         4                5,6,7       8,9,10     11
OpenPI action order:    [ee_pos(3), ee_rot(3), gripper(1), base_motion(4), control_mode(1)]
  reorder indices:       5,6,7      8,9,10     11          0,1,2,3         4
```

After reordering, both state (16D) and actions (12D) are padded to 32 dimensions:
- Mean is zero-padded
- Std is one-padded (so padding dims are not normalized)

**`_load_norm_stats_from_groot_mixture_dataset()`**

Merges statistics across multiple datasets using `compute_overall_statistics()`, which computes weighted mean and std:
```
overall_mean = Σ(w_i * mean_i)
overall_var  = Σ(w_i * (std_i² + mean_i²)) - overall_mean²
overall_std  = sqrt(overall_var)
```
Currently uses uniform weights (all 1.0) across datasets.

---

## 4. Policy Transforms

**File:** `src/openpi/policies/robocasa_policy.py`

### 4.1 `RobocasaInputs`

A frozen dataclass implementing `transforms.DataTransformFn`. Applied to every training sample and inference input.

**What it does:**
1. **State padding:** Pads 16D state to `action_dim` (default 32) with zeros via `transforms.pad_to_dim()`
2. **Image parsing:** Converts images to uint8 format, handles both:
   - Float images in [0, 1] ��� multiply by 255
   - Channels-first (C, H, W) → rearrange to (H, W, C)
3. **Image dict construction:** Maps images to the Pi-0.5 expected keys:
   - `base_0_rgb` ← base camera image
   - `left_wrist_0_rgb` ← wrist camera image
   - `right_wrist_0_rgb` ← zeros (RoboCasa only has one wrist camera)
4. **Image mask:** For Pi0 model type, masks `right_wrist_0_rgb` as False (tells the model to ignore the padding image). For other model types, all True.
5. **Action padding:** If `actions` present (training), pads 12D actions to `action_dim`
6. **Prompt passthrough:** Copies the language prompt as-is

### 4.2 `RobocasaOutputs`

Extracts only the first 12 action dimensions from the model's output (which is 32D due to padding):
```python
{"actions": data["actions"][:, :12]}
```
This is the inverse of the input padding — slicing back to RoboCasa's native 12D action space.

### 4.3 `make_robocasa_example()`

Creates a random input example for testing the policy pipeline:
- `observation/state`: random float64 array of shape (16,)
- `observation/image`: random uint8 array of shape (224, 224, 3)
- `observation/wrist_image`: random uint8 array of shape (224, 224, 3)
- `prompt`: "do something"

---

## 5. Training Configuration

**File:** `src/openpi/training/config.py`

### 5.1 `LeRobotRobocasaDataConfig` (lines 645-754)

A frozen dataclass extending `DataConfigFactory`. This is the main configuration class for RoboCasa training.

**Fields:**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `data_dirs` | `list[dict]` | None | List of dataset dicts, each with `"path"` and `"filter_key"` |
| `dataset_weights` | `list[float]` | None | Custom weights for multi-dataset mixing |
| `layout_and_style_ids` | `list[tuple]` | None | Filter to specific (layout, style) pairs |
| `num_demos` | `int` | None | Max demos after filtering |
| `obj_category` | `str` | None | Filter by primary object category |
| `fixture_refs` | `dict` | None | Pin exact fixture locations (cabinet/counter names) |
| `match_episode_id` | `int` | None | Match full ep_meta of a reference episode |
| `object_categories` | `list[str]` | None | Filter by allowed object categories |
| `episode_ids` | `list[int]` | None | Explicit episode allowlist |
| `eval_init_mode` | `str` | None | How to initialize eval environments |
| `eval_pool_*` | various | — | Pool configuration for eval resets |
| `eval_keep_robot_pose` | `bool` | False | Preserve recorded robot pose on eval reset |
| `eval_robot_pose_noise` | `float` | 0.0 | Noise on robot XY + yaw (meters/radians) |
| `eval_object_pose_noise` | `float` | 0.0 | Noise on object XY position |
| `eval_object_ori_noise` | `float` | 0.0 | Noise on object yaw |

**`create()` method logic:**

1. Creates `RobocasaInputs` and `RobocasaOutputs` transform groups
2. Creates the model transform factory (standard Pi-0.5 transforms)
3. **Enriches data_dirs:** Copies all filter fields (`match_episode_id`, `layout_and_style_ids`, `num_demos`, etc.) into each data_dir dict. This passes filtering params down to the dataset constructor.
4. **Computes fallback norm stats:** If no pre-computed norm stats exist in the assets directory, it calls `compute_norm_stats_from_filtered_dataset()` on the enriched data_dirs. For multi-dataset configs, it computes per-dataset stats and merges them with `compute_overall_statistics()`.
5. Returns a `DataConfig` with all transforms, enriched data_dirs, and norm stats.

### 5.2 Training Configs (15 configs)

All RoboCasa configs share this base setup:
- **Model:** `Pi0Config(pi05=True, action_horizon=10, paligemma_variant="gemma_2b_lora")`
- **Checkpoint:** `gs://openpi-assets/checkpoints/pi05_base/params` (pre-trained Pi-0.5)
- **Output dir:** `/data/hf_cache/models/pi05_robocasa_exps/`
- **Freeze filter:** LoRA on vision backbone, full fine-tune on action head
- **Training:** 100K steps, batch_size=32, save every 2K-5K steps

**Config categories:**

#### Debug configs with `match_episode_id` (1-demo overfitting)

| Config | Task | Dataset Split | match_episode_id | Scene | GPUs |
|---|---|---|---|---|---|
| `..._layout2style2_ep_meta_debug_v1` | PickPlaceCounterToCabinet | target/20250811 | 23 | apple, layout=2, style=2 | 4 |
| `..._sink2counter_ep_meta_debug_v1` | PickPlaceSinkToCounter | target/20250813 | 10 | orange, layout=7, style=7 | 4 |
| `..._cabinet2counter_ep_meta_debug_v1` | PickPlaceCabinetToCounter | target/20250819 | 53 | marshmallow, layout=46, style=15 | 4 |

These are for sanity-checking: train on a single demo matched by full environment state.

#### Few-demo configs with `layout_and_style_ids`

| Config | Task | Layout/Style | Demos | Eval Mode |
|---|---|---|---|---|
| `..._single_task_lora` | PickPlaceCounterToCabinet | (1,1) | 5 | fixture_pair_same_category |
| `..._pick_place_stove` | PickPlaceStoveToCounter | (1,1) | 5 | fixture_pair_same_category |
| `..._exact_replay` | PickPlaceCounterToCabinet | (1,1) | 1 | exact_state_replay |
| `..._sink_to_counter` | PickPlaceSinkToCounter | (5,5) | 5 (episode_ids filtered) | — |
| `..._sink_to_counter_eval` | PickPlaceSinkToCounter | (5,5) | 5 | fixture_pair_same_category |

#### Fixed-fixture configs (largest matched groups)

| Config | Task | match_episode_id | Matched demos | Dataset |
|---|---|---|---|---|
| `..._turn_on_sink_faucet` | TurnOnSinkFaucet | 0 | 107 eps (all same scene) | pretrain/20250819 |
| `..._slide_dishwasher_rack` | SlideDishwasherRack | 0 | 24 eps | pretrain/20250820 |
| `..._pick_place_counter_to_cabinet` | PickPlaceCounterToCabinet | 1 | 6 eps | pretrain/20250819 |
| `..._prepare_coffee` | PrepareCoffee | 5 | 8 eps | composite/20250716 |
| `..._deliver_straw` | DeliverStraw | 51 | 3 eps | composite/20250723 |
| `..._get_toasted_bread` | GetToastedBread | 12 | 9 eps | composite/20250731 |

#### Multitask config

| Config | Notes |
|---|---|
| `pi05_robocasa_rlds_multitask` | No data_dirs set — placeholder for future multi-task setup |

---

## 6. Data Loading Integration

**File:** `src/openpi/training/data_loader.py` (lines 551-576)

The `create_torch_dataset()` function routes to RoboCasa datasets based on the `data_dirs` attribute:

```python
if getattr(data_config, "data_dirs", None):
    data_dirs = data_config.data_dirs
    if len(data_dirs) == 1:
        return GrootOpenpiSingleDataset(
            dataset_meta=data_dirs[0],
            action_horizon=action_horizon,
        )
    elif len(data_dirs) > 1:
        return GrootOpenpiMultiDataset(
            dataset_meta_list=data_dirs,
            dataset_weights=...,
            dataset_weights_alpha=0.4,
            action_horizon=action_horizon,
        )
```

This check runs before the RLDS, HDF5, and LeRobot-HuggingFace paths, giving RoboCasa/Groot datasets first priority when `data_dirs` is set.

---

## 7. Normalization Statistics Pipeline

**File:** `scripts/compute_norm_stats.py`

Computes per-dimension mean, std, q01, and q99 for `state` (32D padded) and `actions` (32D padded). Applied before every training run.

**RoboCasa-specific behavior:**
- When `data_config.repo_id` is None and `asset_id` is "robocasa", stats are written to `<assets_base_dir>/robocasa/`
- The `enforce_min_quantile_range` check (used in LIBERO configs) also applies to RoboCasa: if the gripper action dim has a collapsed range (< 0.5), it's forced to [-1, 1]

**Two stat computation paths:**
1. **From Groot's stats.json** (fallback in `LeRobotRobocasaDataConfig.create()`): Fast, reads pre-computed stats from the dataset's `meta/stats.json` and reorders indices
2. **From the actual data** (`compute_norm_stats.py` script): Iterates over the full filtered dataset, more accurate for small subsets

---

## 8. Evaluation Pipeline

**File:** `examples/robocasa/main.py`

A standalone evaluation script that:

1. **Creates a RoboCasa Gym environment** via `gymnasium.make(f"robocasa/{env_name}", **env_kwargs)`
2. **Connects to a model server** via `WebsocketClientPolicy(host, port)`
3. **Runs N rollout episodes** with action chunking (replan every `replan_steps` timesteps)
4. **Records videos** as .mp4 for each episode (success/failure labeled)
5. **Saves stats** to `stats.json` (success rate, episode count)

### Observation processing

At each timestep:
```python
# Rotate 180 degrees to match train preprocessing
img = obs["video.robot0_agentview_left"]
wrist_img = obs["video.robot0_eye_in_hand"]
# Resize with padding to 224x224
img = resize_with_pad(img, 224, 224)
wrist_img = resize_with_pad(wrist_img, 224, 224)

# State assembly (same order as training)
state = concat([
    obs["state.end_effector_position_relative"],
    obs["state.end_effector_rotation_relative"],
    obs["state.base_position"],
    obs["state.base_rotation"],
    obs["state.gripper_qpos"],
])
```

### Action execution

The model returns an action chunk of shape (action_horizon, 12). Only the first `replan_steps` (default 5) actions are executed before re-querying the model. Each action is converted via `robocasa.utils.env_utils.convert_action()` before stepping the environment.

### Eval initialization modes

The script supports controlled evaluation resets via `RoboCasaEvalResetController` (imported from the `dsrl_pi0` project):

| Mode | Description |
|---|---|
| `exact_state_replay` | Restore the exact simulator state from a recorded demo |
| `fixture_pair_fresh_placement` | Same fixture pair, fresh random object placement |
| `fixture_pair_same_category` | Same fixture pair, fresh placement with same object category |
| `fixture_pair_object_pool` | Same fixture pair, sample from an object pool |

Noise parameters (`eval_robot_pose_noise`, `eval_object_pose_noise`, `eval_object_ori_noise`) add uniform perturbations on top of the reset state.

### Oracle video copying

When using eval init modes, the script copies the corresponding oracle demo video from the dataset for side-by-side comparison.

---

## 9. Eval Statistics Aggregation

**File:** `examples/robocasa/get_eval_stats.py`

Reads `stats.json` files from completed evaluations and aggregates success rates across task groups defined in `robocasa.utils.dataset_registry`:

| Group | Tasks |
|---|---|
| `atomic_seen` | 19 atomic tasks from TARGET_TASKS |
| `atomic_seen_no_nav` | Same minus NavigateKitchen |
| `composite_seen` | Seen composite tasks |
| `composite_unseen` | Unseen composite tasks |
| `lifelong_learning_phase1-4` | Progressive lifelong learning splits |

Prints a table of per-task and average success rates for both pretrain and target splits.

---

## 10. Visualization Scripts

### `scripts/visualize_robocasa_demos.py`

Loads a training config, builds the filtered dataset (honoring all scene/object filters), and writes one .mp4 per episode showing:
- Side-by-side base and wrist camera views
- Overlay: task prompt, per-dim action values, timestep counter, camera labels

Usage:
```bash
python scripts/visualize_robocasa_demos.py \
    --config-name pi05_robocasa_single_task_lora \
    --output-dir vis_demos/ \
    --num-episodes 3
```

### `scripts/visualize_all_robocasa_tasks.py`

Visualizes one demo per task across all 18 PickPlace tasks. Scans dataset directories, generates side-by-side .mp4 videos using ffmpeg.

### Inspection scripts

| Script | Purpose |
|---|---|
| `scripts/inspect_episodes.py` | Frame montages (12 sampled frames/episode) with metadata |
| `scripts/inspect_dense.py` | Dense 5x4 grid montages for specific episodes |
| `scripts/dense_occlusion_check.py` | High-density montage (20 frames) for checking object visibility |
| `scripts/occlusion_analysis.py` | Frame montages with object name extraction |
| `scripts/generate_episode_video.py` | Single episode .mp4 with side-by-side cameras |
| `scripts/dump_single_episode.py` | Parametric single-episode video dumper |

---

## 11. SLURM Submission Scripts

**Directory:** `slurm/robocasa/`

| Script | Config | Partition | GPUs | Notes |
|---|---|---|---|---|
| `debug.slurm` | `..._layout2style2_ep_meta_debug_v1` | general | 4x L40S | PickPlaceCounterToCabinet, match_episode_id=23 |
| `debug_pickplacesinkcounter.sh` | `..._sink2counter_ep_meta_debug_v1` | general | 2x L40S | PickPlaceSinkToCounter, match_episode_id=10 |
| `debug_pickplacecabinetcounter.sh` | `..._cabinet2counter_ep_meta_debug_v1` | general | 2x L40S | PickPlaceCabinetToCounter, match_episode_id=53 |
| `slurm_robocasa_single_task.sh` | `..._sink_to_counter` | general | 4 GPUs | Supports `NUM_DEMOS` env var override |
| `slurm_robocasa_single_task_rl.sh` | `..._sink_to_counter` | rl | 2x RTX_PRO_6000 | Same but RL partition |

All scripts follow the pattern:
1. Set environment variables (XLA mem fraction, WANDB entity, conda env)
2. Run `compute_norm_stats.py` (some scripts skip this, relying on cached stats)
3. Run `train.py` with the config name and experiment name

---

## 12. Dataset Directory Structure

RoboCasa LeRobot datasets live at `/data/hf_cache/datasets/robocasa/v1.0/` with the structure:

```
v1.0/
├── pretrain/
│   ├── atomic/
│   │   ├── PickPlaceCounterToCabinet/20250819/lerobot/
│   │   ├── PickPlaceStoveToCounter/20250819/lerobot/
│   │   ├── PickPlaceSinkToCounter/20250819/lerobot/
│   │   ├── TurnOnSinkFaucet/20250819/lerobot/
│   │   ├── SlideDishwasherRack/20250820/lerobot/
│   │   └── ...
│   └── composite/
│       ├── PrepareCoffee/20250716/lerobot/
│       ├── DeliverStraw/20250723/lerobot/
│       └── GetToastedBread/20250731/lerobot/
└── target/
    └── atomic/
        ├── PickPlaceCounterToCabinet/20250811/lerobot/
        ├── PickPlaceSinkToCounter/20250813/lerobot/
        ├── PickPlaceCabinetToCounter/20250819/lerobot/
        └── ...
```

Each `lerobot/` directory contains:
- `meta/episodes.jsonl` — Episode index metadata
- `meta/stats.json` — Pre-computed normalization statistics (Groot format)
- `extras/episode_NNNNNN/ep_meta.json` — Per-episode scene metadata (layout_id, style_id, fixture_refs, object_cfgs)
- `videos/chunk-000/` — Video files per camera per episode
- Parquet data files with state/action data

---

## 13. Key Design Decisions

### Why reorder state/action dimensions?

Groot's native ordering puts base position/rotation before end-effector. OpenPI (inherited from Pi-0's convention) puts end-effector first. The reordering happens in two places:
- `GrootOpenpiSingleDataset.__getitem__()` — for training data
- Eval script observation assembly — for inference

Both must use the same order, or normalization statistics will be misapplied.

### Why pad to 32 dimensions?

Pi-0.5 expects a fixed action dimension (default 32) across all tasks. RoboCasa's 16D state and 12D actions are padded with zeros (state) or zeros+ones for std (norm stats). The `RobocasaOutputs` transform slices back to 12D after model inference.

### Why compute norm stats from filtered demos?

When training on a small subset (e.g., 5 demos from layout 1, style 1), the global dataset statistics may not represent the actual training distribution well. Computing stats from the filtered subset ensures normalization matches what the model actually sees. The config's `create()` method handles this as a fallback when no pre-computed stats exist.

### Why override `sample_step` in `GrootOpenpiMultiDataset`?

The parent class's sampling uses deterministic seed-based hashing. The override switches to fully random sampling for training, which provides better data diversity per epoch. The `index` parameter is effectively ignored.

---

## 14. Robot and Environment Details

### Robot: PandaOmron

The mobile manipulator used in all RoboCasa tasks:
- **Arm:** Franka Panda 7-DOF
- **Base:** Omron mobile base (holonomic, 3-DOF: x, y, yaw)
- **Gripper:** Parallel-jaw gripper (2 finger joints)
- **Cameras:** `robot0_agentview_left` (third-person), `robot0_eye_in_hand` (wrist-mounted)

### Action Space (12D)

| Dims | Component | Range | Description |
|---|---|---|---|
| 0-2 | `ee_pos` | continuous | End-effector position delta (dx, dy, dz) |
| 3-5 | `ee_rot` | continuous | End-effector rotation delta (axis-angle) |
| 6 | `gripper_close` | [-1, 1] | Gripper command (-1=open, 1=close) |
| 7-10 | `base_motion` | continuous | Mobile base velocity (vx, vy, vyaw, spare) |
| 11 | `control_mode` | 0 or 1 | Whether base is actively controlled |

The `convert_action()` function in the eval script converts from the model's raw output to the environment's expected format.

### Control frequency

All environments run at **20 Hz**. The policy predicts `action_horizon=10` future actions (500ms lookahead) and executes `replan_steps=5` (250ms) before re-querying.
