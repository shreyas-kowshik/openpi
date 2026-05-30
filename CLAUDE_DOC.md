# OpenPI + RoboCasa Training Guide

This document explains how to set up data, configure, and run openpi training on robocasa environments.

---

## Quick Start

```bash
# 1. Set up dataset path
cd /home/skowshik/vla/codebase/openpi_robocasa/robocasa
python robocasa/scripts/setup_macros.py
# Edit robocasa/macros_private.py → set DATASET_BASE_PATH="/your/datasets/path"

# 2. Download data
python robocasa/scripts/download_datasets.py \
    --tasks PickPlaceCounterToCabinet --split target --source human

# 3. Edit openpi/src/openpi/training/config.py:
#    - Set assets_dir=None
#    - Set data_dirs[0]["path"] to: DATASET_BASE_PATH/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot
#    - Set checkpoint_base_dir="./checkpoints"

# 4. Train
cd /home/skowshik/vla/codebase/openpi_robocasa/openpi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python scripts/train.py pi05_robocasa_single_task_lora \
    --exp-name sanity_40demos --batch-size 32 --overwrite
```

---

## Repository Layout (Relevant Paths)

```
openpi/
├── scripts/train.py                              # Training entry point
├── src/openpi/training/config.py                 # All training configs (including robocasa)
├── src/openpi/policies/robocasa_policy.py        # RobocasaInputs / RobocasaOutputs transforms
├── src/openpi/groot_utils/groot_openpi_dataset.py # Groot dataset wrapper for training
├── examples/robocasa/
│   ├── convert_robocasa_to_lerobot.py            # Simple LeRobot converter (NOT for Groot)
│   └── main.py                                   # Policy evaluation on robocasa envs
└── slurm_robocasa_single_task.sh                 # Reference SLURM script

robocasa/  (at /home/skowshik/vla/codebase/openpi_robocasa/robocasa)
├── robocasa/scripts/
│   ├── collect_demos.py                          # Human teleoperation data collection
│   ├── download_datasets.py                      # Download pre-made datasets from Box
│   ├── setup_macros.py                           # Set up DATASET_BASE_PATH
│   └── dataset_scripts/
│       ├── dataset_states_to_obs.py              # Convert raw states HDF5 → image HDF5
│       └── convert_hdf5_lerobot.py               # Convert image HDF5 → Groot/LeRobot format
├── robocasa/utils/dataset_registry.py            # Task names, dataset paths registry
└── robocasa/macros.py                            # DATASET_BASE_PATH setting
```

---

## Overview: Training Data Pipeline

```
Option A (Recommended): Download pre-made data
  robocasa download_datasets.py  →  DATASET_BASE_PATH/v1.0/{split}/atomic/{TASK}/{DATE}/lerobot/

Option B: Collect new human demos
  collect_demos.py (human teleoperation)
    → raw states HDF5  (demo.hdf5 - no images)
    → dataset_states_to_obs.py
    → image HDF5 (demo_im128.hdf5)
    → convert_hdf5_lerobot.py (robocasa's version)
    → lerobot/ (Groot format with videos, modality config)

Training:
  lerobot/ (Groot format)  →  GrootOpenpiSingleDataset  →  scripts/train.py
```

**Important**: The training uses the **Groot/LeRobot format** produced by `robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py`. The script at `openpi/examples/robocasa/convert_robocasa_to_lerobot.py` is a simpler/older approach that produces a different format and is NOT what the current training configs use.

---

## Step 1: Set Up DATASET_BASE_PATH

Robocasa uses a private macros file to locate datasets. Run this once:

```bash
cd /home/skowshik/vla/codebase/openpi_robocasa/robocasa
python robocasa/scripts/setup_macros.py
```

This creates `robocasa/macros_private.py`. Edit it to set your dataset path:

```python
# robocasa/macros_private.py
DATASET_BASE_PATH = "/home/skowshik/vla/codebase/openpi_robocasa/datasets"
```

After setting this, datasets will be downloaded to/read from:
```
DATASET_BASE_PATH/v1.0/{split}/atomic/{TASK}/{DATE}/lerobot/
```

> **Note**: If `DATASET_BASE_PATH` is `None`, it defaults to `../datasets` relative to the robocasa package installation directory.

---

## Step 2: Download Pre-Made Data

The robocasa team provides pre-made datasets in Groot/LeRobot format hosted on Box. Downloads are managed via `download_datasets.py`.

### Recommended: PickPlaceCounterToCabinet (target split, human demos)

```bash
cd /home/skowshik/vla/codebase/openpi_robocasa/robocasa
python robocasa/scripts/download_datasets.py \
    --tasks PickPlaceCounterToCabinet \
    --split target \
    --source human
```

This downloads ~500 human demonstrations for the target evaluation split to:
```
DATASET_BASE_PATH/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot/
```

The dataset is in Groot/LeRobot format with:
- H264 video streams for 3 cameras
- State/action arrays
- Language annotations per episode
- Episode metadata with `layout_id` and `style_id`

### Other available splits

```bash
# Pretrain split - human (100 demos)
python robocasa/scripts/download_datasets.py \
    --tasks PickPlaceCounterToCabinet \
    --split pretrain \
    --source human

# Pretrain split - MimicGen (10,000 demos)
python robocasa/scripts/download_datasets.py \
    --tasks PickPlaceCounterToCabinet \
    --split pretrain \
    --source mimicgen
```

### Check what's available (dry run)

```bash
python robocasa/scripts/download_datasets.py \
    --tasks PickPlaceCounterToCabinet \
    --split target pretrain \
    --source human mimicgen \
    --dryrun
```

---

## Step 3: Convert Your Own Data (Optional - if NOT downloading)

If you collect your own demos via `collect_demos.py`, you need to convert them. Skip this if using downloaded data.

### 3a. Collect human demos (requires display + keyboard/SpaceMouse)

```bash
cd /home/skowshik/vla/codebase/openpi_robocasa/robocasa
python robocasa/scripts/collect_demos.py \
    --environment PickPlaceCounterToCabinet \
    --robots PandaOmron \
    --directory /path/to/save/demos \
    --split target \
    --layout 1 \
    --style 1 \
    --device keyboard
```

This saves to `/path/to/save/demos/{TIMESTAMP}_PickPlaceCounterToCabinet/demo.hdf5` (raw states only, no images).

### 3b. Extract image observations from states HDF5

```bash
python robocasa/scripts/dataset_scripts/dataset_states_to_obs.py \
    --dataset /path/to/demo.hdf5 \
    --camera_names robot0_agentview_left robot0_agentview_right robot0_eye_in_hand \
    --camera_height 256 \
    --camera_width 256 \
    --done_mode 2 \
    --num_procs 4 \
    --gpu_ids 0
```

Output: `demo_im256.hdf5` in the same directory.

### 3c. Convert image HDF5 to Groot/LeRobot format

```bash
python robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py \
    --raw_dataset_path /path/to/demo_im256.hdf5 \
    --camera_names robot0_eye_in_hand robot0_agentview_left robot0_agentview_right \
    --camera_height 256 \
    --camera_width 256
```

Output: `/path/to/lerobot/` directory (sibling of the HDF5 file).

---

## Step 4: Understand the Training Configs

There are two robocasa training configs in `src/openpi/training/config.py`:

### Config 1: `pi05_robocasa_single_task_lora` (recommended for sanity checking)

```python
TrainConfig(
    name="pi05_robocasa_single_task_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        paligemma_variant="gemma_2b_lora",   # LoRA on VLM, full fine-tune on action head
    ),
    data=LeRobotRobocasaDataConfig(
        assets=AssetsConfig(
            assets_dir="...",          # <-- HARDCODED PATH: needs update
            asset_id="robocasa",
        ),
        data_dirs=[{
            "path": "...",             # <-- HARDCODED PATH: needs update
            "filter_key": None,
        }],
        layout_and_style_ids=[(1, 1)], # Restrict to layout=1, style=1 (same scene, less variance)
        num_demos=40,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    checkpoint_base_dir="...",         # <-- HARDCODED PATH: needs update
    freeze_filter=...,                 # Freezes VLM backbone, only trains LoRA + action head
    ema_decay=None,
    num_train_steps=100_000,
    batch_size=32,
)
```

**Key parameters to understand:**
- `layout_and_style_ids=[(1, 1)]`: Restricts training data to episodes recorded in kitchen layout 1, style 1. This dramatically reduces task variability and makes learning easier for sanity checking.
- `num_demos=N`: Limits the number of demos used after scene filtering.
- `paligemma_variant="gemma_2b_lora"`: Uses LoRA for memory-efficient fine-tuning.
- `weight_loader`: Loads base pi0.5 weights from GCS (requires `gsutil`/GCS access).

### Config 2: `pi05_robocasa_rlds_multitask`

Multi-task training config. Requires a pre-computed norm stats file. Not suited for quick sanity checking.

---

## Step 5: Update Config Paths

The training configs have hardcoded absolute paths from the original developer's machine. You need to update them.

### Option A: Edit `config.py` directly (recommended for simplicity)

Open `src/openpi/training/config.py` and find the `pi05_robocasa_single_task_lora` config (~line 1139). Update:

```python
TrainConfig(
    name="pi05_robocasa_single_task_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        paligemma_variant="gemma_2b_lora",
    ),
    data=LeRobotRobocasaDataConfig(
        assets=AssetsConfig(
            assets_dir=None,           # ← Set to None (will auto-compute from data_dirs)
            asset_id="robocasa",
        ),
        data_dirs=[{
            "path": "/home/skowshik/vla/codebase/openpi_robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot",
            "filter_key": None,
        }],
        layout_and_style_ids=[(1, 1)],
        num_demos=40,                  # ← Change to 5 for minimal test
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    checkpoint_base_dir="./checkpoints",   # ← Local path
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
    ).get_freeze_filter(),
    ema_decay=None,
    num_train_steps=100_000,
    batch_size=32,
    save_interval=2_000,
    num_workers=4,
    log_interval=100,
),
```

> **Note on `assets_dir=None`**: When `assets_dir` is None, the fallback in `LeRobotRobocasaDataConfig.create()` automatically loads normalization stats (mean, std, q01, q99) from the dataset's `meta/stats.json`. This is computed during dataset creation and already present in downloaded datasets. The stats are reordered to match openpi's expected state/action ordering. Both z-score and quantile normalization (required for pi0.5) are fully supported.

### Option B: CLI overrides (partial - complex args may not work)

For simple overrides:
```bash
python scripts/train.py pi05_robocasa_single_task_lora \
    --exp-name my_exp \
    --checkpoint-base-dir ./checkpoints \
    --data.num-demos 40 \
    --overwrite
```

> **Limitation**: The `data.data_dirs` and `data.assets.assets_dir` require complex nested overrides that may not work cleanly with `tyro`. Editing `config.py` is more reliable.

---

## Step 6: Run Training

### Environment Setup

```bash
# Set GPU rendering for headless environments
export EGL_DEVICE_ID=0
export MUJOCO_EGL_DEVICE_ID=0

# Reduce JAX memory fragmentation
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Optionally disable wandb if not configured
export WANDB_MODE=disabled
```

### Training Command

```bash
cd /home/skowshik/vla/codebase/openpi_robocasa/openpi

# Sanity check with 5 demos
python scripts/train.py pi05_robocasa_single_task_lora \
    --exp-name sanity_5demos \
    --batch-size 8 \
    --num-train-steps 500 \
    --overwrite

# Sanity check with 40-50 demos
python scripts/train.py pi05_robocasa_single_task_lora \
    --exp-name sanity_40demos \
    --batch-size 32 \
    --overwrite
```

### With SLURM

A reference SLURM script is at `slurm_robocasa_single_task.sh`. Adapt the paths:

```bash
# With 40 demos (default)
sbatch slurm_robocasa_single_task.sh

# With custom demo count
NUM_DEMOS=5 sbatch slurm_robocasa_single_task.sh
```

---

## Data Format Reference

### Groot/LeRobot Directory Structure (after download)

```
lerobot/
├── meta/
│   ├── info.json           # Dataset metadata
│   ├── episodes.jsonl      # Per-episode metadata (episode_index, length, task_index)
│   ├── tasks.jsonl         # Task descriptions (task_index, task name)
│   └── stats.json          # Normalization stats (mean, std for all modalities)
├── videos/
│   ├── observation.images.robot0_agentview_left/  # H264 videos, one per episode
│   ├── observation.images.robot0_agentview_right/
│   └── observation.images.robot0_eye_in_hand/
├── data/
│   └── chunk-000/          # Parquet files with state, action, annotation scalars
├── extras/
│   └── episode_{N:06d}/    # Per-episode extra files
│       └── ep_meta.json    # Contains layout_id, style_id, lang description
└── modality.json           # Groot modality configuration (key ordering)
```

### State and Action Dimensions

| Component | Dims | Description |
|-----------|------|-------------|
| `state` (total) | 16 | Proprioception |
| - `end_effector_position_relative` | 3 | EEF pos relative to base |
| - `end_effector_rotation_relative` | 4 | EEF rot (quaternion) relative to base |
| - `base_position` | 3 | Mobile base position (x, y, theta=0,0,0) |
| - `base_rotation` | 4 | Mobile base rotation (quaternion) |
| - `gripper_qpos` | 2 | Gripper joint positions |
| `actions` (total) | 12 | Control signals |
| - `end_effector_position` | 3 | EEF position delta |
| - `end_effector_rotation` | 3 | EEF rotation delta |
| - `gripper_close` | 1 | Gripper open/close signal |
| - `base_motion` | 4 | Mobile base motion |
| - `control_mode` | 1 | Control mode flag |

**Model padding**: Pi0.5 has `action_dim=32`. State and actions are padded to 32 dims.

### Camera Inputs

| Camera | Key in training | Shape |
|--------|-----------------|-------|
| Left agent view | `observation/image` | 224×224×3 |
| Wrist (eye-in-hand) | `observation/wrist_image` | 224×224×3 |
| Right wrist | padded with zeros | 224×224×3 |

---

## Key Configuration Parameters

### Controlling Scene Variability

`layout_and_style_ids` restricts which kitchen scenes are included in training:

```python
layout_and_style_ids = [(1, 1)]   # Single fixed scene (easiest)
layout_and_style_ids = [(1, 1), (2, 2), (3, 3)]  # 3 scenes
layout_and_style_ids = None        # All scenes in dataset (hardest, most generalization)
```

- Layout IDs: 1–10 (different kitchen floor plans)
- Style IDs: 1–10 (different textures/aesthetics)
- "Pretrain" split covers layouts 1–10 × styles 1–10 with some demos per combo
- "Target" split covers a subset of 10 scenes (layout/style matched)

### Controlling Demo Count

```python
num_demos = 5    # Minimal test run
num_demos = 40   # Sanity check (recommended)
num_demos = 500  # Full target split
num_demos = None # Use all demos after scene filtering
```

### LoRA vs Full Fine-tuning

```python
# LoRA (memory efficient, recommended for <100 demos):
paligemma_variant="gemma_2b_lora"
freeze_filter=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora").get_freeze_filter()

# Full fine-tuning (remove freeze_filter):
paligemma_variant=None  # default
freeze_filter=nnx.Nothing()
```

---

## Normalization Stats

Stats are loaded from `LEROBOT_PATH/meta/stats.json` automatically when `assets_dir=None`. The `_load_norm_stats_from_groot_dataset()` function:

1. Reads `meta/stats.json` from the dataset (contains `mean`, `std`, `q01`, `q99` for all modalities)
2. Reorders state from `[base_pos, base_rot, ee_pos, ee_rot, gripper]` → `[ee_pos, ee_rot, base_pos, base_rot, gripper]` (openpi ordering)
3. Reorders actions from `[base_motion, control_mode, ee_pos, ee_rot, gripper]` → `[ee_pos, ee_rot, gripper, base_motion, control_mode]`
4. Pads to 32 dims (model action_dim): zeros for `mean`/`q01`, ones for `std`/`q99`

**Quantile normalization**: pi0.5 (`ModelType.PI05`) uses `use_quantile_norm=True`, which requires `q01` and `q99`. The `meta/stats.json` in downloaded datasets contains these fields, and they are now correctly loaded and reordered by `_load_norm_stats_from_groot_dataset()` in `groot_openpi_dataset.py`.

---

## Checklist Before Running

- [ ] `DATASET_BASE_PATH` set in `robocasa/macros_private.py`
- [ ] Dataset downloaded to correct path (verify the `lerobot/` directory exists with `meta/stats.json`)
- [ ] Config paths updated in `src/openpi/training/config.py`
- [ ] `pi05_base` weights accessible (GCS: `gs://openpi-assets/checkpoints/pi05_base/params`)
- [ ] Wandb configured (or set `WANDB_MODE=disabled`)
- [ ] GPU available with enough VRAM (16GB+ for LoRA, 40GB+ for full)

---

## Known Issues / TODOs in the Codebase

1. **Language prompt is an integer index** (`groot_openpi_dataset.py` line 161): The `annotation.human.task_description` field in the Groot format stores an *integer task index* (not the text string). The text is stored in `tasks.jsonl`. The `LeRobotSingleDataset` from robocasa's groot_utils should decode this back to a string. If you see tokenization errors or empty prompts, this mapping may be broken.

2. **Filter key not fully tested**: The `filter_key` field in `data_dirs` entries (`{"path": "...", "filter_key": None}`) is passed to the dataset's subset-loading logic. `None` means "use all demos" (after `layout_and_style_ids` filtering). Setting this to a string key name would load a specific predefined subset (from the HDF5 `mask` group).

3. **Layout IDs** — special values in `collect_demos.py`:
   - `layout_ids=-1`: All layouts
   - `layout_ids=-2`: Simple layouts (no islands/wall stacks) — used for "pretrain" split
   - `layout_ids=-3`: Layouts with islands/wall stacks
   - `layout_ids=-4`: Layouts with dining areas
   - The downloaded "target" datasets use specific numbered layouts (1–10)

4. **Norm stats from dataset**: When `assets_dir=None`, stats come from the training data's `meta/stats.json`. This means the normalization stats are computed only from the N filtered demos, not the full dataset. For very few demos (5–10), this may give poor normalization estimates.

---

## Troubleshooting

### "No module named 'robosuite'"
Robocasa depends on robosuite. Install it:
```bash
pip install robosuite  # or install from source
```

### Norm stats not found
If `meta/stats.json` is missing from the lerobot directory, the downloaded dataset may be incomplete. Re-download or manually compute stats:
```bash
# The LeRobot dataset library can recompute stats (check lerobot docs)
```

### "No episodes match layout_and_style_ids"
The requested (layout_id, style_id) pairs don't exist in the dataset. For the target split of PickPlaceCounterToCabinet:
- Valid pairs: (1,1) through (10,10), but not all combinations may have data
- The pretrain split uses layout_ids=-2, style_ids=-2 (random sampling)
- Remove `layout_and_style_ids` or use `None` to use all available episodes

### Weight loading (GCS access)
The `pi05_base` weights are loaded from GCS on first run:
```bash
# Check if gcloud credentials are set up
gcloud auth application-default login
# Or set env var
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```
