# RoboCasa Simplified Environment & Train/Eval Matching Plan

## Goal

1. Simplify the RoboCasa environment to minimize randomness for initial experiments.
2. Ensure training demos and evaluation rollouts use the **exact same** environment setup.
3. Filter demos to only those matching a specific setup, and train on those.

---

## Part 1: Simplify the Environment

### Sources of Randomness in RoboCasa

| Source | What it controls | How to fix |
|--------|-----------------|------------|
| `layout_id` | Kitchen floor plan (counter arrangement, appliance positions) | Fix to a single ID, e.g. `layout_ids=1` |
| `style_id` | Kitchen textures/aesthetics (cabinet color, counter material) | Fix to a single ID, e.g. `style_ids=1` |
| `obj_instance_split` | Which physical object instances are used | Fix to `"target"` |
| `clutter_mode` | Whether distractor objects are placed in the scene | Set to `0` (no clutter) for simplest |
| Object category | Which object type is picked (honey_bottle, banana, etc.) | Fixed per-episode via `object_cfgs` in `ep_meta` |
| Object position | Where exactly the object is placed on the fixture | Seeded via `self.rng`, or fixed via `ep_meta` |
| Robot base position | Where the robot starts | Fixed via `init_robot_base_pos` in `ep_meta` |
| Camera positions | Camera viewpoints | Fixed via `cam_configs` in `ep_meta`, or `randomize_cameras=False` |

### Recommended Simplest Configuration

For initial sanity checking, fix everything:

```python
env = robosuite.make(
    env_name="PickPlaceCounterToCabinet",
    robots="PandaOmron",
    layout_ids=1,                     # Fixed layout
    style_ids=1,                      # Fixed style
    obj_instance_split="target",      # Target object pool
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["robot0_agentview_left", "robot0_eye_in_hand"],
    camera_height=256,
    camera_width=256,
    control_freq=20,
    randomize_cameras=False,          # No camera randomization
    seed=42,                          # Fixed seed for reproducibility
)
```

This still has some per-episode randomness (object category, exact placement). To fully fix everything, use `ep_meta` from a specific demo episode.

---

## Part 2: Exact Train/Eval Matching via `ep_meta`

### How `ep_meta` Works

Every RoboCasa demo stores complete environment state in `ep_meta.json`:

```
extras/episode_XXXXXX/ep_meta.json
```

Contains:
- `layout_id`: int (e.g., 1)
- `style_id`: int (e.g., 1)
- `object_cfgs`: Full object specifications (category, placement, model path)
- `fixtures`: All fixtures in the scene
- `fixture_refs`: Named references (e.g., `{"cab": "cab_1_right_group"}`)
- `init_robot_base_pos`: Exact robot starting position `[x, y, z]`
- `init_robot_base_ori`: Exact robot starting orientation `[rx, ry, rz]`
- `cam_configs`: Camera positions/orientations for all 5 cameras
- `gen_textures`: Generated texture mappings
- `lang`: Task instruction string

### Reproducing Exact Environment from `ep_meta`

```python
import json
import robosuite

# Load ep_meta from a specific demo
with open("extras/episode_000042/ep_meta.json") as f:
    ep_meta = json.load(f)

# Create environment
env = robosuite.make(
    env_name="PickPlaceCounterToCabinet",
    robots="PandaOmron",
    layout_ids=ep_meta["layout_id"],
    style_ids=ep_meta["style_id"],
    ...
)

# Set ep_meta BEFORE reset - this forces exact reproduction
env.set_ep_meta(ep_meta)
env.reset()
# Environment is now in the EXACT same state as the demo
```

### What `set_ep_meta` Controls During Reset

In `kitchen.py`'s `_setup_model()` (called during `reset()`):
1. If `layout_id` in `_ep_meta` → uses it (skips random sampling from `layout_and_style_ids`)
2. If `style_id` in `_ep_meta` → uses it
3. If `object_cfgs` in `_ep_meta` → uses exact object configs (skips `_get_obj_cfgs()`)
4. If `init_robot_base_pos` in `_ep_meta` → places robot at exact position
5. If `cam_configs` in `_ep_meta` → uses exact camera configurations

This means **every source of randomness is eliminated** when `ep_meta` is provided.

---

## Part 3: Filtering Demos for Single-Setup Training

### Current Dataset Structure (PickPlaceCounterToCabinet target split)

502 total episodes distributed across 10 (layout, style) pairs:
- (1,1): 40 eps, (2,2): 59 eps, (3,3): 45 eps, (4,4): 43 eps, (5,5): 49 eps
- (6,6): 54 eps, (7,7): 59 eps, (8,8): 54 eps, (9,9): 53 eps, (10,10): 46 eps

Within each (layout, style) pair, there's still randomness in:
- Object category (e.g., honey_bottle, banana, ketchup, etc.)
- Object placement position
- Robot base position (small deviations)

### Filtering Strategy

**Level 1 - Filter by (layout_id, style_id)** (already implemented):
```python
# In training config:
layout_and_style_ids=[(1, 1)]  # Only use layout 1, style 1 demos
num_demos=40                    # All 40 demos from this scene
```
This is done in `get_scene_filtered_demos()` in `groot_openpi_dataset.py`.

**Level 2 - Filter by object category** (IMPLEMENTED):
Within a (layout, style), further restrict to demos with the same object category.
This ensures the policy sees the same object appearance across all training demos.
Use `obj_category="apple"` in config.

**Level 3 - Use single ep_meta for evaluation** (IMPLEMENTED):
Pick one specific ep_meta from the training set and use it during all evaluation rollouts.
This ensures the evaluation environment is **identical** to one of the training episodes.
Reference ep_meta is auto-saved to `{checkpoint_dir}/reference_ep_meta.json`.

### Implementation (DONE)

All filtering and train/eval matching has been implemented:

#### 1. Object category filtering (`groot_openpi_dataset.py`)

`get_scene_filtered_demos()` now accepts `obj_category` parameter:
```python
filtered = get_scene_filtered_demos(
    dataset_path, [(1, 1)], num_demos=40, obj_category="apple"
)
```

#### 2. ep_meta extraction utilities (`groot_openpi_dataset.py`)

```python
# Get ep_meta for a specific episode
ep_meta = get_ep_meta_for_episode(dataset_path, episode_idx=42)

# Get reference ep_meta from first matching episode (for eval)
ref_meta = get_reference_ep_meta(dataset_path, [(1, 1)], obj_category="apple")

# List available object categories
cats = list_available_obj_categories(dataset_path, [(1, 1)])
# Returns: {"apple": 2, "bar": 2, "cinnamon": 2, ...}
```

#### 3. Config support (`config.py`)

`LeRobotRobocasaDataConfig` now has `obj_category` field:
```python
data=LeRobotRobocasaDataConfig(
    data_dirs=[{"path": "...", "filter_key": None}],
    layout_and_style_ids=[(1, 1)],
    num_demos=40,
    obj_category="apple",  # NEW: only train on apple demos
)
```

#### 4. Reference ep_meta saved during training (`train.py`)

Training automatically saves `reference_ep_meta.json` to checkpoint directory.
This contains the ep_meta of the first training episode for eval reproduction.

#### 5. Evaluation uses reference ep_meta

```python
import json

# Load reference from checkpoint
with open(f"{checkpoint_dir}/reference_ep_meta.json") as f:
    ref_ep_meta = json.load(f)

# Create env and set ep_meta before every reset
env.set_ep_meta(ref_ep_meta)
env.reset()  # Exact same setup as training demo
```

#### Object Categories Available (layout=1, style=1)

```
apple: 2 eps, bar: 2 eps, cinnamon: 2 eps, tomato: 2 eps,
boxed_food: 2 eps, teapot: 2 eps, salt_and_pepper_shaker: 1 ep,
tongs: 1 ep, hot_dog: 1 ep, jam: 1 ep, squash: 1 ep, cup: 1 ep,
... (34 total categories across 40 episodes)
```

---

## Part 4: Data Flow Summary

```
Dataset (502 episodes)
    │
    ├─ Filter by (layout_id=1, style_id=1) → 40 episodes
    │
    ├─ [Optional] Filter by obj_category → N episodes
    │
    ├─ Compute norm stats from filtered subset (auto-cached)
    │
    ├─ Train policy on filtered demos
    │
    └─ Save reference ep_meta from first training episode
         │
         └─ Evaluation: env.set_ep_meta(reference_ep_meta) → identical setup
```

---

## Part 5: Norm Stats

### How norm stats work for RoboCasa

1. **Auto-computation**: `LeRobotRobocasaDataConfig.create()` automatically computes norm stats from the filtered dataset if not provided via assets. Results are cached to:
   ```
   {dataset_path}/computed_norm_stats/{tag}.json
   ```
   where `tag` encodes the filter params (e.g., `ls_11__n40.json`).

2. **Standalone script**: `scripts/compute_norm_stats.py` can also compute stats:
   ```bash
   python scripts/compute_norm_stats.py --config-name=pi05_robocasa_single_task_lora_fresh_debug_v1
   ```

3. **Stats structure** (padded to 32 dims):
   - State: 16 real dims [ee_pos_rel(3), ee_rot_rel(4), base_pos(3), base_rot(4), gripper_qpos(2)] + 16 padding
   - Actions: 12 real dims [ee_pos(3), ee_rot(3), gripper_close(1), base_motion(4), control_mode(1)] + 20 padding

4. **Stats are logged to wandb** at step 0 (q01/q99/range per dimension).

---

## Part 6: Action Logging

Action values for batch0 are now logged at training start (step 0):
- Per-dimension: mean, min, max, std for first batch element
- Per-dimension: mean, min, max, std across full batch
- Both logged to wandb under `init_batch0_action/` and `init_fullbatch_action/` prefixes
- Also logged to console for quick inspection

Continuing per-dimension logging happens at every `log_interval` steps under `batch0_action/` prefix.
