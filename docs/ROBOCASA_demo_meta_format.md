# RoboCasa Exact-Replay Reset Data Format

This document describes the output format of `scripts/extract_eval_reset_data.py`
and how each file is consumed during evaluation to reproduce a training episode's
environment identically — same robot pose, same object positions, same scene geometry.

## Output Directory Structure

```
{output_dir}/{config_name}/
├── manifest.json
├── extras/
│   ├── episode_000000/
│   │   ├── ep_meta.json
│   │   ├── states.npz
│   │   └── model.xml.gz
│   ├── episode_000005/
│   │   ├── ep_meta.json
│   │   ├── states.npz
│   │   └── model.xml.gz
│   └── ...
└── videos/                          (optional, for oracle comparison)
    └── chunk-000/
        ├── observation.images.robot0_agentview_left/
        │   ├── episode_000000.mp4
        │   └── ...
        └── observation.images.robot0_eye_in_hand/
            ├── episode_000000.mp4
            └── ...
```

## Per-Episode Files

### `ep_meta.json`

Full environment specification. This single file defines everything needed to
reconstruct the scene from scratch (without exact physics state):

```json
{
  "layout_id": 21,
  "style_id": 49,
  "fixture_refs": {
    "sink": "sink_main_group",
    "counter": "counter_1_main_group"
  },
  "object_cfgs": [
    {
      "name": "obj",
      "obj_groups": "all",
      "graspable": true,
      "placement": {
        "fixture": "counter_1_main_group",
        "sample_region_kwargs": { ... },
        "size": [0.3, 0.3],
        "pos": ["ref", -1.0],
        "offset": [0.0, 0.1]
      },
      "type": "object",
      "info": {
        "cat": "apple",
        "mjcf_path": "/path/to/apple/model.xml"
      }
    },
    ...
  ],
  "fixtures": { ... },
  "gen_textures": true,
  "lang": "Pick up the apple from the counter and place it in the cabinet.",
  "cam_configs": { ... },
  "init_robot_base_pos": [x, y, z],
  "init_robot_base_ori": [roll, pitch, yaw]
}
```

**Key fields:**

| Field | Description |
|-------|-------------|
| `layout_id` | Kitchen floor plan index (determines counter/cabinet/sink positions) |
| `style_id` | Visual style index (textures, materials) |
| `fixture_refs` | Maps task-level names (e.g. "sink", "cab") to specific fixture instances in the layout |
| `object_cfgs` | List of object configurations. `object_cfgs[0]` is the main task object |
| `object_cfgs[i].info.cat` | Object category (e.g. "apple", "banana") |
| `object_cfgs[i].info.mjcf_path` | Path to the specific object mesh/model |
| `object_cfgs[i].placement` | Where and how the object is placed on its fixture |
| `init_robot_base_pos` | Robot base position `[x, y, z]` at episode start |
| `init_robot_base_ori` | Robot base orientation `[roll, pitch, yaw]` at episode start |
| `lang` | Natural language task description |

### `states.npz`

MuJoCo simulator states for the full trajectory.

```python
data = np.load("states.npz")
states = data["states"]  # shape: (T, state_dim)
initial_state = states[0]  # the state at timestep 0
```

- `states[0]` is the flattened MuJoCo state at the start of the episode
- Contains joint positions (qpos), joint velocities (qvel), and actuator states
- This is the state vector that `sim.set_state_from_flattened()` accepts

### `model.xml.gz`

Gzipped MuJoCo XML model definition. This is the exact XML that was used when
the demo was recorded, including all asset paths, body definitions, and
geom/joint parameters.

```python
import gzip
with gzip.open("model.xml.gz", "rt") as f:
    model_xml = f.read()  # XML string
```

## `manifest.json`

Top-level index of all extracted episodes:

```json
{
  "config_name": "pi05_robocasa_single_task_lora_turn_on_sink_faucet",
  "num_episodes": 107,
  "episodes": [
    {
      "episode_id": 0,
      "layout_id": 21,
      "style_id": 49,
      "fixture_refs": {},
      "lang": "Turn on the sink faucet.",
      "init_robot_base_pos": [0.12, -0.03, 0.91],
      "init_robot_base_ori": [0.0, 0.0, 1.57],
      "files": ["ep_meta.json", "states.npz", "model.xml.gz"]
    },
    ...
  ]
}
```

## How Exact State Replay Works

During evaluation, `RoboCasaEvalResetController` (from `dsrl_pi0/examples/robocasa_eval_reset.py`)
loads this data and the gymnasium wrapper's `reset()` method executes these steps:

```
1. env.set_ep_meta(ep_meta)
   └── Stores the scene config so the environment knows which layout/style/fixtures to use

2. model_xml = env.edit_model_xml(model_xml_string)
   └── Post-processes asset paths in the XML for the current machine

3. env.reset_from_xml_string(model_xml)
   └── Tears down the current MuJoCo model and rebuilds from the XML
   └── This sets up the exact scene geometry (counters, cabinets, objects)

4. env.sim.reset()
   └── Soft reset of the simulator

5. env.sim.set_state_from_flattened(initial_state)
   └── Restores the exact physics state: all joint positions, velocities,
       object poses, robot configuration — everything
   └── After this call, the simulator is in the exact same state as
       when the demo was recorded at timestep 0

6. env.sim.forward()
   └── Propagates the state to derived quantities (contacts, sensor readings)
```

After step 5, the environment is **byte-for-byte identical** to the recorded demo's
initial state. The robot arm pose, gripper opening, object positions, object orientations —
nothing has changed.

## Evaluation Modes

The reset controller supports four modes, each using different subsets of the data:

| Mode | ep_meta | states.npz | model.xml.gz | What varies |
|------|---------|------------|--------------|-------------|
| `exact_state_replay` | Yes | Yes | Yes | Nothing — fully deterministic |
| `fixture_pair_fresh_placement` | Yes | No | No | Object placement position |
| `fixture_pair_same_category` | Yes (modified) | No | No | Object instance + placement |
| `fixture_pair_object_pool` | Yes (modified) | No | No | Object category + instance + placement |

## Usage

### Extract reset data from a training config

```bash
uv run scripts/extract_eval_reset_data.py \
    --config-name pi05_robocasa_single_task_lora_turn_on_sink_faucet \
    --output-dir ./eval_reset_data
```

### Robocasa State Replay Dump

Extract exact-replay reset data for all 6 fixed-fixture RoboCasa configs:

```bash
uv run python3 scripts/extract_eval_reset_data.py --config-name pi05_robocasa_single_task_lora_turn_on_sink_faucet --output-dir /home/skowshik/vla/codebase/openpi/data_dumps/robocasa/
uv run python3 scripts/extract_eval_reset_data.py --config-name pi05_robocasa_single_task_lora_slide_dishwasher_rack --output-dir /home/skowshik/vla/codebase/openpi/data_dumps/robocasa/
uv run python3 scripts/extract_eval_reset_data.py --config-name pi05_robocasa_single_task_lora_pick_place_counter_to_cabinet --output-dir /home/skowshik/vla/codebase/openpi/data_dumps/robocasa/
uv run python3 scripts/extract_eval_reset_data.py --config-name pi05_robocasa_single_task_lora_prepare_coffee --output-dir /home/skowshik/vla/codebase/openpi/data_dumps/robocasa/
uv run python3 scripts/extract_eval_reset_data.py --config-name pi05_robocasa_single_task_lora_deliver_straw --output-dir /home/skowshik/vla/codebase/openpi/data_dumps/robocasa/
uv run python3 scripts/extract_eval_reset_data.py --config-name pi05_robocasa_single_task_lora_get_toasted_bread --output-dir /home/skowshik/vla/codebase/openpi/data_dumps/robocasa/
```

### Run evaluation with exact state replay

```bash
python examples/robocasa/main.py \
    --eval-init-mode exact_state_replay \
    --dataset-path ./eval_reset_data/pi05_robocasa_single_task_lora_turn_on_sink_faucet \
    --env-name TurnOnSinkFaucet \
    --num-trials 10 \
    --log-dir ./eval_logs
```
