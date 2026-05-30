# Robomimic Reference Guide

Complete reference for robomimic: setup, datasets, training, evaluation, visualization, and integration with openpi pi0.5 fine-tuning.

---

## Table of Contents
- [1. What Was Done (Setup Log)](#1-what-was-done-setup-log)
- [2. Installation](#2-installation)
- [3. Available Tasks and Datasets](#3-available-tasks-and-datasets)
- [4. HDF5 Dataset Structure](#4-hdf5-dataset-structure)
- [5. Downloading Data](#5-downloading-data)
- [6. Extracting Observations from Raw Data](#6-extracting-observations-from-raw-data)
- [7. Visualization](#7-visualization)
- [8. Training with Robomimic (Native)](#8-training-with-robomimic-native)
- [9. Evaluation (Native)](#9-evaluation-native)
- [10. Environment Wrapper API](#10-environment-wrapper-api)
- [11. Pi0.5 Fine-Tuning Integration](#11-pi05-fine-tuning-integration)
- [12. Key File Paths](#12-key-file-paths)

---

## 1. What Was Done (Setup Log)

### 1.1 Cloned Robomimic
```bash
git clone https://github.com/ARISE-Initiative/robomimic.git /home/skowshik/vla/codebase/robomimic
```

### 1.2 Installed in `openpi` Conda Env
```bash
cd /home/skowshik/vla/codebase/robomimic
conda run -n openpi pip install -e . --no-deps
```
`--no-deps` was required because robomimic pins `huggingface_hub==0.23.4` and `transformers==4.41.2` which would downgrade and break other packages in `openpi`. All actual runtime dependencies (numpy, h5py, torch, imageio, robosuite, mujoco) were already present.

### 1.3 Patched egl_probe
`egl_probe` (GPU detection for EGL offscreen rendering) fails to build due to CMake version incompatibility on this system. Patched `robomimic/envs/env_robosuite.py` lines 99-110 to fall back to `MUJOCO_EGL_DEVICE_ID` env var (defaults to GPU 0):

```python
# In EnvRobosuite.__init__, replaced hard import with try/except:
try:
    import egl_probe
    valid_gpu_devices = egl_probe.get_available_devices()
    if len(valid_gpu_devices) > 0:
        kwargs["render_gpu_device_id"] = valid_gpu_devices[0]
except ImportError:
    import os
    kwargs["render_gpu_device_id"] = int(os.environ.get("MUJOCO_EGL_DEVICE_ID", 0))
```

### 1.4 Downloaded tool_hang Datasets
```bash
cd /home/skowshik/vla/codebase/robomimic
conda run -n openpi python robomimic/scripts/download_datasets.py \
    --tasks tool_hang --dataset_types ph --hdf5_types raw low_dim \
    --download_dir /data/hf_cache/datasets/robomimic/
```
Result:
```
/data/hf_cache/datasets/robomimic/tool_hang/ph/
├── demo_v15.hdf5      # 154 MB - raw MuJoCo states + actions (200 demos)
└── low_dim_v15.hdf5   # 190 MB - extracted low-dim observations (200 demos)
```

### 1.5 Visualized All 200 Demos
Created custom per-episode visualization script (`visualize_robomimic_demos.py`) because the built-in `playback_dataset.py` only writes a single combined video. Rendered all 200 episodes:
```bash
MUJOCO_GL=egl conda run -n openpi python visualize_robomimic_demos.py
```
Result: **200 mp4 files** (139 MB total) in `robomimic_data_vis/tool_hang_ep_0.mp4` through `tool_hang_ep_199.mp4`. Each video shows agentview + wrist camera side-by-side at 512x512 per camera, 20 FPS.

---

## 2. Installation

### Repo and Environment
| Item | Value |
|------|-------|
| Robomimic repo | `/home/skowshik/vla/codebase/robomimic/` |
| Version | 0.5.0 (editable install) |
| Source | `https://github.com/ARISE-Initiative/robomimic.git` |
| Conda env | `openpi` |

### Key Packages in `openpi` Env
| Package | Version | Purpose |
|---------|---------|---------|
| robosuite | 1.5.2 | MuJoCo robot manipulation simulator |
| robocasa | (installed) | Kitchen manipulation environments |
| mujoco | (installed) | Google DeepMind MuJoCo bindings |
| torch / torchvision | (installed) | PyTorch |
| imageio / imageio-ffmpeg | (installed) | Video I/O |
| h5py | (installed) | HDF5 file I/O |

### Offscreen Rendering
Any script that renders frames (visualization, obs extraction, evaluation) requires:
```bash
MUJOCO_GL=egl conda run -n openpi python <script.py>
```
To select a specific GPU: `MUJOCO_EGL_DEVICE_ID=<gpu_id>` (default: 0).

---

## 3. Available Tasks and Datasets

### Simulation Tasks

| Task | Dataset Types | Rollout Horizon | Action Dim | Description |
|------|:------------:|:---------------:|:----------:|-------------|
| `lift` | ph, mh, mg | 400 | 7 | Lift a cube from table |
| `can` | ph, mh, mg, paired | 400 | 7 | Pick can, place in bin |
| `square` | ph, mh | 400 | 7 | Pick square nut, place on rod |
| `transport` | ph, mh | 700 | 14 | Two-arm: pass object between arms |
| `tool_hang` | ph | 700 | 7 | Hang tool on rack (hardest) |

### Real Robot Tasks

| Task | Dataset Types | Horizon |
|------|:------------:|:-------:|
| `lift_real` | ph | 1000 |
| `can_real` | ph | 1000 |
| `tool_hang_real` | ph | 1000 |

### Dataset Types
- **ph** (Proficient Human): 200 successful trajectories from 1 skilled operator
- **mh** (Multi-Human): 300 trajectories from 6 operators of varying skill
- **mg** (Machine Generated): SAC agent rollouts (lift, can only)
- **paired**: Paired good/bad trajectories (can only)

### HDF5 Observation Types
| Type | Contents | Direct Download? | Notes |
|------|----------|:----------------:|-------|
| `raw` | Full MuJoCo states + actions | Yes | Needed for sim replay and obs extraction |
| `low_dim` | Extracted robot/object states | Yes | Ready for state-based training |
| `image` | RGB camera observations | No | Generate from raw via `dataset_states_to_obs.py` |
| `low_dim_sparse/dense` | State + sparse/dense rewards | Yes (mg only) | For offline RL |
| `image_sparse/dense` | Images + sparse/dense rewards | No | Generate from raw |

### HuggingFace Registry
All sim datasets are hosted on HuggingFace repo `robomimic/robomimic_datasets`. URL format: `v1.5/<task>/<dataset_type>/<filename>.hdf5`. Real robot datasets are hosted on Stanford servers.

---

## 4. HDF5 Dataset Structure

### Overall Layout
```
dataset.hdf5
├── data/
│   ├── demo_0/
│   │   ├── actions              (T, 7)     # 7-DOF: 6 OSC + 1 gripper
│   │   ├── states               (T, 58)    # full MuJoCo state vector
│   │   ├── rewards              (T,)
│   │   ├── dones                (T,)
│   │   ├── obs/                             # only in low_dim/image datasets
│   │   │   ├── robot0_eef_pos       (T, 3)       # end-effector position
│   │   │   ├── robot0_eef_quat      (T, 4)       # end-effector quaternion
│   │   │   ├── robot0_gripper_qpos  (T, 2)       # gripper joint positions
│   │   │   ├── robot0_joint_pos     (T, 7)       # joint positions
│   │   │   ├── object               (T, 44)      # object state (task-dependent)
│   │   │   ├── agentview_image      (T, H, W, 3) # only in image datasets
│   │   │   └── robot0_eye_in_hand_image (T, H, W, 3)
│   │   ├── next_obs/                        # for offline RL
│   │   ├── controller_info/
│   │   └── attrs:
│   │       ├── model_file           # MuJoCo XML string (per-episode model)
│   │       ├── ep_meta              # episode metadata JSON (optional)
│   │       └── num_samples          # timestep count
│   ├── demo_1/
│   └── ...demo_199/
├── mask/                                    # filter keys
│   ├── train                                # demo key list for training
│   └── valid                                # demo key list for validation
└── data.attrs:
    └── env_args                             # JSON-serialized environment config
```

### tool_hang Specifics
- **200 demos**, episode lengths ~500-800 steps
- **Action dim**: 7 (6-DOF OSC position/orientation + 1 gripper)
- **State dim**: 58 (full MuJoCo state)
- **Low-dim obs**: `robot0_eef_pos` (3), `robot0_eef_quat` (4), `robot0_gripper_qpos` (2), `robot0_joint_pos` (7), `object` (44)
- **Cameras**: `agentview` (third-person), `robot0_eye_in_hand` (wrist-mounted)

### Quick Inspection
```bash
conda run -n openpi python -c "
import h5py, json
f = h5py.File('/data/hf_cache/datasets/robomimic/tool_hang/ph/demo_v15.hdf5', 'r')
print('Demos:', len(list(f['data'].keys())))
d = f['data/demo_0']
print('Keys:', list(d.keys()))
print('Actions:', d['actions'].shape)
print('States:', d['states'].shape)
env_args = json.loads(f['data'].attrs['env_args'])
print('Env:', env_args['env_name'], '| type:', env_args['type'])
f.close()
"
# Output: 200 demos, actions (681,7), states (681,58), env ToolHang type 1
```

---

## 5. Downloading Data

### Download Script
```bash
cd /home/skowshik/vla/codebase/robomimic

# Dry run (preview what will be downloaded):
conda run -n openpi python robomimic/scripts/download_datasets.py \
    --tasks tool_hang --dataset_types ph --hdf5_types raw low_dim \
    --download_dir /data/hf_cache/datasets/robomimic/ --dry_run

# Actual download:
conda run -n openpi python robomimic/scripts/download_datasets.py \
    --tasks tool_hang --dataset_types ph --hdf5_types raw low_dim \
    --download_dir /data/hf_cache/datasets/robomimic/
```

### Arguments
| Arg | Values | Default |
|-----|--------|---------|
| `--tasks` | `lift`, `can`, `square`, `transport`, `tool_hang`, `sim`, `real`, `all` | `lift` |
| `--dataset_types` | `ph`, `mh`, `mg`, `paired`, `all` | `ph` |
| `--hdf5_types` | `raw`, `low_dim`, `image`, `low_dim_sparse`, `low_dim_dense`, etc. | `low_dim` |
| `--download_dir` | Custom path (default: `robomimic/../datasets/`) | None |
| `--dry_run` | Flag to preview without downloading | |

### Download All Sim Tasks (Low-Dim)
```bash
conda run -n openpi python robomimic/scripts/download_datasets.py \
    --tasks sim --dataset_types ph --hdf5_types low_dim \
    --download_dir /data/hf_cache/datasets/robomimic/
```

---

## 6. Extracting Observations from Raw Data

The `raw` HDF5 contains MuJoCo states, not camera images. To generate image observations for pi0.5 fine-tuning:

```bash
cd /home/skowshik/vla/codebase/robomimic

# Extract low-dim observations
MUJOCO_GL=egl conda run -n openpi python robomimic/scripts/dataset_states_to_obs.py \
    --dataset /data/hf_cache/datasets/robomimic/tool_hang/ph/demo_v15.hdf5 \
    --output_name low_dim_v15.hdf5 \
    --done_mode 2

# Extract IMAGE observations (critical for pi0.5)
MUJOCO_GL=egl conda run -n openpi python robomimic/scripts/dataset_states_to_obs.py \
    --dataset /data/hf_cache/datasets/robomimic/tool_hang/ph/demo_v15.hdf5 \
    --output_name image_224_v15.hdf5 \
    --camera_names agentview robot0_eye_in_hand \
    --camera_height 224 --camera_width 224 \
    --done_mode 2 --compress --exclude-next-obs
```

### Arguments
| Arg | Description |
|-----|-------------|
| `--dataset` | Input raw HDF5 (must have `states` arrays) |
| `--output_name` | Output filename (written to same directory as input) |
| `--camera_names` | Camera names to render (e.g., `agentview robot0_eye_in_hand`) |
| `--camera_height/width` | Image resolution (use 224 for pi0.5) |
| `--done_mode` | 0: success-based, 1: trajectory-end, 2: both |
| `--compress` | Compress HDF5 (saves space for images) |
| `--exclude-next-obs` | Skip `next_obs` extraction (not needed for BC/pi0.5) |
| `--depth` | Also extract depth observations |

The output HDF5 will have `obs/agentview_image` (T, 224, 224, 3) and `obs/robot0_eye_in_hand_image` (T, 224, 224, 3) arrays per episode.

---

## 7. Visualization

### Per-Episode Visualization (Custom Script)
Location: `/home/skowshik/vla/codebase/openpi/visualize_robomimic_demos.py`

```bash
MUJOCO_GL=egl conda run -n openpi python visualize_robomimic_demos.py \
    --dataset /data/hf_cache/datasets/robomimic/tool_hang/ph/demo_v15.hdf5 \
    --output_dir robomimic_data_vis \
    --camera_names agentview robot0_eye_in_hand \
    --video_skip 5 --fps 20 \
    --n 10   # limit to first 10 episodes (omit for all 200)
```

Output: `robomimic_data_vis/tool_hang_ep_<idx>.mp4` (200 files, 139 MB total).

Each video: two cameras concatenated horizontally at 512x512 per camera, 20 FPS, every 5th sim frame.

### Built-in Playback (Single Combined Video)
```bash
MUJOCO_GL=egl conda run -n openpi python \
    /home/skowshik/vla/codebase/robomimic/robomimic/scripts/playback_dataset.py \
    --dataset /data/hf_cache/datasets/robomimic/tool_hang/ph/demo_v15.hdf5 \
    --render_image_names agentview robot0_eye_in_hand \
    --video_path /tmp/tool_hang_all.mp4 --n 5
```
Limitation: writes ALL episodes into one video file.

---

## 8. Training with Robomimic (Native)

### Available Algorithms

**Imitation Learning:**
| Algorithm | Class | Key Config |
|-----------|-------|-----------|
| BC | `BC` | `algo.actor_layer_dims` |
| BC-Gaussian | `BC_Gaussian` | `algo.gaussian.enabled=True` |
| BC-GMM | `BC_GMM` | `algo.gmm.enabled=True, algo.gmm.num_modes` |
| BC-VAE | `BC_VAE` | `algo.vae.enabled=True, algo.vae.latent_dim` |
| BC-RNN | `BC_RNN` | `algo.rnn.enabled=True, algo.rnn.hidden_dim` |
| BC-RNN-GMM | `BC_RNN_GMM` | RNN + GMM combined |
| BC-Transformer | `BC_Transformer` | `algo.transformer.enabled=True` |
| Diffusion Policy | `DiffusionPolicyUNet` | Separate config class |
| HBC | `HBC` | Hierarchical planner + actor |

**Offline RL:**
| Algorithm | Class | Description |
|-----------|-------|-------------|
| BCQ | `BCQ` | Batch Constrained Q-learning |
| CQL | `CQL` | Conservative Q-Learning |
| IQL | `IQL` | Implicit Q-Learning |
| IRIS | `IRIS` | Intrinsic reward offline RL |
| TD3-BC | `TD3_BC` | TD3 with BC constraint |

### Config Structure
```json
{
  "algo_name": "bc",
  "experiment": {
    "name": "tool_hang_bc",
    "validate": true,
    "save": {"enabled": true, "every_n_epochs": 50, "on_best_rollout_success_rate": true},
    "rollout": {"enabled": true, "n": 50, "horizon": 700, "rate": 50, "terminate_on_success": true}
  },
  "train": {
    "data": [{"path": "/data/hf_cache/datasets/robomimic/tool_hang/ph/low_dim_v15.hdf5"}],
    "output_dir": "/path/to/output",
    "batch_size": 100,
    "num_epochs": 2000,
    "hdf5_cache_mode": "all",
    "seq_length": 1,
    "cuda": true,
    "seed": 1
  },
  "algo": {
    "actor_layer_dims": [1024, 1024],
    "optim_params": {"policy": {"learning_rate": {"initial": 0.0001}}}
  },
  "observation": {
    "modalities": {
      "obs": {
        "low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
        "rgb": []
      }
    }
  }
}
```

Config templates at: `/home/skowshik/vla/codebase/robomimic/robomimic/exps/templates/`

### Training Command
```bash
conda run -n openpi python /home/skowshik/vla/codebase/robomimic/robomimic/scripts/train.py \
    --config /path/to/config.json

# Debug mode (1 epoch, small batch, no saving):
conda run -n openpi python /home/skowshik/vla/codebase/robomimic/robomimic/scripts/train.py \
    --config /path/to/config.json --debug
```

---

## 9. Evaluation (Native)

### Running a Trained Agent
```bash
MUJOCO_GL=egl conda run -n openpi python \
    /home/skowshik/vla/codebase/robomimic/robomimic/scripts/run_trained_agent.py \
    --agent /path/to/model.pth \
    --n_rollouts 50 --horizon 700 --seed 0 \
    --video_path /path/to/output.mp4 \
    --camera_names agentview robot0_eye_in_hand
```

### Programmatic Rollout
```python
from robomimic.utils.file_utils import policy_from_checkpoint, env_from_checkpoint

policy, ckpt_dict = policy_from_checkpoint(ckpt_path="/path/to/model.pth", device="cuda:0")
env, _ = env_from_checkpoint(ckpt_dict, render=False, render_offscreen=True)

policy.start_episode()
obs = env.reset()
for step in range(700):
    action = policy(ob=obs)
    obs, reward, done, info = env.step(action)
    if done or info.get("is_success", False):
        break
```

---

## 10. Environment Wrapper API

`EnvRobosuite` at `robomimic/envs/env_robosuite.py`:

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `step` | `step(action)` | `(obs, reward, done, info)` | Execute action; `info["is_success"]` for task success |
| `reset` | `reset()` | `obs` | Reset to new random initial state |
| `reset_to` | `reset_to(state_dict)` | `obs` | Deterministic reset to specific state |
| `render` | `render(mode, height, width, camera_name)` | `np.ndarray` or `None` | `"rgb_array"` for offscreen |
| `get_state` | `get_state()` | `{"states": np.array, "model": str}` | Current sim state |
| `get_observation` | `get_observation()` | `dict` | Current obs dict |
| `is_success` | `is_success()` | `{"task": bool}` | Task completion check |
| `get_reward` | `get_reward()` | `float` | Current reward |

**State dict format** (for `reset_to`):
```python
{
    "states": np.array,     # MuJoCo state vector (58-dim for tool_hang)
    "model": str,           # MuJoCo XML model (changes between episodes)
    "ep_meta": str,         # Episode metadata JSON (optional)
}
```

**Creating env from dataset metadata:**
```python
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

# Required initialization
ObsUtils.initialize_obs_utils_with_obs_specs(
    obs_modality_specs=dict(obs=dict(low_dim=["robot0_eef_pos"], rgb=[]))
)

env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path="path/to/dataset.hdf5")
env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True)
is_robosuite = EnvUtils.is_robosuite_env(env_meta)
```

---

## 11. Pi0.5 Fine-Tuning Integration

### Overview: What Needs to Happen

To fine-tune pi0.5 on robomimic tasks, you need to bridge robomimic's HDF5 data into openpi's training pipeline. The pipeline follows this pattern (from existing libero integration):

```
HDF5 Data → Custom Dataset class → Repack Transform → Data Transform → Model Transform → Training
```

### 11.1 Existing Pattern to Follow: Libero

The Libero integration is the closest reference. Key files:

| Component | File | What It Does |
|-----------|------|--------------|
| Policy transforms | `src/openpi/policies/libero_policy.py` | `LiberoInputs` / `LiberoOutputs` - maps data to model format |
| Data config | `src/openpi/training/config.py:376` | `LeRobotLiberoDataConfig` - wires transforms + norm stats |
| HDF5 dataset | `src/openpi/training/data_loader.py:311` | `LiberoProHDF5Dataset` - reads HDF5, returns standardized dicts |
| Training config | `src/openpi/training/config.py:2030+` | `TrainConfig` entries for pi0.5 + libero |
| Norm stats | `scripts/compute_norm_stats.py` | Computes action/state normalization statistics |

### 11.2 Step-by-Step Integration Guide

#### Step A: Extract Image Observations from Raw Data

Pi0.5 needs images. Robomimic raw datasets only have MuJoCo states. Generate images at 224x224 (openpi's `IMAGE_RESOLUTION`):

```bash
MUJOCO_GL=egl conda run -n openpi python \
    /home/skowshik/vla/codebase/robomimic/robomimic/scripts/dataset_states_to_obs.py \
    --dataset /data/hf_cache/datasets/robomimic/tool_hang/ph/demo_v15.hdf5 \
    --output_name image_224_v15.hdf5 \
    --camera_names agentview robot0_eye_in_hand \
    --camera_height 224 --camera_width 224 \
    --done_mode 2 --compress --exclude-next-obs
```

This produces an HDF5 with `obs/agentview_image` (T, 224, 224, 3) and `obs/robot0_eye_in_hand_image` (T, 224, 224, 3) per episode, plus actions and states.

#### Step B: Write a Custom HDF5 Dataset Class

Model after `LiberoProHDF5Dataset` (data_loader.py:311). The key contract:

```python
class RobomimicHDF5Dataset(Dataset):
    """Reads robomimic HDF5 and returns items compatible with openpi transforms."""

    def __init__(self, hdf5_path: str, action_horizon: int, task_description: str, num_episodes: int = -1):
        self._action_horizon = action_horizon
        self._prompt = task_description  # e.g., "hang the tool on the rack"
        self._samples = []  # (demo_key, timestep_idx)
        self._data = {}

        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
            if num_episodes > 0:
                demo_keys = demo_keys[:num_episodes]

            for demo_key in demo_keys:
                demo = f["data"][demo_key]
                T = demo["actions"].shape[0]
                self._data[demo_key] = {
                    "agentview_image": demo["obs/agentview_image"][:],              # (T, 224, 224, 3) uint8
                    "wrist_image": demo["obs/robot0_eye_in_hand_image"][:],         # (T, 224, 224, 3) uint8
                    "eef_pos": demo["obs/robot0_eef_pos"][:].astype(np.float32),    # (T, 3)
                    "eef_quat": demo["obs/robot0_eef_quat"][:].astype(np.float32),  # (T, 4)
                    "gripper_qpos": demo["obs/robot0_gripper_qpos"][:].astype(np.float32),  # (T, 2)
                    "actions": demo["actions"][:].astype(np.float32),               # (T, 7)
                }
                for t in range(T):
                    self._samples.append((demo_key, t))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        demo_key, t = self._samples[index.__index__()]
        data = self._data[demo_key]
        T = data["actions"].shape[0]

        # Build action chunk [t, t+action_horizon), pad with last action at episode end
        actions = np.empty((self._action_horizon, 7), dtype=np.float32)
        for i in range(self._action_horizon):
            actions[i] = data["actions"][min(t + i, T - 1)]

        # State: concat eef_pos (3) + eef_quat (4) + gripper_qpos (2) = 9-dim
        state = np.concatenate([
            data["eef_pos"][t], data["eef_quat"][t], data["gripper_qpos"][t]
        ], axis=-1)  # (9,)

        return {
            "image": data["agentview_image"][t],         # (224, 224, 3) uint8
            "wrist_image": data["wrist_image"][t],       # (224, 224, 3) uint8
            "state": state,                              # (9,) float32
            "actions": actions,                          # (action_horizon, 7) float32
            "prompt": self._prompt,
        }
```

**Key points:**
- Return keys must match what the repack transform expects (see `LeRobotLiberoDataConfig.create()`)
- Images must be uint8 (H, W, C)
- Actions are chunked into `(action_horizon, action_dim)` with last-action padding
- State is whatever proprioceptive info you want the model to see (concat of eef + gripper)
- Prompt is a fixed string since robomimic has no language annotations

#### Step C: Write Input/Output Transforms

Model after `LiberoInputs` / `LiberoOutputs` in `libero_policy.py`:

```python
# In src/openpi/policies/robomimic_policy.py

@dataclasses.dataclass(frozen=True)
class RobomimicInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),  # padding
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RobomimicOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}  # 7-DOF
```

#### Step D: Write a Data Config

Model after `LeRobotLiberoDataConfig`:

```python
# In src/openpi/training/config.py

@dataclasses.dataclass(frozen=True)
class RobomimicDataConfig(DataConfigFactory):
    hdf5_path: str = ""
    task_description: str = ""
    num_episodes: int = -1

    @override
    def create(self, assets_dirs, model_config):
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform({
                "observation/image": "image",
                "observation/wrist_image": "wrist_image",
                "observation/state": "state",
                "actions": "actions",
                "prompt": "prompt",
            })]
        )
        data_transforms = _transforms.Group(
            inputs=[robomimic_policy.RobomimicInputs(model_type=model_config.model_type)],
            outputs=[robomimic_policy.RobomimicOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
```

#### Step E: Add a Training Config

```python
# In src/openpi/training/config.py, add to CONFIGS list:

TrainConfig(
    name="pi05_robomimic_tool_hang_ep200",
    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
    ),
    data=RobomimicDataConfig(
        hdf5_path="/data/hf_cache/datasets/robomimic/tool_hang/ph/image_224_v15.hdf5",
        task_description="hang the tool on the rack",
        num_episodes=200,
        base_config=DataConfig(
            prompt_from_task=False,
        ),
    ),
    assets_base_dir="/data/user_data/skowshik/openpi_cache/robomimic_ft/assets",
    checkpoint_base_dir="/data/hf_cache/models/",
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=100_000,
    action_l1_loss_interval=500,
    save_interval=4000,
    keep_period=1000,
    action_dim=7,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=50,
        peak_lr=2.5e-5,
        decay_steps=100_000,
        decay_lr=2.5e-6,
    ),
    batch_size=32,
    log_interval=50,
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
        pi05=True,
    ).get_freeze_filter(),
    num_workers=0,
    enforce_min_quantile_range=True,
),
```

#### Step F: Compute Norm Stats

Before training, you must compute normalization statistics:
```bash
conda run -n openpi python scripts/compute_norm_stats.py \
    --config-name pi05_robomimic_tool_hang_ep200
```
This iterates over the dataset, computes per-dimension mean/std and quantile (q01/q99) statistics for `state` and `actions`, and saves them to `{assets_base_dir}/{config_name}/{repo_id_or_asset_id}/`. The `enforce_min_quantile_range=True` flag prevents gripper dim blow-up on few-episode training.

#### Step G: Train
```bash
conda run -n openpi python scripts/train.py --config-name pi05_robomimic_tool_hang_ep200
```

### 11.3 Action Space Comparison

| Environment | action_dim | State Dim (for model) | Actions Are | Gripper |
|-------------|:----------:|:---------------------:|:-----------:|:-------:|
| **Libero** | 7 | 8 (eef_pos:3 + eef_quat:4 + gripper:1) | Delta (relative) | Last dim |
| **RoboCasa** | 12 | 16 (padded) | Varies | Last dim |
| **Robomimic tool_hang** | 7 | 9 (eef_pos:3 + eef_quat:4 + gripper_qpos:2) | Delta (OSC) | Last dim |

Robomimic tool_hang action dim matches Libero exactly (7-DOF). The state dim differs slightly (9 vs 8) but the model pads states internally.

### 11.4 Key Considerations

1. **Delta vs Absolute Actions**: Robomimic tool_hang uses OSC (Operational Space Controller) which outputs delta position/orientation. These are already relative actions, similar to Libero. You likely do NOT need `extra_delta_transform=True`.

2. **Language Instructions**: Robomimic datasets have no language annotations. Provide a fixed task description string as the prompt. Example: `"hang the tool on the rack"`.

3. **Image Resolution**: Pi0.5 expects 224x224 images. Make sure to extract observations at that resolution (`--camera_height 224 --camera_width 224`).

4. **Image Orientation**: Robomimic images from `dataset_states_to_obs.py` are already in the correct orientation (not flipped like LIBERO-PRO HDF5). Do NOT apply the `[::-1, ::-1]` flip that `LiberoProHDF5Dataset` applies.

5. **Action Horizon**: The existing libero pi0.5 configs use `action_horizon=10`. Start with this.

6. **Quantile Normalization**: Pi0.5 uses quantile normalization (`use_quantile_norm` is auto-set for non-PI0 model types). Set `enforce_min_quantile_range=True` especially for few-episode settings.

7. **Base Weights**: Load from `gs://openpi-assets/checkpoints/pi05_base/params` (the pretrained pi0.5 base).

8. **Freeze Filter**: Use LoRA fine-tuning via:
   ```python
   pi0_config.Pi0Config(
       paligemma_variant="gemma_2b_lora",
       action_expert_variant="gemma_300m",
       pi05=True,
   ).get_freeze_filter()
   ```
   This freezes base model parameters and only trains LoRA adapters + action head.

---

## 12. Key File Paths

### Robomimic
| Resource | Path |
|----------|------|
| Robomimic repo | `/home/skowshik/vla/codebase/robomimic/` |
| Download script | `robomimic/scripts/download_datasets.py` |
| Training script | `robomimic/scripts/train.py` |
| Evaluation script | `robomimic/scripts/run_trained_agent.py` |
| Playback script | `robomimic/scripts/playback_dataset.py` |
| Obs extraction | `robomimic/scripts/dataset_states_to_obs.py` |
| Dataset info | `robomimic/scripts/get_dataset_info.py` |
| Config templates | `robomimic/exps/templates/` |
| Env wrapper (patched) | `robomimic/envs/env_robosuite.py` |
| Algo implementations | `robomimic/algo/` |
| Config definitions | `robomimic/config/` |
| Dataset registry | `robomimic/__init__.py` |

### Downloaded Data
| Resource | Path |
|----------|------|
| tool_hang raw | `/data/hf_cache/datasets/robomimic/tool_hang/ph/demo_v15.hdf5` (154 MB) |
| tool_hang low_dim | `/data/hf_cache/datasets/robomimic/tool_hang/ph/low_dim_v15.hdf5` (190 MB) |

### Visualization
| Resource | Path |
|----------|------|
| Per-episode vis script | `/home/skowshik/vla/codebase/openpi/visualize_robomimic_demos.py` |
| Vis output | `/home/skowshik/vla/codebase/openpi/robomimic_data_vis/tool_hang_ep_<idx>.mp4` (200 files, 139 MB) |

### OpenPI (for pi0.5 integration reference)
| Resource | Path |
|----------|------|
| Libero policy transforms | `src/openpi/policies/libero_policy.py` |
| LiberoProHDF5Dataset | `src/openpi/training/data_loader.py:311` |
| LeRobotLiberoDataConfig | `src/openpi/training/config.py:376` |
| Pi0.5 training configs | `src/openpi/training/config.py:2030+` |
| Norm stats script | `scripts/compute_norm_stats.py` |
| Training entry point | `scripts/train.py` |
| RoboCasa policy transforms | `src/openpi/policies/robocasa_policy.py` |
| Transforms library | `src/openpi/transforms.py` |
| Model config (Pi0/Pi0.5) | `src/openpi/models/pi0_config.py` |
