# Real World (YAM Robot) Integration Plan

## Context

Port real-world YAM bimanual robot features from the source repo (`/home/skowshik/vla/codebase/sreyas_openpi/openpi`, commits `a87d90c`..`a261b28`) into this repository. The source repo is an older fork with real-world additions on top; this repo has diverged significantly with additional features (LIBERO HDF5/Pro datasets, RoboCasa Groot, robomimic, richer filtering, episode-level filtering, etc.). This plan selectively ports only the real-world features without regressing existing functionality.

Full documentation of the real-world features is at: `/home/skowshik/vla/codebase/sreyas_openpi/openpi/real_world_docs.md`

---

## What Already Exists (No Changes Needed)

| Item | Status |
|------|--------|
| `src/openpi/policies/yam_policy.py` | Identical between repos |
| 7 YAM training configs (simpletest, combined, pickplace_a/b, arrange_a/all, subtask) | Already in config.py |
| `dataset_filter_prompt`, `dataset_filter_orig_traj_id_6_*` fields in DataConfig | Already exist |
| `_FilteredLeRobotDataset` + `_has_episode_filter()` in data_loader.py | Already exist (destination's own version) |
| `return_normalized` parameter on `Policy.infer()` | Already exists |
| 5 SLURM scripts (combined, pickplace_a/b, arrange_a/all) | Already exist |
| examples/yam/ base files (data converters, docs, serve_yam_residual.py, etc.) | Already exist |

---

## Changes to Port (7 Steps)

### Step 1: Model-Level Changes (Foundation)

These are the lowest-level changes that policy.py depends on.

#### 1a. `src/openpi/models/model.py`

Add `return_vlm_embedding: bool = False` parameter to the abstract `sample_actions()` method signature.

**Source reference**: `/home/skowshik/vla/codebase/sreyas_openpi/openpi/src/openpi/models/model.py:280`

#### 1b. `src/openpi/models/pi0.py`

Three changes:

1. **`__call__()` method** (training loss):
   - Add `return_vlm_embedding: bool = False` parameter
   - Capture `kv_cache` from LLM forward pass (change `_` to `kv_cache` in the return)
   - When `return_vlm_embedding=True`, return `(loss, (prefix_out, kv_cache))` instead of just `loss`

2. **`sample_actions()` method** (inference):
   - Add `return_vlm_embedding: bool = False` parameter
   - Capture `prefix_hidden_state` from prefix-only LLM call (change `(_, _), kv_cache` to `(prefix_hidden_state, _), kv_cache`)
   - When `return_vlm_embedding=True`, return `(x_0, (prefix_hidden_state, kv_cache))` instead of just `x_0`

3. **New `get_prefix_rep()` method** (after `sample_actions`):
   ```python
   def get_prefix_rep(self, observation: _model.Observation):
       """Returns VLM hidden-state representations [B, S_prefix, W] and kv_cache."""
       observation = _model.preprocess_observation(None, observation, train=False)
       prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
       prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
       positions = jnp.cumsum(prefix_mask, axis=1) - 1
       (hidden_state, _), kv_cache = self.PaliGemma.llm(
           [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
       )
       return hidden_state, kv_cache
   ```

**Source reference**: `/home/skowshik/vla/codebase/sreyas_openpi/openpi/src/openpi/models/pi0.py:190-301`

---

### Step 2: Policy-Level Changes

#### `src/openpi/policies/policy.py`

**`__init__()` — add 3 lines after `self._pytorch_device = pytorch_device` (current line 56):**
```python
self.action_dim = model.action_dim
self.action_horizon = model.action_horizon
self._get_prefix_rep = nnx_utils.module_jit(model.get_prefix_rep)
```

**`__init__()` — modify JIT call (current line 64):**
```python
# FROM:
self._sample_actions = nnx_utils.module_jit(model.sample_actions)
# TO:
self._sample_actions = nnx_utils.module_jit(model.sample_actions, static_argnames=("return_vlm_embedding",))
```

**`infer()` — add `return_vlm_embedding: bool = False` parameter.**

**`infer()` — replace the single-sample batching with batch-aware logic:**
```python
# Instead of always doing:
#   inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
# Detect batch:
if inputs["state"].ndim > 1:
    batch_size = inputs["state"].shape[0]
    def _add_batch_dim(x):
        return jnp.broadcast_to(x[jnp.newaxis, ...], (batch_size,) + x.shape)
    inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
    for key in inputs:
        if key not in ["image", "state"]:
            inputs[key] = jax.tree.map(lambda x: _add_batch_dim(x), inputs[key])
else:
    batch_size = 1
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
```

**`infer()` — handle VLM embedding in sample_kwargs and result unpacking:**
```python
if return_vlm_embedding:
    sample_kwargs["return_vlm_embedding"] = True

sample_result = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
if return_vlm_embedding:
    actions, vlm_embedding = sample_result
else:
    actions = sample_result

# ... build outputs dict with actions ...

# Only squeeze if batch_size == 1:
if batch_size == 1:
    outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

# Attach vlm_embedding if requested:
if return_vlm_embedding:
    outputs["vlm_embedding"] = vlm_embedding
```

**New `get_prefix_rep()` method (after `infer`):**
```python
@override
def get_prefix_rep(self, obs: dict):
    inputs = jax.tree.map(lambda x: x, obs)
    inputs = self._input_transform(inputs)
    inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
    if inputs["state"].ndim > 1:
        batch_size = inputs["state"].shape[0]
        def _add_batch_dim(x):
            return jnp.broadcast_to(x[jnp.newaxis, ...], (batch_size,) + x.shape)
        for key in inputs:
            if key not in ["image", "state"]:
                inputs[key] = jax.tree.map(lambda x: _add_batch_dim(x), inputs[key])
    else:
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    return self._get_prefix_rep(_model.Observation.from_dict(inputs))
```

**Source reference**: `/home/skowshik/vla/codebase/sreyas_openpi/openpi/src/openpi/policies/policy.py` (full file)

---

### Step 3: WebSocket Server Changes

#### `src/openpi/serving/websocket_policy_server.py`

**Add imports:**
```python
import jax
import numpy as np
```

**Update docstring** to mention `get_prefix_rep` method.

**Modify message handling in the websocket loop** (around the `obs = msgpack_numpy.unpackb(...)` line):
```python
# FROM:
obs = msgpack_numpy.unpackb(await websocket.recv())
# ...
action = self._policy.infer(obs)

# TO:
message = msgpack_numpy.unpackb(await websocket.recv())
method = message.get("method", "infer")  # backward compat
obs = message.get("obs", message)        # old clients send obs directly
# ...
if method == "infer":
    noise = obs.pop("noise", None)
    result = self._policy.infer(obs, noise=noise)
elif method == "get_prefix_rep":
    result = self._policy.get_prefix_rep(obs)
else:
    raise ValueError(f"Unknown method: {method}")
```

**Before sending result, convert to float32 numpy:**
```python
result = jax.tree.map(lambda x: np.asarray(x).astype(np.float32), result)
await websocket.send(packer.pack(result))
```

**Rename `action` → `result`** throughout the rest of the handler (timing dict, etc.).

**Source reference**: `/home/skowshik/vla/codebase/sreyas_openpi/openpi/src/openpi/serving/websocket_policy_server.py`

---

### Step 4: ActionChunkBroker Changes

#### `packages/openpi-client/src/openpi_client/action_chunk_broker.py`

**Remove** `import tree` dependency.

**Modify `infer()` signature** to accept and forward `noise`:
```python
# FROM:
def infer(self, obs: Dict) -> Dict:
    ...
    self._last_results = self._policy.infer(obs)
# TO:
def infer(self, obs: Dict, noise: float = None) -> Dict:
    ...
    self._last_results = self._policy.infer(obs, noise=noise)
```

**Replace the inline slicer + `tree.map_structure`** with new `_slice_action_chunks()` method:
```python
# FROM:
def slicer(x):
    if isinstance(x, np.ndarray):
        return x[self._cur_step, ...]
    else:
        return x
results = tree.map_structure(slicer, self._last_results)

# TO:
results = self._slice_action_chunks(self._last_results)
```

**Add the new method:**
```python
def _slice_action_chunks(self, value, key_path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        return {
            key: self._slice_action_chunks(child, (*key_path, str(key)))
            for key, child in value.items()
        }
    is_action_field = any(
        key in {"action", "actions"} or key.endswith("_actions")
        for key in key_path
    )
    if is_action_field and isinstance(value, np.ndarray) and value.ndim > 0:
        if self._cur_step >= value.shape[0]:
            raise ValueError(
                f"Action field {'/'.join(key_path)!r} has chunk length "
                f"{value.shape[0]}, shorter than requested step {self._cur_step}."
            )
        return value[self._cur_step, ...]
    return value
```

**Add `get_prefix_rep()` delegation:**
```python
@override
def get_prefix_rep(self, observation: Dict) -> Dict:
    return self._policy.get_prefix_rep(observation)
```

**Source reference**: `/home/skowshik/vla/codebase/sreyas_openpi/openpi/packages/openpi-client/src/openpi_client/action_chunk_broker.py`

---

### Step 5: Add Missing YAM Wipe Training Configs

#### `src/openpi/training/config.py`

Insert 3 new `TrainConfig` entries between `pi05_yam_arrange_all_lora` and `pi05_yam_subtask_lora` (after line 7400):

| Config Name | `dataset_filter_prompt` | Traj ID Filter | Steps |
|-------------|-------------------------|----------------|-------|
| `pi05_yam_wipe_a_lora` | `"Wipe the black tray with the white cloth"` | `dataset_filter_orig_traj_id_6_eq=43505` | 20k |
| `pi05_yam_wipe_b_lora` | `"Wipe the black tray with the white cloth"` | `dataset_filter_orig_traj_id_6_max=43214` | 20k |
| `pi05_yam_wipe_all_lora` | `"Wipe the black tray with the white cloth"` | (none) | 20k |

All three follow the exact same structure as existing filtered YAM configs: `SimpleDataConfig`, `repo_id="local/yam_combined"`, `AssetsConfig(assets_dir="./assets/pi05_yam_combined_lora", asset_id="local/yam_combined")`, same model/freeze/lr/schedule settings.

**Source reference**: `/home/skowshik/vla/codebase/sreyas_openpi/openpi/src/openpi/training/config.py` (search for `pi05_yam_wipe`)

---

### Step 6: Update examples/yam/ Files

#### 6a. Update existing files

**`examples/yam/eval.py`** — Major update. Copy from source. Key additions:
- `SafetyDebugSubscriber` class (prints robot safety diagnostics every N steps)
- 12+ new CLI arguments: `--init-traj-glob`, `--init-timestep`, `--init-seed`, `--init-traj-id-min/max`, `--save-rollouts`, `--save-dir`, `--save-run-tag`, `--safety-debug-every`, `--label-rollouts`, `--label-timeout-sec`, `--scene-reset-sec`, `--no-diagnostics`
- Subscriber wiring: SceneReset → SafetyDebug → SuccessLabel → RolloutSaver → EvalDiagnostics → EvalSummary
- Label-driven retry loop replacing simple `runtime.run()` (supports success/failure/abort/retry)

**`examples/yam/env.py`** — Major update. Copy from source. Key additions:
- Imports: `glob`, `logging`, `random`, `pathlib.Path`, `h5py`, `PIL.Image`
- `_load_state_at()`: reads 14D state from training HDF5 at given timestep
- `_resize_like_training()`: BICUBIC resize to match training resolution
- Init-state parameters: `init_traj_globs`, `init_timestep`, `init_seed`, `init_traj_id_min/max`, `terminate_check`
- `_expand_globs()`, `_traj_id_from_path()`, `_filter_by_traj_id()` static methods
- `_drive_to_init_state()`: picks random training trajectory, drives robot to target state
- Fixed image pipeline: BGR→RGB conversion, direct BICUBIC resize instead of `resize_with_pad`
- Safety diagnostic logging and observation output
- `terminate_check` support for mid-rollout operator input

**`examples/yam/serve_yam_residual.py`** — Copy from source. Key changes:
- `update_actor_params()` accepts optional `batch_stats` with 4-case handling (server has/hasn't stats × client sends/doesn't)
- Action chunk shape uses `chunk_len` (not `query_freq`) — fixes checkpoint portability
- Base action truncated to `chunk_len` before residual actor
- Additional PARL hyperparams: `b_o_n`, `grad_a_q`, `parl_a_star_delta_clip_norm`
- `_warned_missing_bs` flag for one-time BatchNorm warning

**`examples/yam/EVAL.md`** — Minor: update image pipeline description to mention BGR-to-RGB and direct-resize.

#### 6b. Copy new files from source

| Source File | Lines | Purpose |
|-------------|-------|---------|
| `examples/yam/eval_diagnostics.py` | 362 | Per-query diagnostics subscriber (actor_version, infer_ms, action diffs) |
| `examples/yam/eval_saver.py` | 227 | Saves rollouts to HDF5 + per-camera MP4 videos |
| `examples/yam/eval_summary.py` | 115 | Aggregates eval labels into summary JSON |
| `examples/yam/scene_reset_subscriber.py` | 86 | Countdown timer before policy starts stepping |
| `examples/yam/success_label_subscriber.py` | 198 | Operator labeling (1/0/2/r keys) during rollouts |
| `examples/yam/convert_yam_eval_rollouts_to_lerobot.py` | 227 | Convert saved eval rollouts to LeRobot format |

---

### Step 7: SLURM Scripts + Norm Stats

#### 7a. Copy new SLURM scripts from source
- `slurm_yam_wipe_a.sh`
- `slurm_yam_wipe_b.sh`
- `slurm_yam_wipe_all.sh`

#### 7b. Copy norm stats
- **Source**: `/home/skowshik/vla/codebase/sreyas_openpi/openpi/real-world-assets/pi05_yam_combined_lora/local/yam_combined/norm_stats.json`
- **Dest**: `assets/pi05_yam_combined_lora/local/yam_combined/norm_stats.json`

This is needed because the filtered YAM configs reference `assets_dir="./assets/pi05_yam_combined_lora"` to share norm stats.

---

## DO NOT PORT

| Item | Reason |
|------|--------|
| data_loader.py class removals (FilteredDataset, EpisodeFilteredDataset, LiberoProHDF5Dataset, etc.) | Dest has richer implementations needed for LIBERO/robomimic |
| DataConfig field removals (filter_prompt, hdf5_path, episode_ids, libero10_data_dir, flip_images, etc.) | Dest needs these for LIBERO/robomimic/RoboCasa |
| Config factory removals (CustomLiberoDataConfig, LiberoProHDF5DataConfig, Libero10HDF5DataConfig, RobomimicDataConfig) | Dest-only features |
| `download.py` DEFAULT_CACHE_DIR change | Machine-specific path |
| `pyproject.toml` dependency changes | Dest has different version pins, mujoco override; `jaxrl2` is a local-only dep |
| `lerobot` import path change in data_loader.py | Dest already handles this |
| `pi0.py` comment typo ("actiPaliGemmaon") | Typo in source, skip |
| `serve_policy.py` LIBERO default change (`pi05_libero` → `pi05_libero_finetuned`) | Dest has its own default |
| LOG comments removed from config.py | Dest has these for documentation |

---

## Verification Checklist

After all changes:

```bash
# 1. Verify new configs load
python -c "from openpi.training.config import get_config; print(get_config('pi05_yam_wipe_a_lora').name)"
python -c "from openpi.training.config import get_config; print(get_config('pi05_yam_wipe_b_lora').name)"
python -c "from openpi.training.config import get_config; print(get_config('pi05_yam_wipe_all_lora').name)"

# 2. Verify all existing configs still load
python -c "from openpi.training.config import _CONFIGS; print(f'{len(_CONFIGS)} configs loaded')"

# 3. Verify policy imports
python -c "from openpi.policies.policy import Policy; print('Policy OK')"

# 4. Verify server imports
python -c "from openpi.serving.websocket_policy_server import WebsocketPolicyServer; print('Server OK')"

# 5. Verify client imports
python -c "from openpi_client.action_chunk_broker import ActionChunkBroker; print('Client OK')"

# 6. Verify model changes
python -c "from openpi.models.pi0 import Pi0Model; print('Model OK')"

# 7. Verify norm stats exist
python -c "import json; json.load(open('assets/pi05_yam_combined_lora/local/yam_combined/norm_stats.json')); print('Norm stats OK')"

# 8. Run existing tests
python -m pytest tests/ -x -q
```
