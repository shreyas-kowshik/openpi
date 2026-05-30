# OpenPI Code Index

Generated on 2026-05-11 for the local repository at `/home/skowshik/vla/codebase/openpi`.

This index covers the source, examples, scripts, and operational config that are useful for code navigation. It intentionally does not enumerate generated experiment artifacts such as checkpoints, videos, W&B runs, cache folders, local virtualenvs, and data dumps.

## Repository at a Glance

- `src/openpi/` - main OpenPI Python package: model definitions, data transforms, policy wrappers, training config, data loaders, checkpointing, and serving.
- `packages/openpi-client/` - lightweight client/runtime package used by robots or local clients to call a policy server.
- `scripts/` - CLI entry points for training, PyTorch training, serving, stats computation, checkpoint utilities, dataset inspection, and visualization.
- `examples/` - platform-specific adapters and workflows for ALOHA, DROID, LIBERO, RoboCasa, UR5, and a simple websocket client.
- `docs/` - operational docs for Docker, normalization stats, remote inference, and RoboCasa metadata.
- `slurm/` and `slurm_orchestrator/` - HPC launch scripts and a resubmitting Slurm watchdog.
- `third_party/` - vendored/submodule code, notably ALOHA and LIBERO.
- `assets/`, `logs/`, `data*`, `videos/`, `vis_*`, `robocasa_*`, `wandb/` - local assets and experiment outputs, mostly not source.

## Setup and Tooling

- Package metadata: [`pyproject.toml`](../pyproject.toml)
  - Python `>=3.11`.
  - Managed with `uv`.
  - Workspace member: `packages/*`, including `openpi-client`.
  - Core dependencies include JAX, Flax, Orbax, PyTorch, Transformers, LeRobot, OpenCV, WandB, and Tyro.
  - Test discovery: `src`, `scripts`, and `packages`.
  - Ruff is configured for Python 3.11 with line length 120.
- Lockfile: [`uv.lock`](../uv.lock)
- Primary README: [`README.md`](../README.md)
- Contribution guide: [`CONTRIBUTING.md`](../CONTRIBUTING.md)

Common commands from the README:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv run scripts/compute_norm_stats.py --config-name pi05_libero
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

## Core Data Flow

1. A `TrainConfig` from [`src/openpi/training/config.py`](../src/openpi/training/config.py) selects the model, dataset/data transforms, optimizer, assets, and checkpoint settings.
2. Data is loaded by [`src/openpi/training/data_loader.py`](../src/openpi/training/data_loader.py) from LeRobot, RLDS, HDF5, fake data, or local custom datasets.
3. Dictionaries are mapped by transform groups from [`src/openpi/transforms.py`](../src/openpi/transforms.py) and platform policy files in [`src/openpi/policies/`](../src/openpi/policies/).
4. Transforms produce an [`Observation`](../src/openpi/models/model.py) and action tensor consumed by a model in [`src/openpi/models/`](../src/openpi/models/).
5. Training is run by [`scripts/train.py`](../scripts/train.py) for JAX/Flax or [`scripts/train_pytorch.py`](../scripts/train_pytorch.py) for PyTorch.
6. Inference policies are created by [`src/openpi/policies/policy_config.py`](../src/openpi/policies/policy_config.py), wrapped by [`src/openpi/policies/policy.py`](../src/openpi/policies/policy.py), and optionally exposed over websocket by [`src/openpi/serving/websocket_policy_server.py`](../src/openpi/serving/websocket_policy_server.py).
7. Remote clients use [`packages/openpi-client/src/openpi_client/websocket_client_policy.py`](../packages/openpi-client/src/openpi_client/websocket_client_policy.py).

## Main Package: `src/openpi`

### Models

Directory: [`src/openpi/models/`](../src/openpi/models/)

- [`model.py`](../src/openpi/models/model.py) - base model contracts and shared data structures:
  - `ModelType`: `pi0`, `pi0_fast`, `pi05`.
  - `Observation`: normalized model input container for images, masks, state, prompt tokens, and FAST masks.
  - `Actions`: action tensor type alias.
  - `preprocess_observation`: image resizing/augmentation and mask filling.
  - `BaseModelConfig` / `BaseModel`: abstract config/model API.
  - `restore_params`: Orbax checkpoint parameter loading.
- [`pi0.py`](../src/openpi/models/pi0.py) - JAX/Flax pi0 and pi0.5 flow-matching VLA implementation.
- [`pi0_config.py`](../src/openpi/models/pi0_config.py) - `Pi0Config`, including action horizon/dim, pi0.5 mode, discrete state input, LoRA/freeze filters, and model specs.
- [`pi0_fast.py`](../src/openpi/models/pi0_fast.py) - autoregressive pi0-FAST model and `Pi0FASTConfig`.
- [`gemma.py`](../src/openpi/models/gemma.py) and [`gemma_fast.py`](../src/openpi/models/gemma_fast.py) - Gemma/PaliGemma-style transformer building blocks.
- [`siglip.py`](../src/openpi/models/siglip.py) and [`vit.py`](../src/openpi/models/vit.py) - vision encoder components.
- [`lora.py`](../src/openpi/models/lora.py) - LoRA adapters for einsum and feed-forward layers.
- [`tokenizer.py`](../src/openpi/models/tokenizer.py) - Paligemma, FAST, binning, and FSQ tokenizers.
- [`utils/fsq_tokenizer.py`](../src/openpi/models/utils/fsq_tokenizer.py) - FSQ tokenizer utility implementation.

PyTorch models live in [`src/openpi/models_pytorch/`](../src/openpi/models_pytorch/):

- [`pi0_pytorch.py`](../src/openpi/models_pytorch/pi0_pytorch.py) - PyTorch pi0/pi0.5 implementation.
- [`gemma_pytorch.py`](../src/openpi/models_pytorch/gemma_pytorch.py) - PyTorch Gemma pieces.
- [`preprocessing_pytorch.py`](../src/openpi/models_pytorch/preprocessing_pytorch.py) - PyTorch input preprocessing.
- [`transformers_replace/`](../src/openpi/models_pytorch/transformers_replace/) - local patches for Transformers Gemma/SigLIP/PaliGemma behavior.

### Transforms

File: [`src/openpi/transforms.py`](../src/openpi/transforms.py)

This is the central data-shaping toolkit. Important pieces:

- `DataTransformFn`, `Group`, `CompositeTransform`, `compose`
- Structural transforms: `RepackTransform`, `InjectDefaultPrompt`, `FilterPrompt`
- Normalization: `Normalize`, `Unnormalize`
- Vision/action prep: `ResizeImages`, `SubsampleActions`, `DeltaActions`, `AbsoluteActions`
- Tokenization: `TokenizePrompt`, `TokenizeFASTInputs`, `ExtractFASTActions`
- Dataset helpers: `PromptFromLeRobotTask`, `PadStatesAndActions`
- Tree utilities: `flatten_dict`, `unflatten_dict`, `transform_dict`, `apply_tree`, `pad_to_dim`, `make_bool_mask`

Tests: [`src/openpi/transforms_test.py`](../src/openpi/transforms_test.py)

### Policies

Directory: [`src/openpi/policies/`](../src/openpi/policies/)

- [`policy.py`](../src/openpi/policies/policy.py) - `Policy` applies input transforms, calls the JAX or PyTorch model, applies output transforms, and returns actions plus timing. `PolicyRecorder` records inputs/outputs to disk.
- [`policy_config.py`](../src/openpi/policies/policy_config.py) - `create_trained_policy`, which loads a checkpoint, detects JAX vs PyTorch weights, creates transforms, loads norm stats, and returns a `Policy`.
- Platform mappings:
  - [`aloha_policy.py`](../src/openpi/policies/aloha_policy.py) - ALOHA input/output mappings and gripper conversions.
  - [`droid_policy.py`](../src/openpi/policies/droid_policy.py) - DROID observations/actions.
  - [`libero_policy.py`](../src/openpi/policies/libero_policy.py) - LIBERO image/state/action/prompt mapping.
  - [`robocasa_policy.py`](../src/openpi/policies/robocasa_policy.py) - RoboCasa mapping.
  - [`robomimic_policy.py`](../src/openpi/policies/robomimic_policy.py) - RoboMimic mapping.
- Tests: [`policy_test.py`](../src/openpi/policies/policy_test.py)

### Training

Directory: [`src/openpi/training/`](../src/openpi/training/)

- [`config.py`](../src/openpi/training/config.py) - the largest control surface in the repo.
  - Asset/data configs: `AssetsConfig`, `DataConfig`, `DataConfigFactory`.
  - Data factories: `CustomLiberoDataConfig`, `SimpleDataConfig`, `LeRobotAlohaDataConfig`, `LeRobotLiberoDataConfig`, `LiberoProHDF5DataConfig`, `Libero10HDF5DataConfig`, `RobomimicDataConfig`, `RLDSDroidDataConfig`, `LeRobotDROIDDataConfig`, `LeRobotRobocasaDataConfig`.
  - Model transform selection: `ModelTransformFactory`.
  - Training config: `TrainConfig`.
  - Registry: `_CONFIGS`, `_CONFIGS_DICT`, `cli()`, `get_config()`.
  - Config families include base `pi0`, `pi0_fast`, `pi05`, ALOHA, DROID, LIBERO, LIBERO10, RoboCasa, RoboMimic, debug, LoRA/full-finetune, low-memory, and filtered-episode variants.
- [`data_loader.py`](../src/openpi/training/data_loader.py) - dataset protocols and concrete loaders:
  - `FilteredDataset`, `EpisodeFilteredDataset`, `TransformedDataset`, `IterableTransformedDataset`, `FakeDataset`.
  - HDF5 datasets for LIBERO Pro, LIBERO10, and RoboMimic.
  - `ConcatDataset`, torch/RLDS loader creation, transform wrapping, `TorchDataLoader`, `RLDSDataLoader`, `DataLoaderImpl`.
- [`checkpoints.py`](../src/openpi/training/checkpoints.py) - checkpoint directory initialization, save/restore, norm stats loading, Orbax callback support.
- [`weight_loaders.py`](../src/openpi/training/weight_loaders.py) - no-op, checkpoint, and PaliGemma weight loaders plus parameter merging.
- [`optimizer.py`](../src/openpi/training/optimizer.py) - cosine/rsqrt schedules and AdamW/SGD factories.
- [`sharding.py`](../src/openpi/training/sharding.py) - JAX mesh and FSDP/data sharding helpers.
- [`utils.py`](../src/openpi/training/utils.py) - `TrainState` and tree introspection utilities.
- [`droid_rlds_dataset.py`](../src/openpi/training/droid_rlds_dataset.py) - DROID RLDS dataset wrapper.
- [`misc/roboarena_config.py`](../src/openpi/training/misc/roboarena_config.py) - RoboArena-related config helpers.

### Serving

Directory: [`src/openpi/serving/`](../src/openpi/serving/)

- [`websocket_policy_server.py`](../src/openpi/serving/websocket_policy_server.py) - `WebsocketPolicyServer`, serving metadata on connect and msgpack-numpy inference requests over websocket.

### Shared Utilities

Directory: [`src/openpi/shared/`](../src/openpi/shared/)

- [`array_typing.py`](../src/openpi/shared/array_typing.py) - jaxtyping/beartype helpers and aliases.
- [`download.py`](../src/openpi/shared/download.py) - local/cache download helpers for checkpoint/assets paths.
- [`image_tools.py`](../src/openpi/shared/image_tools.py) - image conversion and resize-with-padding.
- [`normalize.py`](../src/openpi/shared/normalize.py) - norm stat computation/application helpers.
- [`nnx_utils.py`](../src/openpi/shared/nnx_utils.py) - NNX/JAX helper utilities.

### GROOT Utilities

Directory: [`src/openpi/groot_utils/`](../src/openpi/groot_utils/)

- [`groot_openpi_dataset.py`](../src/openpi/groot_utils/groot_openpi_dataset.py) - GROOT/OpenPI dataset adapter utilities.

## Client Package: `packages/openpi-client`

Directory: [`packages/openpi-client/`](../packages/openpi-client/)

- [`pyproject.toml`](../packages/openpi-client/pyproject.toml) - package metadata for the client library.
- [`base_policy.py`](../packages/openpi-client/src/openpi_client/base_policy.py) - minimal policy interface.
- [`websocket_client_policy.py`](../packages/openpi-client/src/openpi_client/websocket_client_policy.py) - websocket client implementation for remote policy inference.
- [`action_chunk_broker.py`](../packages/openpi-client/src/openpi_client/action_chunk_broker.py) - chunks model outputs into per-step actions.
- [`image_tools.py`](../packages/openpi-client/src/openpi_client/image_tools.py) - client-side image conversion/resizing.
- [`msgpack_numpy.py`](../packages/openpi-client/src/openpi_client/msgpack_numpy.py) - numpy-aware msgpack serialization.
- Runtime loop:
  - [`runtime/runtime.py`](../packages/openpi-client/src/openpi_client/runtime/runtime.py) - episode loop connecting environment, agent, and subscribers.
  - [`runtime/environment.py`](../packages/openpi-client/src/openpi_client/runtime/environment.py) - robot/environment interface.
  - [`runtime/agent.py`](../packages/openpi-client/src/openpi_client/runtime/agent.py) - decision-making interface.
  - [`runtime/subscriber.py`](../packages/openpi-client/src/openpi_client/runtime/subscriber.py) - hooks for logging/display/saving.
  - [`runtime/agents/policy_agent.py`](../packages/openpi-client/src/openpi_client/runtime/agents/policy_agent.py) - wraps a policy as a runtime agent.

## CLI Scripts

Directory: [`scripts/`](../scripts/)

Training and serving:

- [`train.py`](../scripts/train.py) - main JAX training loop with WandB, checkpointing, sharding, validation, and resume support.
- [`train_pytorch.py`](../scripts/train_pytorch.py) - PyTorch/DDP training loop and safetensors checkpoint handling.
- [`serve_policy.py`](../scripts/serve_policy.py) - starts a local websocket policy server from a default or checkpoint-backed policy.
- [`compute_norm_stats.py`](../scripts/compute_norm_stats.py) - computes dataset normalization statistics for a training config.

Checkpoint/data utilities:

- [`check_checkpoints.py`](../scripts/check_checkpoints.py) - inspects checkpoint availability from script/config metadata.
- [`merge_checkpoints.py`](../scripts/merge_checkpoints.py) - merges selected task/base parameters and writes checkpoint params.
- [`dump_filtered_data.py`](../scripts/dump_filtered_data.py) - extracts filtered episodes.
- [`extract_eval_reset_data.py`](../scripts/extract_eval_reset_data.py) - creates evaluation reset data from filtered episode config.
- [`unpack_hf5_episodes.py`](../scripts/unpack_hf5_episodes.py) - unpacks HDF5 episodes.
- [`inspect_episodes.py`](../scripts/inspect_episodes.py), [`inspect_dense.py`](../scripts/inspect_dense.py), [`dense_occlusion_check.py`](../scripts/dense_occlusion_check.py), [`occlusion_analysis.py`](../scripts/occlusion_analysis.py) - inspection/diagnostic scripts.

Visualization:

- [`visualize_demos.py`](../scripts/visualize_demos.py) - LeRobot/demo video visualization.
- [`visualize_robocasa_demos.py`](../scripts/visualize_robocasa_demos.py) - RoboCasa episode video rendering with prompts/actions.
- [`visualize_all_robocasa_tasks.py`](../scripts/visualize_all_robocasa_tasks.py) and [`visualize_composite_robocasa_tasks.py`](../scripts/visualize_composite_robocasa_tasks.py) - batch RoboCasa visualization.
- [`dump_hdf5_video.py`](../scripts/dump_hdf5_video.py), [`dump_single_episode.py`](../scripts/dump_single_episode.py), [`generate_episode_video.py`](../scripts/generate_episode_video.py) - video extraction/rendering helpers.
- [`vis_alphareq.py`](../scripts/vis_alphareq.py) - activation extraction/effective-rank analysis.

Docker:

- [`scripts/docker/serve_policy.Dockerfile`](../scripts/docker/serve_policy.Dockerfile)
- [`scripts/docker/compose.yml`](../scripts/docker/compose.yml)
- [`scripts/docker/install_docker_ubuntu22.sh`](../scripts/docker/install_docker_ubuntu22.sh)
- [`scripts/docker/install_nvidia_container_toolkit.sh`](../scripts/docker/install_nvidia_container_toolkit.sh)

## Examples

Directory: [`examples/`](../examples/)

- [`simple_client/`](../examples/simple_client/) - random-observation client for testing remote inference without a robot.
- [`libero/`](../examples/libero/) - LIBERO evaluation, Docker compose, and data conversion to LeRobot.
- [`droid/`](../examples/droid/) - DROID robot eval/client code, DROID data conversion, and training README.
- [`aloha_real/`](../examples/aloha_real/) - real ALOHA runtime environment, robot utilities, video display, and data conversion.
- [`aloha_sim/`](../examples/aloha_sim/) - ALOHA simulation runtime and video saving.
- [`robocasa/`](../examples/robocasa/) - RoboCasa eval, conversion to LeRobot, GROOT demo generation, and stats.
- [`ur5/`](../examples/ur5/) - UR5 example README.
- [`convert_jax_model_to_pytorch.py`](../examples/convert_jax_model_to_pytorch.py) - checkpoint conversion entry point.

## Documentation

Directory: [`docs/`](../docs/)

- [`docker.md`](../docs/docker.md) - Docker setup.
- [`norm_stats.md`](../docs/norm_stats.md) - normalization statistics workflow.
- [`remote_inference.md`](../docs/remote_inference.md) - websocket policy server/client usage.
- [`ROBOCASA_demo_meta_format.md`](../docs/ROBOCASA_demo_meta_format.md) - RoboCasa demo metadata format.

Additional root-level notes in this checkout:

- [`get_started.md`](../get_started.md), [`train.md`](../train.md), [`eval.md`](../eval.md)
- [`UNDERSTANDING.md`](../UNDERSTANDING.md), [`transformer_understanding.md`](../transformer_understanding.md)
- [`LoRA_Attention_Analysis.md`](../LoRA_Attention_Analysis.md)
- `CLAUDE_*.md` files - local working notes/session docs.

## Slurm and Orchestration

- [`slurm/`](../slurm/) - Slurm launch scripts grouped by task family:
  - `libero10/`
  - `robocasa/`
  - `robomimic/`
- [`slurm_orchestrator/`](../slurm_orchestrator/) - lightweight watchdog for resubmitting Slurm jobs.
  - [`orchestrator.py`](../slurm_orchestrator/orchestrator.py) - stdlib-only monitor that uses `sbatch`, `squeue`, and `sacct`.
  - [`README.md`](../slurm_orchestrator/README.md) - usage, config fields, status meanings, retire/reset commands.
  - `config*.json` and `state*.json` - job sets and persisted state.
  - [`launch.sh`](../slurm_orchestrator/launch.sh) - tmux launcher.

## Code Flow: `slurm/robocasa/fixed_fixture/load_dishwasher.slurm`

Batch entry point: [`slurm/robocasa/fixed_fixture/load_dishwasher.slurm`](../slurm/robocasa/fixed_fixture/load_dishwasher.slurm)

This Slurm file trains a one-demo RoboCasa `LoadDishwasher` pi0.5 LoRA policy on 4 GPUs. The selected config is:

```bash
CONFIG_NAME="pi05_robocasa_single_task_lora_load_dishwasher_action_dim12"
EXP_NAME="${CONFIG_NAME}-v1"
```

### 1. Slurm and Environment Setup

The batch script requests:

- 1 node
- 4 GPUs
- 16 CPUs
- 256 GB memory
- 48 hour wall time
- `general` partition / `normal` QoS
- stdout/stderr under `/data/user_data/$USER/pi05_robocasa_exps/logs/`

It then:

1. Enables `set -e`, so the job exits on the first failing command.
2. Exports JAX/XLA memory settings:
   - `XLA_PYTHON_CLIENT_PREALLOCATE=false`
   - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`
3. Sets WandB entity:
   - `WANDB_ENTITY=skowshik-carnegie-mellon-university`
4. Prepends OpenPI source to `PYTHONPATH`:
   - `/home/skowshik/vla/codebase/openpi/src`
5. Activates the `openpi` conda environment.
6. Changes working directory to the repository root.
7. Creates the run log directory.

### 2. Stage One: Compute Normalization Stats

The first Python command is:

```bash
python scripts/compute_norm_stats.py --config-name="${CONFIG_NAME}"
```

Code path:

1. [`scripts/compute_norm_stats.py`](../scripts/compute_norm_stats.py) calls `config = _config.get_config(config_name)`.
2. [`src/openpi/training/config.py`](../src/openpi/training/config.py) resolves `pi05_robocasa_single_task_lora_load_dishwasher_action_dim12` from `_CONFIGS`.
3. That config uses `LeRobotRobocasaDataConfig` with:
   - dataset path: `/data/hf_cache/datasets/robocasa/v1.0/pretrain/composite/LoadDishwasher/20250717/lerobot`
   - `asset_id="robocasa"`
   - layout/style filter: `[(59, 36)]`
   - fixture filter: `{"dishwasher": "dishwasher_1_island_group_1", "counter": "island_island_group_1"}`
   - `num_demos=1`
4. `LeRobotRobocasaDataConfig.create()` builds:
   - `RobocasaInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)`
   - `RobocasaOutputs()`
   - pi0.5 model transforms from `ModelTransformFactory`
   - enriched `data_dirs` containing the layout/style, fixture, and demo-count filters.
5. `compute_norm_stats.py` calls `create_torch_dataloader(...)`, which calls [`src/openpi/training/data_loader.py`](../src/openpi/training/data_loader.py) `create_torch_dataset(...)`.
6. Because `data_config.data_dirs` is set, `create_torch_dataset(...)` constructs [`GrootOpenpiSingleDataset`](../src/openpi/groot_utils/groot_openpi_dataset.py).
7. `GrootOpenpiSingleDataset`:
   - reads modality metadata from the LeRobot/GROOT dataset
   - selects video, state, action, and language modality keys
   - calls `get_scene_filtered_demos(...)`
   - filters episodes by layout/style `(59, 36)`
   - filters by exact fixture refs for the dishwasher and counter
   - truncates to the first matching demo because `num_demos=1`
8. Each dataset item is remapped in `GrootOpenpiSingleDataset.__getitem__()` to OpenPI-style keys:
   - `observation/image` from `video.robot0_agentview_left`
   - `observation/wrist_image` from `video.robot0_eye_in_hand`
   - `observation/image_right` from `video.robot0_agentview_right`
   - `observation/state` from concatenated end-effector, base, and gripper state
   - `actions` from concatenated end-effector, gripper, base-motion, and control-mode action fields
   - `prompt` from `annotation.human.task_description`
9. For norm stats only, the script applies:
   - repack transforms
   - `RobocasaInputs`
   - `RemoveStrings`
   - `RemoveImages`
10. It computes running stats for `state` and `actions`, then writes them under the config assets directory:

```text
assets/pi05_robocasa_single_task_lora_load_dishwasher_action_dim12/robocasa/
```

These stats are required by the training data loader unless `skip_norm_stats=True`.

### 3. Stage Two: Launch Training

The second Python command is:

```bash
python scripts/train.py "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --project-name=pi05_robocasa \
    --resume
```

Tyro parses this into the same `TrainConfig`, overriding:

- `exp_name = "pi05_robocasa_single_task_lora_load_dishwasher_action_dim12-v1"`
- `project_name = "pi05_robocasa"`
- `resume = True`

The checkpoint directory becomes:

```text
/data/hf_cache/models/pi05_robocasa_exps/
  pi05_robocasa_single_task_lora_load_dishwasher_action_dim12/
  pi05_robocasa_single_task_lora_load_dishwasher_action_dim12-v1/
```

### 4. Training Config Details

The config block in [`src/openpi/training/config.py`](../src/openpi/training/config.py) sets:

- model: `Pi0Config(pi05=True, action_horizon=10, paligemma_variant="gemma_2b_lora")`
- action dimension: `12`
- base weights: `gs://openpi-assets/checkpoints/pi05_base/params`
- checkpoint base: `/data/hf_cache/models/pi05_robocasa_exps/`
- freeze filter: `Pi0Config(paligemma_variant="gemma_2b_lora").get_freeze_filter()`
- no EMA: `ema_decay=None`
- train steps: `100_000`
- batch size: `64`
- save interval: `4000`
- keep period: `1000`
- workers: `4`
- log interval: `100`
- action L1 logging interval: `1000`

Because the model uses `paligemma_variant="gemma_2b_lora"`, the freeze filter freezes the base Gemma/PaliGemma parameters while leaving LoRA parameters trainable. `TrainConfig.trainable_filter` is `nnx.Param` excluding that freeze filter.

### 5. Dataset and Transform Flow During Training

In [`scripts/train.py`](../scripts/train.py), `main(config)` does the following:

1. Initializes logging and checks that global batch size is divisible by the JAX device count.
2. Creates a JAX mesh via [`src/openpi/training/sharding.py`](../src/openpi/training/sharding.py).
3. Initializes or resumes the checkpoint manager via [`src/openpi/training/checkpoints.py`](../src/openpi/training/checkpoints.py).
4. Initializes WandB. With `--resume`, it reuses `wandb_id.txt` if present.
5. Calls `config.data.create(...)` to materialize the RoboCasa `DataConfig`.
6. If this is not a resume and a filtered RoboCasa dataset is present, saves `reference_ep_meta.json` in the checkpoint directory. This uses `get_reference_ep_meta(...)` from [`src/openpi/groot_utils/groot_openpi_dataset.py`](../src/openpi/groot_utils/groot_openpi_dataset.py) so evaluation can reproduce the same layout/style/fixture setup.
7. Logs norm-stat quantiles to WandB.
8. Calls `create_data_loader(...)`.

The training data loader follows this transform sequence:

1. `GrootOpenpiSingleDataset.__getitem__()` returns raw OpenPI-style RoboCasa sample dictionaries.
2. [`RobocasaInputs`](../src/openpi/policies/robocasa_policy.py) maps them to model-facing keys:
   - `state`
   - `image/base_0_rgb`
   - `image/left_wrist_0_rgb`
   - `image/right_wrist_0_rgb`
   - image masks
   - `actions`
   - `prompt`
3. [`Normalize`](../src/openpi/transforms.py) applies quantile normalization using the stats computed in stage one. For pi0.5, `DataConfigFactory.create_base_config(...)` sets `use_quantile_norm=True`.
4. `ModelTransformFactory` applies pi0.5 model transforms:
   - `InjectDefaultPrompt`
   - `ResizeImages(224, 224)`
   - `TokenizePrompt(PaligemmaTokenizer(max_token_len=200), discrete_state_input=True)`
   - `PadStatesAndActions(action_dim=12)`
5. `DataLoaderImpl` converts transformed dictionaries into:
   - [`Observation`](../src/openpi/models/model.py)
   - action tensor shaped by batch, action horizon `10`, and action dim `12`

### 6. Model Initialization and Resume Behavior

[`scripts/train.py`](../scripts/train.py) calls `init_train_state(config, init_rng, mesh, resume=resuming)`.

- If no checkpoint is being resumed, it:
  1. creates [`Pi0`](../src/openpi/models/pi0.py) from `Pi0Config`
  2. loads partial base weights through `CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`
  3. initializes optimizer state for trainable parameters only.
- If resuming, it first builds only the train-state shape/sharding, then later calls `restore_state(...)` to load the latest checkpoint.

The model architecture is the pi0.5 flow-matching path:

- image encoder: SigLIP
- language/VLA backbone: PaliGemma/Gemma 2B with LoRA
- action expert: Gemma 300M-style expert
- action horizon: 10
- action dimension: 12
- pi0.5 mode uses discrete state input and AdaRMS conditioning for the flow timestep.

### 7. Training Step Flow

Each JIT-compiled `train_step(...)` in [`scripts/train.py`](../scripts/train.py):

1. Reconstructs the model with `nnx.merge(state.model_def, state.params)`.
2. Calls [`Pi0.compute_loss(...)`](../src/openpi/models/pi0.py).
3. `compute_loss(...)`:
   - preprocesses observations through `preprocess_observation(..., train=True)`
   - samples Gaussian noise
   - samples a diffusion/flow timestep
   - constructs noisy actions `x_t`
   - embeds image and prompt prefix tokens
   - embeds noisy action suffix tokens with timestep conditioning
   - runs the PaliGemma/Gemma backbone
   - projects suffix outputs to action velocity `v_t`
   - returns MSE against the flow target `u_t = noise - actions`
4. Gradients are computed only for `config.trainable_filter`.
5. Optax updates the trainable parameters.
6. The loop logs:
   - loss
   - gradient norm
   - action output projection gradient norm
   - parameter norms
   - per-dimension action stats
   - periodic action L1 losses from sampled actions.

### 8. Checkpoints and Outputs

During the run:

- Checkpoints are saved every `4000` steps through [`src/openpi/training/checkpoints.py`](../src/openpi/training/checkpoints.py).
- Every checkpoint whose step is divisible by `keep_period=1000` is preserved according to the checkpoint manager settings.
- Trainable parameter names are dumped to:

```text
logs/trainable_pi05_robocasa_single_task_lora_load_dishwasher_action_dim12.pkl
```

- WandB logs go to project `pi05_robocasa`.
- If this is the first non-resumed run, `reference_ep_meta.json` is written into the checkpoint directory.

### 9. Inference Path After Training

After a checkpoint exists, the serving path is:

1. [`scripts/serve_policy.py`](../scripts/serve_policy.py) parses a checkpoint-backed policy request.
2. [`src/openpi/policies/policy_config.py`](../src/openpi/policies/policy_config.py) `create_trained_policy(...)`:
   - loads the model checkpoint
   - loads norm stats from checkpoint assets
   - recreates the same data/model transforms
   - returns [`Policy`](../src/openpi/policies/policy.py)
3. [`Policy.infer(...)`](../src/openpi/policies/policy.py) applies input transforms, calls `model.sample_actions(...)`, applies `RobocasaOutputs`, and returns only the first 12 RoboCasa action dimensions.
4. [`src/openpi/serving/websocket_policy_server.py`](../src/openpi/serving/websocket_policy_server.py) can expose that policy remotely.
5. [`packages/openpi-client/src/openpi_client/websocket_client_policy.py`](../packages/openpi-client/src/openpi_client/websocket_client_policy.py) can call the remote policy from an evaluation or robot runtime.

## Tests

Project tests are under `src`, `scripts`, and `packages` per `pyproject.toml`.

- Model tests:
  - [`src/openpi/models/model_test.py`](../src/openpi/models/model_test.py)
  - [`src/openpi/models/pi0_test.py`](../src/openpi/models/pi0_test.py)
  - [`src/openpi/models/lora_test.py`](../src/openpi/models/lora_test.py)
  - [`src/openpi/models/tokenizer_test.py`](../src/openpi/models/tokenizer_test.py)
- Transform/policy/shared tests:
  - [`src/openpi/transforms_test.py`](../src/openpi/transforms_test.py)
  - [`src/openpi/policies/policy_test.py`](../src/openpi/policies/policy_test.py)
  - [`src/openpi/shared/download_test.py`](../src/openpi/shared/download_test.py)
  - [`src/openpi/shared/image_tools_test.py`](../src/openpi/shared/image_tools_test.py)
  - [`src/openpi/shared/normalize_test.py`](../src/openpi/shared/normalize_test.py)
- Training/script tests:
  - [`src/openpi/training/data_loader_test.py`](../src/openpi/training/data_loader_test.py)
  - [`scripts/train_test.py`](../scripts/train_test.py)
- Client tests:
  - [`packages/openpi-client/src/openpi_client/image_tools_test.py`](../packages/openpi-client/src/openpi_client/image_tools_test.py)
  - [`packages/openpi-client/src/openpi_client/msgpack_numpy_test.py`](../packages/openpi-client/src/openpi_client/msgpack_numpy_test.py)

Typical test command:

```bash
uv run pytest
```

## Generated or Local-Only Content to Treat Carefully

These directories/files can be large, machine-specific, or generated during experiments:

- `assets/` - downloaded/copied policy assets and norm stats.
- `logs/` - many local pickle artifacts.
- `wandb/` - local WandB run directories.
- `data/`, `data_dumps/`, `eval_reset_data/` - datasets or derived data.
- `videos/`, `robocasa_vis/`, `vis_demos/`, `vis_train_data_dump*/`, `robocasa_groot/` - rendered rollouts and visualizations.
- `.venv/`, `.pytest_cache/`, `__pycache__/` - local environment/cache.
- `examples/libero/.venv/` - local nested virtualenv in this checkout; ignore for source navigation.

## Where to Start for Common Tasks

- Add a new training run/config: start in [`src/openpi/training/config.py`](../src/openpi/training/config.py), then check the matching policy mapping in [`src/openpi/policies/`](../src/openpi/policies/).
- Add a new dataset source: start with `DataConfigFactory` subclasses in [`config.py`](../src/openpi/training/config.py) and dataset implementations in [`data_loader.py`](../src/openpi/training/data_loader.py).
- Add a new robot/task adapter: add input/output transforms in [`src/openpi/policies/`](../src/openpi/policies/) and an example runner under [`examples/`](../examples/).
- Debug model input shapes: inspect [`src/openpi/models/model.py`](../src/openpi/models/model.py), [`src/openpi/transforms.py`](../src/openpi/transforms.py), and the selected `DataConfig`.
- Serve a checkpoint: use [`scripts/serve_policy.py`](../scripts/serve_policy.py) plus [`src/openpi/policies/policy_config.py`](../src/openpi/policies/policy_config.py).
- Call a remote policy: use [`packages/openpi-client/src/openpi_client/websocket_client_policy.py`](../packages/openpi-client/src/openpi_client/websocket_client_policy.py) or [`examples/simple_client/main.py`](../examples/simple_client/main.py).
- Work on PyTorch support: inspect [`src/openpi/models_pytorch/`](../src/openpi/models_pytorch/), [`scripts/train_pytorch.py`](../scripts/train_pytorch.py), and [`examples/convert_jax_model_to_pytorch.py`](../examples/convert_jax_model_to_pytorch.py).
