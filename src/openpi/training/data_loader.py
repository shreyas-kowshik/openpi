from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import h5py
import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class FilteredDataset(Dataset[T_co]):
    """A dataset that filters samples based on a prompt match.
    
    This implementation uses dataset metadata for fast filtering when available,
    avoiding the need to load full samples during index building.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        filter_prompt: str,
        lerobot_dataset=None,
        dataset_meta=None,
        num_episodes: int = -1,
        exclude_filter_prompt: bool = False,
    ):
        """Initialize filtered dataset.
        
        Args:
            dataset: The dataset to filter (may be wrapped with transforms)
            filter_prompt: The prompt/task string to filter by
            lerobot_dataset: Optional raw LeRobot dataset for fast filtering
            dataset_meta: Optional LeRobotDatasetMetadata for fast filtering
            num_episodes: Number of episodes to keep. -1 means keep all, positive value limits episodes.
            exclude_filter_prompt: If True, exclude samples matching filter_prompt instead of including them
        """
        self._dataset = dataset
        self._filter_prompt = filter_prompt
        self._lerobot_dataset = lerobot_dataset
        self._dataset_meta = dataset_meta
        self._num_episodes = num_episodes
        self._exclude_filter_prompt = exclude_filter_prompt
        self._valid_indices = self._build_index()
    
    def _build_index(self) -> list[int]:
        """Build an index of valid samples that match the filter prompt."""
        filter_type = "exclude" if self._exclude_filter_prompt else "include"
        logging.info(f"Building index for prompt filter ({filter_type}): '{self._filter_prompt}'")
        logging.info(f"Dataset length: {len(self._dataset)}")
        
        # Try fast path using metadata
        valid_indices = self._try_fast_filter()
        if valid_indices is not None:
            logging.info(
                f"Filtered dataset (fast): {len(valid_indices)} / {len(self._dataset)} samples "
                f"({filter_type} prompt '{self._filter_prompt}')"
            )
        else:
            # Fallback to slow path
            logging.info("Using slow path: loading samples individually...")
            valid_indices = []
            for i in range(len(self._dataset)):
                if i % 1000 == 0 and i > 0:
                    logging.info(f"Processing sample {i} of {len(self._dataset)}")
                
                sample = self._dataset[i]
                sample_prompt = sample.get('task', sample.get('prompt', None))
                
                matches = (sample_prompt == self._filter_prompt)
                if (matches and not self._exclude_filter_prompt) or (not matches and self._exclude_filter_prompt):
                    valid_indices.append(i)
            
            logging.info(
                f"Filtered dataset: {len(valid_indices)} / {len(self._dataset)} samples "
                f"({filter_type} prompt '{self._filter_prompt}')"
            )
        
        return valid_indices
    
    def _try_fast_filter(self) -> list[int] | None:
        """Try fast filtering using dataset metadata.
        
        Returns:
            List of valid indices if successful, None otherwise.
        """
        # Check if we have everything needed for fast filtering
        if self._lerobot_dataset is None or self._dataset_meta is None:
            return None
        
        if not hasattr(self._lerobot_dataset, 'hf_dataset'):
            return None
        
        if not hasattr(self._dataset_meta, 'tasks'):
            return None
        
        try:
            logging.info("Using fast path: accessing dataset metadata directly...")
            
            hf_dataset = self._lerobot_dataset.hf_dataset
            
            # Check if task_index and episode_index columns exist
            if 'task_index' not in hf_dataset.column_names:
                logging.warning("No 'task_index' column found, falling back to slow path")
                return None
            
            if 'episode_index' not in hf_dataset.column_names:
                logging.warning("No 'episode_index' column found, falling back to slow path")
                return None
            
            # Filter using task indices
            task_indices = hf_dataset['task_index']
            episode_indices = hf_dataset['episode_index']
            tasks_map = self._dataset_meta.tasks
            
            # First pass: find all samples matching the prompt and track unique episodes per task
            valid_indices = []
            episodes_seen = []  # Track episode order
            episodes_set = set()  # For fast lookup
            task_episode_counts = {}  # Track episodes per task
            
            for i, (task_idx, ep_idx) in enumerate(zip(task_indices, episode_indices)):
                if i % 10000 == 0 and i > 0:
                    logging.info(f"  Processed {i} / {len(task_indices)} samples")
                
                task_prompt = tasks_map.get(int(task_idx))
                task_idx_int = int(task_idx)
                ep_idx_int = int(ep_idx)
                
                # Track episodes per task
                if task_idx_int not in task_episode_counts:
                    task_episode_counts[task_idx_int] = set()
                task_episode_counts[task_idx_int].add(ep_idx_int)
                
                matches = (task_prompt == self._filter_prompt)
                if (matches and not self._exclude_filter_prompt) or (not matches and self._exclude_filter_prompt):
                    # Track episodes in order
                    if ep_idx_int not in episodes_set:
                        episodes_seen.append(ep_idx_int)
                        episodes_set.add(ep_idx_int)
                    
                    valid_indices.append(i)
            
            # Print all tasks loaded with episode counts
            logging.info("\n=== Tasks Loaded ===")
            total_episodes = 0
            for task_idx in sorted(task_episode_counts.keys()):
                task_name = tasks_map.get(task_idx, f"Unknown_{task_idx}")
                num_eps = len(task_episode_counts[task_idx])
                total_episodes += num_eps
                logging.info(f"  Task '{task_name}': {num_eps} episodes")
            logging.info(f"Total episodes loaded: {total_episodes}")
            logging.info("=" * 50)
            
            # Limit by number of episodes if specified (only when not excluding)
            if not self._exclude_filter_prompt and self._num_episodes > 0 and len(episodes_seen) > self._num_episodes:
                # Keep only samples from the first N episodes
                episodes_to_keep = set(episodes_seen[:self._num_episodes])
                
                # Filter valid_indices to only include samples from selected episodes
                filtered_indices = []
                for idx in valid_indices:
                    ep_idx = int(episode_indices[idx])
                    if ep_idx in episodes_to_keep:
                        filtered_indices.append(idx)
                
                logging.info(
                    f"Limited dataset from {len(episodes_seen)} episodes to {self._num_episodes} episodes "
                    f"({len(valid_indices)} samples -> {len(filtered_indices)} samples)"
                )
                valid_indices = filtered_indices
            else:
                logging.info(f"Dataset contains {len(episodes_seen)} episodes with {len(valid_indices)} total samples")
            
            return valid_indices
            
        except Exception as e:
            logging.warning(f"Fast filtering failed: {e}. Falling back to slow path.")
            return None
    
    def __getitem__(self, index: SupportsIndex) -> T_co:
        if index >= len(self._valid_indices):
            raise IndexError(f"Index {index} out of range for filtered dataset")
        actual_index = self._valid_indices[index]
        return self._dataset[actual_index]
    
    def __len__(self) -> int:
        return len(self._valid_indices)


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class LiberoProHDF5Dataset(Dataset):
    """Dataset that reads directly from a LIBERO-PRO HDF5 demo file.

    Returns items with the same keys as the ``physical-intelligence/libero`` LeRobot dataset so
    they can be combined with LeRobot samples and pass through the same repack / data transforms:
      - image:       main camera    (H, W, 3) uint8
      - wrist_image: wrist camera   (H, W, 3) uint8
      - state:       ee_states (6,) concat gripper_states (2,) → (8,) float32
      - actions:     (action_horizon, 7) float32, last action padded at episode end
      - prompt:      task language instruction string
    """

    def __init__(self, hdf5_path: str, action_horizon: int, num_episodes: int = -1):
        import json as _json

        self._action_horizon = action_horizon
        self._samples: list[tuple[str, int]] = []  # (demo_key, timestep_idx)
        self._data: dict = {}  # demo_key -> {arrays}

        with h5py.File(hdf5_path, "r") as f:
            raw_prompt = f["data"].attrs.get("problem_info", "")
            try:
                self._prompt = _json.loads(raw_prompt).get("language_instruction", "")
            except Exception:
                self._prompt = str(raw_prompt)

            demo_keys = sorted(f["data"].keys())
            if num_episodes > 0:
                demo_keys = demo_keys[:num_episodes]

            logging.info(f"LiberoProHDF5Dataset: loading {len(demo_keys)} episodes from {hdf5_path}")

            for demo_key in demo_keys:
                demo = f["data"][demo_key]
                T = demo["actions"].shape[0]
                self._data[demo_key] = {
                    "agentview_rgb": demo["obs/agentview_rgb"][:],                          # (T, H, W, 3) uint8
                    "eye_in_hand_rgb": demo["obs/eye_in_hand_rgb"][:],                      # (T, H, W, 3) uint8
                    "ee_states": demo["obs/ee_states"][:].astype(np.float32),              # (T, 6)
                    "gripper_states": demo["obs/gripper_states"][:].astype(np.float32),    # (T, 2)
                    "actions": demo["actions"][:].astype(np.float32),                      # (T, 7)
                }
                for t in range(T):
                    self._samples.append((demo_key, t))

        logging.info(
            f"LiberoProHDF5Dataset: {len(demo_keys)} episodes, {len(self._samples)} timestep samples, "
            f"prompt='{self._prompt}'"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: SupportsIndex) -> dict:
        demo_key, t = self._samples[index.__index__()]
        data = self._data[demo_key]
        T = data["actions"].shape[0]

        # Build action chunk [t, t+action_horizon); pad with last action at episode end.
        actions = np.empty((self._action_horizon, 7), dtype=np.float32)
        for i in range(self._action_horizon):
            actions[i] = data["actions"][min(t + i, T - 1)]

        state = np.concatenate([data["ee_states"][t], data["gripper_states"][t]], axis=-1)  # (8,)

        # LIBERO-PRO HDF5 images are stored vertically and horizontally flipped; flip both axes to match LeRobot orientation.
        image = data["agentview_rgb"][t][::-1, ::-1].copy()
        wrist_image = data["eye_in_hand_rgb"][t][::-1, ::-1].copy()

        # Keys match the physical-intelligence/libero LeRobot dataset so the same repack
        # transform (image→observation/image, etc.) works for both data sources.
        return {
            "image": image,
            "wrist_image": wrist_image,
            "state": state,
            "actions": actions,
            "prompt": self._prompt,
        }


class ConcatDataset(Dataset):
    """Concatenates multiple datasets into one."""

    def __init__(self, datasets: Sequence[Dataset]):
        self._datasets = list(datasets)
        self._lengths = [len(d) for d in self._datasets]
        self._total = sum(self._lengths)

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        for dataset, length in zip(self._datasets, self._lengths):
            if idx < length:
                return dataset[idx]
            idx -= length
        raise IndexError(f"Index {index.__index__()} out of range for ConcatDataset of size {self._total}")


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # Load dataset metadata
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    
    # Print dataset metadata
    if hasattr(dataset_meta, 'tasks') and hasattr(dataset_meta, 'total_episodes'):
        logging.info(f"Dataset: {repo_id} — {dataset_meta.total_episodes} episodes, {len(dataset_meta.tasks)} tasks")
    
    # Load raw LeRobot dataset
    lerobot_ds = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    # Apply prompt filtering BEFORE transforms for efficiency
    dataset = lerobot_ds
    if data_config.filter_prompt is not None:
        dataset = FilteredDataset(
            dataset,
            filter_prompt=data_config.filter_prompt,
            lerobot_dataset=lerobot_ds,
            dataset_meta=dataset_meta,
            num_episodes=data_config.num_episodes,
            exclude_filter_prompt=data_config.exclude_filter_prompt,
        )
    
    # Apply transforms after filtering
    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    # If an HDF5 path is also specified, create an additional dataset from it and concatenate.
    hdf5_path = getattr(data_config, "hdf5_path", None)
    if hdf5_path is not None:
        hdf5_num_episodes = getattr(data_config, "hdf5_num_episodes", -1)
        hdf5_dataset = LiberoProHDF5Dataset(hdf5_path, action_horizon, hdf5_num_episodes)
        logging.info(
            f"Combining LeRobot dataset ({len(dataset)} samples) with "
            f"HDF5 dataset ({len(hdf5_dataset)} samples) from {hdf5_path}"
        )
        dataset = ConcatDataset([dataset, hdf5_dataset])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
