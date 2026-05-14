from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import h5py
import jax
import jax.numpy as jnp
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.groot_utils.groot_openpi_dataset as _groot_openpi_dataset
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


class EpisodeFilteredDataset(Dataset[T_co]):
    """A dataset that keeps only samples belonging to a specific set of episode IDs."""

    def __init__(self, dataset: Dataset, episode_ids: Sequence[int], lerobot_dataset=None):
        self._dataset = dataset
        self._episode_ids = set(episode_ids)
        self._valid_indices = self._build_index(lerobot_dataset)

    def _build_index(self, lerobot_dataset) -> list[int]:
        logging.info(f"Filtering dataset to episode IDs: {sorted(self._episode_ids)}")

        # Fast path: use the raw LeRobot hf_dataset to look up episode_index,
        # but iterate over the *wrapped* dataset's index space.
        if lerobot_dataset is not None and hasattr(lerobot_dataset, 'hf_dataset'):
            hf_dataset = lerobot_dataset.hf_dataset
            if 'episode_index' in hf_dataset.column_names:
                episode_indices = hf_dataset['episode_index']

                # If the wrapped dataset is a FilteredDataset, map through its valid_indices
                # to get the original LeRobot sample index for each wrapped index.
                if isinstance(self._dataset, FilteredDataset):
                    inner_indices = self._dataset._valid_indices
                    valid_indices = [
                        i for i, orig_idx in enumerate(inner_indices)
                        if int(episode_indices[orig_idx]) in self._episode_ids
                    ]
                else:
                    valid_indices = [
                        i for i, ep_idx in enumerate(episode_indices)
                        if int(ep_idx) in self._episode_ids
                    ]

                # Log which episodes were actually found
                if isinstance(self._dataset, FilteredDataset):
                    found_eps = sorted({int(episode_indices[inner_indices[i]]) for i in valid_indices})
                else:
                    found_eps = sorted({int(episode_indices[i]) for i in valid_indices})
                missing_eps = sorted(self._episode_ids - set(found_eps))
                logging.info(
                    f"\n=== EpisodeFilteredDataset ===\n"
                    f"  Requested episode IDs: {sorted(self._episode_ids)}\n"
                    f"  Found episode IDs:     {found_eps}\n"
                    f"  Missing episode IDs:   {missing_eps}\n"
                    f"  Samples kept: {len(valid_indices)} / {len(self._dataset)}\n"
                    f"{'=' * 50}"
                )
                return valid_indices

        # Slow fallback: load each sample and check
        logging.info("EpisodeFilteredDataset: using slow path (no episode_index column)")
        valid_indices = []
        for i in range(len(self._dataset)):
            sample = self._dataset[i]
            ep_idx = sample.get('episode_index', None)
            if ep_idx is not None and int(ep_idx) in self._episode_ids:
                valid_indices.append(i)
        logging.info(
            f"EpisodeFilteredDataset (slow): {len(valid_indices)} / {len(self._dataset)} samples"
        )
        return valid_indices

    def __getitem__(self, index: SupportsIndex) -> T_co:
        actual_index = self._valid_indices[index.__index__()]
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


class Libero10HDF5Dataset(Dataset):
    """Dataset that reads from original LIBERO benchmark HDF5 files (one file per task).

    Scans a directory for ``.hdf5`` files (e.g. the ``libero_10/`` folder produced by the
    LIBERO download script) and loads all demos from every file.  Each file corresponds
    to a single task; the language instruction is extracted from the ``problem_info``
    attribute stored inside the HDF5.

    Returns items with the same keys as ``LiberoProHDF5Dataset`` so the standard Libero
    repack / data transforms work without modification:
      - image:       agentview camera   (H, W, 3) uint8
      - wrist_image: wrist camera       (H, W, 3) uint8
      - state:       ee_states (6,) concat gripper_states (2,) → (8,) float32
      - actions:     (action_horizon, 7) float32, last action padded at episode end
      - prompt:      task language instruction string
    """

    def __init__(
        self,
        data_dir: str,
        action_horizon: int,
        num_episodes_per_task: int = -1,
        flip_images: bool = True,
        filter_prompt: str | None = None,
        hdf5_filenames: Sequence[str] | None = None,
    ):
        """
        Args:
            data_dir: Directory containing LIBERO .hdf5 files.
            action_horizon: Number of future actions to chunk per sample.
            num_episodes_per_task: Max demos to load per file (-1 = all).
            flip_images: Flip images (robosuite stores them flipped).
            filter_prompt: If set, only load files whose language instruction
                contains this substring (case-insensitive).
            hdf5_filenames: If set, only load these specific filenames
                (e.g. ``["KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5"]``).
        """
        import json as _json
        import glob as _glob

        self._action_horizon = action_horizon
        self._flip_images = flip_images
        # (file_idx, demo_key, timestep) → flat sample index
        self._samples: list[tuple[int, str, int]] = []
        # file_idx → {demo_key → {arrays}}
        self._file_data: list[dict] = []
        # file_idx → prompt string
        self._file_prompts: list[str] = []

        if hdf5_filenames:
            hdf5_paths = sorted([os.path.join(data_dir, fn) for fn in hdf5_filenames])
            missing = [p for p in hdf5_paths if not os.path.isfile(p)]
            if missing:
                raise FileNotFoundError(f"HDF5 files not found: {missing}")
        else:
            hdf5_paths = sorted(_glob.glob(os.path.join(data_dir, "*.hdf5")))
        if not hdf5_paths:
            raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")

        total_demos = 0
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, "r") as f:
                # Extract language instruction from problem_info attribute.
                raw_info = f["data"].attrs.get("problem_info", "")
                try:
                    info = _json.loads(raw_info)
                    lang = info.get("language_instruction", "")
                    if isinstance(lang, list):
                        lang = "".join(lang)
                    prompt = lang.strip('"').strip()
                except Exception:
                    prompt = str(raw_info)

                # Skip this file if it doesn't match the filter_prompt.
                if filter_prompt is not None and filter_prompt.lower() not in prompt.lower():
                    logging.info(
                        f"Libero10HDF5Dataset: skipping {os.path.basename(hdf5_path)} "
                        f"(prompt '{prompt}' doesn't match filter '{filter_prompt}')"
                    )
                    continue

                demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo")])
                if num_episodes_per_task > 0:
                    demo_keys = demo_keys[:num_episodes_per_task]

                # Use current list length as the file index (accounts for skipped files).
                file_idx = len(self._file_data)
                file_data: dict = {}
                for demo_key in demo_keys:
                    demo = f["data"][demo_key]
                    T = demo["actions"].shape[0]
                    file_data[demo_key] = {
                        "agentview_rgb": demo["obs/agentview_rgb"][:],
                        "eye_in_hand_rgb": demo["obs/eye_in_hand_rgb"][:],
                        "ee_states": demo["obs/ee_states"][:].astype(np.float32),
                        "gripper_states": demo["obs/gripper_states"][:].astype(np.float32),
                        "actions": demo["actions"][:].astype(np.float32),
                    }
                    for t in range(T):
                        self._samples.append((file_idx, demo_key, t))

                total_demos += len(demo_keys)
                self._file_data.append(file_data)
                self._file_prompts.append(prompt)

                logging.info(
                    f"Libero10HDF5Dataset: loaded {len(demo_keys)} demos from "
                    f"{os.path.basename(hdf5_path)}, prompt='{prompt}'"
                )

        logging.info(
            f"Libero10HDF5Dataset: {len(hdf5_paths)} files, {total_demos} total demos, "
            f"{len(self._samples)} timestep samples"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: SupportsIndex) -> dict:
        file_idx, demo_key, t = self._samples[index.__index__()]
        data = self._file_data[file_idx][demo_key]
        T = data["actions"].shape[0]

        # Build action chunk [t, t+action_horizon); pad with last action at episode end.
        act_dim = data["actions"].shape[-1]
        actions = np.empty((self._action_horizon, act_dim), dtype=np.float32)
        for i in range(self._action_horizon):
            actions[i] = data["actions"][min(t + i, T - 1)]

        state = np.concatenate([data["ee_states"][t], data["gripper_states"][t]], axis=-1)

        image = data["agentview_rgb"][t]
        wrist_image = data["eye_in_hand_rgb"][t]
        if self._flip_images:
            image = image[::-1, ::-1].copy()
            wrist_image = wrist_image[::-1, ::-1].copy()

        return {
            "image": image,
            "wrist_image": wrist_image,
            "state": state,
            "actions": actions,
            "prompt": self._file_prompts[file_idx],
        }


class RobomimicHDF5Dataset(Dataset):
    """Dataset that reads directly from a robomimic image HDF5 file.

    Expects an HDF5 produced by robomimic's ``dataset_states_to_obs.py`` with
    ``--camera_names agentview robot0_eye_in_hand``.

    Returns items with keys:
      - image:       agentview camera   (H, W, 3) uint8
      - wrist_image: wrist camera       (H, W, 3) uint8
      - state:       eef_pos (3) + eef_quat (4) + gripper_qpos (2) → (9,) float32
      - actions:     (action_horizon, 7) float32
      - prompt:      task language instruction string
    """

    def __init__(self, hdf5_path: str, action_horizon: int, task_description: str, num_episodes: int = -1):
        self._action_horizon = action_horizon
        self._prompt = task_description
        self._samples: list[tuple[str, int]] = []
        self._data: dict = {}

        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
            if num_episodes > 0:
                demo_keys = demo_keys[:num_episodes]

            logging.info(f"RobomimicHDF5Dataset: loading {len(demo_keys)} episodes from {hdf5_path}")

            for demo_key in demo_keys:
                demo = f["data"][demo_key]
                T = demo["actions"].shape[0]
                self._data[demo_key] = {
                    "agentview_image": demo["obs/agentview_image"][:],
                    "wrist_image": demo["obs/robot0_eye_in_hand_image"][:],
                    "eef_pos": demo["obs/robot0_eef_pos"][:].astype(np.float32),
                    "eef_quat": demo["obs/robot0_eef_quat"][:].astype(np.float32),
                    "gripper_qpos": demo["obs/robot0_gripper_qpos"][:].astype(np.float32),
                    "actions": demo["actions"][:].astype(np.float32),
                }
                for t in range(T):
                    self._samples.append((demo_key, t))

        logging.info(
            f"RobomimicHDF5Dataset: {len(demo_keys)} episodes, {len(self._samples)} timestep samples, "
            f"prompt='{self._prompt}'"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: SupportsIndex) -> dict:
        demo_key, t = self._samples[index.__index__()]
        data = self._data[demo_key]
        T = data["actions"].shape[0]

        actions = np.empty((self._action_horizon, 7), dtype=np.float32)
        for i in range(self._action_horizon):
            actions[i] = data["actions"][min(t + i, T - 1)]

        state = np.concatenate([
            data["eef_pos"][t], data["eef_quat"][t], data["gripper_qpos"][t]
        ], axis=-1)  # (9,)

        return {
            "image": data["agentview_image"][t],
            "wrist_image": data["wrist_image"][t],
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

    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # Groot/RoboCasa dataset loading
    if getattr(data_config, "data_dirs", None):
        data_dirs = data_config.data_dirs
        if len(data_dirs) == 1:
            return _groot_openpi_dataset.GrootOpenpiSingleDataset(
                dataset_meta=data_dirs[0],
                action_horizon=action_horizon,
            )
        elif len(data_dirs) > 1:
            return _groot_openpi_dataset.GrootOpenpiMultiDataset(
                dataset_meta_list=data_dirs,
                dataset_weights=getattr(data_config, "dataset_weights", None),
                dataset_weights_alpha=0.4,
                action_horizon=action_horizon,
            )
        else:
            raise ValueError("data_dirs is empty")

    # Robomimic HDF5-only dataset loading
    if repo_id == "robomimic":
        hdf5_path = data_config.hdf5_path
        if hdf5_path is None:
            raise ValueError("robomimic config requires hdf5_path to be set.")
        return RobomimicHDF5Dataset(
            hdf5_path, action_horizon, data_config.task_description, data_config.hdf5_num_episodes
        )

    # LIBERO-10 HDF5 directory loading
    if repo_id == "libero_10_hdf5":
        libero10_dir = getattr(data_config, "libero10_data_dir", None)
        if libero10_dir is None:
            raise ValueError("libero_10_hdf5 config requires libero10_data_dir to be set.")
        return Libero10HDF5Dataset(
            data_dir=libero10_dir,
            action_horizon=action_horizon,
            num_episodes_per_task=data_config.hdf5_num_episodes,
            flip_images=getattr(data_config, "flip_images", True),
            filter_prompt=data_config.filter_prompt,
            hdf5_filenames=getattr(data_config, "hdf5_filenames", None),
        )

    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")

    # Standard (openpi) LeRobot dataset loading
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

    # Validate: episode_ids and num_episodes cannot be used together
    if data_config.episode_ids and data_config.num_episodes > 0:
        raise ValueError(
            "Cannot specify both 'episode_ids' and 'num_episodes'. "
            "Use 'episode_ids' to select specific episodes, or 'num_episodes' to limit the count, but not both."
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

    # Apply episode ID filtering AFTER prompt filtering so it operates on the correct subset
    if data_config.episode_ids:
        dataset = EpisodeFilteredDataset(dataset, data_config.episode_ids, lerobot_dataset=lerobot_ds)
    
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
