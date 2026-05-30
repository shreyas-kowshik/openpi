"""Dump dataloader episodes to .hdf5 files in dsrl_pi0 expert format.

Iterates the raw dataset for a given training config, groups samples by
episode, and writes one .hdf5 file per episode with the structure:

    episode_XXXXXXX/
        actions:      (T, action_dim)  float32
        image:        (T, H, W, 3)     uint8
        rewards:      (T,)             float32
        state:        (T, state_dim)   float32
        wrist_image:  (T, H, W, 3)     uint8
    metadata/
        attrs: {action_dim, image_size, num_episodes, query_frequency,
                state_dim, task_description}
    episode attrs: {is_success, num_steps, source_ep_idx}

Usage:
    uv run scripts/dump_hdf5_demos.py \
        --config-name pi05_robocasa_single_task_lora_load_dishwasher_action_dim12_discrete_state \
        --output-dir /tmp/robocasa_hdf5_dumps
"""

import logging
import os
import pathlib
from collections import OrderedDict

import cv2
import h5py
import numpy as np
import tyro

if "OPENPI_DATA_HOME" not in os.environ:
    _fallback = pathlib.Path.home() / ".cache" / "openpi"
    _fallback.mkdir(parents=True, exist_ok=True)
    os.environ["OPENPI_DATA_HOME"] = str(_fallback)

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def _get_episode_groups(dataset) -> list[tuple[str, list[int]]]:
    """Pre-compute episode groups as (label, [sample_indices])."""
    if isinstance(dataset, _data_loader.ConcatDataset):
        groups = []
        offset = 0
        for sub_ds in dataset._datasets:
            sub_groups = _get_episode_groups(sub_ds)
            for label, indices in sub_groups:
                groups.append((label, [idx + offset for idx in indices]))
            offset += len(sub_ds)
        return groups

    if isinstance(dataset, _data_loader.TransformedDataset):
        return _get_episode_groups(dataset._dataset)

    if isinstance(dataset, _data_loader.EpisodeFilteredDataset):
        inner_groups = _get_episode_groups(dataset._dataset)
        valid_set = set(dataset._valid_indices)
        inner_to_filtered = {}
        for filtered_idx, inner_idx in enumerate(dataset._valid_indices):
            inner_to_filtered[inner_idx] = filtered_idx
        groups = []
        for label, inner_indices in inner_groups:
            mapped = [inner_to_filtered[i] for i in inner_indices if i in valid_set]
            if mapped:
                groups.append((label, mapped))
        return groups

    if isinstance(dataset, _data_loader.FilteredDataset):
        inner_groups = _get_episode_groups(dataset._dataset)
        valid_set = set(dataset._valid_indices)
        inner_to_filtered = {}
        for filtered_idx, inner_idx in enumerate(dataset._valid_indices):
            inner_to_filtered[inner_idx] = filtered_idx
        groups = []
        for label, inner_indices in inner_groups:
            mapped = [inner_to_filtered[i] for i in inner_indices if i in valid_set]
            if mapped:
                groups.append((label, mapped))
        return groups

    if isinstance(dataset, _data_loader.LiberoProHDF5Dataset):
        demo_groups: dict[str, list[int]] = OrderedDict()
        for idx, (demo_key, _t) in enumerate(dataset._samples):
            demo_groups.setdefault(demo_key, []).append(idx)
        return [(f"hdf5_{dk}", indices) for dk, indices in demo_groups.items()]

    if isinstance(dataset, _data_loader.Libero10HDF5Dataset):
        demo_groups_l10: dict[str, list[int]] = OrderedDict()
        for idx, (file_idx, demo_key, _t) in enumerate(dataset._samples):
            label = f"file{file_idx}_{demo_key}"
            demo_groups_l10.setdefault(label, []).append(idx)
        return [(label, indices) for label, indices in demo_groups_l10.items()]

    # Groot / LeRobot datasets
    if hasattr(dataset, "hf_dataset") and "episode_index" in dataset.hf_dataset.column_names:
        ep_groups: dict[int, list[int]] = OrderedDict()
        episode_indices = dataset.hf_dataset["episode_index"]
        for idx, ep_idx in enumerate(episode_indices):
            ep_groups.setdefault(int(ep_idx), []).append(idx)
        return [(f"lerobot_ep{ep_idx}", indices) for ep_idx, indices in ep_groups.items()]

    logging.warning(f"Cannot detect episodes for {type(dataset).__name__}; treating as single episode.")
    return [("episode_0", list(range(len(dataset))))]


def _to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    """Convert image to uint8 HWC."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).clip(0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    """Resize image to (size, size) if needed."""
    if image.shape[0] != size or image.shape[1] != size:
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return image


def main(
    config_name: str,
    output_dir: str = "/tmp/robocasa_hdf5_dumps",
    image_size: int = 224,
    num_episodes: int | None = None,
):
    """Dump training episodes as .hdf5 files in dsrl_pi0 expert format.

    Args:
        config_name: Training config name.
        output_dir: Root output directory for .hdf5 files.
        image_size: Resize images to this square size (default 224).
        num_episodes: Max episodes to dump. None = all.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    action_horizon = config.model.action_horizon

    logging.info(f"Config: {config_name}, action_horizon={action_horizon}")

    raw_dataset = _data_loader.create_torch_dataset(data_config, action_horizon, config.model)
    logging.info(f"Dataset size: {len(raw_dataset)} samples")

    episode_groups = _get_episode_groups(raw_dataset)
    if num_episodes is not None:
        episode_groups = episode_groups[:num_episodes]

    logging.info(f"Found {len(episode_groups)} episode(s) to dump")

    out_dir = pathlib.Path(output_dir) / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep_num, (label, indices) in enumerate(episode_groups):
        # Collect all timesteps for this episode
        images = []
        wrist_images = []
        states = []
        actions_list = []
        prompt = ""

        for idx in indices:
            sample = raw_dataset[idx]
            img = _to_uint8_hwc(sample["observation/image"])
            img = _resize(img, image_size)
            images.append(img)

            wrist = _to_uint8_hwc(sample["observation/wrist_image"])
            wrist = _resize(wrist, image_size)
            wrist_images.append(wrist)

            states.append(np.asarray(sample["observation/state"], dtype=np.float32))

            # Take only the first action from the chunked horizon
            act = np.asarray(sample["actions"], dtype=np.float32)
            if act.ndim == 2:
                act = act[0]
            actions_list.append(act)

            if not prompt:
                prompt = str(sample.get("prompt", ""))

        T = len(images)
        action_dim = actions_list[0].shape[-1]
        state_dim = states[0].shape[-1]

        # Extract source episode index from label if available
        source_ep_idx = -1
        if label.startswith("lerobot_ep"):
            try:
                source_ep_idx = int(label.replace("lerobot_ep", ""))
            except ValueError:
                pass

        # Stack into arrays
        images_arr = np.stack(images)         # (T, H, W, 3)
        wrist_arr = np.stack(wrist_images)    # (T, H, W, 3)
        states_arr = np.stack(states)         # (T, state_dim)
        actions_arr = np.stack(actions_list)  # (T, action_dim)
        rewards_arr = np.ones(T, dtype=np.float32)  # expert demos assumed success

        # Write HDF5
        safe_label = label.replace("/", "_")
        out_path = out_dir / f"{safe_label}.hdf5"

        with h5py.File(out_path, "w") as f:
            ep_key = "episode_0000000"
            ep_grp = f.create_group(ep_key)
            ep_grp.attrs["is_success"] = 1
            ep_grp.attrs["num_steps"] = T
            ep_grp.attrs["source_ep_idx"] = source_ep_idx

            ep_grp.create_dataset("actions", data=actions_arr)
            ep_grp.create_dataset("image", data=images_arr)
            ep_grp.create_dataset("rewards", data=rewards_arr)
            ep_grp.create_dataset("state", data=states_arr)
            ep_grp.create_dataset("wrist_image", data=wrist_arr)

            md_grp = f.create_group("metadata")
            md_grp.attrs["action_dim"] = action_dim
            md_grp.attrs["image_size"] = image_size
            md_grp.attrs["num_episodes"] = 1
            md_grp.attrs["query_frequency"] = action_horizon
            md_grp.attrs["state_dim"] = state_dim
            md_grp.attrs["task_description"] = prompt

        logging.info(
            f"  [{ep_num+1}/{len(episode_groups)}] {out_path.name}: "
            f"{T} steps, actions={actions_arr.shape}, state={states_arr.shape}, prompt='{prompt[:60]}'"
        )

    logging.info(f"Done. Wrote {len(episode_groups)} file(s) to {out_dir}")


if __name__ == "__main__":
    tyro.cli(main)
