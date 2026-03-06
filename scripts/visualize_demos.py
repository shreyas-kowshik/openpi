"""Visualize demo episodes from a training config as .mp4 videos.

Iterates through the raw dataset (before repack/model transforms) and dumps
each episode as an mp4 with side-by-side base + wrist camera views.

Usage:
    uv run scripts/visualize_demos.py \
        --config-name pi05_libero_lora_vision_fullft_action_placebookincaddy_task_ep1_bs32_v2_icml_v2 \
        --output-dir ./data_dumps/viz \
        --num-episodes-to-dump 3
"""

import logging
import pathlib

import cv2
import imageio
import numpy as np
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def _to_uint8_hwc(image) -> np.ndarray:
    """Convert an image (torch CHW float or numpy HWC uint8) to uint8 HWC."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).clip(0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def _get_image(sample: dict) -> np.ndarray:
    """Extract the base camera image from a raw sample."""
    for key in ("image", "observation/image"):
        if key in sample:
            return _to_uint8_hwc(sample[key])
    raise KeyError(f"No image key found in sample. Keys: {list(sample.keys())}")


def _get_wrist_image(sample: dict) -> np.ndarray | None:
    """Extract the wrist camera image from a raw sample."""
    for key in ("wrist_image", "observation/wrist_image"):
        if key in sample:
            return _to_uint8_hwc(sample[key])
    return None


def _resize_to_match(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize image to target dimensions."""
    if img.shape[0] != target_h or img.shape[1] != target_w:
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return img


def _write_episode_video(frames: list[np.ndarray], path: pathlib.Path, fps: int = 10):
    """Write a list of RGB uint8 frames to an H.264 mp4 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    logging.info(f"  Wrote {len(frames)} frames -> {path}")


def _get_episode_groups(dataset) -> list[tuple[str, list[int]]]:
    """Pre-compute episode groups as (label, [sample_indices]) from the dataset structure.

    Handles LeRobot datasets (via episode_index), HDF5 datasets (via _samples),
    ConcatDataset (recurses into sub-datasets), and generic wrappers like
    FilteredDataset / TransformedDataset.
    """
    # -- ConcatDataset: recurse into sub-datasets --
    if isinstance(dataset, _data_loader.ConcatDataset):
        groups = []
        offset = 0
        for sub_ds in dataset._datasets:
            sub_groups = _get_episode_groups(sub_ds)
            for label, indices in sub_groups:
                groups.append((label, [idx + offset for idx in indices]))
            offset += len(sub_ds)
        return groups

    # -- Unwrap TransformedDataset --
    if isinstance(dataset, _data_loader.TransformedDataset):
        return _get_episode_groups(dataset._dataset)

    # -- FilteredDataset: map back to underlying indices --
    if isinstance(dataset, _data_loader.FilteredDataset):
        inner_groups = _get_episode_groups(dataset._dataset)
        valid_set = set(dataset._valid_indices)
        # Build reverse map: inner_index -> filtered_index
        inner_to_filtered = {}
        for filtered_idx, inner_idx in enumerate(dataset._valid_indices):
            inner_to_filtered[inner_idx] = filtered_idx
        groups = []
        for label, inner_indices in inner_groups:
            mapped = [inner_to_filtered[i] for i in inner_indices if i in valid_set]
            if mapped:
                groups.append((label, mapped))
        return groups

    # -- LiberoProHDF5Dataset: group by demo_key --
    if isinstance(dataset, _data_loader.LiberoProHDF5Dataset):
        from collections import OrderedDict
        demo_groups: dict[str, list[int]] = OrderedDict()
        for idx, (demo_key, _t) in enumerate(dataset._samples):
            demo_groups.setdefault(demo_key, []).append(idx)
        return [(f"hdf5_{dk}", indices) for dk, indices in demo_groups.items()]

    # -- LeRobot dataset: group by episode_index --
    if hasattr(dataset, 'hf_dataset') and 'episode_index' in dataset.hf_dataset.column_names:
        from collections import OrderedDict
        ep_groups: dict[int, list[int]] = OrderedDict()
        episode_indices = dataset.hf_dataset['episode_index']
        for idx, ep_idx in enumerate(episode_indices):
            ep_groups.setdefault(int(ep_idx), []).append(idx)
        return [(f"lerobot_ep{ep_idx}", indices) for ep_idx, indices in ep_groups.items()]

    # -- Fallback: treat entire dataset as one episode --
    logging.warning(f"Cannot detect episodes for {type(dataset).__name__}; treating as single episode.")
    return [("episode_0", list(range(len(dataset))))]


def main(
    config_name: str,
    output_dir: str = "./data_dumps/viz",
    num_episodes_to_dump: int = 5,
    fps: int = 10,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # num_episodes_to_dump controls how many episodes to visualize.
    # data_config.num_episodes controls dataset filtering (not visualization), so don't cap by it.
    max_episodes = num_episodes_to_dump

    logging.info(f"Config: {config_name}")
    logging.info(f"Will dump up to {max_episodes} episodes to {output_dir}")

    # Create the raw dataset (same as training, before repack/model transforms).
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    logging.info(f"Dataset size: {len(dataset)} samples")

    out_dir = pathlib.Path(output_dir) / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute episode boundaries from dataset structure.
    episode_groups = _get_episode_groups(dataset)
    logging.info(f"Found {len(episode_groups)} episodes in dataset, dumping {max_episodes}")

    for ep_num, (label, indices) in enumerate(episode_groups[:max_episodes]):
        sample_0 = dataset[indices[0]]
        prompt = sample_0.get("prompt", sample_0.get("task", ""))
        logging.info(f"Episode {ep_num}/{max_episodes}: {label} ({len(indices)} frames), prompt='{prompt}'")

        frames = []
        for idx in indices:
            sample = dataset[idx]
            base_img = _get_image(sample)
            wrist_img = _get_wrist_image(sample)

            if wrist_img is not None:
                wrist_img = _resize_to_match(wrist_img, base_img.shape[0], base_img.shape[1])
                frame = np.concatenate([base_img, wrist_img], axis=1)
            else:
                frame = base_img

            frames.append(frame)

        vid_path = out_dir / f"{label}.mp4"
        _write_episode_video(frames, vid_path, fps=fps)

    logging.info(f"Done. Wrote {min(max_episodes, len(episode_groups))} episode videos to {out_dir}")


if __name__ == "__main__":
    tyro.cli(main)
