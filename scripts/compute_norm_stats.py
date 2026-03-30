"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import logging
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class RemoveImages(transforms.DataTransformFn):
    """Remove image-related keys that aren't needed for norm stats and may have inconsistent shapes."""
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if k not in ("image", "image_mask")}

def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)

def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None and not getattr(data_config, "data_dirs", None):
        raise ValueError("Data config must have a repo_id or data_dirs")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings and images since they are not needed for norm stats.
            # Images may also have inconsistent shapes across data sources (e.g. LeRobot 256x256 vs HDF5 128x128).
            RemoveStrings(),
            RemoveImages(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings and images since they are not needed for norm stats.
            # Images may also have inconsistent shapes across data sources (e.g. LeRobot 256x256 vs HDF5 128x128).
            RemoveStrings(),
            RemoveImages(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    init_logging()
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    if config.enforce_min_quantile_range:
        # Fix gripper action dim (last dim of actions) if its range has collapsed.
        # In LIBERO, gripper actions range from -1 (close) to 1 (open).  When training
        # on very few episodes the gripper may barely move, giving a near-zero range
        # that blows up quantile normalization.
        GRIPPER_Q01 = -1.0
        GRIPPER_Q99 = 1.0
        GRIPPER_RANGE_THRESHOLD = 0.5  # healthy range is ~2.0; flag anything far below
        if "actions" in norm_stats:
            ns = norm_stats["actions"]
            if ns.q01 is not None and ns.q99 is not None:
                gripper_dim = ns.q01.shape[-1] - 1  # last dim
                gripper_range = ns.q99[gripper_dim] - ns.q01[gripper_dim]
                if gripper_range < GRIPPER_RANGE_THRESHOLD:
                    logging.warning(
                        f"Gripper action (dim {gripper_dim}) has collapsed range: "
                        f"q01={ns.q01[gripper_dim]:.4f}, q99={ns.q99[gripper_dim]:.4f} "
                        f"(range={gripper_range:.4f}). Overriding to [{GRIPPER_Q01}, {GRIPPER_Q99}]."
                    )
                    ns.q01[gripper_dim] = GRIPPER_Q01
                    ns.q99[gripper_dim] = GRIPPER_Q99

    # Determine output path: use repo_id if available, else use asset_id or "robocasa"
    if data_config.repo_id is not None:
        output_path = config.assets_dirs / data_config.repo_id
    elif data_config.asset_id is not None:
        output_path = config.assets_dirs / data_config.asset_id
    else:
        output_path = config.assets_dirs / "robocasa"
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    # Also print per-dimension stats for visibility
    for key in keys:
        ns = norm_stats[key]
        print(f"\n=== {key} norm stats ===")
        ndims = len(ns.mean) if hasattr(ns.mean, '__len__') else 1
        for d in range(ndims):
            m = ns.mean[d] if hasattr(ns.mean, '__len__') else ns.mean
            s = ns.std[d] if hasattr(ns.std, '__len__') else ns.std
            q01_v = ns.q01[d] if ns.q01 is not None and hasattr(ns.q01, '__len__') else None
            q99_v = ns.q99[d] if ns.q99 is not None and hasattr(ns.q99, '__len__') else None
            # Stop printing once we hit padding (mean=0, std=1)
            if abs(m) < 1e-10 and abs(s - 1.0) < 1e-10:
                print(f"  dim {d}+: padding (zeros)")
                break
            line = f"  dim {d}: mean={m:.6f}, std={s:.6f}"
            if q01_v is not None:
                line += f", q01={q01_v:.6f}, q99={q99_v:.6f}, range={q99_v - q01_v:.6f}"
            print(line)


if __name__ == "__main__":
    tyro.cli(main)
