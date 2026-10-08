"""Trace a RoboCasa config from filtered demonstrations to the BC model batch."""

import dataclasses
import json

import jax
import numpy as np
import tyro

from openpi.training import config as configs
from openpi.training import data_loader


def main(config_name: str, batch_size: int = 2, *, skip_norm_stats: bool = False):
    config = dataclasses.replace(configs.get_config(config_name), batch_size=batch_size, num_workers=0)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    if not data_config.data_dirs:
        raise ValueError("This tracer requires a local RoboCasa config")
    first, last = dataset[0], dataset[len(dataset) - 1]
    transformed = data_loader.transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    reference = transformed[0]
    loader = data_loader.create_data_loader(config, skip_norm_stats=skip_norm_stats, num_batches=1, shuffle=False)
    observation, actions = next(iter(loader))
    np.testing.assert_array_equal(np.asarray(observation.tokenized_prompt[0]), reference["tokenized_prompt"])
    assert observation.tokenized_prompt.shape == (batch_size, config.model.max_token_len)
    assert actions.shape == (batch_size, config.model.action_horizon, config.model.action_dim)
    assert all(image.shape == (batch_size, 224, 224, 3) for image in observation.images.values())
    np.testing.assert_array_equal(last["action"], np.repeat(last["action"][:1], config.model.action_horizon, axis=0))
    report = {
        "config": config_name,
        "episodes": dataset.episodes,
        "frames": len(dataset),
        "first_prompt": first["prompt"],
        "last_prompt": last["prompt"],
        "norm_stats_loaded": data_config.norm_stats is not None,
        "norm_stats_path": str(config.assets_dirs / data_config.asset_id),
        "checkpoint_parent": str(config.checkpoint_base_dir) + "/" + config.name,
        "images": {key: list(value.shape) for key, value in observation.images.items()},
        "state_shape": list(observation.state.shape),
        "token_shape": list(observation.tokenized_prompt.shape),
        "action_shape": list(actions.shape),
        "first_tokens_match_sample": True,
        "last_action_chunk_stays_in_episode": True,
        "model": dataclasses.asdict(config.model),
        "devices": [str(device) for device in jax.devices()],
    }
    print(json.dumps(report, indent=2))
    dataset.close()


if __name__ == "__main__":
    tyro.cli(main)
