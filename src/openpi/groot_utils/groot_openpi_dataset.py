"""Groot-LeRobot dataset implementation for openpi training."""

import glob
import json
import os
from collections.abc import Iterator, Sequence
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import cv2  # Add OpenCV for video frame extraction

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms
import openpi.shared.normalize as _normalize

import pathlib
from pathlib import Path

T_co = TypeVar("T_co", covariant=True)

from robocasa.utils.groot_utils.groot_dataset import LeRobotSingleDataset, LeRobotMixtureDataset, LE_ROBOT_MODALITY_FILENAME, ModalityConfig, LE_ROBOT_EPISODE_FILENAME
from robocasa.utils.groot_utils.embodiment_tags import EmbodimentTag


def get_scene_filtered_demos(
    dataset_path: pathlib.Path,
    layout_and_style_ids: list[tuple[int, int]],
    num_demos: int | None = None,
    obj_category: str | None = None,
    fixture_refs: dict[str, str] | None = None,
) -> list[int]:
    """Filter episodes by scene parameters, then take first N.

    Args:
        dataset_path: Path to the LeRobot dataset.
        layout_and_style_ids: List of (layout_id, style_id) tuples to allow.
        num_demos: Max number of demos to keep after filtering.
        obj_category: If provided, only keep episodes where the primary object
            (object_cfgs[0]) matches this category (e.g. "honey_bottle", "banana").
        fixture_refs: If provided, only keep episodes whose fixture_refs match
            exactly (e.g. {"cab": "cab_2_main_group", "counter": "counter_main_main_group"}).
            This ensures all demos use the exact same counter/cabinet location.
    """
    import logging as _logging

    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    with open(episodes_path) as f:
        episodes = [json.loads(line) for line in f]

    allowed = set(map(tuple, layout_and_style_ids))
    filtered = []
    skipped_by_category = 0
    skipped_by_fixture_refs = 0
    for ep in episodes:
        idx = ep["episode_index"]
        meta_path = dataset_path / "extras" / f"episode_{idx:06d}" / "ep_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        if (meta["layout_id"], meta["style_id"]) not in allowed:
            continue
        if fixture_refs is not None:
            ep_fixture_refs = meta.get("fixture_refs", {})
            if ep_fixture_refs != fixture_refs:
                skipped_by_fixture_refs += 1
                continue
        if obj_category is not None:
            ep_cat = meta.get("object_cfgs", [{}])[0].get("info", {}).get("cat")
            if ep_cat != obj_category:
                skipped_by_category += 1
                continue
        filtered.append(idx)

    filtered.sort()

    filter_desc_parts = [f"layout_and_style_ids={layout_and_style_ids}"]
    if fixture_refs is not None:
        filter_desc_parts.append(f"fixture_refs={fixture_refs}")
        _logging.info(
            f"Fixture refs filter: kept {len(filtered)} episodes, "
            f"skipped {skipped_by_fixture_refs}"
        )
    if obj_category is not None:
        filter_desc_parts.append(f"obj_category={obj_category}")
        _logging.info(
            f"Object category filter '{obj_category}': kept {len(filtered)} episodes, "
            f"skipped {skipped_by_category}"
        )

    if not filtered:
        raise ValueError(
            f"No episodes match {', '.join(filter_desc_parts)} "
            f"in dataset at {dataset_path}. Check that the requested filters "
            f"exist in the per-episode ep_meta.json files."
        )

    if num_demos is not None and num_demos < len(filtered):
        filtered = filtered[:num_demos]

    _logging.info(
        f"Scene filtering: {len(filtered)} episodes after filters "
        f"({', '.join(filter_desc_parts)})"
    )

    return filtered


def get_ep_meta_for_episode(dataset_path: pathlib.Path, episode_idx: int) -> dict:
    """Load the ep_meta.json for a specific episode index."""
    meta_path = dataset_path / "extras" / f"episode_{episode_idx:06d}" / "ep_meta.json"
    with open(meta_path) as f:
        return json.load(f)


def _extract_obj_categories(ep_meta: dict) -> list[str]:
    """Extract the ordered list of object categories from an ep_meta dict."""
    return [
        cfg.get("info", {}).get("cat", "")
        for cfg in ep_meta.get("object_cfgs", [])
    ]


def get_ep_meta_matched_demos(
    dataset_path: pathlib.Path,
    match_episode_id: int,
) -> list[int]:
    """Filter episodes to those matching the full ep_meta state of a reference episode.

    Matches on: layout_id, style_id, fixture_refs, and ALL object categories
    (target + distractors). This ensures the filtered demos use the exact same
    environment setup (kitchen layout, cabinet/counter, and all objects).

    Returns:
        List of matching episode indices (sorted). Includes the reference episode itself.
    """
    import logging as _logging

    ref_meta = get_ep_meta_for_episode(dataset_path, match_episode_id)
    ref_layout = ref_meta["layout_id"]
    ref_style = ref_meta["style_id"]
    ref_fixture_refs = ref_meta.get("fixture_refs", {})
    ref_obj_cats = _extract_obj_categories(ref_meta)

    _logging.info(
        f"Matching full ep_meta of episode {match_episode_id}: "
        f"layout={ref_layout}, style={ref_style}, "
        f"fixture_refs={ref_fixture_refs}, "
        f"objects={ref_obj_cats}"
    )

    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    with open(episodes_path) as f:
        episodes = [json.loads(line) for line in f]

    filtered = []
    for ep in episodes:
        idx = ep["episode_index"]
        meta = get_ep_meta_for_episode(dataset_path, idx)
        if meta["layout_id"] != ref_layout:
            continue
        if meta["style_id"] != ref_style:
            continue
        if meta.get("fixture_refs", {}) != ref_fixture_refs:
            continue
        if _extract_obj_categories(meta) != ref_obj_cats:
            continue
        filtered.append(idx)

    filtered.sort()

    _logging.info(
        f"Full ep_meta matching: {len(filtered)} episodes match episode {match_episode_id}"
    )

    if not filtered:
        raise ValueError(
            f"No episodes match the full ep_meta of episode {match_episode_id} "
            f"in dataset at {dataset_path}. This should not happen since the "
            f"reference episode itself should always match."
        )

    return filtered


def get_reference_ep_meta(
    dataset_path: pathlib.Path,
    layout_and_style_ids: list[tuple[int, int]] | None = None,
    num_demos: int | None = None,
    obj_category: str | None = None,
    fixture_refs: dict[str, str] | None = None,
    match_episode_id: int | None = None,
) -> dict:
    """Get the ep_meta of the first episode matching the given filters.

    This is used to get a "reference" ep_meta that can be passed to
    env.set_ep_meta() during evaluation to reproduce the exact same
    environment setup as one of the training demos.
    """
    if match_episode_id is not None:
        return get_ep_meta_for_episode(dataset_path, match_episode_id)
    if layout_and_style_ids is not None:
        filtered = get_scene_filtered_demos(
            dataset_path, layout_and_style_ids, num_demos, obj_category, fixture_refs
        )
        first_ep = filtered[0]
    else:
        first_ep = 0
    return get_ep_meta_for_episode(dataset_path, first_ep)


def list_available_fixture_refs(
    dataset_path: pathlib.Path,
    layout_and_style_ids: list[tuple[int, int]] | None = None,
) -> dict[str, int]:
    """List all fixture_refs groups in the dataset with episode counts.

    Returns a dict mapping fixture_refs (as JSON string) to episode count,
    sorted by count descending.
    """
    from collections import Counter

    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    with open(episodes_path) as f:
        episodes = [json.loads(line) for line in f]

    allowed = set(map(tuple, layout_and_style_ids)) if layout_and_style_ids else None
    fxref_counts: Counter = Counter()
    for ep in episodes:
        idx = ep["episode_index"]
        meta_path = dataset_path / "extras" / f"episode_{idx:06d}" / "ep_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        if allowed is not None and (meta["layout_id"], meta["style_id"]) not in allowed:
            continue
        fxrefs = meta.get("fixture_refs", {})
        fxref_counts[json.dumps(fxrefs, sort_keys=True)] += 1

    return {k: v for k, v in fxref_counts.most_common()}


def list_available_obj_categories(
    dataset_path: pathlib.Path,
    layout_and_style_ids: list[tuple[int, int]] | None = None,
) -> dict[str, int]:
    """List all object categories available in the dataset (optionally filtered by scene).

    Returns a dict mapping category name to episode count.
    """
    from collections import Counter

    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    with open(episodes_path) as f:
        episodes = [json.loads(line) for line in f]

    allowed = set(map(tuple, layout_and_style_ids)) if layout_and_style_ids else None
    cat_counts: Counter = Counter()
    for ep in episodes:
        idx = ep["episode_index"]
        meta_path = dataset_path / "extras" / f"episode_{idx:06d}" / "ep_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        if allowed is not None and (meta["layout_id"], meta["style_id"]) not in allowed:
            continue
        cat = meta.get("object_cfgs", [{}])[0].get("info", {}).get("cat", "unknown")
        cat_counts[cat] += 1

    return dict(cat_counts.most_common())


def get_modality_keys(dataset_path: pathlib.Path) -> dict[str, list[str]]:
    """
    Get the modality keys from the dataset path.
    Returns a dictionary with modality types as keys and their corresponding modality keys as values,
    maintaining the order: video, state, action, annotation
    """
    modality_path = dataset_path / LE_ROBOT_MODALITY_FILENAME
    with open(modality_path, "r") as f:
        modality_meta = json.load(f)

    # Initialize dictionary with ordered keys
    modality_dict = {}
    for key in modality_meta.keys():
        modality_dict[key] = []
        for modality in modality_meta[key]:
            modality_dict[key].append(f"{key}.{modality}")
    return modality_dict


class GrootOpenpiSingleDataset(LeRobotSingleDataset):
    def __init__(
        self,
        dataset_meta: dict,
        action_horizon: int,
    ):
        # this part copied from Abhi's DP codebasee
        dataset_path = dataset_meta["path"]
        dataset_path = pathlib.Path(dataset_path)
        filter_key = dataset_meta["filter_key"]
        delta_indices = list(range(0, action_horizon))
        delta_indices_obs = [0]
        modality_keys_dict = get_modality_keys(dataset_path)
        video_modality_keys = modality_keys_dict["video"]
        language_modality_keys = modality_keys_dict["annotation"]
        state_modality_keys = modality_keys_dict["state"]
        action_modality_keys = modality_keys_dict["action"]
        state_modality_keys = [key for key in state_modality_keys if key != "state.dummy_tensor"]
        modality_configs = {
            "video": ModalityConfig(
                delta_indices=delta_indices_obs,
                modality_keys=video_modality_keys,  # we will include all video modalities
            ),
            "state": ModalityConfig(
                delta_indices=delta_indices_obs,
                modality_keys=state_modality_keys,
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=action_modality_keys,
            ),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=language_modality_keys,
            ),
        }

        # Scene filtering: restrict to specific episodes
        match_episode_id = dataset_meta.get("match_episode_id")
        layout_and_style_ids = dataset_meta.get("layout_and_style_ids")
        num_demos = dataset_meta.get("num_demos")
        obj_category = dataset_meta.get("obj_category")
        fixture_refs = dataset_meta.get("fixture_refs")
        subset_demos = None
        if match_episode_id is not None:
            subset_demos = get_ep_meta_matched_demos(dataset_path, match_episode_id)
        elif layout_and_style_ids is not None:
            subset_demos = get_scene_filtered_demos(
                dataset_path, layout_and_style_ids, num_demos, obj_category, fixture_refs
            )

        super().__init__(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            video_backend="opencv",
            video_backend_kwargs=None,
            transforms=None,
            filter_key=filter_key if subset_demos is None else None,
            subset_demos=subset_demos,
        )

    def __getitem__(self, index: SupportsIndex) -> dict:
        item = super().__getitem__(index)

        state = np.concatenate([
            item["state.end_effector_position_relative"],
            item["state.end_effector_rotation_relative"],
            item["state.base_position"],
            item["state.base_rotation"],
            item["state.gripper_qpos"],
        ], axis=1)
        actions = np.concatenate([
            item["action.end_effector_position"],
            item["action.end_effector_rotation"],
            item["action.gripper_close"],
            item["action.base_motion"],
            item["action.control_mode"],
        ], axis=1)

        new_item = {
            "observation/image": item["video.robot0_agentview_left"][0],
            "observation/wrist_image": item["video.robot0_eye_in_hand"][0],
            "observation/state": state[0],
            "actions": actions,
            "prompt": item["annotation.human.task_description"][0],
        }
        return new_item


class GrootOpenpiMultiDataset(LeRobotMixtureDataset):
    def __init__(
            self,
            dataset_meta_list,
            action_horizon: int,
            dataset_weights=None,
            dataset_weights_alpha=0.4,
            metadata_config: dict = {
                "percentile_mixing_method": "weighted_average",
            },
        ):
        datasets = []
        for ds_meta in dataset_meta_list:
            ds_path = ds_meta["path"]
            ds_path = pathlib.Path(ds_path)
            filter_key = ds_meta["filter_key"]
            delta_indices = list(range(0, action_horizon))
            delta_indices_obs = [0]
            modality_keys_dict = get_modality_keys(ds_path)
            video_modality_keys = modality_keys_dict["video"]
            language_modality_keys = modality_keys_dict["annotation"]
            state_modality_keys = modality_keys_dict["state"]
            action_modality_keys = modality_keys_dict["action"]
            state_modality_keys = [key for key in state_modality_keys if key != "state.dummy_tensor"]
            modality_configs = {
                "video": ModalityConfig(
                    delta_indices=delta_indices_obs,
                    modality_keys=video_modality_keys,
                ),
                "state": ModalityConfig(
                    delta_indices=delta_indices_obs,
                    modality_keys=state_modality_keys,
                ),
                "action": ModalityConfig(
                    delta_indices=delta_indices,
                    modality_keys=action_modality_keys,
                ),
                "language": ModalityConfig(
                    delta_indices=[0],
                    modality_keys=language_modality_keys,
                ),
            }

            # Scene filtering
            match_episode_id = ds_meta.get("match_episode_id")
            layout_and_style_ids = ds_meta.get("layout_and_style_ids")
            num_demos_val = ds_meta.get("num_demos")
            obj_category = ds_meta.get("obj_category")
            fixture_refs = ds_meta.get("fixture_refs")
            subset_demos = None
            if match_episode_id is not None:
                subset_demos = get_ep_meta_matched_demos(ds_path, match_episode_id)
            elif layout_and_style_ids is not None:
                subset_demos = get_scene_filtered_demos(
                    ds_path, layout_and_style_ids, num_demos_val, obj_category, fixture_refs
                )

            this_dataset = LeRobotSingleDataset(
                dataset_path=ds_path,
                modality_configs=modality_configs,
                embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                video_backend="opencv",
                video_backend_kwargs=None,
                transforms=None,
                filter_key=filter_key if subset_demos is None else None,
                subset_demos=subset_demos,
            )
            datasets.append(this_dataset)

        if not dataset_weights:
            ds_weights = np.array([np.power(len(dataset), dataset_weights_alpha) for dataset in datasets])
            # the groot dataloader requires that at least one dataset has weight 1.0
            ds_weights = ds_weights / ds_weights[0]
        dataset_mixture = list(zip(datasets, ds_weights))
        # set balance_dataset_weights to False, since we are calculating weights ourselves
        super().__init__(
            data_mixture=dataset_mixture,
            mode="train",
            balance_dataset_weights=False,
            balance_trajectory_weights=False,
            metadata_config=metadata_config,
        )

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the dataset (ignores index, samples randomly)."""
        # Sample dataset
        dataset_index = np.random.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        # Sample trajectory
        trajectory_index = np.random.choice(
            len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index]
        )
        trajectory_id = dataset.trajectory_ids[trajectory_index]

        # Sample step
        base_index = np.random.choice(dataset.trajectory_lengths[trajectory_index])
        return dataset, trajectory_id, base_index

    def __getitem__(self, index: SupportsIndex) -> dict:
        item = super().__getitem__(index)

        state = np.concatenate([
            item["state.end_effector_position_relative"],
            item["state.end_effector_rotation_relative"],
            item["state.base_position"],
            item["state.base_rotation"],
            item["state.gripper_qpos"],
        ], axis=1)
        actions = np.concatenate([
            item["action.end_effector_position"],
            item["action.end_effector_rotation"],
            item["action.gripper_close"],
            item["action.base_motion"],
            item["action.control_mode"],
        ], axis=1)

        new_item = {
            "observation/image": item["video.robot0_agentview_left"][0],
            "observation/wrist_image": item["video.robot0_eye_in_hand"][0],
            "observation/state": state[0],
            "actions": actions,
            "prompt": item["annotation.human.task_description"][0],
        }
        return new_item


def _load_norm_stats_from_groot_dataset(ds_meta: dict) -> dict[str, _transforms.NormStats] | None:
    def pad_zeros(input, targ_len):
        return np.concatenate([input, np.zeros(targ_len - len(input))])

    def pad_ones(input, targ_len):
        return np.concatenate([input, np.ones(targ_len - len(input))])

    dataset_path = ds_meta["path"]
    dataset_path = pathlib.Path(dataset_path)
    path = dataset_path / "meta" / "stats.json"
    data = json.loads(path.read_text())

    """
    the groot state ordering
    "state.base_position" 0, 1, 2
    "state.base_rotation" 3, 4, 5, 6
    "state.end_effector_position_relative" 7, 8, 9
    "state.end_effector_rotation_relative" 10, 11, 12, 13
    "state.gripper_qpos" 14, 15

    the desired state ordering
    "state.end_effector_position_relative" 7, 8, 9
    "state.end_effector_rotation_relative" 10, 11, 12, 13
    "state.base_position" 0, 1, 2
    "state.base_rotation" 3, 4, 5, 6
    "state.gripper_qpos" 14, 15
    """
    raw_states_stats = data["observation.state"]
    raw_states_mean = np.array(raw_states_stats["mean"])
    raw_states_std = np.array(raw_states_stats["std"])
    raw_states_q01 = np.array(raw_states_stats["q01"])
    raw_states_q99 = np.array(raw_states_stats["q99"])

    states_indices = [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 14, 15]
    states_mean = raw_states_mean[states_indices]
    states_std = raw_states_std[states_indices]
    states_q01 = raw_states_q01[states_indices]
    states_q99 = raw_states_q99[states_indices]

    states_stats = _normalize.NormStats(
        mean=pad_zeros(states_mean, targ_len=32),
        std=pad_ones(states_std, targ_len=32),
        q01=pad_zeros(states_q01, targ_len=32),
        q99=pad_ones(states_q99, targ_len=32),
    )

    """
    the groot action ordering
    "action.base_motion" 0, 1, 2, 3
    "action.control_mode" 4
    "action.end_effector_position" 5, 6, 7
    "action.end_effector_rotation" 8, 9, 10
    "action.gripper_close" 11

    the desired action ordering
    "action.end_effector_position" 5, 6, 7
    "action.end_effector_rotation" 8, 9, 10
    "action.gripper_close" 11
    "action.base_motion" 0, 1, 2, 3
    "action.control_mode" 4
    """
    raw_actions_stats = data["action"]
    raw_actions_mean = np.array(raw_actions_stats["mean"])
    raw_actions_std = np.array(raw_actions_stats["std"])
    raw_actions_q01 = np.array(raw_actions_stats["q01"])
    raw_actions_q99 = np.array(raw_actions_stats["q99"])

    actions_indices = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]
    actions_mean = raw_actions_mean[actions_indices]
    actions_std = raw_actions_std[actions_indices]
    actions_q01 = raw_actions_q01[actions_indices]
    actions_q99 = raw_actions_q99[actions_indices]

    actions_stats = _normalize.NormStats(
        mean=pad_zeros(actions_mean, targ_len=32),
        std=pad_ones(actions_std, targ_len=32),
        q01=pad_zeros(actions_q01, targ_len=32),
        q99=pad_ones(actions_q99, targ_len=32),
    )

    return {
        "state": states_stats,
        "actions": actions_stats,
    }

def compute_overall_statistics(
    per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
    dataset_sampling_weights: list[float] | np.ndarray,
) -> dict[str, dict[str, list[float]]]:
    """
    Computes overall statistics from per-task statistics using dataset sample weights.
    """
    dataset_sampling_weights = np.array(dataset_sampling_weights)
    normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

    overall_stats: dict[str, dict[str, list[float]]] = {}
    modality_keys = per_task_stats[0].keys()

    for modality in modality_keys:
        num_dims = len(per_task_stats[0][modality].mean)
        weighted_means = np.zeros(num_dims)
        weighted_squares = np.zeros(num_dims)
        weighted_q01 = np.zeros(num_dims)
        weighted_q99 = np.zeros(num_dims)
        has_quantiles = per_task_stats[0][modality].q01 is not None

        for task_idx, task_stats in enumerate(per_task_stats):
            w_i = normalized_weights[task_idx]
            stats = task_stats[modality]
            means = np.array(stats.mean)
            stds = np.array(stats.std)
            weighted_means += w_i * means
            weighted_squares += w_i * (stds**2 + means**2)
            if has_quantiles:
                weighted_q01 += w_i * np.array(stats.q01)
                weighted_q99 += w_i * np.array(stats.q99)

        overall_mean = weighted_means.tolist()
        overall_variance = weighted_squares - weighted_means**2
        overall_std = np.sqrt(overall_variance).tolist()

        overall_stats[modality] = _normalize.NormStats(
            mean=overall_mean,
            std=overall_std,
            q01=weighted_q01.tolist() if has_quantiles else None,
            q99=weighted_q99.tolist() if has_quantiles else None,
        )

    return overall_stats


def _load_norm_stats_from_groot_mixture_dataset(dataset_meta_list) -> dict[str, _transforms.NormStats] | None:
    per_dataset_norm_stats = []
    for ds_meta in dataset_meta_list:
        per_dataset_norm_stats.append(_load_norm_stats_from_groot_dataset(ds_meta))

    return compute_overall_statistics(
        per_dataset_norm_stats,
        dataset_sampling_weights=np.ones(len(dataset_meta_list)),
    )


def _stats_cache_path(ds_meta: dict) -> pathlib.Path:
    """Deterministic cache path based on dataset path and filter params."""
    import hashlib

    dataset_path = pathlib.Path(ds_meta["path"])
    match_episode_id = ds_meta.get("match_episode_id")
    layout_and_style_ids = ds_meta.get("layout_and_style_ids")
    num_demos = ds_meta.get("num_demos")
    obj_category = ds_meta.get("obj_category")
    fixture_refs = ds_meta.get("fixture_refs")
    parts = []
    if match_episode_id is not None:
        parts.append(f"ep{match_episode_id}")
    if layout_and_style_ids is not None:
        parts.append("ls_" + "_".join(f"{l}{s}" for l, s in sorted(layout_and_style_ids)))
    if fixture_refs is not None:
        # Use a short hash of the fixture_refs dict to keep filenames reasonable
        fxr_str = json.dumps(fixture_refs, sort_keys=True)
        fxr_hash = hashlib.md5(fxr_str.encode()).hexdigest()[:8]
        parts.append(f"fxr_{fxr_hash}")
    if num_demos is not None:
        parts.append(f"n{num_demos}")
    if obj_category is not None:
        parts.append(f"obj_{obj_category}")
    tag = "__".join(parts) if parts else "all"
    return dataset_path / "computed_norm_stats" / f"{tag}.json"


def compute_norm_stats_from_filtered_dataset(
    ds_meta: dict,
    action_horizon: int,
) -> dict[str, _normalize.NormStats]:
    """Compute norm stats by iterating over the actual filtered dataset.

    Builds a GrootOpenpiSingleDataset with the given meta (including scene
    filtering), iterates every item, and computes mean/std/q01/q99 for
    state and actions.  Results are cached to disk so subsequent runs with
    the same filter params skip recomputation.
    """
    import logging as _logging

    cache_path = _stats_cache_path(ds_meta)
    if cache_path.exists():
        _logging.info(f"Loading cached filtered norm stats from {cache_path}")
        return _load_cached_norm_stats(cache_path)

    _logging.info(f"Computing norm stats from filtered dataset ({ds_meta})...")

    dataset = GrootOpenpiSingleDataset(dataset_meta=ds_meta, action_horizon=action_horizon)

    all_states = []
    all_actions = []
    for i in range(len(dataset)):
        item = dataset[i]
        all_states.append(np.asarray(item["observation/state"]))       # (state_dim,)
        all_actions.append(np.asarray(item["actions"]).reshape(-1, item["actions"].shape[-1]))  # (horizon, action_dim) -> flatten horizon

    states = np.stack(all_states, axis=0)    # (N, state_dim)
    actions = np.concatenate(all_actions, axis=0)  # (N*horizon, action_dim)

    def _make_stats(arr, pad_len):
        """Compute stats for arr of shape (N, D), pad to pad_len."""
        def pad_zeros(v, n):
            return np.concatenate([v, np.zeros(n - len(v))])
        def pad_ones(v, n):
            return np.concatenate([v, np.ones(n - len(v))])

        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        q01 = np.percentile(arr, 1, axis=0)
        q99 = np.percentile(arr, 99, axis=0)
        return _normalize.NormStats(
            mean=pad_zeros(mean, pad_len),
            std=pad_ones(std, pad_len),
            q01=pad_zeros(q01, pad_len),
            q99=pad_ones(q99, pad_len),
        )

    norm_stats = {
        "state": _make_stats(states, 32),
        "actions": _make_stats(actions, 32),
    }

    # Save to cache
    _save_norm_stats(norm_stats, cache_path)
    _logging.info(f"Saved filtered norm stats to {cache_path}")
    return norm_stats


def _save_norm_stats(norm_stats: dict[str, _normalize.NormStats], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, ns in norm_stats.items():
        out[key] = {
            "mean": np.asarray(ns.mean).tolist(),
            "std": np.asarray(ns.std).tolist(),
            "q01": np.asarray(ns.q01).tolist() if ns.q01 is not None else None,
            "q99": np.asarray(ns.q99).tolist() if ns.q99 is not None else None,
        }
    path.write_text(json.dumps(out, indent=2))


def _load_cached_norm_stats(path: pathlib.Path) -> dict[str, _normalize.NormStats]:
    data = json.loads(path.read_text())
    result = {}
    for key, v in data.items():
        result[key] = _normalize.NormStats(
            mean=np.array(v["mean"]),
            std=np.array(v["std"]),
            q01=np.array(v["q01"]) if v["q01"] is not None else None,
            q99=np.array(v["q99"]) if v["q99"] is not None else None,
        )
    return result
