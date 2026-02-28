"""Dump the exact filtered data indices used for training a given config.

This script replicates the filtering logic from FilteredDataset to identify
which episodes from the LeRobot dataset are used for training.
It saves the episode indices and per-episode sample ranges to a JSON file
so the same data can be loaded in another codebase (e.g., dsrl_pi0).

Usage:
    python scripts/dump_filtered_data.py \
        --repo_id physical-intelligence/libero \
        --filter_prompt "put both moka pots on the stove" \
        --num_episodes 5 \
        --output_dir /tmp/openpi_data_dumps
"""

import argparse
import json
import logging
import os

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_filtered_episodes(
    repo_id: str,
    filter_prompt: str,
    num_episodes: int = -1,
    exclude_filter_prompt: bool = False,
):
    """Use dataset metadata to find matching episodes.

    The metadata.episodes dict contains per-episode info including tasks,
    so we don't need to load the full HF dataset.
    """
    logger.info(f"Loading dataset metadata for '{repo_id}'...")
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    tasks_map = meta.tasks
    episodes = meta.episodes  # dict: ep_idx -> {episode_index, tasks, length}

    filter_type = "exclude" if exclude_filter_prompt else "include"
    logger.info(f"Filtering ({filter_type}) for prompt: '{filter_prompt}'")
    logger.info(f"Total episodes in dataset: {meta.total_episodes}")
    logger.info(f"Total frames in dataset: {meta.total_frames}")

    # Find matching episodes in order
    matching_episodes = []
    for ep_idx in sorted(episodes.keys()):
        ep_info = episodes[ep_idx]
        ep_tasks = ep_info.get("tasks", [])
        matches = filter_prompt in ep_tasks

        if (matches and not exclude_filter_prompt) or (not matches and exclude_filter_prompt):
            matching_episodes.append(ep_idx)

    logger.info(f"Found {len(matching_episodes)} matching episodes")
    logger.info(f"All matching episode indices (in order): {matching_episodes}")

    # Limit by number of episodes (only when including, not excluding)
    if not exclude_filter_prompt and num_episodes > 0 and len(matching_episodes) > num_episodes:
        logger.info(
            f"Limiting from {len(matching_episodes)} episodes to {num_episodes} episodes"
        )
        matching_episodes = matching_episodes[:num_episodes]

    # Compute sample index ranges per episode
    # Episodes are stored contiguously; compute cumulative offsets from episode lengths
    cum_offset = 0
    ep_start = {}
    for ep_idx in sorted(episodes.keys()):
        ep_start[ep_idx] = cum_offset
        cum_offset += episodes[ep_idx]["length"]

    episode_to_sample_indices = {}
    total_samples = 0
    for ep_idx in matching_episodes:
        start = ep_start[ep_idx]
        length = episodes[ep_idx]["length"]
        episode_to_sample_indices[ep_idx] = list(range(start, start + length))
        total_samples += length

    # Build flat list of all valid sample indices
    valid_indices = []
    for ep_idx in matching_episodes:
        valid_indices.extend(episode_to_sample_indices[ep_idx])

    return valid_indices, matching_episodes, tasks_map, episodes, episode_to_sample_indices


def main():
    parser = argparse.ArgumentParser(description="Dump filtered training data indices")
    parser.add_argument("--repo_id", type=str, default="physical-intelligence/libero")
    parser.add_argument("--filter_prompt", type=str, default="put both moka pots on the stove")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--exclude_filter_prompt", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/openpi_data_dumps",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    valid_indices, matching_episodes, tasks_map, episodes, episode_to_sample_indices = get_filtered_episodes(
        repo_id=args.repo_id,
        filter_prompt=args.filter_prompt,
        num_episodes=args.num_episodes,
        exclude_filter_prompt=args.exclude_filter_prompt,
    )

    # Summary
    logger.info("\n=== Dump Summary ===")
    logger.info(f"Config: filter_prompt='{args.filter_prompt}', num_episodes={args.num_episodes}")
    logger.info(f"Total filtered samples: {len(valid_indices)}")
    logger.info(f"Episodes kept: {matching_episodes}")
    for ep in matching_episodes:
        n = len(episode_to_sample_indices[ep])
        logger.info(f"  Episode {ep}: {n} samples (length={episodes[ep]['length']})")

    # Save the dump
    safe_prompt = args.filter_prompt.replace(" ", "_")[:60]
    dump_filename = f"filtered_{safe_prompt}_ep{args.num_episodes}.json"
    dump_path = os.path.join(args.output_dir, dump_filename)

    dump = {
        "repo_id": args.repo_id,
        "filter_prompt": args.filter_prompt,
        "num_episodes": args.num_episodes,
        "exclude_filter_prompt": args.exclude_filter_prompt,
        "total_samples": len(valid_indices),
        "episode_indices": matching_episodes,
        "sample_indices": valid_indices,
        "episode_to_sample_indices": {str(k): v for k, v in episode_to_sample_indices.items()},
        "tasks_map": {str(k): v for k, v in tasks_map.items()},
    }

    with open(dump_path, "w") as f:
        json.dump(dump, f, indent=2)

    logger.info(f"\nDump saved to: {dump_path}")
    logger.info(
        f"\nTo load the same data in dsrl_pi0, use the episode_indices {matching_episodes} "
        f"when loading the LeRobot dataset '{args.repo_id}'."
    )


if __name__ == "__main__":
    main()
