"""Extract exact-replay reset data for a training config's filtered episodes.

For each filtered episode in the config's dataset, copies the three files
needed to reproduce the environment identically during evaluation:

  - ep_meta.json   — scene config (layout, style, fixtures, objects, robot pose)
  - states.npz     — MuJoCo states; states[0] is the initial state
  - model.xml.gz   — the exact MuJoCo model XML

The output directory mirrors the structure that RoboCasaEvalResetController
expects, so it can be passed directly as --dataset-path to the eval script.

See docs/ROBOCASA_demo_meta_format.md for a detailed description of the output
format and how each file is used during evaluation.

Usage:
    uv run scripts/extract_eval_reset_data.py \
        --config-name pi05_robocasa_single_task_lora_turn_on_sink_faucet \
        --output-dir ./eval_reset_data

Then evaluate with exact state replay:
    python examples/robocasa/main.py \
        --eval-init-mode exact_state_replay \
        --dataset-path ./eval_reset_data/pi05_robocasa_single_task_lora_turn_on_sink_faucet
"""

import json
import logging
import pathlib
import shutil

import numpy as np

import tyro

import openpi.training.config as _config
from openpi.groot_utils.groot_openpi_dataset import (
    get_scene_filtered_demos,
    load_ep_meta,
)


def get_filtered_episode_ids(config: _config.TrainConfig) -> list[tuple[pathlib.Path, list[int]]]:
    """Resolve the filtered episode IDs for each data_dir in the config.

    Uses the same filtering logic as GrootOpenpiSingleDataset to determine
    exactly which episodes the model trains on.

    Returns a list of (dataset_path, [episode_ids]) tuples.
    """
    from openpi.groot_utils.groot_openpi_dataset import GrootOpenpiSingleDataset

    data_config = config.data
    # Build enriched data_dirs the same way config.create() does
    data_dirs = data_config.data_dirs or []
    filter_fields = {}
    for attr in ("match_episode_id", "layout_and_style_ids", "num_demos",
                 "obj_category", "fixture_refs", "object_categories", "episode_ids"):
        val = getattr(data_config, attr, None)
        if val is not None:
            filter_fields[attr] = val

    results = []
    for d in data_dirs:
        d_enriched = dict(d)
        d_enriched.update(filter_fields)

        dataset_path = pathlib.Path(d["path"])

        # Create the dataset — this applies the exact same filtering as training
        ds = GrootOpenpiSingleDataset(d_enriched, config.model.action_horizon)

        # Extract the episode IDs from the dataset's subset_demos or trajectory_ids
        if hasattr(ds, '_subset_demos') and ds._subset_demos is not None:
            ep_ids = sorted(ds._subset_demos)
        elif hasattr(ds, 'trajectory_ids'):
            ep_ids = sorted(ds.trajectory_ids)
        else:
            # Fallback: read from episodes.jsonl
            episodes_path = dataset_path / "meta" / "episodes.jsonl"
            with open(episodes_path) as f:
                ep_ids = [json.loads(line)["episode_index"] for line in f]

        results.append((dataset_path, ep_ids))

    return results


def main(config_name: str, output_dir: str = "./eval_reset_data") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = _config.get_config(config_name)
    per_dataset = get_filtered_episode_ids(config)

    out_base = pathlib.Path(output_dir) / config_name
    out_extras = out_base / "extras"
    out_extras.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    episode_manifest = []

    for dataset_path, ep_ids in per_dataset:
        logging.info(f"Dataset: {dataset_path} — {len(ep_ids)} filtered episodes")

        for ep_id in ep_ids:
            src_dir = dataset_path / "extras" / f"episode_{ep_id:06d}"
            dst_dir = out_extras / f"episode_{ep_id:06d}"
            dst_dir.mkdir(parents=True, exist_ok=True)

            files_copied = []
            for filename in ("ep_meta.json", "states.npz", "model.xml.gz"):
                src = src_dir / filename
                dst = dst_dir / filename
                if src.exists():
                    shutil.copy2(src, dst)
                    files_copied.append(filename)
                else:
                    logging.warning(f"  Missing {src}")

            meta = load_ep_meta(dataset_path, ep_id)
            episode_manifest.append({
                "episode_id": ep_id,
                "layout_id": meta.get("layout_id"),
                "style_id": meta.get("style_id"),
                "fixture_refs": meta.get("fixture_refs"),
                "lang": meta.get("lang"),
                "init_robot_base_pos": meta.get("init_robot_base_pos"),
                "init_robot_base_ori": meta.get("init_robot_base_ori"),
                "files": files_copied,
            })
            total_episodes += 1

    # Also copy the videos directory structure (for oracle video comparison during eval)
    for dataset_path, ep_ids in per_dataset:
        videos_src = dataset_path / "videos"
        if videos_src.exists():
            videos_dst = out_base / "videos"
            for ep_id in ep_ids:
                for chunk_dir in videos_src.iterdir():
                    if not chunk_dir.is_dir():
                        continue
                    for cam_dir in chunk_dir.iterdir():
                        if not cam_dir.is_dir():
                            continue
                        vid_file = cam_dir / f"episode_{ep_id:06d}.mp4"
                        if vid_file.exists():
                            dst = videos_dst / chunk_dir.name / cam_dir.name / vid_file.name
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(vid_file, dst)

    # Write manifest (convert numpy types for JSON)
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    manifest_path = out_base / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "config_name": config_name,
            "num_episodes": total_episodes,
            "episodes": episode_manifest,
        }, f, indent=2, cls=NumpyEncoder)

    logging.info(f"Extracted {total_episodes} episodes to {out_base}")
    logging.info(f"Manifest: {manifest_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"Episodes: {total_episodes}")
    print(f"Output: {out_base}")
    print(f"\nTo evaluate with exact state replay:")
    print(f"  python examples/robocasa/main.py \\")
    print(f"    --eval-init-mode exact_state_replay \\")
    print(f"    --dataset-path {out_base}")
    print(f"{'='*60}")

    for ep in episode_manifest:
        print(f"  ep {ep['episode_id']:4d}: L={ep['layout_id']}, S={ep['style_id']}, "
              f"fixtures={ep['fixture_refs']}, lang=\"{ep['lang']}\"")


if __name__ == "__main__":
    tyro.cli(main)
