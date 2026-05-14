"""
Merge task-specific checkpoints with pi05 base checkpoint.

Takes PaLiGemma VLM parameters (vision encoder + language model) from a
task-specific fine-tuned checkpoint and combines them with the pi05 base
diffusion/action parameters.

Architecture reference (Pi05):
  - PaLiGemma VLM: llm (gemma, index 0) + img (SigLIP)
  - Action expert: llm (gemma, index 1, identified by '_1' in flattened keys)
  - Action layers: action_in_proj, action_out_proj, time_mlp_in, time_mlp_out

Usage:
    python scripts/merge_checkpoints.py \
        --task-checkpoint /path/to/task/checkpoint/step/params \
        --base-checkpoint /path/to/base/params \
        --output-dir /path/to/output

    # Or batch mode for all full_data tasks:
    python scripts/merge_checkpoints.py --batch \
        --checkpoint-base-dir /data/hf_cache/models/pi05_checkpoints_libero10 \
        --base-checkpoint gs://openpi-assets/checkpoints/pi05_base/params \
        --output-dir /data/hf_cache/models/pi05_merged_checkpoints
"""

import argparse
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
import orbax.checkpoint as ocp

from openpi.models import model as _model
from openpi.shared import download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tasks in full_data that we want to merge
FULL_DATA_TASKS = [
    "pi05_libero10_both_mokapots_stove_full_data_hdf5_libero10",
    "pi05_libero10_alphabet_soup_tomato_sauce_basket_full_data_hdf5_libero10",
    "pi05_libero10_black_bowl_bottom_drawer_full_data_hdf5_libero10",
]


def is_vlm_param(key_tuple):
    """Check if a flattened parameter key belongs to the PaLiGemma VLM.

    VLM params are:
    - All params without '_1' in the key (these are the main gemma / vision backbone)
    - img params even if they have '_1' (SigLIP encoder)

    Action/diffusion params (from base) are:
    - Action expert params (contain '_1' but not 'img')
    - action_in_proj, action_out_proj, time_mlp_in, time_mlp_out
    """
    key_str = '.'.join(str(k) for k in key_tuple)

    # Action projection / time MLP layers -> NOT vlm
    action_layer_prefixes = ('action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out')
    for prefix in action_layer_prefixes:
        if prefix in key_str:
            return False

    # Action expert (gemma _1) -> NOT vlm, UNLESS it's img
    if '_1' in key_str:
        if 'img' in key_str:
            return True
        return False

    # Everything else is VLM
    return True


def load_params(checkpoint_path: str):
    """Load checkpoint parameters."""
    checkpoint_path = str(download.maybe_download(str(checkpoint_path)))
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    params = _model.restore_params(
        checkpoint_path,
        restore_type=jax.Array,
        dtype=jnp.bfloat16,
    )

    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return params


def merge_params(task_params, base_params):
    """Merge VLM params from task checkpoint with action/diffusion params from base.

    Args:
        task_params: Fine-tuned task checkpoint (source of VLM params).
        base_params: Pi05 base checkpoint (source of action/diffusion params).

    Returns:
        Merged parameter dict.
    """
    task_flat = flatten_dict(task_params)
    base_flat = flatten_dict(base_params)

    merged_flat = {}
    vlm_count = 0
    base_count = 0

    # Use task params as the reference for all keys
    all_keys = set(task_flat.keys()) | set(base_flat.keys())

    for key in all_keys:
        if is_vlm_param(key):
            if key in task_flat:
                merged_flat[key] = task_flat[key]
                vlm_count += 1
            else:
                logger.warning(f"VLM key {'.'.join(str(k) for k in key)} not in task checkpoint, using base")
                merged_flat[key] = base_flat[key]
                vlm_count += 1
        else:
            if key in base_flat:
                merged_flat[key] = base_flat[key]
                base_count += 1
            else:
                logger.warning(f"Base key {'.'.join(str(k) for k in key)} not in base checkpoint, using task")
                merged_flat[key] = task_flat[key]
                base_count += 1

    logger.info(f"Merged: {vlm_count} VLM params from task, {base_count} action/diffusion params from base")

    # Log which keys came from base (action/diffusion)
    base_keys = ['.'.join(str(k) for k in key) for key in all_keys if not is_vlm_param(key)]
    logger.info(f"Action/diffusion keys from base: {base_keys[:10]}{'...' if len(base_keys) > 10 else ''}")

    return unflatten_dict(merged_flat)


def save_params(params, output_dir: str):
    """Save merged parameters using orbax."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    params_path = output_path / "params"
    logger.info(f"Saving merged checkpoint to {params_path}")

    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(
            params_path,
            {"params": params},
            force=True,
        )

    logger.info(f"Saved merged checkpoint to {params_path}")


def get_latest_checkpoint(task_dir: Path) -> Path | None:
    """Find the latest checkpoint step in a task directory."""
    # Structure: task_dir/<exp_name>/<step>/params
    exp_dirs = [d for d in task_dir.iterdir() if d.is_dir()]
    if not exp_dirs:
        return None

    exp_dir = exp_dirs[0]  # Usually only one experiment
    step_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not step_dirs:
        return None

    latest = max(step_dirs, key=lambda d: int(d.name))
    params_path = latest / "params"
    if params_path.exists():
        return params_path
    return None


def main():
    parser = argparse.ArgumentParser(description="Merge task VLM params with pi05 base action params")

    # Single merge mode
    parser.add_argument("--task-checkpoint", type=str, help="Path to task checkpoint params dir")
    parser.add_argument("--base-checkpoint", type=str,
                        default="gs://openpi-assets/checkpoints/pi05_base/params",
                        help="Path to pi05 base checkpoint params")
    parser.add_argument("--output-dir", type=str, help="Output directory for merged checkpoint")

    # Batch mode
    parser.add_argument("--batch", action="store_true", help="Run in batch mode for all full_data tasks")
    parser.add_argument("--checkpoint-base-dir", type=str,
                        default="/data/hf_cache/models/pi05_checkpoints_libero10",
                        help="Base directory containing task checkpoint directories")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Task names to process (default: all full_data tasks)")

    args = parser.parse_args()

    if args.batch:
        tasks = args.tasks or FULL_DATA_TASKS
        checkpoint_base = Path(args.checkpoint_base_dir)

        if not args.output_dir:
            args.output_dir = str(checkpoint_base.parent / "pi05_merged_checkpoints")

        # Load base params once
        logger.info("Loading base pi05 checkpoint (shared across all tasks)...")
        base_params = load_params(args.base_checkpoint)

        for task_name in tasks:
            task_dir = checkpoint_base / task_name
            if not task_dir.exists():
                logger.warning(f"Task directory not found: {task_dir}, skipping")
                continue

            params_path = get_latest_checkpoint(task_dir)
            if params_path is None:
                logger.warning(f"No checkpoint found for {task_name}, skipping")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing task: {task_name}")
            logger.info(f"Using checkpoint: {params_path}")
            logger.info(f"{'='*60}")

            task_params = load_params(str(params_path))
            merged = merge_params(task_params, base_params)

            output = Path(args.output_dir) / task_name
            save_params(merged, str(output))
            logger.info(f"Saved merged checkpoint for {task_name} -> {output}")

            # Free memory
            del task_params, merged
            jax.clear_caches()

    else:
        if not args.task_checkpoint or not args.output_dir:
            parser.error("--task-checkpoint and --output-dir are required in single mode")

        logger.info("Loading task checkpoint...")
        task_params = load_params(args.task_checkpoint)

        logger.info("Loading base pi05 checkpoint...")
        base_params = load_params(args.base_checkpoint)

        logger.info("Merging parameters...")
        merged = merge_params(task_params, base_params)

        save_params(merged, args.output_dir)
        logger.info("Done!")


if __name__ == "__main__":
    main()
