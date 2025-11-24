"""
Load two checkpoints on different GPUs, transfer vision-related parameters 
(PaliGemma.llm and PaliGemma.img) from GPU 0 model to GPU 1 model, and save the result.
"""

import logging
from pathlib import Path
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
import orbax.checkpoint as ocp

from openpi.training import config as _config
from openpi.models import model as _model
from openpi.shared import download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SOURCE_CONFIG_NAME = "pi0_libero"  # Config for checkpoint with vision params to transfer
TARGET_CONFIG_NAME = "pi0_libero"  # Config for checkpoint to receive vision params

# SOURCE_CHECKPOINT_DIR = "/data/user_data/skowshik/openpi_cache/pi0_libero_lora_moka_pots_task_ep20/checkpoints/pi0_libero_lora_moka_pots_task_ep20/pi0_libero_lora_moka_pots_task_ep20-v1/20000/"
SOURCE_CHECKPOINT_DIR = "gs://openpi-assets/checkpoints/pi05_libero"
TARGET_CHECKPOINT_DIR = "/data/user_data/skowshik/openpi_cache/pi0_libero_fullft_moka_pots_task_ep29/debug-v1/0/"

OUTPUT_DIR = "/data/user_data/skowshik/openpi_cache/merged_checkpoint/"


def load_params_on_device(checkpoint_dir: str, device: jax.Device):
    """Load checkpoint parameters and place them on a specific device."""
    checkpoint_dir = Path(download.maybe_download(str(checkpoint_dir)))
    params_path = checkpoint_dir / "params"
    
    print(f"Loading checkpoint from {params_path} onto {device}")
    logger.info(f"Loading checkpoint from {params_path} onto {device}")
    
    # Create single-device sharding for the specified device
    sharding = jax.sharding.SingleDeviceSharding(device)
    
    # Load params with specific device sharding
    params = _model.restore_params(
        params_path,
        restore_type=jax.Array,
        dtype=jnp.bfloat16,
        sharding=sharding
    )
    
    print(f"Successfully loaded checkpoint on {device}")
    logger.info(f"Successfully loaded checkpoint on {device}")
    return params


def transfer_vision_params(source_params, target_params):
    """
    Transfer vision-related parameters (PaliGemma.llm and PaliGemma.img) 
    from source to target.
    
    Returns modified target_params.
    """
    print("Transferring vision-related parameters...")
    logger.info("Transferring vision-related parameters...")
    
    # Flatten the parameter trees for easier manipulation
    source_flat = flatten_dict(source_params)
    target_flat = flatten_dict(target_params)
    
    # Identify and transfer PaliGemma parameters (both llm and img)
    transferred_count = 0
    vision_param_keys = []
    
    for _key, value in source_flat.items():
        # Check if key contains PaliGemma (which includes both llm and img)
        key = '.'.join(_key)
        # if ('llm' in _key or 'img' in _key) and '_1' not in _key:
        if '_1' in key:
            if 'img' in key:
                vision_param_keys.append(key)
                target_flat[_key] = value
                transferred_count += 1
            else:
                print(key)
        else:
            vision_param_keys.append(key)
            target_flat[_key] = value
            transferred_count += 1
    
    print(f"Transferred {transferred_count} vision-related parameters")
    logger.info(f"Transferred {transferred_count} vision-related parameters")
    print(f"Sample keys transferred: {list(vision_param_keys)[:5]}")
    logger.info(f"Sample keys transferred: {list(vision_param_keys)[:5]}")

    # Print all keys in target that were not transferred
    not_transferred_keys = ['.'.join(k) for k in target_flat.keys() if '.'.join(k) not in vision_param_keys]
    print(f"\n\n\nKeys not transferred: {list(not_transferred_keys)}\n\n\n")
    logger.info(f"Keys not transferred: {not_transferred_keys}")
    
    # Count llm and img params separately for verification
    llm_count = sum(1 for k in vision_param_keys if 'llm' in k)
    img_count = sum(1 for k in vision_param_keys if 'img' in k)
    print(f"  - llm parameters: {llm_count}")
    print(f"  - img parameters: {img_count}")
    logger.info(f"  - llm parameters: {llm_count}")
    logger.info(f"  - img parameters: {img_count}")
    
    # Reconstruct the parameter tree
    modified_target_params = unflatten_dict(target_flat)
    
    return modified_target_params


def save_params(params, output_dir: str):
    """Save parameters to output directory using orbax checkpoint."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    params_path = output_path / "params"
    
    print(f"Saving merged checkpoint to {params_path}")
    logger.info(f"Saving merged checkpoint to {params_path}")
    
    # Use orbax to save the checkpoint
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(
            params_path,
            {"params": params},
            force=True  # Overwrite if exists
        )
    
    print(f"Successfully saved checkpoint to {output_path}")
    logger.info(f"Successfully saved checkpoint to {output_path}")


def main():
    """Main function to load, merge, and save checkpoints."""
    
    # Print to ensure script is running
    print("=" * 80)
    print("CHECKPOINT MERGE SCRIPT STARTING")
    print("=" * 80)
    
    # Check available devices
    devices = jax.devices()
    print(f"Available devices: {devices}")
    logger.info(f"Available devices: {devices}")
    # breakpoint()
    
    if len(devices) < 2:
        print(f"WARNING: Only {len(devices)} device(s) available. Will use available devices.")
        logger.warning(f"Only {len(devices)} device(s) available. Will use available devices.")
        gpu0 = devices[0]
        gpu1 = devices[0] if len(devices) == 1 else devices[1]
    else:
        gpu0 = devices[0]
        gpu1 = devices[1]
    
    print(f"Using GPU 0: {gpu0}")
    print(f"Using GPU 1: {gpu1}")
    logger.info(f"Using GPU 0: {gpu0}")
    logger.info(f"Using GPU 1: {gpu1}")
    # breakpoint()
    
    # Load source checkpoint (with vision params to transfer) on GPU 0
    print("\n=== Loading source checkpoint on GPU 0 ===")
    logger.info("\n=== Loading source checkpoint on GPU 0 ===")
    source_params = load_params_on_device(SOURCE_CHECKPOINT_DIR, gpu0)
    
    # Load target checkpoint (to receive vision params) on GPU 1
    print("\n=== Loading target checkpoint on GPU 1 ===")
    logger.info("\n=== Loading target checkpoint on GPU 1 ===")
    target_params = load_params_on_device(TARGET_CHECKPOINT_DIR, gpu1)
    
    # Transfer vision parameters from source to target
    print("\n=== Transferring vision parameters ===")
    logger.info("\n=== Transferring vision parameters ===")
    merged_params = transfer_vision_params(source_params, target_params)
    
    # Save the merged checkpoint
    print("\n=== Saving merged checkpoint ===")
    logger.info("\n=== Saving merged checkpoint ===")
    save_params(merged_params, OUTPUT_DIR)
    
    print("\n=== Process completed successfully ===")
    print(f"Merged checkpoint saved to: {OUTPUT_DIR}")
    logger.info("\n=== Process completed successfully ===")
    logger.info(f"Merged checkpoint saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    print("Starting checkpoint merge script...")
    print(f"Source: {SOURCE_CHECKPOINT_DIR}")
    print(f"Target: {TARGET_CHECKPOINT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    main()
