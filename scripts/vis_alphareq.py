"""
Effective Rank Computation for Model Activations

This script extracts intermediate activations from multiple layers of a trained Pi0/Pi0.5 model
and computes the effective rank of their covariance matrices.

Process:
1. Load trained model checkpoint
2. Process 10 batches (batch_size samples each)
3. Extract activations from:
   - SIGLIP vision encoder output (per image)
   - PaliGemma vision transformer output (per image)
   - Action transformer output (first action token)
4. Compute covariance matrix for each activation type across all samples
5. Compute effective rank = exp(entropy(normalized_singular_values))

The effective rank provides a measure of the intrinsic dimensionality of the activations,
indicating how many dimensions are effectively used by the model.
"""

import dataclasses
import logging
import pickle
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import orbax.checkpoint as ocp


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

def _initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> ocp.CheckpointManager:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": _checkpoints.CallbackHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1000000,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    return mngr


def extract_all_layer_activations(
    model: _model.BaseModel,
    rng: at.KeyArrayLike,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, dict[str, at.Array] | at.Array]:
    """Extract activations from SIGLIP, vision LLM, and action transformer.
    
    Returns:
        Dictionary with keys 'siglip', 'vision_llm', and 'action'.
        'siglip' and 'vision_llm' contain nested dicts mapping image names to activations.
        'action' contains activations for the first action token.
    """
    
    observation, actions = batch
    
    # Check if model has the methods needed to extract activations (Pi0/Pi0.5 models)
    has_activation_extraction = (
        hasattr(model, 'embed_prefix') and 
        hasattr(model, 'embed_suffix') and 
        hasattr(model, 'PaliGemma') and
        hasattr(model, 'action_out_proj')
    )
    
    if not has_activation_extraction:
        raise ValueError("Model does not support activation extraction. Only Pi0/Pi0.5 models are supported.")
    
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=False)
    
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    
    # ===== Extract SIGLIP activations for each image =====
    siglip_activations = {}
    image_token_counts = {}
    
    for image_name in observation.images:
        # Get SIGLIP output (last hidden layer)
        image_tokens, _ = model.PaliGemma.img(observation.images[image_name], train=False)
        # image_tokens shape: (batch, num_patches, hidden_dim)
        # Average pool over patches to get per-image representation
        siglip_activations[image_name] = jnp.mean(image_tokens, axis=1)  # (batch, hidden_dim)
        image_token_counts[image_name] = image_tokens.shape[1]
    
    # ===== Forward pass to get vision LLM and action transformer activations =====
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    
    from openpi.models.pi0 import make_attn_mask
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    
    # Get intermediate activations from transformers
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    
    # ===== Extract vision LLM activations per image =====
    # prefix_out contains outputs for all image patches + language tokens
    # We need to extract the portion corresponding to each image
    vision_llm_activations = {}
    start_idx = 0
    
    for image_name in observation.images:
        num_tokens = image_token_counts[image_name]
        end_idx = start_idx + num_tokens
        # Extract tokens for this image and average pool
        image_output = prefix_out[:, start_idx:end_idx, :]  # (batch, num_patches, hidden_dim)
        vision_llm_activations[image_name] = jnp.mean(image_output, axis=1)  # (batch, hidden_dim)
        start_idx = end_idx
    
    # ===== Extract action transformer activation for first action =====
    # suffix_out shape: (batch, action_horizon, hidden_dim)
    # We want the first action's activation
    action_activations = suffix_out[:, 0, :]  # (batch, hidden_dim)
    
    return {
        "siglip": siglip_activations,
        "vision_llm": vision_llm_activations,
        "action": action_activations,
    }


def compute_effective_rank(activations: np.ndarray) -> float:
    """
    Compute effective rank of a set of activations.
    
    Args:
        activations: Array of shape (num_samples, hidden_dim)
    
    Returns:
        Effective rank computed as exp(entropy(normalized_singular_values))
    """
    # Compute covariance matrix: (hidden_dim, hidden_dim)
    # Center the data first
    activations_centered = activations - np.mean(activations, axis=0, keepdims=True)
    cov_matrix = np.cov(activations_centered, rowvar=False)
    
    # Compute SVD of covariance matrix
    singular_values = scipy.linalg.svdvals(cov_matrix)
    
    # Normalize singular values to get probabilities
    singular_values = np.maximum(singular_values, 0)  # Ensure non-negative
    sum_sv = np.sum(singular_values)
    
    if sum_sv == 0:
        return 0.0
    
    pi = singular_values / sum_sv
    
    # Compute entropy: -sum(pi * log(pi))
    # Filter out zero values to avoid log(0)
    pi_nonzero = pi[pi > 0]
    entropy = -np.sum(pi_nonzero * np.log(pi_nonzero))
    
    # Effective rank is exp(entropy)
    effective_rank = np.exp(entropy)
    
    return effective_rank


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info("=" * 80)
    logging.info("EFFECTIVE RANK COMPUTATION FOR LAYER ACTIVATIONS")
    logging.info(f"Processing 10 batches (batch size: {config.batch_size})")
    logging.info("=" * 80)

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    sample_rng = rng

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    
    # Initialize wandb for logging (simplified for inference)
    # if config.wandb_enabled:
    #     wandb.init(
    #         name=f"{config.exp_name}_effective_rank",
    #         config=dataclasses.asdict(config),
    #         project=config.project_name,
    #     )
    # else:
    #     wandb.init(mode="disabled")

    # Create data loader
    logging.info("Creating data loader...")
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    first_batch = next(data_iter)
    logging.info(f"Data loader initialized with batch size: {config.batch_size}")

    # Find the checkpoint step to load
    logging.info(f"Looking for checkpoint in {config.checkpoint_dir}...")
    checkpoint_manager = _initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=None, overwrite=False, resume=False,
    )
    # step = checkpoint_manager.latest_step()
    step = 750
    if step is None:
        raise ValueError(f"No checkpoint found in {config.checkpoint_dir}")
    
    checkpoint_path = epath.Path(config.checkpoint_dir) / str(step) / "params"
    logging.info(f"Loading checkpoint from step {step}: {checkpoint_path}")
    
    # Use restore_params to load checkpoint with proper sharding
    checkpoint_params = _model.restore_params(
        checkpoint_path,
        dtype=jnp.bfloat16,
        restore_type=jax.Array,
    )
    logging.info("Checkpoint parameters loaded")
    
    # Load model with checkpoint parameters
    model = config.model.load(checkpoint_params)
    model.eval()  # Set model to evaluation mode
    logging.info("Model loaded successfully")
    
    # No JIT compilation needed for inference-only script with small number of batches
    # (JIT would require the model to be hashable, which is not the case)
    logging.info("Using non-JIT activation extraction (inference mode)...")
    pextract_activations = extract_all_layer_activations
    
    # ===== Collect activations from 10 batches =====
    num_batches = 10
    logging.info(f"\nExtracting activations from {num_batches} batches...")
    logging.info(f"Total samples: {num_batches * config.batch_size}")
    
    # Initialize collectors for each activation type
    collected_siglip = {}  # {image_name: list of activations}
    collected_vision_llm = {}  # {image_name: list of activations}
    collected_action = []  # list of activations
    
    for batch_idx in tqdm.tqdm(range(num_batches), desc="Processing batches"):
        # Get a batch
        batch = next(data_iter)
        
        # Generate random key for this batch
        batch_rng = jax.random.fold_in(sample_rng, batch_idx)
        
        # Extract activations
        with sharding.set_mesh(mesh):
            activations = pextract_activations(model, batch_rng, batch)
        
        # Convert to numpy and collect
        # SIGLIP activations
        for image_name, image_acts in activations["siglip"].items():
            if image_name not in collected_siglip:
                collected_siglip[image_name] = []
            collected_siglip[image_name].append(np.array(image_acts))
        
        # Vision LLM activations
        for image_name, image_acts in activations["vision_llm"].items():
            if image_name not in collected_vision_llm:
                collected_vision_llm[image_name] = []
            collected_vision_llm[image_name].append(np.array(image_acts))
        
        # Action activations
        collected_action.append(np.array(activations["action"]))
    
    # ===== Concatenate all activations =====
    logging.info("\nConcatenating activations from all batches...")
    
    # Concatenate across batches: (num_batches, batch_size, hidden_dim) -> (total_samples, hidden_dim)
    siglip_activations = {name: np.concatenate(acts, axis=0) for name, acts in collected_siglip.items()}
    vision_llm_activations = {name: np.concatenate(acts, axis=0) for name, acts in collected_vision_llm.items()}
    action_activations = np.concatenate(collected_action, axis=0)
    
    total_samples = action_activations.shape[0]
    logging.info(f"Total samples collected: {total_samples}")
    
    # ===== Compute effective rank for each activation type =====
    logging.info("\n" + "=" * 80)
    logging.info("COMPUTING EFFECTIVE RANK")
    logging.info("=" * 80)
    
    effective_ranks = {}
    
    # Compute for SIGLIP activations (per image)
    logging.info("\nSIGLIP (Vision Encoder) Activations:")
    for image_name, activations in siglip_activations.items():
        eff_rank = compute_effective_rank(activations)
        effective_ranks[f"siglip_{image_name}"] = eff_rank
        logging.info(f"  {image_name}: Effective Rank = {eff_rank:.4f} "
                    f"(shape: {activations.shape})")
    
    # Compute for Vision LLM activations (per image)
    logging.info("\nVision LLM (PaliGemma) Activations:")
    for image_name, activations in vision_llm_activations.items():
        eff_rank = compute_effective_rank(activations)
        effective_ranks[f"vision_llm_{image_name}"] = eff_rank
        logging.info(f"  {image_name}: Effective Rank = {eff_rank:.4f} "
                    f"(shape: {activations.shape})")
    
    # Compute for Action transformer activations
    logging.info("\nAction Transformer (First Action) Activations:")
    eff_rank = compute_effective_rank(action_activations)
    effective_ranks["action_transformer"] = eff_rank
    logging.info(f"  Effective Rank = {eff_rank:.4f} "
                f"(shape: {action_activations.shape})")
    
    # ===== Summary =====
    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY OF EFFECTIVE RANKS")
    logging.info("=" * 80)
    for key, value in effective_ranks.items():
        logging.info(f"  {key}: {value:.4f}")
    
    # Log to wandb
    wandb.log({"effective_rank/" + k: v for k, v in effective_ranks.items()})
    
    # Save results to pickle file
    logs_dir = epath.Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_file = logs_dir / f"effective_ranks_{config.name}.pkl"
    
    results = {
        "effective_ranks": effective_ranks,
        "siglip_activations": siglip_activations,
        "vision_llm_activations": vision_llm_activations,
        "action_activations": action_activations,
    }
    
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    
    logging.info(f"\nSaved results to {results_file}")
    logging.info("\n" + "=" * 80)
    logging.info("EFFECTIVE RANK COMPUTATION COMPLETE")
    logging.info("=" * 80)


if __name__ == "__main__":
    main(_config.cli())
