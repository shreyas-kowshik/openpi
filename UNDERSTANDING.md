# OpenPI Model Architecture and Data Flow

This document describes how the model class is structured and how data flows from the entry point through the model to the loss function and backpropagation, specifically for the `pi0_libero_low_mem_finetune` configuration.

## Table of Contents
1. [Overview](#overview)
2. [Entry Point](#entry-point)
3. [Configuration](#configuration)
4. [Model Architecture](#model-architecture)
5. [Data Pipeline](#data-pipeline)
6. [Forward Pass](#forward-pass)
7. [Loss Computation](#loss-computation)
8. [Backpropagation & Optimization](#backpropagation--optimization)
9. [Key Design Patterns](#key-design-patterns)

---

## Overview

OpenPI (π₀) is a Vision-Language-Action (VLA) model for robotic manipulation. The `pi0_libero_low_mem_finetune` config performs **LoRA (Low-Rank Adaptation) fine-tuning** on the LIBERO benchmark dataset, using a memory-efficient approach that freezes most of the base model weights.

**Key Model Components:**
- **Vision Encoder**: SigLIP (vision transformer)
- **Language Model**: PaliGemma (Gemma 2B with LoRA adapters)
- **Action Expert**: Gemma 300M (with LoRA adapters)
- **Action Prediction**: Flow matching diffusion model

---

## Entry Point

### `scripts/train.py`

The training starts in the `main()` function:

```python
# scripts/train.py:194
def main(config: _config.TrainConfig):
    init_logging()
    
    # 1. Setup JAX sharding for distributed training
    mesh = sharding.make_mesh(config.fsdp_devices)
    
    # 2. Initialize checkpoint manager
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(...)
    
    # 3. Create data loader
    data_loader = _data_loader.create_data_loader(config, ...)
    
    # 4. Initialize train state (model + optimizer)
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    
    # 5. JIT compile the training step
    ptrain_step = jax.jit(functools.partial(train_step, config), ...)
    
    # 6. Training loop
    for step in range(start_step, config.num_train_steps):
        train_state, info = ptrain_step(train_rng, train_state, batch)
        # Log metrics, save checkpoints
```

**Flow:**
1. Configuration is loaded via `tyro.cli()` 
2. Data loader is created with transforms and normalization
3. Model is initialized (with optional weight loading from checkpoint)
4. Training loop runs JIT-compiled training steps

---

## Configuration

### `pi0_libero_low_mem_finetune` Config

Located in `src/openpi/training/config.py:669-694`:

```python
TrainConfig(
    name="pi0_libero_low_mem_finetune",
    
    # Model with LoRA adapters
    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",      # Language model with LoRA
        action_expert_variant="gemma_300m_lora"  # Action expert with LoRA
    ),
    
    # Dataset configuration
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=True,
    ),
    
    # Load base model weights
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    
    # Training hyperparameters
    num_train_steps=30_000,
    
    # Freeze all non-LoRA weights
    freeze_filter=pi0_config.Pi0Config(...).get_freeze_filter(),
    
    # Disable EMA for LoRA
    ema_decay=None,
)
```

**Key Configuration Parameters:**
- **Action Dimension**: 32 (default for pi0)
- **Action Horizon**: 50 (number of timesteps to predict)
- **Max Token Length**: 48 (for tokenized prompts)
- **Batch Size**: 32 (global batch size)
- **Learning Rate**: AdamW with cosine decay schedule

---

## Model Architecture

### Class Hierarchy

```
BaseModel (abstract)
    ├── Pi0 (src/openpi/models/pi0.py)
    │   └── Used by pi0_libero_low_mem_finetune
    └── Pi0FAST (alternative architecture)
```

### Pi0 Model Structure

Located in `src/openpi/models/pi0.py:66-103`:

```python
class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        # Vision + Language Model (PaliGemma)
        self.PaliGemma = nnx.Dict(
            llm=nnx_bridge.ToNNX(_gemma.Module([
                paligemma_config,      # Gemma 2B (with LoRA)
                action_expert_config   # Gemma 300M (with LoRA)
            ])),
            img=nnx_bridge.ToNNX(_siglip.Module(...))  # SigLIP vision encoder
        )
        
        # Action processing layers
        self.action_in_proj = nnx.Linear(action_dim, action_expert_width)
        self.state_proj = nnx.Linear(action_dim, action_expert_width)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_width, action_expert_width)
        self.action_time_mlp_out = nnx.Linear(action_expert_width, action_expert_width)
        self.action_out_proj = nnx.Linear(action_expert_width, action_dim)
```

**Components:**

1. **SigLIP Vision Encoder** (`self.PaliGemma.img`)
   - Pre-trained vision transformer (ViT)
   - Input: 224×224 RGB images
   - Output: 256 image tokens (16×16 patches)
   - Parameters: **FROZEN** (not updated during LoRA training)

2. **Gemma Language Model** (`self.PaliGemma.llm[0]`)
   - 2B parameter transformer with LoRA adapters
   - Processes concatenated image + text tokens
   - **LoRA layers**: trainable low-rank matrices in attention
   - Base parameters: **FROZEN**

3. **Action Expert** (`self.PaliGemma.llm[1]`)
   - 300M parameter transformer with LoRA adapters
   - Processes noisy action tokens
   - **LoRA layers**: trainable
   - Base parameters: **FROZEN**

4. **Action Projection Layers**
   - `action_in_proj`: Projects actions to expert embedding space
   - `state_proj`: Projects robot state to embedding space
   - `action_time_mlp_*`: Combines action and timestep information
   - `action_out_proj`: Projects expert output to action space
   - **ALL TRAINABLE** (not frozen)

### LoRA Freeze Filter

The freeze filter determines which parameters are trainable:

```python
# src/openpi/models/pi0_config.py:79-108
def get_freeze_filter(self) -> nnx.filterlib.Filter:
    # Freeze all LLM parameters EXCEPT LoRA layers
    return nnx.All(
        nnx_utils.PathRegex(".*llm.*"),           # Match all LLM params
        nnx.Not(nnx_utils.PathRegex(".*lora.*"))  # Exclude LoRA params
    )
```

**Trainable Parameters:**
- LoRA adapters in Gemma 2B language model
- LoRA adapters in Gemma 300M action expert
- All action projection layers (linear layers outside LLM)

**Frozen Parameters:**
- SigLIP vision encoder (entire module)
- Gemma 2B base weights
- Gemma 300M base weights

---

## Data Pipeline

### Data Flow

```
LeRobot Dataset (HuggingFace)
    ↓
[1] Repack Transform (key mapping)
    ↓
[2] Data Transforms (robot-specific)
    ↓
[3] Normalization (quantile normalization)
    ↓
[4] Model Transforms (tokenization, resizing)
    ↓
DataLoader (batched, sharded)
    ↓
Training Loop
```

### Transform Pipeline

Located in `src/openpi/training/data_loader.py:183-191`:

```python
def transform_dataset(dataset, data_config):
    return TransformedDataset(dataset, [
        *data_config.repack_transforms.inputs,      # [1] Key remapping
        *data_config.data_transforms.inputs,         # [2] Robot transforms
        _transforms.Normalize(norm_stats, ...),      # [3] Normalization
        *data_config.model_transforms.inputs,        # [4] Tokenization
    ])
```

### Transform Details

#### [1] Repack Transform
```python
# Maps dataset keys to model input keys
{
    "observation/image": "image",
    "observation/wrist_image": "wrist_image",
    "observation/state": "state",
    "actions": "actions",
    "prompt": "prompt",
}
```

#### [2] LIBERO Data Transforms
```python
# src/openpi/policies/libero_policy.py
class LiberoInputs:
    def __call__(self, data):
        # Concatenate joint positions and gripper state
        state = jnp.concatenate([
            data["observation/state"][:7],  # joint positions
            data["observation/state"][7:],  # gripper state
        ])
        
        # Return transformed data
        return {
            "images": {"cam_high": image, "wrist": wrist_image},
            "state": state,
            "actions": actions,
            "prompt": prompt,
        }
```

#### [3] Normalization
Uses pre-computed quantile statistics from the dataset:
```python
# Normalize state and actions to [-1, 1] range
normalized = (value - stats.q01) / (stats.q99 - stats.q01) * 2 - 1
```

#### [4] Model Transforms
```python
# src/openpi/training/config.py:112-124
ModelTransformFactory:
    - InjectDefaultPrompt()        # Add default prompt if missing
    - ResizeImages(224, 224)       # Resize all images
    - TokenizePrompt()              # Tokenize text with PaliGemma tokenizer
    - PadStatesAndActions()         # Pad to action_dim
```

### Batch Structure

After transforms, each batch contains:

```python
Observation = {
    "images": {
        "base_0_rgb": Float[B, 224, 224, 3],      # Primary camera
        "wrist_0_rgb": Float[B, 224, 224, 3],     # Wrist camera (if available)
    },
    "image_masks": {
        "base_0_rgb": Bool[B],                     # Image validity mask
        "wrist_0_rgb": Bool[B],
    },
    "state": Float[B, 32],                         # Robot proprioceptive state
    "tokenized_prompt": Int[B, 48],                # Tokenized instruction
    "tokenized_prompt_mask": Bool[B, 48],          # Token validity mask
}

Actions = Float[B, 50, 32]  # [batch, action_horizon, action_dim]
```

---

## Forward Pass

### Training Forward Pass

The forward pass for training is in `Pi0.compute_loss()`:

```python
# src/openpi/models/pi0.py:189-214
def compute_loss(self, rng, observation, actions, *, train=False):
    # [1] Preprocess observations (data augmentation if training)
    observation = _model.preprocess_observation(rng, observation, train=train)
    
    # [2] Flow matching: add noise to actions
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    x_t = time * noise + (1 - time) * actions
    u_t = noise - actions  # Target flow field
    
    # [3] Embed prefix (images + language)
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    
    # [4] Embed suffix (state + noisy actions)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
        observation, x_t, time
    )
    
    # [5] Create attention mask (prefix-LM style)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    
    # [6] Forward pass through dual LLMs
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond]
    )
    
    # [7] Project action expert output to action space
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
    
    # [8] Compute MSE loss
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

### Detailed Flow

#### [1] Observation Preprocessing

```python
# src/openpi/models/model.py:98-117
def preprocess_observation(rng, observation, *, train=False):
    if train:
        # Random image augmentation
        for key in observation.images:
            observation.images[key] = random_crop_resize(
                rng, observation.images[key]
            )
    
    # Normalize images to [-1, 1]
    for key in observation.images:
        observation.images[key] = observation.images[key] * 2.0 - 1.0
    
    return observation
```

#### [2] Flow Matching Noise

Uses **Conditional Flow Matching** for action generation:
- Sample time `t ~ Beta(1.5, 1)` (biased toward t=1)
- Create noisy action: `x_t = t * noise + (1-t) * action`
- Target: `u_t = noise - action` (flow field)

#### [3] Embed Prefix (Vision + Language)

```python
# src/openpi/models/pi0.py:105-137
def embed_prefix(self, obs):
    tokens = []
    input_mask = []
    ar_mask = []
    
    # Encode images with SigLIP
    for name in obs.images:
        image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
        # Shape: [B, 256, embed_dim]  (16×16 patches)
        tokens.append(image_tokens)
        input_mask.append(obs.image_masks[name])
        ar_mask += [False] * 256  # Full attention within images
    
    # Embed tokenized prompt with Gemma
    if obs.tokenized_prompt is not None:
        tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        # Shape: [B, 48, embed_dim]
        tokens.append(tokenized_inputs)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * 48  # Full attention within language
    
    # Concatenate: [B, 256+256+48, embed_dim] for dual cameras + text
    return jnp.concatenate(tokens, axis=1), ...
```

**Attention Pattern**: Full bidirectional attention within prefix (images and text can attend to each other).

#### [4] Embed Suffix (State + Noisy Actions)

```python
# src/openpi/models/pi0.py:139-186
def embed_suffix(self, obs, noisy_actions, timestep):
    tokens = []
    
    # Project state to embedding space
    state_token = self.state_proj(obs.state)[:, None, :]  # [B, 1, embed_dim]
    tokens.append(state_token)
    
    # Project noisy actions
    action_tokens = self.action_in_proj(noisy_actions)  # [B, 50, embed_dim]
    
    # Embed timestep with sinusoidal positional encoding
    time_emb = posemb_sincos(timestep, embed_dim, min_period=4e-3, max_period=4.0)
    
    # Combine action and time information via MLP
    time_tokens = einops.repeat(time_emb, "b d -> b 50 d")
    action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
    action_time_tokens = self.action_time_mlp_in(action_time_tokens)
    action_time_tokens = nnx.swish(action_time_tokens)
    action_time_tokens = self.action_time_mlp_out(action_time_tokens)
    
    tokens.append(action_time_tokens)
    
    # Autoregressive mask: state attends to prefix, actions are causal
    ar_mask = [True] + [True] + [False] * 49
    
    return jnp.concatenate(tokens, axis=1), ...
```

#### [5] Attention Mask

```python
# src/openpi/models/pi0.py:19-44
def make_attn_mask(input_mask, ar_mask):
    """
    Creates prefix-LM style attention:
    - Prefix tokens (images + text): full bidirectional attention
    - Suffix tokens (state + actions): can attend to prefix + causal self-attention
    
    Example for 3 prefix + 3 suffix tokens with ar_mask=[0,0,0,1,1,1]:
    
    Attention Matrix:
         P0  P1  P2  S0  S1  S2
    P0 [  1   1   1   0   0   0 ]  # Prefix attends to prefix only
    P1 [  1   1   1   0   0   0 ]
    P2 [  1   1   1   0   0   0 ]
    S0 [  1   1   1   1   0   0 ]  # Suffix attends to prefix + causal
    S1 [  1   1   1   1   1   0 ]
    S2 [  1   1   1   1   1   1 ]
    """
    cumsum = jnp.cumsum(ar_mask, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    return jnp.logical_and(attn_mask, input_mask)
```

#### [6] Dual LLM Forward Pass

The model uses **two separate LLM instances** that share the same attention mechanism but have different parameters:

```python
# Dual tower architecture
(prefix_out, suffix_out), kv_cache = self.PaliGemma.llm(
    [prefix_tokens, suffix_tokens],  # Two separate token streams
    mask=attn_mask,
    positions=positions,
    adarms_cond=[None, adarms_cond]
)
```

**LLM 0 (Language Model)**: Processes prefix tokens (images + text)
- Uses Gemma 2B with LoRA adapters
- Output: contextualized image and text embeddings

**LLM 1 (Action Expert)**: Processes suffix tokens (state + actions)
- Uses Gemma 300M with LoRA adapters
- Output: denoised action embeddings

Both LLMs can attend to each other via the shared attention mask.

#### [7] Action Projection

```python
v_t = self.action_out_proj(suffix_out[:, -50:])  # [B, 50, 32]
```

Projects the action expert output back to action space.

---

## Loss Computation

### Flow Matching Loss

```python
# src/openpi/models/pi0.py:214
loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

**Loss per timestep**: Mean squared error between predicted flow field `v_t` and target flow field `u_t`.

**Final loss**: Averaged over action horizon (50 steps) and action dimensions (32 dims).

```
Loss = (1/50) * (1/32) * Σ ||v_t - u_t||²
```

**Intuition**: The model learns to predict the "direction" to denoise noisy actions back to clean actions. This is more stable than directly predicting clean actions.

---

## Backpropagation & Optimization

### Training Step

Located in `scripts/train.py:137-192`:

```python
def train_step(config, rng, state, batch):
    # [1] Merge model definition with parameters
    model = nnx.merge(state.model_def, state.params)
    model.train()  # Set to training mode
    
    # [2] Define loss function
    def loss_fn(model, rng, observation, actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)
    
    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    
    # [3] Compute gradients (only for trainable parameters)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions
    )
    
    # [4] Apply optimizer (AdamW with gradient clipping)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    # [5] Update model with new parameters
    nnx.update(model, new_params)
    new_params = nnx.state(model)
    
    # [6] Update train state
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state
    )
    
    return new_state, {"loss": loss, "grad_norm": optax.global_norm(grads)}
```

### Optimizer Configuration

```python
# AdamW with cosine decay
optimizer = optax.adamw(
    learning_rate=cosine_schedule,
    weight_decay=0.01,
    b1=0.9,
    b2=0.999,
)

# Gradient clipping
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip gradients to norm=1
    optimizer,
)
```

**Learning Rate Schedule**:
```python
CosineDecaySchedule(
    warmup_steps=2000,       # Linear warmup
    peak_lr=3e-4,            # Peak learning rate
    decay_steps=30_000,      # Total training steps
    decay_lr=1e-5,           # Final learning rate
)
```

### LoRA-Specific Training

**Key Point**: Only LoRA adapter parameters and action projection layers are updated.

**Trainable Parameters** (~1-5% of total):
- LoRA matrices in Gemma 2B attention layers
- LoRA matrices in Gemma 300M attention layers  
- Action projection layers

**Frozen Parameters** (~95-99% of total):
- All SigLIP vision encoder weights
- Gemma 2B base weights (except LoRA)
- Gemma 300M base weights (except LoRA)

**Memory Savings**: 
- Forward pass: Full model
- Backward pass: Only gradients for ~5% of parameters
- Optimizer states: Only for trainable parameters

---

## Key Design Patterns

### 1. Prefix-LM Attention

The model uses a **prefix language model** attention pattern:
- **Prefix** (images + text): Bidirectional attention (fully visible)
- **Suffix** (actions): Causal attention + can attend to prefix

This allows the model to:
- Process visual and language context in parallel
- Generate actions autoregressively while conditioning on context

### 2. Dual Tower Architecture

Two separate LLM instances:
- **Tower 1**: Processes visual and language inputs (Gemma 2B)
- **Tower 2**: Processes actions (Gemma 300M)

Benefits:
- Specialized processing for different modalities
- Action expert can be smaller (fewer parameters)
- Both towers can attend to each other

### 3. Flow Matching for Actions

Instead of predicting actions directly, the model learns to denoise:
- More stable training (avoids mode collapse)
- Better multi-modal action distributions
- Iterative refinement at inference time

### 4. LoRA Fine-Tuning

Memory-efficient adaptation:
- Freeze base model weights (pre-trained knowledge)
- Add trainable low-rank adapters
- Fine-tune on task-specific data

Trade-off:
- ✅ Much lower memory footprint
- ✅ Faster training
- ✅ Less prone to catastrophic forgetting
- ❌ Slightly lower final performance vs. full fine-tuning

### 5. Quantile Normalization

Pre-computed statistics for robust normalization:
```python
normalized = (x - q01) / (q99 - q01) * 2 - 1
```

Benefits:
- Robust to outliers (uses 1st and 99th percentiles)
- Brings all inputs to similar scale
- Better training stability

---

## Summary

**Data Flow Diagram**:

```
Dataset (LIBERO)
    ↓
Transforms (repack, normalize, tokenize)
    ↓
Batch: {images, state, prompt_tokens, actions}
    ↓
Model.compute_loss()
    ├─→ Preprocess (augment if training)
    ├─→ Add flow matching noise: x_t = t*noise + (1-t)*action
    ├─→ Embed prefix: SigLIP(images) + Gemma.embed(prompt)
    ├─→ Embed suffix: proj(state) + proj(x_t) + time_mlp(t)
    ├─→ Create prefix-LM attention mask
    ├─→ Forward pass: Dual LLMs with shared attention
    │   ├─→ LLM 0 (Gemma 2B + LoRA): process prefix
    │   └─→ LLM 1 (Gemma 300M + LoRA): process suffix
    ├─→ Project: v_t = action_out_proj(suffix_output)
    └─→ Loss: MSE(v_t, u_t) where u_t = noise - action
        ↓
Gradients (only for LoRA + projection layers)
    ↓
Optimizer (AdamW with cosine decay)
    ↓
Updated parameters
```

**Training Loop**: 30,000 steps × 32 batch size = 960K examples

**Output**: Fine-tuned checkpoint with LoRA adapters for LIBERO tasks.

---

## Pi0 FAST: Alternative Architecture

Pi0 FAST is an alternative model architecture that uses **discrete action tokenization** instead of flow matching. This makes it similar to language modeling approaches like RT-2 and OpenVLA.

### Key Differences from Pi0

| Aspect | Pi0 (Standard) | Pi0 FAST |
|--------|---------------|----------|
| **Action Representation** | Continuous (flow matching) | Discrete (tokenized) |
| **Model Architecture** | Dual LLM towers (language + action expert) | Single LLM tower |
| **Loss Function** | MSE on flow field | Cross-entropy on tokens |
| **Training Objective** | Denoise noisy actions | Predict next action token |
| **Inference** | Iterative denoising (10 steps) | Autoregressive decoding |
| **Action Expert** | Separate Gemma 300M | Integrated into main LLM |
| **State Input** | Continuous embedding | Discretized (256 bins) |

### Configuration

Example configs in `src/openpi/training/config.py`:

```python
# Full fine-tuning
TrainConfig(
    name="pi0_fast_libero",
    model=pi0_fast.Pi0FASTConfig(
        action_dim=7,
        action_horizon=10,
        max_token_len=180,  # Includes prompt + state + action tokens
    ),
    data=LeRobotLiberoDataConfig(...),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
)

# LoRA fine-tuning
TrainConfig(
    name="pi0_fast_libero_low_mem_finetune",
    model=pi0_fast.Pi0FASTConfig(
        action_dim=7,
        action_horizon=10,
        max_token_len=180,
        paligemma_variant="gemma_2b_lora",  # Add LoRA adapters
    ),
    freeze_filter=...,  # Freeze non-LoRA weights
)
```

**Key Parameters**:
- `max_token_len`: Must accommodate prompt + state + action tokens
  - Single-arm robots: ~180 tokens
  - Dual-arm robots: ~250 tokens

---

## FAST Action Tokenization

### Overview

FAST (Factorized Action Space Tokenization) discretizes continuous actions into tokens using a learned VQ-VAE style tokenizer.

**Continuous Actions** → **FAST Tokenizer** → **Discrete Tokens** → **Language Model**

### FASTTokenizer Implementation

Located in `src/openpi/models/tokenizer.py:51-140`:

```python
class FASTTokenizer:
    def __init__(self, max_len=256, fast_tokenizer_path="physical-intelligence/fast"):
        # PaliGemma text tokenizer
        self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(...)
        
        # FAST action tokenizer (learned VQ-VAE)
        self._fast_tokenizer = AutoProcessor.from_pretrained(
            fast_tokenizer_path, trust_remote_code=True
        )
        
    def tokenize(self, prompt, state, actions):
        # [1] Discretize state into 256 bins
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        
        # [2] Create prefix (prompt + state)
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {prompt}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)
        
        # [3] Tokenize actions with FAST
        action_tokens = self._fast_tokenizer(actions[None])[0]  # VQ-VAE encoding
        action_tokens_in_pg = self._map_to_paligemma_vocab(action_tokens)
        
        # [4] Create postfix (action tokens)
        postfix_tokens = (
            self._paligemma_tokenizer.encode("Action: ")
            + action_tokens_in_pg.tolist()
            + self._paligemma_tokenizer.encode("|", add_eos=True)
        )
        
        # [5] Concatenate and create masks
        tokens = prefix_tokens + postfix_tokens
        ar_mask = [0]*len(prefix_tokens) + [1]*len(postfix_tokens)  # Prefix-LM
        loss_mask = [False]*len(prefix_tokens) + [True]*len(postfix_tokens)
        
        return tokens, token_mask, ar_mask, loss_mask
```

### Token Sequence Structure

```
[BOS] Task: fold towel, State: 120 45 89 ... 230;\n Action: <tok1> <tok2> ... <tokN> | [EOS]
|<------------ Prefix (bidirectional) ------------->|<----- Postfix (causal) ----->|
|<-------------- No loss on prefix ---------------->|<--- Compute loss on actions -->|
```

**Prefix** (ar_mask=0):
- BOS token
- Task description
- Discretized state (256 bins per dimension)
- Delimiter ";\n"
- Full bidirectional attention within prefix

**Postfix** (ar_mask=1):
- "Action:" marker
- FAST action tokens (typically 10-30 tokens for horizon=10-16)
- Delimiter "|"
- EOS token
- Causal attention, can attend to full prefix

### Action Token Mapping

FAST tokens are mapped to the **end of PaliGemma's vocabulary**:

```python
def _act_tokens_to_paligemma_tokens(self, tokens):
    # Map FAST tokens to last 128 tokens in PaliGemma vocab
    # (skipping the last 128 special tokens)
    return paligemma_vocab_size - 1 - 128 - tokens
```

**Example**:
- PaliGemma vocab size: 256,128
- Reserved for FAST: tokens 255,872 to 256,000
- FAST token 0 → PaliGemma token 256,000
- FAST token 127 → PaliGemma token 255,873

---

## Pi0 FAST Model Architecture

### Class Structure

```python
# src/openpi/models/pi0_fast.py:134-157
class Pi0FAST(_model.BaseModel):
    def __init__(self, config: Pi0FASTConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        
        # Single LLM (Gemma 2B) - no separate action expert
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        
        # Vision encoder (same as Pi0)
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
```

**Key Architectural Difference**: 
- Pi0: Dual LLM towers (language model + action expert)
- Pi0 FAST: Single LLM tower (processes everything)

### Advantages

1. **Simpler Architecture**: Single LLM instead of dual towers
2. **No Action Expert**: Reduces parameter count
3. **Language-Like Training**: Standard next-token prediction
4. **Faster Inference**: Single forward pass vs. iterative denoising

---

## FAST Forward Pass (Training)

### Compute Loss

Located in `src/openpi/models/pi0_fast.py:198-233`:

```python
def compute_loss(self, rng, observation, actions, *, train=False):
    # [1] Preprocess observation
    observation = _model.preprocess_observation(
        rng, observation, train=train, image_keys=list(observation.images.keys())
    )
    
    # [2] Embed inputs (images + tokenized prompt + state + actions)
    input_token_embeddings, input_mask, ar_mask = self.embed_inputs(observation)
    # Shape: [B, seq_len, embed_dim]
    # seq_len = num_image_tokens + num_text_tokens + num_action_tokens
    
    # [3] Create prefix-LM attention mask
    attn_mask = make_attn_mask(input_mask, ar_mask)
    
    # [4] Prepare targets (shifted by 1 for next-token prediction)
    targets = jax.nn.one_hot(
        observation.tokenized_prompt[:, 1:],  # Shift left by 1
        self.PaliGemma.llm.module.vocab_size,
    )
    
    # [5] Forward pass through LLM
    pre_logits, _, _ = self.PaliGemma.llm(
        embedded_prefix=input_token_embeddings[:, :-1],  # Don't input last token
        mask=attn_mask[:, :-1, :-1],
        return_prelogits=True,
    )
    
    # [6] Decode only action tokens (memory optimization)
    logits, _ = self.PaliGemma.llm(
        pre_logits=pre_logits[:, -targets.shape[1]:],
    )
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    
    # [7] Compute cross-entropy loss (only on action tokens)
    loss_mask = observation.token_loss_mask[:, 1:]  # Only action tokens
    token_log_probs = jnp.sum(targets * log_probs, axis=-1)
    loss = -jnp.sum(token_log_probs * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)
    
    return loss  # Shape: [B]
```

### Detailed Steps

#### [2] Embed Inputs

```python
# src/openpi/models/pi0_fast.py:159-195
def embed_inputs(self, obs):
    token_embeddings = []
    input_mask = []
    ar_mask = []
    
    # Embed images (same as Pi0)
    for name in obs.images:
        image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)
        token_embeddings.append(image_token_embeddings)  # [B, 256, emb]
        input_mask.append(obs.image_masks[name])
        ar_mask.append(0 * input_mask[-1])  # Bidirectional attention
    
    # Embed tokenized inputs (prompt + state + action tokens)
    tokenized_inputs_embeddings = self.PaliGemma.llm(
        obs.tokenized_prompt, embed_only=True
    )
    token_embeddings.append(tokenized_inputs_embeddings)  # [B, max_token_len, emb]
    input_mask.append(obs.tokenized_prompt_mask)
    ar_mask.append(obs.token_ar_mask)  # Prefix=0, Postfix=1
    
    return (
        jnp.concatenate(token_embeddings, axis=1),  # [B, total_len, emb]
        jnp.concatenate(input_mask, axis=1),
        jnp.concatenate(ar_mask, axis=1),
    )
```

**Total Sequence**:
```
[image_tokens_cam1] [image_tokens_cam2] [prompt_tokens] [state_tokens] ["Action:"] [action_tokens] ["|"] [EOS]
|<------- 256 ------->|<------ 256 ------>|<---- ~20 --->|<--- ~10 ---->|<-- 2 -->|<--- ~20 --->|<1>|<1>|
```

For LIBERO with 2 cameras + prompt + state + actions:
- Camera 1: 256 tokens (16×16 patches)
- Camera 2: 256 tokens
- Prompt: ~20 tokens ("fold the towel")
- State: ~10 tokens (7 joints discretized)
- Action marker: 2 tokens ("Action: ")
- Actions: ~20 tokens (10 timesteps × 7 dims encoded by FAST)
- Delimiter: 1 token ("|")
- EOS: 1 token
- **Total**: ~566 tokens (but many configs use shorter sequences)

#### [7] Cross-Entropy Loss

```python
# Standard language modeling loss
Loss = -Σ log P(action_token_i | prefix, action_token_0, ..., action_token_{i-1})
```

**Loss is computed ONLY on action tokens**, not on prompt or state tokens.

---

## FAST Inference (Autoregressive Decoding)

Located in `src/openpi/models/pi0_fast.py:236-313`:

```python
def sample_actions(self, rng, observation, *, max_decoding_steps=256, temperature=0.0):
    # [1] Embed prefix (images + prompt + state)
    prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    
    # [2] Left-to-right align (pack to end of sequence)
    prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
        prefix_token_embeddings, prefix_mask, prefix_attn_mask
    )
    
    # [3] Prefill KV cache with prefix
    prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
    prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
    prefix_logits, kv_cache, _ = self.PaliGemma.llm(
        embedded_prefix=prefix_token_embeddings,
        mask=prefix_attn_mask,
        positions=prefix_positions,
        decode=True  # Enable KV caching
    )
    
    # [4] Decode action tokens autoregressively
    last_logit = prefix_logits[:, -1:]
    output_tokens = jnp.zeros((batch_size, max_decoding_steps))
    
    def step(carry):
        rng, last_logit, output_tokens, cache, all_eos, step = carry
        
        # Sample next token
        rng, rng_step = jax.random.split(rng)
        token = jax.lax.cond(
            temperature > 0.0,
            lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1),
            lambda _: jnp.argmax(last_logit, axis=-1),  # Greedy decoding
            operand=None,
        )
        output_tokens = put_along_last_axis(output_tokens, step, token)
        
        # Check for early stopping (EOS token)
        has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
        all_eos = jnp.all(has_eos)
        
        # Decode one step with cached KV
        token_embedding = self.PaliGemma.llm(token, embed_only=True)
        positions = prefill_len[:, None] + step + 1
        mask = create_causal_mask(...)
        last_logit, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=token_embedding,
            mask=mask,
            positions=positions,
            decode=True,
            kv_cache=cache  # Reuse cached key-values
        )
        
        return rng, last_logit, output_tokens, kv_cache, all_eos, step + 1
    
    # [5] Decode until EOS or max_decoding_steps
    def cond(carry):
        _, _, _, _, all_eos, step = carry
        return (~all_eos) & (step < max_decoding_steps)
    
    _, _, output_tokens, _, _, _ = jax.lax.while_loop(
        cond, step, (rng, last_logit, output_tokens, kv_cache, False, 0)
    )
    
    return output_tokens  # [B, max_decoding_steps] (integer tokens)
```

### Decoding Strategy

1. **Prefill**: Process entire prefix in one forward pass, cache KV states
2. **Autoregressive**: Generate tokens one at a time
   - Each step: predict next token given all previous tokens
   - Use cached KV states for efficiency (don't recompute prefix)
3. **Early Stopping**: Stop when EOS token is generated
4. **Greedy vs. Sampling**: 
   - `temperature=0`: Always pick most likely token (deterministic)
   - `temperature>0`: Sample from probability distribution (stochastic)

### KV Caching

**Without KV Caching**:
```
Step 1: Process [prefix] → predict token_1
Step 2: Process [prefix, token_1] → predict token_2
Step 3: Process [prefix, token_1, token_2] → predict token_3
...
```
Cost: O(N²) where N = sequence length

**With KV Caching**:
```
Step 0: Process [prefix] → cache KV states
Step 1: Process [token_1] with cached [prefix] → predict token_2
Step 2: Process [token_2] with cached [prefix, token_1] → predict token_3
...
```
Cost: O(N) - much faster!

### Post-Processing: Token to Action

After decoding, convert discrete tokens back to continuous actions:

```python
# Extract action tokens from generated sequence
decoded_tokens = tokenizer.decode(output_tokens)
# "Task: fold towel, State: 120 45 ...; Action: 242 189 ... | [EOS]"

# Parse action tokens
action_tokens = extract_between("Action: ", "|")
# [242, 189, 56, 231, ...]

# Decode with FAST tokenizer
actions = fast_tokenizer.decode(
    action_tokens,
    time_horizon=action_horizon,
    action_dim=action_dim
)
# Shape: [action_horizon, action_dim]
```

---

## FAST vs Pi0: Training Comparison

### Loss Functions

**Pi0 (Flow Matching)**:
```python
# MSE loss on continuous flow field
loss = MSE(predicted_flow, target_flow)
     = MSE(v_t, noise - action)
```

**Pi0 FAST (Cross-Entropy)**:
```python
# Cross-entropy loss on discrete tokens
loss = -Σ log P(token_i | token_0, ..., token_{i-1})
```

### Gradient Flow

**Pi0**:
- Gradients flow through continuous action space
- Dense supervision on all action dimensions simultaneously
- Can predict all timesteps in parallel

**Pi0 FAST**:
- Gradients flow through discrete token embeddings
- Sparse supervision (one token at a time)
- Must learn sequential dependencies

### Training Efficiency

**Pi0**:
- ✅ Parallel loss computation (all timesteps at once)
- ✅ Dense gradients on continuous space
- ❌ Requires iterative sampling at inference

**Pi0 FAST**:
- ✅ Simple language modeling objective
- ✅ Single forward pass at inference
- ❌ Sequential loss computation (autoregressive)
- ❌ Quantization error from discretization

---

## Comparison: When to Use Each

### Use Pi0 (Flow Matching) When:

1. **Precision Matters**: Need continuous, high-precision actions
2. **Multimodal Actions**: Task has multiple valid solutions (flow matching captures distribution better)
3. **Complex Dynamics**: Rich continuous dynamics benefit from continuous modeling
4. **Training Data is Limited**: Flow matching is more sample-efficient

**Example Tasks**: Dexterous manipulation, contact-rich tasks, fine-grained control

### Use Pi0 FAST (Discrete Tokens) When:

1. **Fast Inference Required**: Single forward pass vs. 10 iterative steps
2. **Interpretability Needed**: Can inspect generated token sequence
3. **Language-Like Actions**: Natural fit for discrete waypoints
4. **Simpler Training**: Standard LM training infrastructure

**Example Tasks**: Pick-and-place, navigation, high-level planning

---

## Pi0 FAST Configs Summary

### Available Configurations

```python
# 1. Full fine-tuning on LIBERO
TrainConfig(
    name="pi0_fast_libero",
    model=Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
    data=LeRobotLiberoDataConfig(...),
    weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
)

# 2. LoRA fine-tuning on LIBERO
TrainConfig(
    name="pi0_fast_libero_low_mem_finetune",
    model=Pi0FASTConfig(
        action_dim=7,
        action_horizon=10,
        max_token_len=180,
        paligemma_variant="gemma_2b_lora",
    ),
    freeze_filter=...,
)

# 3. DROID evaluation
TrainConfig(
    name="pi0_fast_droid",
    model=Pi0FASTConfig(action_dim=8, action_horizon=10),
    data=SimpleDataConfig(assets=AssetsConfig(asset_id="droid"), ...),
)

# 4. Full DROID fine-tuning
TrainConfig(
    name="pi0_fast_full_droid_finetune",
    model=Pi0FASTConfig(action_dim=8, action_horizon=16, max_token_len=180),
    data=RLDSDroidDataConfig(rlds_data_dir="<path>", ...),
    batch_size=256,
    num_train_steps=100_000,
)
```

### Hyperparameter Guidelines

**max_token_len**: Must fit entire sequence
- Count tokens: images + prompt + state + actions + markers
- Single-arm: ~180 tokens usually sufficient
- Dual-arm: ~250 tokens may be needed
- If warnings appear during training, increase this value

**action_horizon**: How many timesteps to predict
- Shorter horizons (10-16): Faster training, simpler tasks
- Longer horizons (32-50): Better long-horizon planning, slower

**action_dim**: Must match your robot
- LIBERO (single-arm): 7 (6 joints + 1 gripper)
- DROID (single-arm): 8 (7 joints + 1 gripper)
- ALOHA (dual-arm): 14 (2×7)

---

## Summary: Pi0 vs Pi0 FAST

```
┌─────────────────────────────────────────────────────────────────┐
│                         Pi0 (Standard)                           │
├─────────────────────────────────────────────────────────────────┤
│ Images → SigLIP → [Image Tokens]                                │
│ Prompt → Tokenize → [Prompt Tokens]                             │
│ State → Embed → [State Token]                                   │
│ Actions → Add Noise → [Noisy Actions] → Action Expert           │
│                                                                  │
│ Dual LLM Forward Pass:                                          │
│   LLM 0: Process [Images, Prompt]                               │
│   LLM 1: Process [State, Noisy Actions] + attend to LLM 0       │
│                                                                  │
│ Loss: MSE(predicted_flow, noise - action)                       │
│                                                                  │
│ Inference: Iterative denoising (10 steps)                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         Pi0 FAST                                 │
├─────────────────────────────────────────────────────────────────┤
│ Images → SigLIP → [Image Tokens]                                │
│ Prompt → Tokenize → [Prompt Tokens]                             │
│ State → Discretize → [State Tokens]                             │
│ Actions → FAST Tokenizer → [Action Tokens]                      │
│                                                                  │
│ Single LLM Forward Pass:                                        │
│   Process [Images, Prompt, State, "Action:", Action Tokens]     │
│   Prefix-LM attention (prefix bidirectional, suffix causal)     │
│                                                                  │
│ Loss: CrossEntropy(predicted_tokens, action_tokens)             │
│                                                                  │
│ Inference: Autoregressive decoding with KV caching              │
└─────────────────────────────────────────────────────────────────┘
```

**Trade-offs**:
- Pi0: Better for continuous control, multimodal policies
- Pi0 FAST: Faster inference, simpler training, better for discrete actions

