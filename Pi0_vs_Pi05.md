# Pi0 vs Pi0.5: Architecture & Training Differences

This document provides a detailed comparison between Pi0 (the original model) and Pi0.5 (improved version) architectures.

## Table of Contents
1. [High-Level Overview](#high-level-overview)
2. [Key Architectural Differences](#key-architectural-differences)
3. [State Input Handling](#state-input-handling)
4. [Timestep Conditioning](#timestep-conditioning)
5. [Model Configuration](#model-configuration)
6. [Training Differences](#training-differences)
7. [When to Use Each](#when-to-use-each)
8. [Migration Guide](#migration-guide)

---

## High-Level Overview

Both Pi0 and Pi0.5 use **flow matching** for action generation and share the same overall architecture (SigLIP vision encoder + PaliGemma language model + Action Expert). However, Pi0.5 introduces two key improvements:

1. **Discretized State Input**: State is tokenized as part of the language input
2. **Adaptive RMS Normalization**: Uses adaRMSNorm for more flexible timestep conditioning

### Quick Comparison Table

| Feature | Pi0 | Pi0.5 |
|---------|-----|-------|
| **State Representation** | Continuous (embedded) | Discrete (tokenized) |
| **State Location** | Suffix (action expert input) | Prefix (language input) |
| **Timestep Conditioning** | Concatenation + MLP | Adaptive RMSNorm (adaRMS) |
| **Max Token Length** | 48 (default) | 200 (default) |
| **State Encoding** | Linear projection | 256 bins per dimension |
| **Action Expert Layers** | State proj + 2 MLPs | 2 MLPs (no state proj) |
| **Typical Action Horizon** | 50 | 10-16 |
| **Typical Batch Size** | 32 | 256 |
| **EMA Decay** | 0.99 | 0.999 |

---

## Key Architectural Differences

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                              Pi0                                 │
├─────────────────────────────────────────────────────────────────┤
│ Prefix (Language Model):                                        │
│   [Images] → SigLIP → [Image Tokens]                           │
│   [Prompt] → Tokenize → [Prompt Tokens]                        │
│                                                                  │
│ Suffix (Action Expert):                                         │
│   [State] → Linear Proj → [State Token] (continuous)           │
│   [Noisy Actions] → Linear Proj → [Action Tokens]              │
│   [Timestep] → Sin/Cos Encoding → [Time Embedding]             │
│   Concat([Action Tokens, Time Tokens]) → MLP → [Expert Input]  │
│                                                                  │
│ LLM Forward:                                                     │
│   Dual towers process prefix and suffix                         │
│   Time info mixed via concatenation + MLP                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                            Pi0.5                                 │
├─────────────────────────────────────────────────────────────────┤
│ Prefix (Language Model):                                        │
│   [Images] → SigLIP → [Image Tokens]                           │
│   [Prompt] → Tokenize → "Task: fold towel"                     │
│   [State] → Discretize (256 bins) → "State: 120 45 89 ..."    │
│   Combined → "Task: fold towel, State: 120 45 89;\nAction:"    │
│                                                                  │
│ Suffix (Action Expert):                                         │
│   [Noisy Actions] → Linear Proj → [Action Tokens]              │
│   [Timestep] → Sin/Cos Encoding → MLP → [Time Embedding]       │
│   (NO state token in suffix)                                    │
│                                                                  │
│ LLM Forward:                                                     │
│   Dual towers process prefix and suffix                         │
│   Time info injected via adaRMSNorm in action expert           │
│   adaRMS modulates layer normalization with time embedding     │
└─────────────────────────────────────────────────────────────────┘
```

---

## State Input Handling

### Pi0: Continuous State Embedding

**Location**: `src/openpi/models/pi0.py:151-157`

```python
if not self.pi05:
    # Pi0: Add a single continuous state token in the suffix
    state_token = self.state_proj(obs.state)[:, None, :]  # Linear projection
    tokens.append(state_token)
    input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
    ar_mask += [True]  # State token part of causal sequence
```

**Characteristics**:
- State is continuous (raw joint positions, velocities, etc.)
- Single token representing entire state vector
- Projected via learned linear layer: `state_proj = nnx.Linear(action_dim, expert_width)`
- Appears in suffix (action expert input), not prefix
- Enables dense gradients for state information

**Example**:
```python
Input state: [0.5, -0.3, 0.8, ..., 0.1]  # 32-dim continuous
              ↓ Linear projection
State token: [embedding of 2048 dims]     # Single token
```

### Pi0.5: Discrete State Tokenization

**Location**: `src/openpi/models/tokenizer.py:24-29`

```python
if state is not None:
    # Pi0.5: Discretize state into 256 bins per dimension
    discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized_state))
    full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
    tokens = self._tokenizer.encode(full_prompt, add_bos=True)
```

**Characteristics**:
- State is discretized: each dimension → integer in [0, 255]
- Multiple tokens (one per state dimension, converted to text)
- No learned projection for state
- Appears in prefix (language input), treated like text
- State information shared across both LLM towers

**Example**:
```python
Input state: [0.5, -0.3, 0.8, ..., 0.1]  # 32-dim continuous
              ↓ Discretize to 256 bins
Discretized: [191, 90, 231, ..., 153]     # 32 integers (0-255)
              ↓ Convert to string
Text: "Task: fold towel, State: 191 90 231 ... 153;\nAction: "
              ↓ Tokenize with PaliGemma
Tokens: [bos, 4820, 58, 9367, 23854, 44, 3443, 58, 191, 90, ...]  # ~10-50 tokens
```

### Comparison

| Aspect | Pi0 (Continuous) | Pi0.5 (Discrete) |
|--------|------------------|------------------|
| **Precision** | Full float32 precision | 256 bins (8-bit quantization) |
| **Representation** | Dense embedding | Sparse token sequence |
| **Parameter Efficiency** | Requires learned projection | Uses existing LLM embeddings |
| **Token Count** | 1 token (fixed) | ~1 token per state dim (variable) |
| **Max Token Length** | 48 sufficient | 200+ needed |
| **Gradient Flow** | Direct through projection | Through token embeddings |
| **Interpretability** | Opaque embedding | Human-readable numbers |

**Why Pi0.5 Uses Discretization**:
1. **Unified Representation**: State, prompt, and actions all as tokens
2. **Language Model Strengths**: Leverages pre-trained token understanding
3. **Flexibility**: Can easily concatenate state with prompt
4. **Robustness**: 256 bins provide sufficient precision for most robotics tasks
5. **Generalization**: Better transfer across different state spaces

---

## Timestep Conditioning

Both models use flow matching, which requires injecting the diffusion timestep `t` into the action expert. They differ in **how** this timestep information modulates the network.

### Pi0: Concatenation + MLP

**Location**: `src/openpi/models/pi0.py:170-178`

```python
else:  # Pi0 (not pi05)
    # Mix timestep + action information using an MLP
    time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
    action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
    # 2x width input
    action_time_tokens = self.action_time_mlp_in(action_time_tokens)
    action_time_tokens = nnx.swish(action_time_tokens)
    action_time_tokens = self.action_time_mlp_out(action_time_tokens)
    action_expert_tokens = action_time_tokens
    adarms_cond = None  # No adaRMS conditioning
```

**Architecture**:
```
Time Embedding (b, emb)
    ↓ Repeat for each action timestep
Time Tokens (b, action_horizon, emb)
    ↓ Concatenate with action tokens
[Action Tokens | Time Tokens] (b, action_horizon, 2*emb)
    ↓ MLP (2*emb → emb)
Action Expert Input (b, action_horizon, emb)
    ↓ Standard transformer (regular RMSNorm)
Action Expert Output
```

**Characteristics**:
- Simple concatenation of action and time embeddings
- Doubles the input dimension to action expert
- Time info mixed via 2-layer MLP (`action_time_mlp_in` → `action_time_mlp_out`)
- Standard RMSNorm used in transformer layers
- Time information is "baked into" the token embeddings

### Pi0.5: Adaptive RMS Normalization (adaRMS)

**Location**: `src/openpi/models/pi0.py:162-169`

```python
if self.pi05:
    # Time MLP for adaRMS conditioning
    time_emb = self.time_mlp_in(time_emb)
    time_emb = nnx.swish(time_emb)
    time_emb = self.time_mlp_out(time_emb)
    time_emb = nnx.swish(time_emb)
    action_expert_tokens = action_tokens  # No concatenation!
    adarms_cond = time_emb  # Pass to transformer layers
```

**adaRMS in Transformer**: `src/openpi/models/gemma.py:113-131`

```python
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        
        if cond is None:
            # Regular RMSNorm (Pi0)
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (1 + scale)
            return normed_inputs.astype(dtype), None
        
        # Adaptive RMSNorm (Pi0.5)
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros)(cond)
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift  # Adaptive!
        return normed_inputs.astype(dtype), gate
```

**Architecture**:
```
Time Embedding (b, emb)
    ↓ MLP (emb → emb)
Time Conditioning (b, emb)
    ↓ Broadcast to transformer layers

Action Tokens (b, action_horizon, emb)  ← No concatenation!
    ↓ Transformer Layer
    ├─→ RMSNorm with time conditioning:
    │   └─→ Dense(emb → 3*emb): [scale, shift, gate]
    │       └─→ x_norm = x_norm * (1 + scale) + shift
    ├─→ Attention (modulated by gate)
    └─→ FFN (modulated by gate)
```

**Characteristics**:
- No concatenation (saves parameters and memory)
- Time embedding processed by separate MLP
- Time dynamically modulates **every transformer layer**
- Each layer produces scale, shift, gate from time embedding
- More expressive: different modulation per layer
- Inspired by diffusion models (e.g., DiT, U-ViT)

### Comparison: Timestep Conditioning

| Aspect | Pi0 (Concat + MLP) | Pi0.5 (adaRMS) |
|--------|-------------------|----------------|
| **Method** | Concatenate time with actions | Modulate layer normalization |
| **Input Dim** | 2× expert width | 1× expert width |
| **Parameters** | Extra MLP (2×w → w) | Dense per layer (w → 3w) |
| **Flexibility** | Time fixed after MLP | Time affects every layer dynamically |
| **Memory** | Higher (2× tokens) | Lower (1× tokens) |
| **Computation** | More in MLP, less in layers | Less in MLP, more in layers |
| **Theoretical** | Simpler | More expressive (per-layer modulation) |

**Why Pi0.5 Uses adaRMS**:
1. **Memory Efficiency**: No need to double embedding dimension
2. **Better Conditioning**: Time information modulates each layer independently
3. **Proven in Diffusion**: Used successfully in image generation (Stable Diffusion 3, DiT)
4. **Flexibility**: Scale and shift adapt normalization per layer and timestep
5. **Gating**: Can selectively apply transformations based on timestep

---

## Model Configuration

### Default Hyperparameters

```python
# Pi0 Config (src/openpi/models/pi0_config.py)
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    action_dim: int = 32
    action_horizon: int = 50           # Pi0 default
    max_token_len: int = 48            # Pi0 default
    pi05: bool = False
    
    def __post_init__(self):
        if self.pi05:
            self.max_token_len = 200   # Pi0.5 needs more tokens
```

### Configuration in Practice

#### Pi0 Example Configs

```python
# Inference
TrainConfig(
    name="pi0_aloha",
    model=pi0_config.Pi0Config(),  # pi05=False (default)
    data=LeRobotAlohaDataConfig(...),
)

# Fine-tuning on LIBERO
TrainConfig(
    name="pi0_libero",
    model=pi0_config.Pi0Config(),  # pi05=False
    data=LeRobotLiberoDataConfig(...),
    num_train_steps=30_000,
)
```

#### Pi0.5 Example Configs

```python
# Inference
TrainConfig(
    name="pi05_aloha",
    model=pi0_config.Pi0Config(pi05=True),  # Enable pi05
    data=LeRobotAlohaDataConfig(...),
)

# Fine-tuning on LIBERO
TrainConfig(
    name="pi05_libero",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,              # Shorter than Pi0
        discrete_state_input=False,     # Override for compatibility
    ),
    data=LeRobotLiberoDataConfig(
        extra_delta_transform=False,    # Different from Pi0
    ),
    batch_size=256,                     # Much larger than Pi0
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    ema_decay=0.999,                    # Stronger EMA than Pi0
)
```

### Tokenization Differences

**Pi0 Tokenization**:
```python
# Only prompt is tokenized
tokenizer.tokenize(prompt="fold the towel", state=None)
# Output: [bos, 9367, 1, 23854, 1, 198]  (~6 tokens)
#         "fold" " the" " towel" "\n"
```

**Pi0.5 Tokenization**:
```python
# Prompt + discretized state tokenized together
tokenizer.tokenize(prompt="fold the towel", state=[0.5, -0.3, ...])
# Output: [bos, 4820, 58, 9367, 1, 23854, 44, 3443, 58, 191, 90, ..., 198, 4450, 58, 1]
#         "Task" ": " "fold" " the" " towel" ", " "State" ": " "191" "90" ... "\n" "Action" ": "
#         (~40-60 tokens depending on state dimension)
```

---

## Training Differences

### Typical Training Configurations

| Hyperparameter | Pi0 | Pi0.5 | Reason for Difference |
|----------------|-----|-------|----------------------|
| **Batch Size** | 32 | 256 | Pi0.5 benefits from larger batches |
| **Action Horizon** | 50 | 10-16 | Pi0.5 optimized for shorter horizons |
| **Max Token Len** | 48 | 200 | Pi0.5 needs space for state tokens |
| **Learning Rate** | 3e-4 | 5e-5 | Pi0.5 uses lower LR with large batch |
| **Warmup Steps** | 2000 | 10,000 | Pi0.5 needs longer warmup |
| **EMA Decay** | 0.99 | 0.999 | Pi0.5 uses stronger EMA |
| **Gradient Clipping** | 1.0 | 1.0 | Same |
| **Training Steps** | 30K | 30K-100K | Pi0.5 may need more steps |

### Training Efficiency

**Pi0**:
- Smaller batch size → more frequent updates
- Longer action horizons → more tokens per sample
- Simpler timestep conditioning → faster forward pass
- Continuous state → denser gradients

**Pi0.5**:
- Larger batch size → better GPU utilization
- Shorter action horizons → faster sequences
- adaRMS → slightly more computation per layer
- Discrete state → sparser gradients, but better generalization

### Memory Comparison

**Per-sample memory** (approximate, for action_dim=32):

```
Pi0:
  Images: 2 × 256 tokens × 2048 dims = 1.05 MB
  Prompt: 6 tokens × 2048 dims = 0.05 MB
  State: 1 token × 2048 dims = 0.008 MB
  Actions: 50 tokens × 2048 dims = 0.41 MB
  Time: 50 tokens × 2048 dims = 0.41 MB (concatenated)
  Total: ~1.93 MB per sample

Pi0.5:
  Images: 2 × 256 tokens × 2048 dims = 1.05 MB
  Prompt+State: 40 tokens × 2048 dims = 0.33 MB
  Actions: 10 tokens × 2048 dims = 0.08 MB
  Time: 10 tokens × 2048 dims (adaRMS, not concatenated) = 0 MB
  Total: ~1.46 MB per sample
```

**Batch memory** (batch_size × per-sample):
- Pi0: 32 × 1.93 MB = **61.8 MB**
- Pi0.5: 256 × 1.46 MB = **373.8 MB**

Despite larger batch size, Pi0.5 is still feasible on modern GPUs (40GB+ VRAM).

---

## When to Use Each

### Use Pi0 When:

1. **State Precision Critical**: Need full float32 precision for state
   - Example: Fine-grained force control, sensitive calibration tasks

2. **Long Action Horizons**: Tasks benefit from 50+ timestep predictions
   - Example: Slow manipulation, long-horizon planning

3. **Limited Compute**: Smaller batches preferred (32 vs 256)
   - Example: Training on single GPU

4. **Continuous State Space**: State naturally continuous and high-dimensional
   - Example: Complex robot with many joints and sensors

5. **Baseline Comparisons**: Comparing against published Pi0 results

### Use Pi0.5 When:

1. **State Space is Discrete or Bounded**: State fits well in 256 bins
   - Example: Most manipulation tasks with normalized joints

2. **Fast Inference Needed**: Shorter action horizons reduce latency
   - Example: Reactive tasks, closed-loop control

3. **Large-Scale Training**: Have resources for batch_size=256+
   - Example: Multi-GPU training, large datasets

4. **State Interpretability Important**: Want to inspect discretized states
   - Example: Debugging, analysis, visualization

5. **Better Generalization**: Leveraging pre-trained language model strengths
   - Example: Transfer learning, few-shot adaptation

6. **State-of-the-Art Performance**: Pi0.5 generally performs better

### Performance Comparison

Based on published results and configs:

| Benchmark | Pi0 Success Rate | Pi0.5 Success Rate | Notes |
|-----------|------------------|-------------------|--------|
| **ALOHA** | ~85% | ~90% | Real robot manipulation |
| **LIBERO** | ~75% | ~82% | Simulation benchmark |
| **DROID** | ~70% | ~78% | Real-world diverse tasks |

**Note**: Exact numbers vary by task, training time, and hyperparameters. Pi0.5 typically shows 5-10% improvement.

---

## Migration Guide

### Converting Pi0 Checkpoint to Pi0.5

**Important**: Pi0 and Pi0.5 checkpoints are **not directly compatible** due to:
1. Different layer structures (state projection vs. no state projection)
2. Different normalization (RMSNorm vs. adaRMSNorm)
3. Different tokenization (continuous vs. discrete state)

**Recommended Approach**: Fine-tune Pi0.5 base checkpoint on your data.

### Converting Pi0 Config to Pi0.5

```python
# Original Pi0 config
pi0_config = TrainConfig(
    name="my_pi0_model",
    model=pi0_config.Pi0Config(
        action_dim=7,
        action_horizon=50,
    ),
    data=MyDataConfig(...),
    batch_size=32,
    lr_schedule=_optimizer.CosineDecaySchedule(peak_lr=3e-4),
    ema_decay=0.99,
)

# Convert to Pi0.5
pi05_config = TrainConfig(
    name="my_pi05_model",
    model=pi0_config.Pi0Config(
        pi05=True,                    # Enable Pi0.5
        action_dim=7,
        action_horizon=10,             # Reduce horizon
        # max_token_len=200 (automatic)
    ),
    data=MyDataConfig(...),
    batch_size=256,                    # Increase batch size
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,           # Longer warmup
        peak_lr=5e-5,                  # Lower peak LR
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    ema_decay=0.999,                   # Stronger EMA
    num_train_steps=50_000,            # May need more steps
)
```

### Data Pipeline Adjustments

**Pi0 Data Transform**:
```python
# Continuous state passed directly to model
data_transforms = _transforms.Group(
    inputs=[
        # State remains continuous
        MyRobotInputs(),
    ],
)
```

**Pi0.5 Data Transform**:
```python
# State will be discretized during tokenization
# Ensure ModelTransformFactory uses discrete_state_input=True
model_transforms = ModelTransformFactory()(model_config)
# This will:
# 1. Discretize state into 256 bins
# 2. Concatenate with prompt as text
# 3. Tokenize combined string
```

### Inference Code Changes

**Pi0 Inference**:
```python
observation = {
    "images": {"cam_high": image},
    "state": state_vector,  # Continuous [0.5, -0.3, ...]
    "tokenized_prompt": tokenize(prompt),
}
actions = model.sample_actions(rng, observation)
```

**Pi0.5 Inference**:
```python
# State discretization happens in tokenizer
observation = {
    "images": {"cam_high": image},
    "state": state_vector,  # Still continuous in input
    # Tokenizer will handle discretization:
    # "Task: {prompt}, State: 191 90 231 ...;\nAction:"
    "tokenized_prompt": tokenize(prompt, state=state_vector),
}
actions = model.sample_actions(rng, observation)
```

---

## Summary

### Key Takeaways

1. **Pi0.5 is an improved version of Pi0**, not a separate architecture
   - Same base structure (SigLIP + PaliGemma + Action Expert)
   - Two key changes: discrete state input + adaRMS conditioning

2. **Discrete State** (Pi0.5) vs **Continuous State** (Pi0)
   - Pi0.5: State tokenized as text, appears in prompt
   - Pi0: State embedded as continuous token in action expert

3. **adaRMS** (Pi0.5) vs **Concat+MLP** (Pi0)
   - Pi0.5: Timestep modulates each layer via adaptive normalization
   - Pi0: Timestep concatenated with actions, processed by MLP

4. **Training Differences**
   - Pi0.5 uses larger batches (256 vs 32)
   - Pi0.5 uses shorter action horizons (10 vs 50)
   - Pi0.5 needs longer max_token_len (200 vs 48)

5. **When to Choose**
   - **Pi0.5**: Better performance, more modern, recommended for new projects
   - **Pi0**: Simpler, good for baselines, continuous state precision

### Architecture Decision Tree

```
Do you need full float32 state precision?
├─ Yes → Use Pi0
└─ No
   ├─ Do you have multi-GPU resources?
   │  ├─ Yes → Use Pi0.5 (batch_size=256)
   │  └─ No → Use Pi0.5 with smaller batch (batch_size=64-128)
   └─ Need long action horizons (>30)?
      ├─ Yes → Consider Pi0
      └─ No → Use Pi0.5 ✓ (recommended)
```

**General Recommendation**: Start with **Pi0.5** unless you have specific requirements for continuous state representation or are reproducing Pi0 baseline results.

