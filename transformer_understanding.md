# Pi0.5 Transformer Forward Pass - Detailed Explanation

This document provides a comprehensive explanation of how a forward pass happens in the Pi0.5 architecture, including input construction, attention mechanisms, and the flow through the dual-tower transformer.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Input Sequence Construction](#input-sequence-construction)
3. [Forward Pass Through Transformer](#forward-pass-through-transformer)
4. [Attention Mechanism](#attention-mechanism)
5. [AdaRMSNorm (Pi0.5 Specific)](#adarmsnorm-pi05-specific)
6. [Output Projection](#output-projection)
7. [Complete Flow Diagram](#complete-flow-diagram)

---

## Architecture Overview

Pi0.5 uses a **dual-tower transformer architecture** with:
- **Tower 1 (PaliGemma)**: Gemma 2B - processes vision and language inputs
- **Tower 2 (Action Expert)**: Gemma 300M - processes noisy action tokens
- **Shared attention**: Both towers share the same attention mechanism (queries, keys, values are concatenated)
- **AdaRMSNorm**: Adaptive normalization conditioned on diffusion timestep (Pi0.5 only)

### Key Components

```python
# From pi0.py lines 73-100
self.PaliGemma = nnx.Dict(
    llm=_gemma.Module([paligemma_config, action_expert_config], adarms=True),
    img=_siglip.Module(...)
)
self.action_in_proj = nnx.Linear(action_dim, 1024)  # Projects actions to expert space
self.time_mlp_in = nnx.Linear(1024, 1024)           # Time embedding MLP (Pi0.5)
self.time_mlp_out = nnx.Linear(1024, 1024)          # Time embedding MLP (Pi0.5)
self.action_out_proj = nnx.Linear(1024, action_dim) # Projects back to action space
```

---

## Input Sequence Construction

### Step 1: Prefix Embedding (`embed_prefix`)

The prefix contains vision and language inputs that provide context.

**Code**: `pi0.py` lines 105-137

```python
def embed_prefix(self, obs: Observation):
    tokens = []
    input_mask = []
    ar_mask = []
    
    # 1. Process images through SigLIP vision encoder
    for name in obs.images:
        image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
        # Shape: [batch, 256, 2048]  (256 image patches)
        tokens.append(image_tokens)
        input_mask.append(obs.image_masks[name])
        ar_mask += [False] * 256  # Full bidirectional attention
    
    # 2. Embed tokenized text prompt through Gemma embedder
    if obs.tokenized_prompt is not None:
        tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        # Shape: [batch, prompt_len, 2048]
        tokens.append(tokenized_inputs)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * prompt_len  # Full bidirectional attention
    
    # Concatenate all prefix tokens
    tokens = jnp.concatenate(tokens, axis=1)  # [batch, prefix_len, 2048]
    return tokens, input_mask, ar_mask
```

**Prefix Structure** (for a 3-camera setup):
```
[IMG_1: 256 tokens] [IMG_2: 256 tokens] [IMG_3: 256 tokens] [PROMPT: ~50 tokens]
Total: ~818 tokens, all with bidirectional attention (ar_mask=False)
```

### Step 2: Suffix Embedding (`embed_suffix`)

The suffix contains action tokens that will be denoised.

**Code**: `pi0.py` lines 139-186

```python
def embed_suffix(self, obs, noisy_actions, timestep):
    # Pi0.5 does NOT include state as a separate token
    # State is discretized and included in the prompt instead
    
    # 1. Project noisy actions to expert embedding space
    action_tokens = self.action_in_proj(noisy_actions)
    # Shape: [batch, action_horizon, 1024]
    
    # 2. Create timestep embedding using sinusoidal encoding
    time_emb = posemb_sincos(timestep, 1024, min_period=4e-3, max_period=4.0)
    # Shape: [batch, 1024]
    
    # 3. Process time embedding through MLP (Pi0.5 specific)
    time_emb = self.time_mlp_in(time_emb)
    time_emb = nnx.swish(time_emb)
    time_emb = self.time_mlp_out(time_emb)
    time_emb = nnx.swish(time_emb)
    # Shape: [batch, 1024]
    
    # For Pi0.5: action_tokens go directly to transformer
    # time_emb is passed as adarms_cond for adaptive normalization
    action_expert_tokens = action_tokens
    adarms_cond = time_emb
    
    # First action token has causal mask, rest have bidirectional
    ar_mask = [True] + [False] * (action_horizon - 1)
    
    return action_tokens, input_mask, ar_mask, adarms_cond
```

**Suffix Structure**:
```
[ACTION_1] [ACTION_2] ... [ACTION_10]
Total: 10 tokens (for horizon=10), causal attention for first, bidirectional for rest
```

**Key Difference: Pi0 vs Pi0.5**

| Aspect | Pi0 | Pi0.5 |
|--------|-----|-------|
| **State token** | Separate token in suffix | Discretized in prompt |
| **Time conditioning** | Concatenated with actions, MLP processes both | AdaRMSNorm in each transformer layer |
| **Suffix tokens** | [STATE] + [ACTION_1...ACTION_50] | [ACTION_1...ACTION_10] |
| **Action horizon** | 50 | 10 |

---

## Forward Pass Through Transformer

### Step 1: Combine Prefix and Suffix

**Code**: `pi0.py` lines 202-211

```python
def compute_loss(self, rng, observation, actions, train=False):
    # ... noise and time sampling ...
    
    # Embed inputs
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
    
    # Combine masks
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    
    # Create attention mask
    attn_mask = make_attn_mask(input_mask, ar_mask)
    # Shape: [batch, total_len, total_len]
    
    # Create position indices
    positions = jnp.cumsum(input_mask, axis=1) - 1
    # Shape: [batch, total_len]
    
    # Forward pass through dual-tower transformer
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],      # List of inputs for each tower
        mask=attn_mask,                      # Attention mask
        positions=positions,                  # Position indices for RoPE
        adarms_cond=[None, adarms_cond]      # Pi0.5: timestep conditioning for action expert
    )
```

**Combined Sequence**:
```
┌────────────────── Prefix (Tower 1) ──────────────────┐ ┌─ Suffix (Tower 2) ─┐
[IMG_1] [IMG_2] [IMG_3] [PROMPT] [STATE_TOKENS*] [ACT_1] [ACT_2] ... [ACT_10]
  256     256     256      ~50         ~14             10 action tokens
                                   (*discretized in prompt)

Total sequence length: ~828 tokens
```

### Step 2: Attention Mask Construction

**Code**: `pi0.py` lines 19-44

```python
def make_attn_mask(input_mask, mask_ar):
    """
    Creates prefix-LM attention mask:
    - Prefix tokens (ar_mask=False): full bidirectional attention
    - Suffix tokens (ar_mask=True/False): causal or bidirectional based on mask
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)
```

**Attention Pattern**:
```
Query\Key  IMG_1  IMG_2  IMG_3  PROMPT  ACT_1  ACT_2  ACT_3  ...  ACT_10
─────────────────────────────────────────────────────────────────────────
IMG_1       ✓      ✓      ✓      ✓       ✗      ✗      ✗    ...   ✗
IMG_2       ✓      ✓      ✓      ✓       ✗      ✗      ✗    ...   ✗
IMG_3       ✓      ✓      ✓      ✓       ✗      ✗      ✗    ...   ✗
PROMPT      ✓      ✓      ✓      ✓       ✗      ✗      ✗    ...   ✗
ACT_1       ✓      ✓      ✓      ✓       ✓      ✗      ✗    ...   ✗     (Causal: ar_mask=True)
ACT_2       ✓      ✓      ✓      ✓       ✓      ✓      ✗    ...   ✗     (Bidir: ar_mask=False)
ACT_3       ✓      ✓      ✓      ✓       ✓      ✓      ✓    ...   ✗     (Bidir: ar_mask=False)
...
ACT_10      ✓      ✓      ✓      ✓       ✓      ✓      ✓    ...   ✓     (Bidir: ar_mask=False)

✓ = Can attend
✗ = Cannot attend
```

**Key Property**: Prefix tokens cannot attend to suffix tokens (to prevent information leakage), but suffix tokens can attend to all prefix tokens.

---

## Transformer Forward Pass Details

### Dual-Tower Processing

**Code**: `gemma.py` lines 433-455

```python
def __call__(self, embedded, positions, mask, adarms_cond=None, kv_cache=None, deterministic=True):
    # embedded: [prefix_tokens, suffix_tokens]
    # adarms_cond: [None, time_emb]  (for Pi0.5)
    
    embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
    mask = jnp.asarray(mask)[:, None, :, :]  # Add head dimension
    
    if adarms_cond is None:
        adarms_cond = [None] * len(self.configs)
    
    # Pass through 18 transformer layers
    embedded, kv_cache = self.layers(
        embedded,           # [prefix_tokens, suffix_tokens]
        kv_cache,          # None during training, populated during inference
        positions,         # Position indices for RoPE
        mask,              # Attention mask
        adarms_cond,       # [None, time_emb] for Pi0.5
        deterministic
    )
    
    # Apply final layer norm
    return [
        f(e, a)[0] if e is not None else e 
        for f, e, a in zip(self.final_norms, embedded, adarms_cond)
    ], kv_cache
```

### Single Transformer Block

**Code**: `gemma.py` lines 336-377

Each of the 18 layers processes both towers simultaneously:

```python
def Block.__call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):
    # xs = [prefix_tokens, suffix_tokens]
    # adarms_cond = [None, time_emb] for Pi0.5
    
    # ─────────────────────────────────────────────────────
    # 1. PRE-ATTENTION NORMALIZATION (with AdaRMS for Pi0.5)
    # ─────────────────────────────────────────────────────
    pre_attn = []
    gates = []
    for i, x in enumerate(xs):
        if x is not None:
            # RMSNorm with optional adaptive modulation
            x, gate = RMSNorm()(x, adarms_cond[i])
            # For action expert in Pi0.5: x is modulated by time_emb
        pre_attn.append(x)
        gates.append(gate)
    
    # ─────────────────────────────────────────────────────
    # 2. MULTI-HEAD ATTENTION (Shared across both towers)
    # ─────────────────────────────────────────────────────
    post_attn, kv_cache = Attention()(pre_attn, positions, attn_mask, kv_cache)
    
    # ─────────────────────────────────────────────────────
    # 3. RESIDUAL CONNECTION (with optional gating from AdaRMS)
    # ─────────────────────────────────────────────────────
    xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates)]
    # If gate is not None: xs = xs + post_attn * gate
    # Otherwise: xs = xs + post_attn
    
    # ─────────────────────────────────────────────────────
    # 4. PRE-FFN NORMALIZATION (with AdaRMS for Pi0.5)
    # ─────────────────────────────────────────────────────
    out = []
    gates = []
    for i, (x, config) in enumerate(zip(xs, self.configs)):
        if x is not None:
            x, gate = RMSNorm()(x, adarms_cond[i])
            x = FeedForward(config.width, config.mlp_dim)(x)
        out.append(x)
        gates.append(gate)
    
    # ─────────────────────────────────────────────────────
    # 5. RESIDUAL CONNECTION (with optional gating)
    # ─────────────────────────────────────────────────────
    xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates)]
    
    return xs, kv_cache
```

---

## AdaRMSNorm (Pi0.5 Specific)

AdaRMSNorm (Adaptive RMS Normalization) is a key innovation in Pi0.5 that conditions each transformer layer on the diffusion timestep.

### How It Works

**Code**: `gemma.py` lines 157-176

```python
class RMSNorm(nn.Module):
    def __call__(self, x, cond):
        # Compute RMS normalization
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = x * jnp.reciprocal(jnp.sqrt(var + 1e-06))
        
        if cond is None:
            # Regular RMSNorm (for PaliGemma tower)
            scale = self.param("scale", ...)
            normed_inputs = normed_inputs * (1 + scale)
            return normed_inputs.astype(dtype), None
        
        # Adaptive RMSNorm (for Action Expert in Pi0.5)
        # Project time_emb to 3 * hidden_dim
        modulation = nn.Dense(x.shape[-1] * 3)(cond)  # [batch, 3072]
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        # scale, shift, gate: each [batch, 1, 1024]
        
        # Apply adaptive modulation
        normed_inputs = normed_inputs * (1 + scale) + shift
        return normed_inputs.astype(dtype), gate
```

### AdaRMS Flow

```
Input: x [batch, seq_len, 1024], time_emb [batch, 1024]

Step 1: RMS Normalization
  x_norm = x / sqrt(mean(x²) + ε)

Step 2: Time-dependent Modulation
  modulation = Dense(3072)(time_emb)  → [batch, 3072]
  scale, shift, gate = split(modulation, 3)  → each [batch, 1, 1024]

Step 3: Adaptive Transformation
  output = x_norm * (1 + scale) + shift

Step 4: Gated Residual (used later)
  residual = input + attn_output * gate
```

**Why AdaRMS?**
1. **Per-layer timestep conditioning**: Each layer can adapt differently based on noise level
2. **Memory efficient**: No need to concatenate time with actions
3. **Inspired by DiT**: Proven successful in image diffusion models
4. **Dynamic modulation**: Scale and shift vary per timestep, allowing fine-grained control

---

## Attention Mechanism

### Multi-Head Attention with Dual Towers

**Code**: `gemma.py` lines 208-293

```python
def Attention.__call__(self, xs, positions, attn_mask, kv_cache):
    # xs = [prefix_tokens, suffix_tokens]
    
    # ──────────────────────────────────────────────────────
    # 1. COMPUTE Q, K, V FOR EACH TOWER
    # ──────────────────────────────────────────────────────
    qkvs = []
    for i, (x, config) in enumerate(zip(xs, self.configs)):
        if x is None:
            continue
        
        # Separate Q, K, V projections (with optional LoRA)
        q = q_einsum("BTD,NDH->BTNH", x)       # [batch, seq_len, 8, 256]
        k, v = kv_einsum("BSD,2KDH->2BSKH", x) # [batch, seq_len, 1, 256] each
        qkvs.append((q, k, v))
    
    # ──────────────────────────────────────────────────────
    # 2. CONCATENATE Q, K, V FROM BOTH TOWERS
    # ──────────────────────────────────────────────────────
    q = jnp.concatenate([q1, q2], axis=1)  # [batch, total_len, 8, 256]
    k = jnp.concatenate([k1, k2], axis=1)  # [batch, total_len, 1, 256]
    v = jnp.concatenate([v1, v2], axis=1)  # [batch, total_len, 1, 256]
    
    # ──────────────────────────────────────────────────────
    # 3. APPLY RoPE (ROTARY POSITION EMBEDDING)
    # ──────────────────────────────────────────────────────
    q = _apply_rope(q, positions=positions)
    k = _apply_rope(k, positions=positions)
    q *= head_dim ** -0.5  # Scale for dot product
    
    # ──────────────────────────────────────────────────────
    # 4. COMPUTE ATTENTION SCORES
    # ──────────────────────────────────────────────────────
    # Grouped Query Attention (GQA): 8 query heads, 1 kv head
    q = rearrange(q, "B T (K G) H -> B T K G H", K=1)  # [batch, T, 1, 8, 256]
    logits = einsum("BTKGH,BSKH->BKGTS", q, k)          # [batch, 1, 8, T, S]
    
    # Apply attention mask
    masked_logits = where(attn_mask, logits, -2.38e38)
    probs = softmax(masked_logits, axis=-1)
    
    # ──────────────────────────────────────────────────────
    # 5. COMPUTE ATTENTION OUTPUT
    # ──────────────────────────────────────────────────────
    encoded = einsum("BKGTS,BSKH->BTKGH", probs, v)
    encoded = rearrange(encoded, "B T K G H -> B T (K G) H")  # [batch, T, 8, 256]
    
    # ──────────────────────────────────────────────────────
    # 6. OUTPUT PROJECTION (SEPARATE FOR EACH TOWER)
    # ──────────────────────────────────────────────────────
    out = []
    start = 0
    for i, (x, config) in enumerate(zip(xs, self.configs)):
        if x is not None:
            end = start + x.shape[1]
            # Project back to hidden dimension
            out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
            # PaliGemma: [batch, prefix_len, 2048]
            # Action Expert: [batch, suffix_len, 1024]
            start = end
        else:
            out.append(None)
    
    return out, (k, v)
```

### Key Features

**Grouped Query Attention (GQA)**:
- 8 query heads, 1 key/value head
- Reduces memory and computation
- Equivalent to Multi-Query Attention (MQA)

**RoPE (Rotary Position Embedding)**:
- Applied to queries and keys
- Encodes relative positions
- Better extrapolation than absolute position embeddings

**Dual Tower with Shared Attention**:
- Q, K, V are concatenated from both towers
- Attention is computed over the full sequence
- Output is split back to each tower

---

## Output Projection

After the transformer processes all 18 layers:

**Code**: `pi0.py` line 212

```python
def compute_loss(self, ...):
    # ... forward pass through transformer ...
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(...)
    
    # Extract action predictions from suffix output
    # Only take the last action_horizon tokens (10 for Pi0.5)
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
    # Input: [batch, 10, 1024]
    # Output: [batch, 10, 7] (for 7-dim action space in Libero)
    
    # Compute loss (flow matching objective)
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

**Output Shapes**:
```
suffix_out:  [batch, suffix_len, 1024]  (from action expert)
                     ↓ (select last action_horizon tokens)
             [batch, 10, 1024]
                     ↓ (action_out_proj)
v_t:         [batch, 10, 7]  (predicted velocity)
```

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INPUT PREPARATION                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
         ┌──────────────────┐          ┌─────────────────────┐
         │  embed_prefix    │          │  embed_suffix       │
         │                  │          │                     │
         │ • Vision (SigLIP)│          │ • action_in_proj    │
         │ • Text (Gemma)   │          │ • time_mlp (Pi0.5)  │
         │                  │          │                     │
         │ Output:          │          │ Output:             │
         │ [B,~818,2048]    │          │ [B,10,1024]         │
         │ ar_mask: all 0   │          │ ar_mask: [1,0,..,0] │
         └──────────────────┘          │ adarms_cond: [B,1024]
                    │                   └─────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ATTENTION MASK CONSTRUCTION                          │
│                                                                           │
│  make_attn_mask(input_mask, ar_mask)                                    │
│  • Prefix tokens: bidirectional attention                                │
│  • Suffix tokens: causal for first, bidirectional for rest              │
│  • Prefix-to-suffix: blocked                                             │
│  • Suffix-to-prefix: allowed                                             │
│                                                                           │
│  Output: [B, total_len, total_len]                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DUAL-TOWER TRANSFORMER (18 LAYERS)                     │
│                                                                           │
│  For each layer:                                                         │
│    ┌─────────────────────────────────────────────────────────┐         │
│    │ 1. Pre-Attention RMSNorm                                 │         │
│    │    • PaliGemma: Regular RMSNorm                         │         │
│    │    • Action Expert: AdaRMSNorm (conditioned on time)    │         │
│    │                                                          │         │
│    │    scale, shift, gate = Dense(3*d)(time_emb)            │         │
│    │    x_norm = x_norm * (1 + scale) + shift               │         │
│    └─────────────────────────────────────────────────────────┘         │
│                          │                                               │
│                          ▼                                               │
│    ┌─────────────────────────────────────────────────────────┐         │
│    │ 2. Multi-Head Attention (Shared)                        │         │
│    │    • Q, K, V computed for each tower                    │         │
│    │    • Concatenate Q, K, V from both towers               │         │
│    │    • Apply RoPE position encoding                       │         │
│    │    • Compute attention over full sequence               │         │
│    │    • Split output back to each tower                    │         │
│    │                                                          │         │
│    │    Attention pattern:                                   │         │
│    │    ┌───────────┬──────────┐                            │         │
│    │    │  Prefix   │  Suffix  │                            │         │
│    │    ├───────────┼──────────┤                            │         │
│    │    │  Prefix   │    ✓     │    ✗     │                            │         │
│    │    │  Suffix   │    ✓     │   ✓/✗    │  (depends on ar_mask)     │         │
│    │    └───────────┴──────────┘                            │         │
│    └─────────────────────────────────────────────────────────┘         │
│                          │                                               │
│                          ▼                                               │
│    ┌─────────────────────────────────────────────────────────┐         │
│    │ 3. Residual Connection (with optional gating)           │         │
│    │    If gate exists:  x = x + attn_out * gate            │         │
│    │    Otherwise:       x = x + attn_out                   │         │
│    └─────────────────────────────────────────────────────────┘         │
│                          │                                               │
│                          ▼                                               │
│    ┌─────────────────────────────────────────────────────────┐         │
│    │ 4. Pre-FFN RMSNorm (with AdaRMS for action expert)     │         │
│    └─────────────────────────────────────────────────────────┘         │
│                          │                                               │
│                          ▼                                               │
│    ┌─────────────────────────────────────────────────────────┐         │
│    │ 5. Feed-Forward Network                                 │         │
│    │    • PaliGemma: 2048 → 16384 → 2048                    │         │
│    │    • Action Expert: 1024 → 4096 → 1024                 │         │
│    │    • GeGLU activation                                   │         │
│    └─────────────────────────────────────────────────────────┘         │
│                          │                                               │
│                          ▼                                               │
│    ┌─────────────────────────────────────────────────────────┐         │
│    │ 6. Residual Connection (with optional gating)           │         │
│    └─────────────────────────────────────────────────────────┘         │
│                                                                           │
│  Repeat 18 times                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           FINAL LAYER NORM                                │
│                                                                           │
│  final_norms[0](prefix_out, None)          → [B, prefix_len, 2048]      │
│  final_norms[1](suffix_out, adarms_cond)   → [B, suffix_len, 1024]      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT PROJECTION                                │
│                                                                           │
│  Extract last action_horizon tokens from suffix_out                      │
│  suffix_out[:, -10:]  →  [B, 10, 1024]                                  │
│                 ↓                                                         │
│  action_out_proj: Linear(1024 → 7)                                      │
│                 ↓                                                         │
│  v_t: [B, 10, 7]  (predicted velocity for flow matching)                │
│                                                                           │
│  Loss = MSE(v_t, u_t)  where u_t = noise - actions                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary: Pi0 vs Pi0.5 Key Differences

| Component | Pi0 | Pi0.5 |
|-----------|-----|-------|
| **State Input** | Separate token in suffix | Discretized in prompt (prefix) |
| **Action Horizon** | 50 | 10 |
| **Time Conditioning** | Concatenated with actions, MLP | AdaRMSNorm in each layer |
| **Suffix Tokens** | 51 tokens (1 state + 50 actions) | 10 tokens (actions only) |
| **Sequence Length** | ~869 tokens | ~828 tokens |
| **Memory** | Higher (longer suffix) | Lower (shorter suffix) |
| **Time Embedding** | Processed once via MLP | Used in every layer via AdaRMS |
| **Expressiveness** | Time info fixed after MLP | Time modulates each layer dynamically |

---

## Inference with KV Caching

During inference (sampling), Pi0.5 uses KV caching for efficiency:

**Code**: `pi0.py` lines 233-278

```python
def sample_actions(self, rng, observation, num_steps=10):
    # Step 1: Fill KV cache with prefix (only once)
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], ...)
    
    # Step 2: Iterative denoising with cached prefix
    def step(carry):
        x_t, time = carry
        suffix_tokens, ..., adarms_cond = self.embed_suffix(observation, x_t, time)
        
        # Use cached prefix KV, only compute suffix
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],  # None = skip prefix computation
            kv_cache=kv_cache,      # Reuse prefix KV
            adarms_cond=[None, adarms_cond]
        )
        
        v_t = self.action_out_proj(suffix_out[:, -action_horizon:])
        return x_t + dt * v_t, time + dt
    
    # Iterative refinement from noise to clean actions
    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0
```

**Efficiency Gains**:
- Prefix processed once: ~818 tokens
- Each denoising step: only 10 tokens
- With 10 steps: 818 + 10*10 = 918 tokens vs 918*10 = 9180 tokens (11x speedup)

---

## Conclusion

The Pi0.5 architecture cleverly combines:
1. **Dual-tower transformers** for processing context and actions separately
2. **AdaRMSNorm** for dynamic time-dependent modulation
3. **Prefix-LM attention** for efficient context-action interaction
4. **Flow matching** for action denoising
5. **KV caching** for fast inference

This design achieves strong performance on robotic manipulation tasks while being memory-efficient and fast at inference time.



