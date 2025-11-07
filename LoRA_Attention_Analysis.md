# LoRA Application to Attention Weights: Analysis

## Quick Answer

**YES**, setting `lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0)}` **WILL apply LoRA to ALL weight matrices** in the attention mechanism: **Q, K, V, and O (output projection)**.

---

## Code References

### 1. Attention Definition

**File**: `src/openpi/models/gemma.py`

The `Attention` class is defined starting at **line 158**:

```python
@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # ... implementation
```

### 2. Q, K, V Projections with LoRA

**Location**: `src/openpi/models/gemma.py:176-199`

There are two cases depending on whether multi-query attention (MQA) is used:

#### Case 1: Standard Multi-Head Attention (num_kv_heads == num_heads)

```python
# Lines 176-183
if config.num_kv_heads == config.num_heads:
    qkv_einsum = lora.Einsum(
        shape=(3, config.num_heads, config.width, config.head_dim),
        name=_name("qkv_einsum", i),
        init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
        lora_config=config.lora_configs.get("attn"),  # ← LoRA applied here!
    )
    qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
```

**Key Point**: The shape `(3, ...)` means Q, K, V are computed **together** in one einsum operation. The `lora_config=config.lora_configs.get("attn")` applies LoRA to all three projections simultaneously.

#### Case 2: Multi-Query Attention (num_kv_heads < num_heads)

```python
# Lines 185-199
else:
    # Q projection with LoRA
    q_einsum = lora.Einsum(
        shape=(config.num_heads, config.width, config.head_dim),
        name=_name("q_einsum", i),
        init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
        lora_config=config.lora_configs.get("attn"),  # ← LoRA applied to Q
    )
    q = q_einsum("BTD,NDH->BTNH", x)
    
    # K, V projection with LoRA
    kv_einsum = lora.Einsum(
        shape=(2, config.num_kv_heads, config.width, config.head_dim),
        name=_name("kv_einsum", i),
        init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
        lora_config=config.lora_configs.get("attn"),  # ← LoRA applied to K, V
    )
    k, v = kv_einsum("BSD,2KDH->2BSKH", x)
    qkvs.append((q, k, v))
```

**Key Point**: Q and KV are computed separately, but **both** use `lora_config=config.lora_configs.get("attn")`.

### 3. O (Output) Projection with LoRA

**Location**: `src/openpi/models/gemma.py:238-244`

After computing attention, the output projection is also LoRA-enabled:

```python
# Lines 238-244
out_einsum = lora.Einsum(
    shape=(config.num_heads, config.head_dim, config.width),
    name=_name("attn_vec_einsum", i),
    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
    lora_config=config.lora_configs.get("attn"),  # ← LoRA applied to output projection!
)
out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
```

**Key Point**: The output projection (which projects from attention heads back to model dimension) **also** uses the same LoRA config.

---

## How LoRA is Applied

### LoRA Einsum Implementation

**File**: `src/openpi/models/lora.py:33-85`

```python
class Einsum(nn.Module):
    """Einsum with LoRA support."""

    shape: tuple[int, ...]
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    lora_config: LoRAConfig | None = None

    def setup(self):
        # Base weight (frozen during LoRA training)
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # LoRA low-rank matrices
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank  # Reduce dimension to rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    def __call__(self, eqn: str, x):
        dtype = x.dtype
        # Base computation (with frozen weights)
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            # LoRA low-rank adaptation: x @ W_A @ W_B
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            result = result + lora * config.scaling_value  # Add LoRA contribution
        
        return result
```

### LoRA Mathematics

For each weight matrix W:

```
Output = x @ W_base + (x @ W_A @ W_B) * (alpha / rank)
         └─frozen──┘   └────trainable────┘

Where:
- W_base: Original pre-trained weights (frozen)
- W_A: Low-rank matrix (rank × d_model) - trainable
- W_B: Low-rank matrix (d_model × rank) - trainable
- alpha: Scaling factor (typically equal to rank)
- rank: LoRA rank (e.g., 32)
```

---

## Summary: What Gets LoRA'd

When you set:
```python
lora_configs = {
    "attn": lora.LoRAConfig(rank=32, alpha=32.0),
    "ffn": lora.LoRAConfig(rank=32, alpha=32.0)
}
```

### For `"attn"` Key:

| Weight Matrix | LoRA Applied? | Code Location | Shape |
|--------------|---------------|---------------|-------|
| **Q (Query)** | ✅ YES | `gemma.py:177-183` or `185-191` | `(num_heads, width, head_dim)` |
| **K (Key)** | ✅ YES | `gemma.py:177-183` or `192-198` | `(num_kv_heads, width, head_dim)` |
| **V (Value)** | ✅ YES | `gemma.py:177-183` or `192-198` | `(num_kv_heads, width, head_dim)` |
| **O (Output)** | ✅ YES | `gemma.py:238-244` | `(num_heads, head_dim, width)` |

**All four** attention weight matrices are LoRA-adapted!

### For `"ffn"` Key:

| Weight Matrix | LoRA Applied? | Code Location |
|--------------|---------------|---------------|
| **Gate** | ✅ YES | `gemma.py:319-323` |
| **Up** | ✅ YES | `gemma.py:319-323` |
| **Down** | ✅ YES | `gemma.py:319-323` |

---

## Parameter Count Analysis

### Example: Gemma 2B with LoRA

Assuming:
- `width = 2048` (model dimension)
- `num_heads = 16`
- `head_dim = 128`
- `rank = 32`
- `alpha = 32.0`

#### Without LoRA (Full Fine-tuning)

**Attention weights per layer**:
```
Q: 2048 × 2048 = 4,194,304 params
K: 2048 × 2048 = 4,194,304 params
V: 2048 × 2048 = 4,194,304 params
O: 2048 × 2048 = 4,194,304 params
────────────────────────────────
Total: 16,777,216 params (16.8M)
```

#### With LoRA (rank=32)

**LoRA parameters per layer**:
```
Q_lora: (2048 × 32) + (32 × 2048) = 131,072 params
K_lora: (2048 × 32) + (32 × 2048) = 131,072 params
V_lora: (2048 × 32) + (32 × 2048) = 131,072 params
O_lora: (2048 × 32) + (32 × 2048) = 131,072 params
─────────────────────────────────────────────────
Total: 524,288 params (0.52M)

Reduction: 16.8M → 0.52M (96.9% reduction!)
```

**For entire Gemma 2B** (18 layers):
```
Full fine-tuning: ~2B params
LoRA (rank=32):   ~9.4M params (attention only)
                  ~18.8M params (attention + FFN)
```

---

## Practical Example

### Configuration

```python
from openpi.models import lora, pi0_config

# Define LoRA config
lora_config = lora.LoRAConfig(
    rank=32,
    alpha=32.0,
    rslora=False,  # Use standard LoRA scaling
    init_fn=nn.initializers.normal(stddev=0.01),
)

# Create model config with LoRA
model_config = pi0_config.Pi0Config(
    paligemma_variant="gemma_2b_lora",        # Uses LoRA
    action_expert_variant="gemma_300m_lora",  # Uses LoRA
)
```

### Under the Hood

The Gemma config builder (`src/openpi/models/gemma.py:58-109`) automatically sets:

```python
# For "gemma_2b_lora" variant
config = Config(
    # ... other params ...
    lora_configs={
        "attn": LoRAConfig(rank=32, alpha=32.0),
        "ffn": LoRAConfig(rank=32, alpha=32.0),
    }
)
```

This config is then passed to every transformer block, and each block applies LoRA to:
1. All attention weights (Q, K, V, O)
2. All feed-forward weights (gate, up, down)

### Freeze Filter

When training with LoRA, you must freeze the base weights:

```python
# From src/openpi/models/pi0_config.py:79-108
def get_freeze_filter(self):
    return nnx.All(
        nnx_utils.PathRegex(".*llm.*"),           # Match all LLM params
        nnx.Not(nnx_utils.PathRegex(".*lora.*"))  # Except LoRA params
    )
```

This ensures:
- ✅ `w_a` and `w_b` (LoRA matrices) are trainable
- ❌ `w` (base weights) are frozen

---

## Verification: Check Your Model

To verify LoRA is applied correctly, you can inspect the parameters:

```python
import jax
from openpi.training import config as _config

# Load config
cfg = _config.get_config("pi0_libero_low_mem_finetune")

# Initialize model
rng = jax.random.key(0)
model = cfg.model.create(rng)

# Check parameter names
import flax.nnx as nnx
params = nnx.state(model)
param_names = [k for k in traverse_util.flatten_dict(params).keys()]

# Look for LoRA parameters
lora_params = [k for k in param_names if "lora" in str(k)]
print(f"Found {len(lora_params)} LoRA parameters")

# Example output:
# ('PaliGemma', 'llm', 'layers_0', 'attn', 'qkv_einsum', 'lora_a')
# ('PaliGemma', 'llm', 'layers_0', 'attn', 'qkv_einsum', 'lora_b')
# ('PaliGemma', 'llm', 'layers_0', 'attn', 'attn_vec_einsum', 'lora_a')
# ('PaliGemma', 'llm', 'layers_0', 'attn', 'attn_vec_einsum', 'lora_b')
# ... (for all Q, K, V, O in all layers)
```

---

## Conclusion

✅ **YES**, setting `lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0)}` applies LoRA to:

1. ✅ **Q** (Query projection)
2. ✅ **K** (Key projection)
3. ✅ **V** (Value projection)
4. ✅ **O** (Output projection)

All four weight matrices in the attention mechanism receive low-rank adaptation, reducing trainable parameters by ~97% while maintaining most of the model's capacity.

**Code Locations**:
- Attention definition: `src/openpi/models/gemma.py:158-249`
- LoRA Einsum: `src/openpi/models/lora.py:33-85`
- LoRA Config: `src/openpi/models/lora.py:11-30`

