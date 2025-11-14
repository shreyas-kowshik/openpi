# Fine tuning

Specify configuration in `openpi/src/openpi/training/config.py`
For LoRA FT: `pi0_libero_low_mem_finetune`
This seems like a good entry point to understand code

Compute dataset stats
```
export HF_HOME="/data/user_data/skowshik/huggingface"
export HF_DATASETS_CACHE="/data/user_data/skowshik/huggingface"
export HF_TOKEN=
uv run scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune
```

Pi0.5 in general is a 2.3B model

Start fine tuning
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune --exp-name=debug-v2 --overwrite
```

Debug
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_debug --exp-name=debug-v3
```

LoRA

LoRA configs defined in `src/openpi/models/gemma.py`
Gemma VLM, Gemma based transformer for diffusion, both have LoRA layers which are adapted

Default LoRA setting
```
if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
```

Optimizer details can be updated in `src/openpi/training/optimizer.py`
Default parameters
```
@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6
```

All LoRA *AND* attention computation happens in `models/lora.py`

Default `batch_size=32`
This might not be optimal for LoRA, check ThinkingMachines blog for this

# LoRA V2 training experiments on Libero
Default hyperparameters, train longer (100k steps), Unfreeze final layer full/curriculum, Switch off LoRA for VLM/keep present

```
cp -r /data/user_data/skowshik/openpi_cache/libero_custom_lora_ft/assets/pi0_libero_low_mem_finetune_v1 /data/user_data/skowshik/openpi_cache/libero_custom_lora_ft/assets/pi05_libero_low_mem_finetune_V2_v1_unfreeze_action_out_proj

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_low_mem_finetune_V2_v1_unfreeze_action_out_proj --exp-name=debug-v4 --overwrite
```


LoRA + Full FT ablations
Full FT action layers + LoRA gemma layers + Full FT SigLIP
```
cp -r /data/user_data/skowshik/openpi_cache/libero_custom_lora_ft/assets/pi0_libero_low_mem_finetune_v1 /data/user_data/skowshik/openpi_cache/libero_custom_lora_ft/assets/pi05_libero_lora_vision_full_ft_action_full_ft_siglip

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_lora_vision_full_ft_action_full_ft_siglip --exp-name=pi05_libero_lora_vision_full_ft_action_full_ft_siglip-v1
```

Above + Freeze SigLIP
```
cp -r /data/user_data/skowshik/openpi_cache/libero_custom_lora_ft/assets/pi0_libero_low_mem_finetune_v1 /data/user_data/skowshik/openpi_cache/libero_custom_lora_ft/assets/pi05_libero_lora_vision_full_ft_action_freeze_siglip
```

```
cp -r /data/user_data/skowshik/openpi_cache/libero_custom_lora_ft/assets/pi0_libero_low_mem_finetune_v1 /data/user_data/skowshik/openpi_cache/libero_custom_lora_ft/assets/pi05_libero_fullft_vision_lora_action_full_ft_siglip
```

