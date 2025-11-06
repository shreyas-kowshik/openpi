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

Start fine tuning
```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune --exp-name=debug-v1 --overwrite
```
