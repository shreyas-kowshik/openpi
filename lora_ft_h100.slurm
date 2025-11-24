#!/bin/bash
#SBATCH --job-name=openpi-loraft-vision_fullft
#SBATCH --partition=general          # partition name
#SBATCH --constraint=VRAM_80GB
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00          # time (1 day)
#SBATCH --output=/data/user_data/skowshik/openpi-loraft-debug/H100_outcome_%j.out      # Standard output file (%j is job ID)
#SBATCH --error=/data/user_data/skowshik/openpi-loraft-debug/H100_outcome_%j.out       # Standard error file (%j is job ID)

# export HF_HOME="/data/user_data/skowshik/huggingface"
# export HF_DATASETS_CACHE="/data/user_data/skowshik/huggingface"
# # uv run scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune_v2
# # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_v1 --exp-name=loramem_ft_v1 --overwrite
# # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_v2 --exp-name=loramem_ft_v2 --overwrite
# # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_v3 --exp-name=loramem_ft_v3 --overwrite
# # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_v4 --exp-name=loramem_ft_v4 --overwrite
# # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_v5 --exp-name=loramem_ft_v5 --overwrite

# # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_lora_vision_full_ft_action_full_ft_siglip --exp-name=pi05_libero_lora_vision_full_ft_action_full_ft_siglip-v1 2>&1 | tee logs/train/pi05_libero_lora_vision_full_ft_action_full_ft_siglip-v1.log

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_fullft_vision_lora_action_full_ft_siglip --exp-name=pi05_libero_fullft_vision_lora_action_full_ft_siglip-v1_fresh 2>&1 | tee logs/train/pi05_libero_fullft_vision_lora_action_full_ft_siglip-v1_fresh.log

conda run -n openpi python noop.py

