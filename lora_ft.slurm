#!/bin/bash
#SBATCH --job-name=openpi-loraft-debug
#SBATCH --partition=general          # partition name
#SBATCH --qos=normal            # qos
#SBATCH --gres=gpu:A6000:8
#SBATCH --cpus-per-task=8          # number of CPUs (increased)
#SBATCH --time=48:00:00          # time (1 day)
#SBATCH --mem=128G                   # 128GB total RAM (increased from 48GB)
#SBATCH --output=/data/user_data/skowshik/openpi-loraft-debug/outcome_%j.out      # Standard output file (%j is job ID)
#SBATCH --error=/data/user_data/skowshik/openpi-loraft-debug/outcome_%j.out       # Standard error file (%j is job ID)

export HF_HOME="/data/user_data/skowshik/huggingface"
export HF_DATASETS_CACHE="/data/user_data/skowshik/huggingface"
# uv run scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune_v2
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_v1 --exp-name=loramem_ft_v1 --overwrite
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_v2 --exp-name=loramem_ft_v2 --overwrite
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_v3 --exp-name=loramem_ft_v3 --overwrite
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_v4 --exp-name=loramem_ft_v4 --overwrite
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_v5 --exp-name=loramem_ft_v5 --overwrite

# pi05 training #
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_lora_vision_full_ft_action_full_ft_siglip --exp-name=pi05_libero_lora_vision_full_ft_action_full_ft_siglip-v1 --resume 2>&1 | tee logs/train/pi05_libero_lora_vision_full_ft_action_full_ft_siglip-v1.log

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_lora_vision_full_ft_action_freeze_siglip --exp-name=pi05_libero_lora_vision_full_ft_action_freeze_siglip-v1 2>&1 | tee logs/train/pi05_libero_lora_vision_full_ft_action_freeze_siglip-v1.log

# pi0 training #
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_lora_vision_full_ft_action_full_ft_siglip --exp-name=pi0_libero_lora_vision_full_ft_action_full_ft_siglip-v1 --resume

# pi0 lora ft on one task #
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_lora_moka_pots_task_ep29 --exp-name=pi0_libero_lora_moka_pots_task_ep29-v1 --resume 2>&1 | tee logs/train/pi0_libero_lora_moka_pots_task_ep29-v1.log

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_lora_moka_pots_task_ep20 --exp-name=pi0_libero_lora_moka_pots_task_ep20-v1 --resume 2>&1 | tee logs/train/pi0_libero_lora_moka_pots_task_ep20-v1.log

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_lora_moka_pots_task_ep10 --exp-name=pi0_libero_lora_moka_pots_task_ep10-v1 --resume 2>&1 | tee logs/train/pi0_libero_lora_moka_pots_task_ep10-v1.log

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_lora_moka_pots_task_ep5 --exp-name=pi0_libero_lora_moka_pots_task_ep5-v1 --resume 2>&1 | tee logs/train/pi0_libero_lora_moka_pots_task_ep5-v1.log

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_fullft_moka_pots_task_ep29 --exp-name=pi0_libero_fullft_moka_pots_task_ep29-v1 --resume 2>&1 | tee logs/train/pi0_libero_fullft_moka_pots_task_ep29-v1.log

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_fullft_moka_pots_task_ep29 --exp-name=pi0_libero_fullft_moka_pots_task_ep29-v2_ema_default --resume 2>&1 | tee logs/train/pi0_libero_fullft_moka_pots_task_ep29-v2_ema_default.log

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_fullft_moka_pots_task_ep29 --exp-name=pi0_libero_fullft_moka_pots_task_ep29-v2_ema_default --resume 2>&1 | tee logs/train/pi0_libero_fullft_moka_pots_task_ep29-v2_ema_default.log

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_fullft_moka_pots_task_ep29 --exp-name=debug-v1 --overwrite

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_vision_init_libero_lora_vision_fullft_action_onetask_ep29 --exp-name=pi05_libero_vision_init_libero_lora_vision_fullft_action_onetask_ep29-v1 2>&1 | tee logs/train/pi05_libero_vision_init_libero_lora_vision_fullft_action_onetask_ep29-v1.log

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_vision_init_libero_lora_vision_fullft_action_onetask_ep29_v2 --exp-name=pi05_libero_vision_init_libero_lora_vision_fullft_action_onetask_ep29_v2_ema-v1 2>&1 | tee logs/train/pi05_libero_vision_init_libero_lora_vision_fullft_action_onetask_ep29_v2_ema-v1.log

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_lora_moka_pots_task_ep29 --exp-name=pi0_libero_lora_moka_pots_task_ep29-v1-30k_resume 2>&1 | tee logs/train/pi0_libero_lora_moka_pots_task_ep29-v1-30k_resume.log

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_lora_book_compartment_task_ep41 --exp-name=pi0_libero_lora_book_compartment_task_ep41-v1 --resume 2>&1 | tee logs/train/pi0_libero_lora_book_compartment_task_ep41_libero_128.log

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_vision_lora_fullft_action_book_compartment_task_ep41_init_vision_pi05_libero --exp-name=pi05_libero_vision_lora_fullft_action_book_compartment_task_ep41_init_vision_pi05_libero-v1 --resume 2>&1 | tee logs/train/pi05_libero_vision_lora_fullft_action_book_compartment_task_ep41_init_vision_pi05_libero_128.log

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_lora_vision_fullft_action_init_vision_v1 --exp-name=pi05_libero_lora_vision_fullft_action_init_vision_v1-v1 --resume 2>&1 | tee logs/train/pi05_libero_lora_vision_fullft_action_init_vision_v1.log

