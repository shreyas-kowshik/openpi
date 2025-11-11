#!/bin/bash
#SBATCH --job-name=openpi-loraft-debug
#SBATCH --partition=general          # partition name
#SBATCH --qos=normal            # qos
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=16          # number of CPUs (increased)
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
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_v4 --exp-name=loramem_ft_v4 --overwrite
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_v5 --exp-name=loramem_ft_v5 --overwrite

