#!/bin/bash
#SBATCH --job-name=openpi-loraft-debug
#SBATCH --partition=general          # partition name
#SBATCH --qos=normal            # qos
#SBATCH --gres=gpu:A6000:4
#SBATCH --cpus-per-task=8          # number of CPUs (increased)
#SBATCH --time=48:00:00          # time (1 day)
#SBATCH --mem=128G                   # 128GB total RAM (increased from 48GB)
#SBATCH --output=/data/user_data/skowshik/openpi-loraft-debug/outcome_%j.out      # Standard output file (%j is job ID)
#SBATCH --error=/data/user_data/skowshik/openpi-loraft-debug/outcome_%j.out       # Standard error file (%j is job ID)

export HF_HOME="/data/user_data/skowshik/huggingface"
export HF_DATASETS_CACHE="/data/user_data/skowshik/huggingface"

uv run scripts/compute_norm_stats.py --config-name pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep10_bs32_v1_icml
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep10_bs32_v1_icml --exp-name=pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep10_bs32_v1_icml-v1 --resume 2>&1 | tee logs/train/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep10_bs32_v1_icml.log


