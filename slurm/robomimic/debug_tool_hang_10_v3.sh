#!/bin/bash
#SBATCH --job-name=pi05_toolhang10v3
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --partition=preempt
#SBATCH --output=/data/user_data/%u/pi05_robomimic_exps/logs/pi05_toolhang10v3_%j.out
#SBATCH --error=/data/user_data/%u/pi05_robomimic_exps/logs/pi05_toolhang10v3_%j.err

# =============================================================================
# Robomimic: Pi-0.5 fine-tuning on ToolHang (10 demos) - v3 (peak_lr=1e-6)
# LoRA on VLM (gemma_2b_lora), full fine-tune on action head
# Dataset: 10 demos from robomimic tool_hang ph | 2 x L40S GPUs (FSDP)
# Checkpoints: /data/hf_cache/models/pi05_robomimic_exps/
#
# Steps:
#   1. Compute normalization statistics
#   2. Train the policy
#
# Usage:
#   sbatch slurm/robomimic/debug_tool_hang_10_v3.sh
#
# Config: pi05_robomimic_tool_hang_10demo_v3
#   - HDF5: /data/hf_cache/datasets/robomimic/tool_hang/ph/image_224_v15.hdf5
#   - 10 demos, action_dim=7, action_horizon=10
#   - Task: "hang the tool on the rack"
#
# Monitor:
#   squeue -u skowshik
#   tail -f /data/user_data/skowshik/pi05_robomimic_exps/logs/pi05_toolhang10v3_<JOBID>.out
# =============================================================================

REPO_DIR="${SLURM_SUBMIT_DIR}"

export PATH="$HOME/.local/bin:$PATH"

if [[ -f "$REPO_DIR/.env" ]]; then
    set -a
    source "$REPO_DIR/.env"
    set +a
fi

export HF_DATASETS_CACHE="/data/hf_cache/datasets/"

mkdir -p "/data/user_data/$USER/pi05_robomimic_exps/logs"

CONFIG_NAME="pi05_robomimic_tool_hang_10demo_v3"

cd "$REPO_DIR"

uv run scripts/compute_norm_stats.py --config-name ${CONFIG_NAME}
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ${CONFIG_NAME} --exp-name=pi05_toolhang_10demo_v3 --project-name=pi05_robomimic --wandb-entity=skowshik-carnegie-mellon-university --resume
