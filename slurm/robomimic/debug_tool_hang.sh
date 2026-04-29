#!/bin/bash
#SBATCH --job-name=pi05_toolhang
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --constraint=L40S
#SBATCH --output=/data/user_data/skowshik/pi05_robomimic_exps/logs/pi05_toolhang_%j.out
#SBATCH --error=/data/user_data/skowshik/pi05_robomimic_exps/logs/pi05_toolhang_%j.err

# =============================================================================
# Robomimic: Pi-0.5 fine-tuning on ToolHang
# LoRA on VLM (gemma_2b_lora), full fine-tune on action head
# Dataset: 200 demos from robomimic tool_hang ph | 2 x L40S GPUs (FSDP)
# Checkpoints: /data/hf_cache/models/pi05_robomimic_exps/
#
# Steps:
#   1. Compute normalization statistics
#   2. Train the policy
#
# Usage:
#   sbatch slurm/robomimic/debug_tool_hang.sh
#
# Config: pi05_robomimic_tool_hang
#   - HDF5: /data/hf_cache/datasets/robomimic/tool_hang/ph/image_224_v15.hdf5
#   - 200 demos (num_episodes=-1 for all), action_dim=7, action_horizon=10
#   - Task: "hang the tool on the rack"
#
# Monitor:
#   squeue -u skowshik
#   tail -f /data/user_data/skowshik/pi05_robomimic_exps/logs/pi05_toolhang_<JOBID>.out
# =============================================================================

set -e

CONFIG_NAME="pi05_robomimic_tool_hang"
EXP_NAME=${EXP_NAME:-pi05_toolhang_200demo}

echo "======================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURMD_NODENAME}"
echo "GPUs:   ${SLURM_JOB_GPUS}"
echo "Config: ${CONFIG_NAME}"
echo "Exp:    ${EXP_NAME}"
echo "======================================================"

# ---- Environment ----
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export EGL_DEVICE_ID=0
export MUJOCO_EGL_DEVICE_ID=0

# wandb: set project env var so wandb picks it up even if CLI parsing fails
export WANDB_PROJECT=pi05_robomimic
export WANDB_ENTITY=skowshik-carnegie-mellon-university

# Use the openpi source from this repo
export PYTHONPATH=/home/skowshik/vla/codebase/openpi/src:${PYTHONPATH}

# Activate conda (use eval to support non-interactive shells in SLURM)
source /data/user_data/skowshik/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate openpi

cd /home/skowshik/vla/codebase/openpi

echo "Python: $(which python)"
echo "JAX devices: $(python -c 'import jax; print(jax.device_count(), jax.devices())')"

# ---- Step 1: Compute norm stats ----
echo ""
echo "======================================================"
echo "Step 1: Computing normalization statistics"
echo "======================================================"
python scripts/compute_norm_stats.py --config-name="${CONFIG_NAME}"

# ---- Step 2: Training ----
echo ""
echo "======================================================"
echo "Step 2: Training"
echo "======================================================"
python scripts/train.py "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --project-name=pi05_robomimic \
    --wandb-entity=skowshik-carnegie-mellon-university \
    --overwrite

echo "Training complete."
