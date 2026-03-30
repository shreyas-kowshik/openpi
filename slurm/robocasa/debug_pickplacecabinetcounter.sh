#!/bin/bash
#SBATCH --job-name=pi05_cab2ctr
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --constraint=L40S
#SBATCH --output=/data/user_data/skowshik/pi05_robocasa_exps/logs/pi05_cab2ctr_%j.out
#SBATCH --error=/data/user_data/skowshik/pi05_robocasa_exps/logs/pi05_cab2ctr_%j.err

# =============================================================================
# RoboCasa: Pi-0.5 single-task fine-tuning on PickPlaceCabinetToCounter
# LoRA on VLM (gemma_2b_lora), full fine-tune on action head
# Scene: match_episode_id=53 (marshmallow, layout=46, style=15, 1 demo) | 2 x L40S GPUs (FSDP)
# Checkpoints: /data/hf_cache/models/pi05_robocasa_exps/
#
# Steps:
#   1. Compute normalization statistics (from the single matched demo)
#   2. Train the policy
#
# Usage:
#   sbatch slurm/robocasa/debug_pickplacecabinetcounter.sh
#
# Config: pi05_robocasa_single_task_lora_vision_fullft_action_cabinet2counter_ep_meta_debug_v1
#   - match_episode_id=53 (full ep_meta match: layout=46, style=15,
#     fixture_refs=singlecabinet_2_main_group_1/counter_2_main_group_1,
#     objects=marshmallow+baguette+hot_dog). No occlusion throughout episode.
#   - Dataset: /data/hf_cache/datasets/robocasa/v1.0/target/atomic/PickPlaceCabinetToCounter/20250819/lerobot
#
# Monitor:
#   squeue -u skowshik
#   tail -f /data/user_data/skowshik/pi05_robocasa_exps/logs/pi05_cab2ctr_<JOBID>.out
# =============================================================================

set -e

CONFIG_NAME="pi05_robocasa_single_task_lora_vision_fullft_action_cabinet2counter_ep_meta_debug_v1"
EXP_NAME=${EXP_NAME:-pi05_pickplace_cab2ctr_ep53_marshmallow_1demo}

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
export WANDB_PROJECT=pi05_robocasa
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
    --project-name=pi05_robocasa \
    --wandb-entity=skowshik-carnegie-mellon-university \
    --overwrite

echo "Training complete."
