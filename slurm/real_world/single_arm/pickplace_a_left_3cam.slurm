#!/bin/bash
#SBATCH --job-name=yam-pickplace-a-left-3cam
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/%u/pi05_real_world/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/pi05_real_world/logs/%x_%j.err

# Repo root = directory from which sbatch was called
REPO_DIR="${SLURM_SUBMIT_DIR}"

export PATH="$HOME/.local/bin:$PATH"

if [[ -f "$REPO_DIR/.env" ]]; then
    set -a
    source "$REPO_DIR/.env"
    set +a
fi

export HF_DATASETS_CACHE="/data/hf_cache/datasets/"

mkdir -p "/data/user_data/$USER/pi05_real_world/logs"

CONFIG_NAME="pi05_yam_pickplace_a_left_3cam_lora"

cd "$REPO_DIR"

uv run scripts/compute_norm_stats.py --config-name ${CONFIG_NAME}
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ${CONFIG_NAME} --exp-name=${CONFIG_NAME}-v1 --wandb-entity=skowshik-carnegie-mellon-university --resume
