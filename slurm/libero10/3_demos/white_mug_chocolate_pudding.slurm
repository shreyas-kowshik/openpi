#!/bin/bash
#SBATCH --job-name=openpi-libero10-ep3-white_mug_chocolate_pudding
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --partition=preempt
#SBATCH --output=/data/user_data/%u/openpi-libero10/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/openpi-libero10/logs/%x_%j.err

# Repo root = directory from which sbatch was called
REPO_DIR="${SLURM_SUBMIT_DIR}"

export PATH="$HOME/.local/bin:$PATH"

if [[ -f "$REPO_DIR/.env" ]]; then
    set -a
    source "$REPO_DIR/.env"
    set +a
fi

export HF_DATASETS_CACHE="/data/hf_cache/datasets/"

mkdir -p "/data/user_data/$USER/openpi-libero10/logs"

CONFIG_NAME="pi05_libero10_white_mug_chocolate_pudding_ep3"

cd "$REPO_DIR"

uv run scripts/compute_norm_stats.py --config-name ${CONFIG_NAME}
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ${CONFIG_NAME} --exp-name=${CONFIG_NAME}-v1 --resume
