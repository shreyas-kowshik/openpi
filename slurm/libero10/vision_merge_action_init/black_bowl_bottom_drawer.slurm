#!/bin/bash
#SBATCH --job-name=openpi-vision-merge-black-bowl
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --array=0
#SBATCH --exclude=babel-m5-32,babel-p9-28,babel-o9-24,babel-n9-32,babel-p5-24,babel-v9-32,babel-q5-16,babel-t9-28,babel-v5-32,babel-z5-16,babel-z5-20,babel-z5-28,babel-n5-20
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

CONFIG_NAME="pi05_libero10_black_bowl_bottom_drawer_ep1_VISION_MERGE_ACTION_INIT"

cd "$REPO_DIR"

uv run scripts/compute_norm_stats.py --config-name ${CONFIG_NAME}
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ${CONFIG_NAME} --exp-name=${CONFIG_NAME}-v1 --wandb-entity=skowshik-carnegie-mellon-university --resume
