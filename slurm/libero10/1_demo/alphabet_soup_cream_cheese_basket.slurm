#!/bin/bash
#SBATCH --job-name=openpi-libero10-ep1-alphabet_soup_cream_cheese_basket
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --partition=preempt
#SBATCH --output=/data/user_data/%u/openpi-libero10/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/openpi-libero10/logs/%x_%j.err

REPO_ROOT="/data/group_data/maxlab/common_datasets/openpi"

# Load user-specific env (must define HF_HOME)
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

export HF_DATASETS_CACHE="/data/hf_cache/datasets/"

mkdir -p "/data/user_data/$USER/openpi-libero10/logs"

CONFIG_NAME="pi05_libero10_alphabet_soup_cream_cheese_basket_ep1"

cd "$REPO_ROOT"

uv run scripts/compute_norm_stats.py --config-name ${CONFIG_NAME}
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ${CONFIG_NAME} --exp-name=${CONFIG_NAME}-v1 --resume
