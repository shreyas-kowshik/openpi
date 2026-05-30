#!/bin/bash
#SBATCH --job-name=openpi-libero10-full_data-both_mokapots_stove_ep1_init_action_head_only
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=array
#SBATCH --qos=array_qos
#SBATCH --array=0
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

CONFIG_NAME="pi05_libero10_both_mokapots_stove_ep1_init_full_data_action_head_only_hdf5"

cd "$REPO_DIR"

uv run scripts/compute_norm_stats.py --config-name ${CONFIG_NAME}
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ${CONFIG_NAME} --exp-name=${CONFIG_NAME}-v1 --wandb-entity=skowshik-carnegie-mellon-university --resume
