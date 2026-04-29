#!/bin/bash
# Debug script for sanity-checking ep86 filtered training on a single GPU.
# Run from the repo root: bash slurm/libero10/debug/both_mokapots_stove_ep1_selected_ep86.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

export PATH="$HOME/.local/bin:$PATH"

if [[ -f "$REPO_DIR/.env" ]]; then
    set -a
    source "$REPO_DIR/.env"
    set +a
fi

export HF_DATASETS_CACHE="/data/hf_cache/datasets/"
export CUDA_VISIBLE_DEVICES=0

CONFIG_NAME="pi05_libero10_both_mokapots_stove_ep1_ep86_filtered"

cd "$REPO_DIR"

uv run scripts/compute_norm_stats.py --config-name ${CONFIG_NAME}
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ${CONFIG_NAME} --exp-name=${CONFIG_NAME}-debug --resume
