#!/bin/bash
#SBATCH --job-name=openpi-libero10-clear_clutter-alphabet_soup_tomato_sauce_basket_ep1
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --array=0
#SBATCH --output=/data/user_data/%u/openpi-libero10/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/openpi-libero10/logs/%x_%j.err

set -e

CONFIG_NAME="pi05_libero10_alphabet_soup_tomato_sauce_basket_ep1_clear_clutter"
EXP_NAME="${CONFIG_NAME}-v1"

echo "Job ID: ${SLURM_JOB_ID} | Node: ${SLURMD_NODENAME} | Config: ${CONFIG_NAME}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_ENTITY=skowshik-carnegie-mellon-university
export PYTHONPATH=/home/skowshik/vla/codebase/openpi/src:${PYTHONPATH}
export HF_DATASETS_CACHE="/data/hf_cache/datasets/"

source /data/user_data/skowshik/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate openpi

cd /home/skowshik/vla/codebase/openpi
mkdir -p "/data/user_data/$USER/openpi-libero10/logs"

python scripts/compute_norm_stats.py --config-name="${CONFIG_NAME}"
python scripts/train.py "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --resume
