#!/bin/bash
#SBATCH --job-name=openpi-libero10-cream_cheese_butter_basket
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=/data/user_data/skowshik/openpi-libero10/logs/outcome_%j.out
#SBATCH --error=/data/user_data/skowshik/openpi-libero10/logs/outcome_%j.err

export HF_HOME="/data/user_data/skowshik/huggingface"
export HF_DATASETS_CACHE="/data/user_data/skowshik/huggingface"
export WANDB_ENTITY="skowshik-carnegie-mellon-university"

mkdir -p /data/user_data/skowshik/openpi-libero10/logs

echo "Starting training on $(hostname)"
echo "Running at $(date)"
echo $SLURM_JOB_ID
scontrol show job $SLURM_JOB_ID | egrep -i "gres|tres|partition|nodes"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

nvidia-smi

CONFIG_NAME="pi05_libero10_cream_cheese_butter_basket_ep5"

cd /home/skowshik/vla/codebase/openpi

uv run scripts/compute_norm_stats.py --config-name ${CONFIG_NAME}
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ${CONFIG_NAME} --exp-name=${CONFIG_NAME}-v1 --resume
