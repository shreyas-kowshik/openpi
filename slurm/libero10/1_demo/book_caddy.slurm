#!/bin/bash
#SBATCH --job-name=openpi-libero10-ep1-book_caddy
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --partition=preempt
#SBATCH --exclude=babel-l5-16,babel-u5-28
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

CONFIG_NAME="pi05_libero10_book_caddy_ep1"

cd /home/skowshik/vla/codebase/openpi

uv run scripts/compute_norm_stats.py --config-name ${CONFIG_NAME}
cleanup() {
    echo "$(date): Caught signal, letting training save checkpoint..."
    kill -TERM $TRAIN_PID 2>/dev/null
    wait $TRAIN_PID
}
trap cleanup USR1 SIGTERM

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ${CONFIG_NAME} --exp-name=${CONFIG_NAME}-v1 --resume &
TRAIN_PID=$!
wait $TRAIN_PID
