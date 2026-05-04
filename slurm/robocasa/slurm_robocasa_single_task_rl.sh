#!/bin/bash
#SBATCH --job-name=pi05_robocasa_single_task
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=rl
#SBATCH --qos=rl_qos
#SBATCH --constraint=RTX_PRO_6000
#SBATCH --output=/data/user_data/skowshik/robocasa_logs/logs/pi05_single_task_%j.out
#SBATCH --error=/data/user_data/skowshik/robocasa_logs/logs/pi05_single_task_%j.err

# =============================================================================
# RoboCasa: Pi-0.5 single-task fine-tuning (RL partition)
# LoRA on VLM, full fine-tune on action head
#
# Usage:
#   sbatch slurm/robocasa/slurm_robocasa_single_task_rl.sh              # default 5 demos
#   NUM_DEMOS=10 sbatch slurm/robocasa/slurm_robocasa_single_task_rl.sh # override demo count
# =============================================================================

NUM_DEMOS=${NUM_DEMOS:-5}

export OPENPI_DATA_HOME=/data/hf_cache/pi-models/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1

cd /home/skowshik/vla/codebase/openpi
echo "Starting single-task training: num_demos=${NUM_DEMOS}"

python scripts/train.py pi05_robocasa_single_task_lora_sink_to_counter \
    --exp-name=pi05_pickplace_sink_to_counter_${NUM_DEMOS}demos \
    --batch-size 64 \
    --data.num-demos ${NUM_DEMOS} \
    --overwrite
