#!/bin/bash
#SBATCH --job-name=robocasa-get_toasted_bread
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --output=/data/user_data/%u/pi05_robocasa_exps/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/pi05_robocasa_exps/logs/%x_%j.err

set -e

CONFIG_NAME="pi05_robocasa_single_task_lora_get_toasted_bread"
EXP_NAME="${CONFIG_NAME}-v1"

echo "Job ID: ${SLURM_JOB_ID} | Node: ${SLURMD_NODENAME} | Config: ${CONFIG_NAME}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_ENTITY=skowshik-carnegie-mellon-university
export PYTHONPATH=/home/skowshik/vla/codebase/openpi/src:${PYTHONPATH}

source /data/user_data/skowshik/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate openpi

cd /home/skowshik/vla/codebase/openpi
mkdir -p "/data/user_data/$USER/pi05_robocasa_exps/logs"

python scripts/compute_norm_stats.py --config-name="${CONFIG_NAME}"
python scripts/train.py "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --project-name=pi05_robocasa \
    --resume
