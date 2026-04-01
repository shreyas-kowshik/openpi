#!/bin/bash
#SBATCH --job-name=orchestrator
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/user_data/%u/openpi-libero10/logs/orchestrator_%j.out
#SBATCH --error=/data/user_data/%u/openpi-libero10/logs/orchestrator_%j.err

mkdir -p "/data/user_data/$USER/openpi-libero10/logs"

REPO_DIR="${SLURM_SUBMIT_DIR}"
export PATH="$HOME/.local/bin:$PATH"

cd "$REPO_DIR"

# Default to ep1 config; override with: sbatch --export=ALL,CONFIG=slurm_orchestrator/config_ep3.json slurm_orchestrator/orchestrator.slurm
CONFIG="${CONFIG:-slurm_orchestrator/config_ep1.json}"

echo "Starting orchestrator with config: $CONFIG"
echo "Repo dir: $REPO_DIR"
echo "Running on $(hostname) at $(date)"

python3 slurm_orchestrator/orchestrator.py --config "$CONFIG"
