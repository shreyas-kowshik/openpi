#!/usr/bin/env bash
#SBATCH --job-name=openpi_rc_bc_smoke
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

: "${MIDAS_REPO_DIR:?Set MIDAS_REPO_DIR to the MIDAS checkout}"
: "${OPENPI_DATASET_ROOT:?Set OPENPI_DATASET_ROOT to the RoboCasa dataset root}"
: "${OPENPI_ASSETS_ROOT:?Set OPENPI_ASSETS_ROOT to the BC normalization-stat root}"
: "${OPENPI_CHECKPOINT_ROOT:?Set OPENPI_CHECKPOINT_ROOT to a new writable BC output root}"

# Supply an absolute Python executable from the configured RoboCasa environment,
# or activate that environment before submitting with --export=ALL.
PYTHON_BIN="${OPENPI_PYTHON_BIN:-python}"
export PYTHONPATH="$MIDAS_REPO_DIR/openpi/src:$MIDAS_REPO_DIR/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export POLARS_MAX_THREADS=2
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS
cd "$MIDAS_REPO_DIR/openpi"

CONFIGS=(
    pi05_robocasa_bc_counter_to_cabinet_l1_s1_ep32
    pi05_robocasa_bc_fridge_drawer_to_shelf_l50_s37
    pi05_robocasa_bc_prepare_coffee_l25_s29
)
RUN_NAME="${OPENPI_SMOKE_RUN_NAME:-bc_smoke_${SLURM_JOB_ID}}"

for config in "${CONFIGS[@]}"; do
    echo "Testing BC config: $config"
    "$PYTHON_BIN" scripts/trace_robocasa_bc.py --config-name "$config"
    "$PYTHON_BIN" scripts/train.py "$config" \
        --exp-name "$RUN_NAME" \
        --batch-size 1 \
        --num-workers 0 \
        --num-train-steps 1 \
        --log-interval 1 \
        --no-wandb-enabled
    echo "BC_TRAINING_PASSED $config"
done
