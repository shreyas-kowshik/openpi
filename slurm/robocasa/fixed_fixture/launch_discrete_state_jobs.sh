#!/bin/bash
# Launch both discrete_state training jobs from the repo root.
# Usage: bash slurm/robocasa/fixed_fixture/launch_discrete_state_jobs.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting wash_lettuce_discrete_state..."
JOB1=$(sbatch --parsable "$SCRIPT_DIR/wash_lettuce_discrete_state.slurm")
echo "  Job ID: $JOB1"

echo "Submitting load_dishwasher_discrete_state..."
JOB2=$(sbatch --parsable "$SCRIPT_DIR/load_dishwasher_discrete_state.slurm")
echo "  Job ID: $JOB2"

echo ""
echo "Submitted jobs: $JOB1, $JOB2"
echo "Monitor with: squeue -j $JOB1,$JOB2"
