#!/bin/bash
# Launches remaining 5_demos jobs not currently running for user skowshik

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm/libero10/5_demos"

sbatch "$SLURM_DIR/alphabet_soup_tomato_sauce_basket.slurm"
sbatch "$SLURM_DIR/black_bowl_bottom_drawer.slurm"
sbatch "$SLURM_DIR/both_mokapots_stove.slurm"
sbatch "$SLURM_DIR/stove_mokapot.slurm"
sbatch "$SLURM_DIR/white_mug_plates.slurm"
sbatch "$SLURM_DIR/yellow_white_mug_microwave.slurm"
