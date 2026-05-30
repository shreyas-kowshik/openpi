#!/bin/bash
# =============================================================================
# Create a combined LeRobot dataset from YAM teleop rollouts.
#
# Stages success-only episodes (skips FAILED_* dirs) from pick-place and
# arrange-corn-knife, then converts to a single LeRobot dataset with per-
# episode task prompts and orig_traj_id_6 tracking.
#
# Usage:
#   bash scripts/create_real_world_yam_data.sh
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

export PATH="$HOME/.local/bin:$PATH"

if [[ -f "$REPO_DIR/.env" ]]; then
    set -a
    source "$REPO_DIR/.env"
    set +a
fi

SRC=/data/group_data/maxlab/common_datasets/sreyasv/data/data/data
STAGE=/data/user_data/skowshik/tmp/yam_staging_combined
export HF_LEROBOT_HOME=/data/user_data/skowshik/huggingface/lerobot

TASKS=(pick-place arrange-corn-knife)

# --- Stage success-only symlinks ---
rm -rf "$STAGE"
for task in "${TASKS[@]}"; do
    mkdir -p "$STAGE/$task"
    for d in "$SRC/$task"/*/; do
        name=$(basename "$d")
        [[ "$name" == FAILED_* ]] && continue
        ln -s "$SRC/$task/$name" "$STAGE/$task/$name"
    done
    count=$(ls -1 "$STAGE/$task" | wc -l)
    echo "Staged $count episodes for $task"
done

# --- Convert to LeRobot combined dataset ---
uv run python examples/yam/convert_yam_combined_to_lerobot.py \
    --staging-root "$STAGE" \
    --repo-id local/yam_combined

echo "Dataset written to $HF_LEROBOT_HOME/local/yam_combined/"
