#!/bin/bash
# Move ep3/ep5 16k checkpoints from hf_cache to group_data
# Part 1: 16k checkpoints with params only (no train_state)
# Part 2: Latest checkpoint of each dir with params + train_state
#
# Destination format (flat):
#   {name}_{step}/_CHECKPOINT_METADATA
#   {name}_{step}/assets/
#   {name}_{step}/params/
#   {name}_{step}/train_state/   (only for latest checkpoint)

SRC="/data/hf_cache/models/pi05_checkpoints_libero10"
DST="/data/group_data/maxlab/common_datasets/pi05_common/pi05_checkpoints_libero10"

set -e

echo "============================================"
echo "PART 1: Copy 16k checkpoints (params only)"
echo "============================================"

for dir in "$SRC"/*ep[35]; do
    name=$(basename "$dir")
    inner=$(ls "$dir")
    ckpt_path="$dir/$inner/16000"

    if [ ! -d "$ckpt_path" ]; then
        echo "SKIP (no 16000): $name"
        continue
    fi

    dst_dir="$DST/${name}_16000"
    echo ""
    echo ">>> ${name}_16000 (params only)"

    mkdir -p "$dst_dir"

    rsync -ah --progress "$ckpt_path/_CHECKPOINT_METADATA" "$dst_dir/"
    rsync -ah --progress "$ckpt_path/assets/" "$dst_dir/assets/"
    rsync -ah --progress "$ckpt_path/params/" "$dst_dir/params/"

    echo "DONE: ${name}_16000 (params only)"
done

echo ""
echo "============================================"
echo "PART 2: Copy latest checkpoint (params + train_state)"
echo "============================================"

for dir in "$SRC"/*ep[35]; do
    name=$(basename "$dir")
    inner=$(ls "$dir")

    # Find latest numeric checkpoint (exclude tmp dirs)
    latest=$(ls -d "$dir/$inner"/[0-9]* 2>/dev/null | grep -v tmp | sort -t/ -k"$(echo "$dir/$inner/1" | tr '/' '\n' | wc -l)" -n | tail -1)
    latest_step=$(basename "$latest")

    dst_dir="$DST/${name}_${latest_step}"

    if [ "$latest_step" = "16000" ]; then
        # Latest IS 16000 — just add train_state to what we already copied
        echo ""
        echo ">>> ${name}_16000 (adding train_state — this is the latest)"
        rsync -ah --progress "$latest/train_state/" "$dst_dir/train_state/"
        echo "DONE: ${name}_16000 (train_state added)"
    else
        # Latest is a different step — copy full checkpoint
        echo ""
        echo ">>> ${name}_${latest_step} (full checkpoint — latest)"
        mkdir -p "$dst_dir"
        rsync -ah --progress "$latest/_CHECKPOINT_METADATA" "$dst_dir/"
        rsync -ah --progress "$latest/assets/" "$dst_dir/assets/"
        rsync -ah --progress "$latest/params/" "$dst_dir/params/"
        rsync -ah --progress "$latest/train_state/" "$dst_dir/train_state/"
        echo "DONE: ${name}_${latest_step} (full)"
    fi
done

echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
