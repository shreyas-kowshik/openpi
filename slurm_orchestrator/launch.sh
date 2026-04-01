#!/bin/bash
# Launch the orchestrator inside a tmux session so it persists
# after you disconnect from the cluster.
#
# Usage:
#   bash slurm_orchestrator/launch.sh --config slurm_orchestrator/config_ep1.json
#   bash slurm_orchestrator/launch.sh --config slurm_orchestrator/config_ep3.json
#   bash slurm_orchestrator/launch.sh --config slurm_orchestrator/config_ep5.json
#   bash slurm_orchestrator/launch.sh --config slurm_orchestrator/config_ep1.json --attach
#
#   tmux attach -t slurm_orch          # re-attach later
#   tmux kill-session -t slurm_orch    # stop orchestrator

CONFIG=""
ATTACH=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --attach) ATTACH=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required"
    echo "Usage: bash slurm_orchestrator/launch.sh --config <config.json> [--attach]"
    echo ""
    echo "Available configs:"
    echo "  slurm_orchestrator/config_ep1.json   (1 demo)"
    echo "  slurm_orchestrator/config_ep3.json   (3 demos)"
    echo "  slurm_orchestrator/config_ep5.json   (5 demos)"
    exit 1
fi

SESSION_NAME="slurm_orch"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Kill existing session if any
tmux has-session -t "$SESSION_NAME" 2>/dev/null && {
    echo "Killing existing session '$SESSION_NAME'"
    tmux kill-session -t "$SESSION_NAME"
}

tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_DIR" \
    "python slurm_orchestrator/orchestrator.py --config $CONFIG; echo 'Orchestrator exited. Press Enter to close.'; read"

echo "Orchestrator running in tmux session '$SESSION_NAME'"
echo "  Config:  $CONFIG"
echo "  Attach:  tmux attach -t $SESSION_NAME"
echo "  Status:  python slurm_orchestrator/orchestrator.py --config $CONFIG --status"
echo "  Stop:    tmux kill-session -t $SESSION_NAME"

if $ATTACH; then
    tmux attach -t "$SESSION_NAME"
fi
