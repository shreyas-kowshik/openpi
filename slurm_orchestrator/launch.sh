#!/bin/bash
# Launch the orchestrator inside a tmux session so it persists
# after you disconnect from the cluster.
#
# Usage:
#   bash slurm_orchestrator/launch.sh          # start in tmux
#   bash slurm_orchestrator/launch.sh --attach  # start and attach
#   tmux attach -t slurm_orch                   # re-attach later
#   tmux kill-session -t slurm_orch             # stop orchestrator

SESSION_NAME="slurm_orch"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Kill existing session if any
tmux has-session -t "$SESSION_NAME" 2>/dev/null && {
    echo "Killing existing session '$SESSION_NAME'"
    tmux kill-session -t "$SESSION_NAME"
}

tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_DIR" \
    "python slurm_orchestrator/orchestrator.py; echo 'Orchestrator exited. Press Enter to close.'; read"

echo "Orchestrator running in tmux session '$SESSION_NAME'"
echo "  Attach:  tmux attach -t $SESSION_NAME"
echo "  Status:  python slurm_orchestrator/orchestrator.py --status"
echo "  Stop:    tmux kill-session -t $SESSION_NAME"

if [[ "$1" == "--attach" ]]; then
    tmux attach -t "$SESSION_NAME"
fi
