#!/usr/bin/env bash
# Drives the interactive TUI in tmux (mirrors pi's pi-test.sh pattern):
# launch, capture the rendered UI, submit a prompt, interrupt, exit cleanly.
#
# A live end-to-end reply requires DEEPSEEK_API_KEY; without one the run still
# verifies startup rendering, input handling, and a clean exit.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"

SESSION="agent-m-smoke"
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "== launching agent-m in tmux =="
tmux new-session -d -s "$SESSION" -x 100 -y 30 "./agent-m.sh --ui-mode fullscreen"
sleep 4
echo "== initial render =="
tmux capture-pane -t "$SESSION" -p | head -25

echo "== submitting a prompt =="
tmux send-keys -t "$SESSION" "Say exactly: smoke test ok" Enter
sleep 3
tmux capture-pane -t "$SESSION" -p | tail -12

echo "== interrupting =="
tmux send-keys -t "$SESSION" Escape
sleep 1

echo "== exiting with ctrl+d =="
tmux send-keys -t "$SESSION" C-d
sleep 1
tmux capture-pane -t "$SESSION" -p | tail -5

tmux kill-session -t "$SESSION" 2>/dev/null || true
echo "== smoke complete =="
