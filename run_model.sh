#!/usr/bin/env bash
# One-terminal launcher: start a model in the background, then KOMA in this terminal.
#   ./run_model.sh qwen|gemma|ministral|nemotron [koma_voice.py options]
#   e.g. ./run_model.sh ministral --type --no-camera
# The server log goes to logs/<model>-server.log; quitting KOMA stops the server.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-}"
[[ -n "$MODEL" ]] || { echo "Usage: $0 qwen|gemma|ministral|nemotron [koma_voice.py options]" >&2; exit 2; }
shift
LOG="$HERE/logs/$MODEL-server.log"
SERVER_PID=""

cleanup() {
    trap '' INT TERM  # extra Ctrl+C presses must not interrupt the shutdown below
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping $MODEL server..."
        kill "$SERVER_PID" 2>/dev/null || true
        # llama-server can hang while freeing GPU memory; give it 5 s, then force it.
        for _ in {1..50}; do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 0.1
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

mkdir -p "$HERE/logs"
echo "Starting $MODEL (log: $LOG)..."
"$HERE/serve_model.sh" "$MODEL" >"$LOG" 2>&1 &
SERVER_PID=$!
# KOMA_GATE (set by qr_start.py): a command that must succeed before KOMA starts.
# It runs while the model loads, so the VLM is usually ready when the gate opens.
if [[ -n "${KOMA_GATE:-}" ]]; then
    eval "$KOMA_GATE" || exit 1
fi
for _ in $(seq 1 180); do
    curl -sf --max-time 2 http://127.0.0.1:8080/health >/dev/null && break
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 20 "$LOG" >&2; exit 1; }
    sleep 1
done
curl -sf --max-time 2 http://127.0.0.1:8080/health >/dev/null || { echo "$MODEL did not become ready; see $LOG" >&2; exit 1; }
echo "$MODEL ready."

"$HOME/ultralytics-env/bin/python" "$HERE/koma_voice.py" "$@"
