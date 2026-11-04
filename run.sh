#!/usr/bin/env bash
# Start Qwen3.5-2B (if not already running), then the Koma pipeline.
# Extra arguments are passed to koma.py, e.g.  ./run.sh --preview --save-events
# Without YOLO:  KOMA_APP=koma_voice.py ./run.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$HOME/qwen35/workspace/llama.cpp/build/bin"
MODEL_DIR="$HOME/qwen35/models/Qwen3.5-2B"
VENV="$HOME/ultralytics-env"
HEALTH="http://127.0.0.1:8080/health"
LOG="$HERE/logs/qwen-server.log"
QWEN_PID=""

export LD_LIBRARY_PATH="$LLAMA_DIR:$VENV/lib/python3.12/site-packages/nvidia/nccl/lib:$VENV/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cleanup() {
    if [[ -n "$QWEN_PID" ]] && kill -0 "$QWEN_PID" 2>/dev/null; then
        echo "Stopping Qwen server..."
        kill "$QWEN_PID"; wait "$QWEN_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

mkdir -p "$HERE/logs"
if curl -sf --max-time 2 "$HEALTH" >/dev/null; then
    echo "Qwen server already running on :8080."
else
    echo "Starting Qwen3.5-2B (log: $LOG)..."
    "$LLAMA_DIR/llama-server" \
        --model "$MODEL_DIR/Qwen_Qwen3.5-2B-Q4_K_M.gguf" \
        --mmproj "$MODEL_DIR/mmproj-Qwen_Qwen3.5-2B-bf16.gguf" \
        --host 127.0.0.1 --port 8080 \
        --ctx-size 4096 --parallel 1 --reasoning off \
        --threads 2 --n-gpu-layers all \
        --flash-attn on \
        >"$LOG" 2>&1 &
    QWEN_PID=$!
    for _ in $(seq 1 120); do
        curl -sf --max-time 2 "$HEALTH" >/dev/null && break
        kill -0 "$QWEN_PID" 2>/dev/null || { tail -n 30 "$LOG" >&2; exit 1; }
        sleep 1
    done
    curl -sf --max-time 2 "$HEALTH" >/dev/null || { echo "Qwen did not become ready; see $LOG" >&2; exit 1; }
    echo "Qwen ready."
fi

"$VENV/bin/python" "$HERE/${KOMA_APP:-koma.py}" "$@"
