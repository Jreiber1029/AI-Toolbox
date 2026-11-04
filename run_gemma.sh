#!/usr/bin/env bash
# Test launcher: Gemma 4 E2B instead of Qwen3.5-2B, then the Koma pipeline.
# Extra arguments are passed to the app, e.g.  ./run_gemma.sh --preview
# Default app is koma_voice.py (TensorRT YOLO, no PyTorch). Override: KOMA_APP=koma.py ./run_gemma.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$HOME/qwen35/workspace/llama.cpp/build/bin"
MODEL_DIR="$HOME/gemma4/models/gemma-4-E2B-it"
VENV="$HOME/ultralytics-env"
HEALTH="http://127.0.0.1:8080/health"
LOG="$HERE/logs/gemma-server.log"
GEMMA_PID=""

export LD_LIBRARY_PATH="$LLAMA_DIR:$VENV/lib/python3.12/site-packages/nvidia/nccl/lib:$VENV/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cleanup() {
    trap '' INT TERM  # extra Ctrl+C presses must not interrupt the shutdown below
    if [[ -n "$GEMMA_PID" ]] && kill -0 "$GEMMA_PID" 2>/dev/null; then
        echo "Stopping Gemma server..."
        kill "$GEMMA_PID" 2>/dev/null || true
        # llama-server can hang while freeing GPU memory; give it 5 s, then force it.
        for _ in {1..50}; do
            kill -0 "$GEMMA_PID" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$GEMMA_PID" 2>/dev/null; then
            echo "Gemma server did not exit; forcing it."
            kill -KILL "$GEMMA_PID" 2>/dev/null || true
        fi
        wait "$GEMMA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

for f in "$MODEL_DIR/gemma-4-E2B-it-Q4_K_M.gguf" "$MODEL_DIR/mmproj-gemma-4-E2B-it-Q8_0.gguf"; do
    [[ -f "$f" ]] || { echo "Missing $f - download it first (see run_gemma.sh notes)." >&2; exit 1; }
done

mkdir -p "$HERE/logs"
if curl -sf --max-time 2 "$HEALTH" >/dev/null; then
    echo "Something is already serving :8080 (Qwen?). Stop it first so the test really uses Gemma." >&2
    exit 1
fi

# Weights stay memory-mapped (no --no-mmap/--mlock): Gemma's large per-layer embedding
# table only pages in the rows actually used, which keeps resident RAM well below file size.
echo "Starting Gemma 4 E2B (log: $LOG)..."
"$LLAMA_DIR/llama-server" \
    --model "$MODEL_DIR/gemma-4-E2B-it-Q4_K_M.gguf" \
    --mmproj "$MODEL_DIR/mmproj-gemma-4-E2B-it-Q8_0.gguf" \
    --alias gemma --host 127.0.0.1 --port 8080 \
    --ctx-size 4096 --parallel 1 --reasoning off \
    --threads 2 --n-gpu-layers all \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --image-max-tokens 280 \
    --top-k 64 --top-p 0.95 \
    >"$LOG" 2>&1 &
GEMMA_PID=$!
for _ in $(seq 1 180); do
    curl -sf --max-time 2 "$HEALTH" >/dev/null && break
    kill -0 "$GEMMA_PID" 2>/dev/null || { tail -n 30 "$LOG" >&2; exit 1; }
    sleep 1
done
curl -sf --max-time 2 "$HEALTH" >/dev/null || { echo "Gemma did not become ready; see $LOG" >&2; exit 1; }
echo "Gemma ready."

"$VENV/bin/python" "$HERE/${KOMA_APP:-koma_voice.py}" "$@"
