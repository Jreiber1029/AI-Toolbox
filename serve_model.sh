#!/usr/bin/env bash
# Serve one VLM on :8080 for the KOMA pipeline:  ./serve_model.sh qwen|gemma|ministral|nemotron
# nemotron is text-only (no mmproj): run the app with --no-vision.
# Runs in the foreground; Ctrl+C stops it. Then start the app in another terminal:
#   ~/ultralytics-env/bin/python koma_voice.py
# bench_models.py starts this script once per model.
#
# Every model gets the same context, KV cache, image budget and threads, and no
# server-side sampling overrides, so a comparison only changes the model.
# Sampling (temperature 0.4, repeat_penalty 1.1) comes from each request (koma.stream_vlm).
set -euo pipefail

LLAMA_DIR="$HOME/qwen35/workspace/llama.cpp/build/bin"
VENV="$HOME/ultralytics-env"
export LD_LIBRARY_PATH="$LLAMA_DIR:$VENV/lib/python3.12/site-packages/nvidia/nccl/lib:$VENV/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

case "${1:-}" in
    qwen)
        MODEL="$HOME/qwen35/models/Qwen3.5-2B/Qwen_Qwen3.5-2B-Q4_K_M.gguf"
        MMPROJ="$HOME/qwen35/models/Qwen3.5-2B/mmproj-Qwen_Qwen3.5-2B-bf16.gguf" ;;
    gemma)
        MODEL="$HOME/gemma4/models/gemma-4-E2B-it/gemma-4-E2B-it-Q4_K_M.gguf"
        MMPROJ="$HOME/gemma4/models/gemma-4-E2B-it/mmproj-gemma-4-E2B-it-Q8_0.gguf" ;;
    ministral)
        MODEL="$HOME/ministral3/models/Ministral-3-3B-Instruct-2512/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
        MMPROJ="$HOME/ministral3/models/Ministral-3-3B-Instruct-2512/Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf" ;;
    nemotron)
        MODEL="$HOME/nemotron/models/Nemotron-3.5-Bolt-E4B/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf"
        MMPROJ="" ;;
    *)
        echo "Usage: $0 qwen|gemma|ministral|nemotron" >&2
        exit 2 ;;
esac

for f in "$MODEL" ${MMPROJ:+"$MMPROJ"}; do
    [[ -f "$f" ]] || { echo "Missing $f - download it first." >&2; exit 1; }
done
if curl -sf --max-time 2 http://127.0.0.1:8080/health >/dev/null; then
    echo "Something is already serving :8080. Stop it first so the right model answers." >&2
    exit 1
fi

# exec: this script's PID becomes llama-server's, so stopping it stops the server.
exec "$LLAMA_DIR/llama-server" \
    --model "$MODEL" ${MMPROJ:+--mmproj "$MMPROJ"} \
    --alias "$1" --host 127.0.0.1 --port 8080 \
    --ctx-size 4096 --parallel 1 --reasoning off \
    --threads 2 --n-gpu-layers all \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --image-max-tokens 280
