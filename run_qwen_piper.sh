#!/usr/bin/env bash
# KOMA with Qwen vision, the existing HELIOS RAG, and Piper. Start manually.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$HOME/qwen35/workspace/llama.cpp/build/bin"
MODEL_DIR="$HOME/qwen35/models/Qwen3.5-2B"
VLM_PYTHON="$HOME/ultralytics-env/bin/python"
PIPER_PYTHON="$HOME/tts-env/bin/python"
APP="$HERE/koma_qwen_piper.py"
VOICE="$HERE/voices/koma_lessac_high.onnx"
LOG="$HERE/logs/qwen-piper-server.log"
PORT=8080
QWEN_PID=""
APP_PID=""

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash ~/YOLO-VLM/run_qwen_piper.sh [options]

Starts Qwen3.5-2B and the KOMA camera/Piper app. YOLO is not loaded.
Uses the existing KOMA persona and HELIOS RAG, the USB camera /dev/video1,
and voices/koma_lessac_high.onnx at its native pitch (0 semitone shift).
The existing launcher/server must be stopped first if port 8080 is busy.

  --check                 Check local assets and imports without starting models
  --interval SECONDS      Idle visual observations (default: 20; 0 disables)
  --ask TEXT --once       Ask one question, speak its reply, then exit
  --once                  Describe one frame, speak it, then exit
  --no-voice              Disable the Hey Koma microphone listener
  --camera DEVICE         Camera path/index or csi:N (default: /dev/video1)
  --preview               Show a window on the Jetson's graphical desktop

Ctrl-C stops the app and the Qwen server started by this launcher.
HELP
    exit 0
fi

export LD_LIBRARY_PATH="$LLAMA_DIR:$HOME/ultralytics-env/lib/python3.12/site-packages/nvidia/nccl/lib:$HOME/ultralytics-env/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for required in "$LLAMA_DIR/llama-server" "$VLM_PYTHON" "$PIPER_PYTHON" "$APP" \
    "$HERE/brain.py" "$HERE/koma.py" "$HERE/tts_worker.py" \
    "$MODEL_DIR/Qwen_Qwen3.5-2B-Q4_K_M.gguf" \
    "$MODEL_DIR/mmproj-Qwen_Qwen3.5-2B-bf16.gguf" \
    "$VOICE" "$VOICE.json"; do
    [[ -f "$required" ]] || { echo "Required file missing: $required" >&2; exit 1; }
done

if [[ "${1:-}" == "--check" ]]; then
    "$VLM_PYTHON" - "$HERE" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
assert importlib.util.find_spec("cv2"), "OpenCV is missing from the camera environment"
from brain import Brain

brain = Brain()
config = json.loads((root / "voices/koma_lessac_high.onnx.json").read_text())
print(f"KOMA persona and HELIOS index: {len(brain.lore.records)} records")
print(f"Piper voice: koma_lessac_high.onnx ({config['audio']['sample_rate']} Hz)")
print("Voice pitch shift: 0 semitones; pace: 1.0; YOLO: disabled")
PY
    "$PIPER_PYTHON" -c 'from piper import PiperVoice, SynthesisConfig; print("Piper Python imports: OK")'
    "$VLM_PYTHON" "$APP" --help >/dev/null
    echo "Preflight passed. No camera, microphone, playback, or model server started."
    exit 0
fi

# Validate options and camera-path availability before allocating the VLM.
"$VLM_PYTHON" - "$HERE" "$@" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from koma_qwen_piper import build_parser, validate_args

parser = build_parser()
parser.prog = "run_qwen_piper.sh"
validate_args(parser, parser.parse_args(sys.argv[2:]))
PY

for command in curl ss; do
    command -v "$command" >/dev/null || { echo "Required command missing: $command" >&2; exit 1; }
done
if [[ -n "$(ss -H -ltn "sport = :$PORT")" ]]; then
    echo "Port $PORT is already in use. Stop the old pipeline/server in its terminal with Ctrl-C, then retry." >&2
    exit 1
fi

stop_child() {
    local child_pid="$1"
    [[ -n "$child_pid" ]] || return 0
    if kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM "$child_pid" 2>/dev/null || true
        for _ in {1..50}; do
            kill -0 "$child_pid" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$child_pid" 2>/dev/null; then
            kill -KILL "$child_pid" 2>/dev/null || true
        fi
    fi
    wait "$child_pid" 2>/dev/null || true
}

cleanup() {
    trap '' INT TERM
    stop_child "$APP_PID"
    stop_child "$QWEN_PID"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$HERE/logs"
echo "Starting Qwen3.5-2B; server log: $LOG"
"$LLAMA_DIR/llama-server" \
    --model "$MODEL_DIR/Qwen_Qwen3.5-2B-Q4_K_M.gguf" \
    --mmproj "$MODEL_DIR/mmproj-Qwen_Qwen3.5-2B-bf16.gguf" \
    --alias qwen --host 127.0.0.1 --port "$PORT" \
    --ctx-size 4096 --parallel 1 --reasoning off \
    --threads 2 --n-gpu-layers all --flash-attn on --image-max-tokens 512 \
    >"$LOG" 2>&1 &
QWEN_PID=$!

ready=0
for _ in {1..120}; do
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$QWEN_PID" 2>/dev/null; then
        tail -n 30 "$LOG" >&2
        exit 1
    fi
    sleep 1
done
if [[ "$ready" != 1 ]]; then
    echo "Qwen did not become ready. See $LOG" >&2
    exit 1
fi

"$VLM_PYTHON" "$APP" --vlm-url "http://127.0.0.1:$PORT/v1/chat/completions" "$@" &
APP_PID=$!
wait "$APP_PID"
