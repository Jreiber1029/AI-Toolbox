#!/usr/bin/env bash
# Start the INT8 TensorRT Gemma E2B server (YOLO-VLM_RT), then KOMA (koma_voice.py) in this terminal.
# Usually called through run_koma_csi.sh or run_koma_usb.sh; extra arguments go to koma_voice.py.
# The server log goes to logs/gemma-int8-server.log; quitting KOMA stops the server.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/logs/gemma-int8-server.log"
HEALTH="http://127.0.0.1:8080/health"
SERVER_PID=""

cleanup() {
    trap '' INT TERM  # extra Ctrl+C presses must not interrupt the shutdown below
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping Gemma INT8 server..."
        kill "$SERVER_PID" 2>/dev/null || true
        for _ in {1..50}; do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 0.1
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if curl -sf --max-time 2 "$HEALTH" >/dev/null; then
    echo "Something is already serving :8080. Stop it first so KOMA uses INT8 Gemma." >&2
    exit 1
fi

mkdir -p "$HERE/logs"
echo "Starting Gemma E2B INT8 TensorRT (log: $LOG)..."
"$HERE/YOLO-VLM_RT/start_gemma_int8.sh" >"$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 180); do
    curl -sf --max-time 2 "$HEALTH" >/dev/null && break
    kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 30 "$LOG" >&2; exit 1; }
    sleep 1
done
curl -sf --max-time 2 "$HEALTH" >/dev/null || { echo "Gemma did not become ready; see $LOG" >&2; exit 1; }
echo "Gemma ready."

"$HOME/ultralytics-env/bin/python" "$HERE/koma_voice.py" "$@"
