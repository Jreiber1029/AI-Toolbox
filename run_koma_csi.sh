#!/usr/bin/env bash
# KOMA + INT8 Gemma on the IMX219 CSI camera (nvargus sensor 0; set KOMA_CSI_SENSOR=1 for the other port).
# Extra arguments go to koma_voice.py, e.g.  ./run_koma_csi.sh --stream 8090 --rotate 180
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec "$HERE/run_koma_int8.sh" --camera "csi:${KOMA_CSI_SENSOR:-0}" "$@"
