#!/usr/bin/env bash
# Convert a YOLO .pt to a TensorRT FP16 .engine (much faster on the Jetson).
# Usage: ./export_engine.sh models/yolo26s-ppe-tools-trained.pt
set -euo pipefail
PT="${1:?usage: $0 path/to/model.pt}"
"$HOME/ultralytics-env/bin/yolo" export model="$PT" format=engine half=True device=0 imgsz=640
echo "Done: ${PT%.pt}.engine (koma.py picks .engine files in models/ automatically)"
