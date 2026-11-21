#!/usr/bin/env bash
# KOMA + INT8 Gemma on a USB camera. The camera is found by its stable /dev/v4l/by-id name,
# because its /dev/videoN number changes depending on whether the CSI camera is attached.
# Override with KOMA_USB_CAM=/dev/video1 ./run_koma_usb.sh
# The IMX219 color correction is skipped (it was tuned for the CSI lens).
# Extra arguments go to koma_voice.py, e.g.  ./run_koma_usb.sh --stream 8090
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

CAM="${KOMA_USB_CAM:-}"
if [[ -z "$CAM" ]]; then
    for dev in /dev/v4l/by-id/usb-*-video-index0; do
        [[ -e "$dev" ]] && { CAM="$(readlink -f "$dev")"; break; }
    done
fi
[[ -n "$CAM" && -e "$CAM" ]] || {
    echo "No USB camera found. Plug it in, or set KOMA_USB_CAM=/dev/videoN (see: v4l2-ctl --list-devices)." >&2
    exit 1
}
echo "USB camera: $CAM"
# Pass the bare index so OpenCV opens it with V4L2, as it did before.
[[ "$CAM" =~ ^/dev/video([0-9]+)$ ]] && CAM="${BASH_REMATCH[1]}"
exec "$HERE/run_koma_int8.sh" --camera "$CAM" --no-color-fix "$@"
