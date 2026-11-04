#!/home/jasonreiber/ultralytics-env/bin/python
"""Start KOMA fully loaded, waiting for the TRADESMASTER ID card's QR code.

This script becomes run_model.sh, which gets everything after this script's own options:
  ./qr_start.py gemma --camera 0 --no-color-fix --rotate 180 --conf 0.3 --stream 8090
koma_voice.py gets --qr-code, so the model, detector, Piper and wake listener all load first
and Gemma reads KOMA's system prompt. Then KOMA watches its own camera for the card: the moment
the code reads --code it speaks welcome.txt and starts listening, with nothing left to load.
Ctrl+C while waiting stops KOMA and the model.
"""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--code", default="JETSON-START-001", help="QR text that starts the session")
    p.add_argument("launch", nargs=argparse.REMAINDER, help="run_model.sh arguments: model, then koma_voice.py options")
    args = p.parse_args()
    if not args.launch:
        p.error("give the model and options to launch, e.g. gemma --camera 0")
    run_model = str(ROOT / "run_model.sh")
    # A blank --greeting keeps KOMA from saying "KOMA is awake." before the card is shown.
    os.execv(run_model, [run_model, *args.launch, "--qr-code", args.code, "--greeting", ""])  # this process becomes run_model.sh


if __name__ == "__main__":
    main()
