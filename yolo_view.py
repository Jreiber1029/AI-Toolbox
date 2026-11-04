#!/home/jasonreiber/ultralytics-env/bin/python
"""Watch the tool detector in a browser - no language model, speech or Piper loaded.

Draws YOLO's boxes on the live camera at http://<jetson>:PORT and shows, for each tool
held next to a person (or face), how long it has been held against KOMA's trigger time - the same
rule koma_voice.py uses, so a bar reaching 100% is when KOMA would have answered.
Prints a line whenever the set of detected classes changes, plus RAM use.

Collecting training photos: type a folder name (e.g. "plier") in this terminal and press
Enter. The next --save-count clean frames (no boxes, colour fix applied, as the detector
sees them), one every --save-every seconds, go to --save-dir/<name>/. Enter again stops early.
"""
import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from koma import CSI_PIPELINE, apply_color_lut, log, make_color_lut
from koma_voice import ANCHORS, DETECT_GAP, ROTATIONS, near
from yolo_trt import TrtDetector

ROOT = Path(__file__).resolve().parent
COLOURS = [(0, 200, 255), (255, 160, 0), (80, 220, 80), (255, 80, 200), (200, 200, 200)]


def ram_used_mb():
    info = dict(line.split(":") for line in Path("/proc/meminfo").read_text().splitlines())
    kb = lambda key: int(info[key].split()[0])
    return (kb("MemTotal") - kb("MemAvailable")) // 1024


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "YOLO26toolmodel/toolppefinetune.engine")
    p.add_argument("--camera", default="csi:0", help="'csi:N' for nvargus sensor N, or a V4L2 index/path")
    p.add_argument("--rotate", type=int, choices=sorted(ROTATIONS), default=0)
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    p.add_argument("--confirm-seconds", type=float, default=3)
    p.add_argument("--no-color-fix", action="store_true")
    p.add_argument("--save-dir", type=Path, default=ROOT / "captures", help="Where recorded frames go")
    p.add_argument("--save-count", type=int, default=200, help="Frames saved per recording")
    p.add_argument("--save-every", type=float, default=0.5, help="Seconds between saved frames")
    args = p.parse_args()

    detector = TrtDetector(args.model)
    log(f"{args.model.name}: {', '.join(detector.names.values())}")
    if args.camera.startswith("csi:"):
        camera = cv2.VideoCapture(CSI_PIPELINE.format(sensor=args.camera[4:]), cv2.CAP_GSTREAMER)
    else:
        camera = cv2.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    lut = None if args.no_color_fix else make_color_lut()
    latest = {"jpeg": None}
    rec = {"left": 0, "folder": None, "index": 0, "next": 0.0}  # current recording, if any
    lock = threading.Lock()

    def record_keys():
        for line in sys.stdin:
            with lock:
                if rec["left"]:
                    log(f"Recording stopped: {rec['index']} frames in {rec['folder']}")
                    rec["left"] = 0
                    continue
                folder = args.save_dir / (line.strip() or time.strftime("%Y%m%d-%H%M%S"))
                folder.mkdir(parents=True, exist_ok=True)
                # Carry on numbering after frames already in the folder, so a second take adds to it.
                rec.update(left=args.save_count, folder=folder, index=len(list(folder.glob("*.jpg"))), next=0.0)
            log(f"Recording {args.save_count} frames to {folder} "
                f"(~{args.save_count * args.save_every:.0f} s). Enter stops early.")

    def save_frame(frame, now):
        with lock:
            if not rec["left"] or now < rec["next"]:
                return
            path = rec["folder"] / f"{rec['folder'].name}_{rec['index']:04d}.jpg"
            rec["index"] += 1
            rec["left"] -= 1
            rec["next"] = now + args.save_every
            if not rec["left"]:
                log(f"Recording done: {rec['folder']} now has {rec['index']} frames")
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    def detect_loop():
        held_since, last_seen, last_names, fps = {}, {}, None, 0.0
        while True:
            ok, frame = camera.read()
            if not ok:
                log("Camera read failed.")
                return
            if args.rotate:
                frame = cv2.rotate(frame, ROTATIONS[args.rotate])
            frame = apply_color_lut(frame, lut)
            started = now = time.monotonic()
            save_frame(frame, now)  # before any boxes are drawn on it
            found = detector.detect(frame, args.conf)
            fps = 0.9 * fps + 0.1 / max(time.monotonic() - started, 1e-3)
            anchors = [(box, ANCHORS[name.lower()]) for name, _, box in found if name.lower() in ANCHORS]
            for name, _, box in found:
                if name.lower() not in ANCHORS and any(near(box, *a) for a in anchors):
                    held_since.setdefault(name, now)
                    last_seen[name] = now
            for name in [n for n in held_since if now - last_seen[n] > DETECT_GAP]:
                del held_since[name]

            for name, conf, (x1, y1, x2, y2) in found:
                colour = COLOURS[list(detector.names.values()).index(name) % len(COLOURS)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
                cv2.putText(frame, f"{name} {conf:.0%}", (x1, max(y1 - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
            y = 40
            for name, since in held_since.items():
                share = min((now - since) / args.confirm_seconds, 1)
                cv2.rectangle(frame, (20, y - 22), (20 + int(300 * share), y + 6), (80, 220, 80), -1)
                cv2.putText(frame, f"{name} held {now - since:.1f}s{'  -> KOMA would answer' if share >= 1 else ''}",
                            (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y += 40
            cv2.putText(frame, f"detector {fps:.0f} fps | RAM {ram_used_mb()} MB", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            with lock:
                recording = f"REC {rec['folder'].name} {args.save_count - rec['left']}/{args.save_count}" \
                    if rec["left"] else None
            if recording:
                cv2.circle(frame, (frame.shape[1] - 30, 30), 12, (0, 0, 255), -1)
                cv2.putText(frame, recording, (frame.shape[1] - 340, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            with lock:
                latest["jpeg"] = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()

            names = sorted({name for name, _, _ in found})
            if names != last_names:
                log(f"Seen: {', '.join(names) or 'nothing'}   (RAM {ram_used_mb()} MB)")
                last_names = names

    class Stream(BaseHTTPRequestHandler):
        # multipart/x-mixed-replace: the browser swaps in each JPEG as it arrives.
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with lock:
                        jpeg = latest["jpeg"]
                    if jpeg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    threading.Thread(target=detect_loop, daemon=True).start()
    threading.Thread(target=record_keys, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Stream)
    server.daemon_threads = True
    log(f"Open http://localhost:{args.port} (or http://<jetson-ip>:{args.port}). Ctrl+C stops.")
    log(f"To record {args.save_count} training frames: type a folder name (e.g. plier) and press Enter.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        log("Stopped.")


if __name__ == "__main__":
    main()
