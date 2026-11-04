#!/home/jasonreiber/ultralytics-env/bin/python
"""KOMA: YOLO tool detection -> Qwen VLM -> Piper TTS, with a 'Hey Koma' wake phrase.

KOMA's character and lore live in Koma-storyrag/ (see brain.py). YOLO runs on every
camera frame; when a tool is seen confidently for several frames in a row (and wasn't
mentioned recently), the frame goes to the VLM as a "tool" event. Saying "Hey Koma"
interrupts any speech and sends your request as a "question", "safety" or "handover"
event. "Hey Koma, new visitor" starts a fresh session.
"""
import argparse
import base64
import json
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

from brain import SENTENCE_LIMIT, Brain, classify

ROOT = Path(__file__).resolve().parent
HOME = Path.home()
sys.path.insert(0, str(HOME / "voiceAssistant"))
from camera_color import apply_color_lut, make_color_lut  # noqa: E402

STT_PYTHON = HOME / "stt-env/bin/python"
TTS_PYTHON = HOME / "tts-env/bin/python"
FALLBACK_YOLO = HOME / "yolo26n.engine"
FALLBACK_VOICE = HOME / "voiceAssistant/tts/voices/en_US-lessac-medium.onnx"
CSI_PIPELINE = (
    "nvarguscamerasrc sensor-id={sensor} ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
)
MAX_TOKENS = {"tool": 90, "question": 170, "safety": 120}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def pick_file(explicit, folder, patterns, fallback):
    """Use an explicit path, else the first match in folder, else the fallback."""
    if explicit:
        return Path(explicit).expanduser()
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    return fallback


class Speaker:
    """Talks to tts_worker.py running in the Piper venv."""

    def __init__(self, voice, length_scale=1.0, pitch=0.0, volume=1.0, on_change=lambda speaking: None):
        self.on_change = on_change
        self.proc = subprocess.Popen(
            [str(TTS_PYTHON), str(ROOT / "tts_worker.py"), "--voice", str(voice),
             "--length-scale", str(length_scale), "--pitch", str(pitch), "--volume", str(volume)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.idle = threading.Event()
        self.idle.set()
        ready = threading.Event()

        def watch():
            for line in self.proc.stdout:
                line = line.strip()
                if line == "READY":
                    ready.set()
                elif line == "IDLE":
                    self.idle.set()
                    self.on_change(False)
                elif line:
                    log(f"TTS: {line}")
            ready.set()

        threading.Thread(target=watch, daemon=True).start()
        ready.wait(60)
        if self.proc.poll() is not None:
            raise RuntimeError("Piper TTS worker failed to start")

    def _send(self, msg):
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            log("TTS worker is gone; speech disabled")

    def say(self, text):
        text = re.sub(r"[*_`#]", "", text).strip()
        if text:
            self.idle.clear()
            self.on_change(True)
            # One sentence at a time, so the first plays while Piper makes the next;
            # a long scripted line (welcome, tool) would otherwise wait for all of it.
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                self._send({"text": sentence})

    def flush(self):
        self._send({"cmd": "flush"})
        self.idle.set()
        self.on_change(False)

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.terminate()


def encode_frame(frame, width):
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, int(h * width / w)))
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(jpeg).decode("ascii")


def stream_vlm(url, messages, max_tokens, cancel, on_delta=None, **extra):
    """Yield complete sentences from the VLM as they stream in.

    on_delta is called with each raw text piece (bench_models.py times tokens with it);
    extra adds request fields such as a fixed seed.
    """
    payload = {"model": "qwen", "messages": messages, "temperature": 0.4,
               "max_tokens": max_tokens, "stream": True, "cache_prompt": True,
               "repeat_penalty": 1.1, **extra}
    request = urllib.request.Request(url, json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    buffer = ""
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            if cancel.is_set():
                return
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0].get("delta", {}).get("content")
            if not delta:
                continue
            if on_delta:
                on_delta(delta)
            buffer += delta
            while (m := re.search(r"[.!?](\s|$)", buffer)) and m.end() < len(buffer):
                yield buffer[:m.end()].strip()
                buffer = buffer[m.end():]
    if buffer.strip():
        yield buffer.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yolo", help="YOLO .engine/.pt (default: first file in models/)")
    p.add_argument("--voice", help="Piper .onnx (default: first file in voices/)")
    p.add_argument("--voice-speed", type=float, default=1.2,
                   help="Piper length scale: 1.0 is normal, higher is slower (default 1.2)")
    p.add_argument("--voice-pitch", type=float, default=2.0,
                   help="Pitch shift in semitones, speed unchanged (default +2; 0 disables)")
    p.add_argument("--voice-volume", type=float, default=0.9,
                   help="Voice gain; the KOMA voice peaks near clipping at 1.0 (default 0.9)")
    p.add_argument("--vlm-url", default="http://127.0.0.1:8080/v1/chat/completions")
    p.add_argument("--camera", default="csi:0", help="'csi:N' for nvargus sensor N, or a V4L2 index/path")
    p.add_argument("--conf", type=float, default=0.6, help="Detection confidence threshold")
    p.add_argument("--confirm-frames", type=int, default=5,
                   help="Consecutive frames a class must appear in before triggering")
    p.add_argument("--cooldown", type=float, default=45,
                   help="Seconds before the same class can trigger again")
    p.add_argument("--ignore", default="person",
                   help="Comma-separated classes that never trigger the VLM")
    p.add_argument("--image-width", type=int, default=640, help="Resize frames to this width for the VLM")
    p.add_argument("--memory-turns", type=int, default=3, help="Exchanges kept as recent dialogue")
    p.add_argument("--no-tool-lore", action="store_true", help="Don't add a lore passage to tool comments")
    p.add_argument("--handover-on-start", action="store_true",
                   help="Speak KOMA's handover welcome at startup")
    p.add_argument("--no-color-fix", action="store_true", help="Skip the IMX219 lens color correction")
    p.add_argument("--preview", action="store_true", help="Show an OpenCV window (needs a display)")
    p.add_argument("--save-events", action="store_true", help="Save triggering frames to events/")
    p.add_argument("--no-voice", action="store_true", help="Disable the Hey Koma listener")
    args = p.parse_args()

    yolo_path = pick_file(args.yolo, ROOT / "models", ["*.engine", "*.pt", "*.onnx"], FALLBACK_YOLO)
    voice_path = pick_file(args.voice, ROOT / "voices", ["*.onnx"], FALLBACK_VOICE)
    ignore = {c.strip() for c in args.ignore.split(",") if c.strip()}
    log(f"YOLO model:  {yolo_path}{'  (fallback - copy yours to models/)' if yolo_path == FALLBACK_YOLO else ''}")
    log(f"Piper voice: {voice_path}{'  (fallback - copy yours to voices/)' if voice_path == FALLBACK_VOICE else ''}"
        f"  [speed {args.voice_speed}, pitch {args.voice_pitch:+g} st, volume {args.voice_volume}]")

    brain = Brain(tool_lore=not args.no_tool_lore)
    log(f"Lore: {len(brain.lore.records)} records; runtime facts {'loaded' if brain.runtime_facts else 'MISSING'}")

    from ultralytics import YOLO
    model = YOLO(str(yolo_path), task="detect")
    log(f"Classes: {', '.join(model.names.values())}")

    if args.camera.startswith("csi:"):
        camera = cv2.VideoCapture(CSI_PIPELINE.format(sensor=args.camera[4:]), cv2.CAP_GSTREAMER)
    else:
        camera = cv2.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    lut = None if args.no_color_fix else make_color_lut()

    children = []

    def tell_listener(speaking):
        for proc in children:
            try:
                proc.stdin.write("SPEAKING\n" if speaking else "QUIET\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass

    speaker = Speaker(voice_path, length_scale=args.voice_speed, pitch=args.voice_pitch,
                      volume=args.voice_volume, on_change=tell_listener)
    voice_requests = queue.Queue()
    detection_events = queue.Queue(maxsize=1)
    cancel = threading.Event()  # set when a wake word should abort current VLM output
    stopping = threading.Event()
    latest = {"frame": None, "labels": []}
    lock = threading.Lock()
    history = deque(maxlen=2 * args.memory_turns)  # this visitor's session only
    session = {"handover_done": False}
    status = {"text": "watching"}

    def ask(event, frame, labels=(), question=""):
        use_image = frame is not None and brain.wants_image(event, question, labels)
        messages, lore_ids = brain.build_messages(
            event, labels=labels, question=question, history=tuple(history),
            image_b64=encode_frame(frame, args.image_width) if use_image else None)
        started = time.monotonic()
        reply = []
        for sentence in stream_vlm(args.vlm_url, messages, MAX_TOKENS[event], cancel):
            reply.append(sentence)
            speaker.say(sentence)
            if len(reply) >= SENTENCE_LIMIT[event]:
                break  # closing the stream stops generation on the server
        answer = " ".join(reply)
        if answer and not cancel.is_set():
            log(f"KOMA [{event}{', image' if use_image else ''}"
                f"{', lore ' + ','.join(lore_ids) if lore_ids else ''}] "
                f"({time.monotonic() - started:.1f}s): {answer}")
            history.append({"role": "user",
                            "content": question or f"(The camera detector saw: {', '.join(labels)}.)"})
            history.append({"role": "assistant", "content": answer})

    def handover():
        speaker.say(brain.handover_script)
        history.append({"role": "user", "content": "(A new visitor receives KOMA.)"})
        history.append({"role": "assistant", "content": "(KOMA gave its handover welcome.)"})
        session["handover_done"] = True
        log("KOMA [handover script]")

    def handle_voice(text):
        event = classify(text)
        if event == "new_visitor":
            history.clear()
            brain.recent_lore.clear()
            log("New visitor session started.")
            handover()
            return
        if event == "handover" and not session["handover_done"]:
            handover()
            return
        if event == "handover":
            event = "question"  # already introduced; answer it as a normal question
        with lock:
            frame, labels = latest["frame"], list(latest["labels"])
        question = text + (f" (camera detector currently guesses: {', '.join(labels)})" if labels else "")
        ask(event, frame, labels=labels, question=question)

    def vlm_worker():
        while not stopping.is_set():
            try:
                kind, payload = "voice", voice_requests.get(timeout=0.1)
            except queue.Empty:
                try:
                    kind, payload = "tool", detection_events.get_nowait()
                except queue.Empty:
                    continue
            cancel.clear()
            try:
                if kind == "voice":
                    status["text"] = "answering"
                    handle_voice(payload)
                else:
                    frame, labels = payload
                    status["text"] = f"looking at {', '.join(labels)}"
                    ask("tool", frame, labels=labels)
            except Exception as error:
                log(f"VLM request failed: {error}")
            finally:
                status["text"] = "watching"

    def listen_worker():
        proc = subprocess.Popen([str(STT_PYTHON), str(ROOT / "wake_listener.py")],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        children.append(proc)
        for line in proc.stdout:
            line = line.strip()
            if line == "READY":
                log('Say "Hey Koma" to talk to me.')
            elif line == "WAKE":
                cancel.set()
                speaker.flush()
                status["text"] = "listening"
                log("Wake phrase heard - listening...")
            elif line.startswith("REQUEST "):
                text = line[8:]
                log(f"Visitor: {text}")
                voice_requests.put(text)
            elif line == "TIMEOUT":
                status["text"] = "watching"
                log("Didn't catch a request.")
            if stopping.is_set():
                break
        if not stopping.is_set():
            log("Wake listener stopped; voice requests unavailable.")

    threading.Thread(target=vlm_worker, daemon=True).start()
    if not args.no_voice:
        threading.Thread(target=listen_worker, daemon=True).start()
    if args.save_events:
        (ROOT / "events").mkdir(exist_ok=True)

    streak = {}       # class -> consecutive frames seen
    last_fired = {}   # class -> monotonic time of last trigger
    stats_t, stats_frames, stats_infer_ms, fps = time.monotonic(), 0, 0.0, 0.0
    if args.handover_on_start:
        handover()
    else:
        speaker.say("KOMA is awake.")
    log("Running. Ctrl+C to stop.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera read failed")
            frame = apply_color_lut(frame, lut)
            result = model.predict(frame, conf=args.conf, device=0, verbose=False)[0]
            seen = {}
            for box in result.boxes:
                name = result.names[int(box.cls[0])]
                seen[name] = max(seen.get(name, 0.0), float(box.conf[0]))
            with lock:
                latest["frame"] = frame
                latest["labels"] = [n for n in seen if n not in ignore]

            streak = {n: streak.get(n, 0) + 1 for n in seen}
            now = time.monotonic()
            ready = [n for n, count in streak.items()
                     if count >= args.confirm_frames and n not in ignore
                     and now - last_fired.get(n, -1e9) >= args.cooldown]
            # Only trigger when KOMA is idle, so tool comments never talk over a conversation.
            idle = status["text"] == "watching" and speaker.idle.is_set()
            if ready and idle and detection_events.empty() and voice_requests.empty():
                for n in ready:
                    last_fired[n] = now
                log(f"Detected: {', '.join(f'{n} ({seen[n]:.0%})' for n in ready)}")
                detection_events.put_nowait((frame.copy(), ready))
                if args.save_events:
                    cv2.imwrite(str(ROOT / "events" / f"{datetime.now():%Y%m%d-%H%M%S}.jpg"), result.plot())

            stats_frames += 1
            stats_infer_ms += sum(result.speed.values())
            if now - stats_t >= 60:
                fps = stats_frames / (now - stats_t)
                log(f"Perf: {fps:.1f} fps, YOLO {stats_infer_ms / stats_frames:.1f} ms/frame")
                stats_t, stats_frames, stats_infer_ms = now, 0, 0.0
            if args.preview:
                view = result.plot()
                cv2.putText(view, f"{fps:.0f} fps | {status['text']}", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("KOMA", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        stopping.set()
        cancel.set()
        speaker.close()
        for proc in children:
            proc.terminate()
        camera.release()
        cv2.destroyAllWindows()
        log("Stopped.")


if __name__ == "__main__":
    main()
