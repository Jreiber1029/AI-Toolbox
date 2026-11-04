#!/home/jasonreiber/ultralytics-env/bin/python
"""KOMA: 'Koma' wake word -> Qwen VLM -> Piper TTS.

The persona, 2070 lore and handover come from the KomaDemoRag-v4 pack (its
koma_demo.py retriever); --old-lore uses brain.py's HELIOS 2100 lore instead.
Spoken questions that refer to the scene ("what is this?") send the current frame
to Qwen. A light TensorRT detector (yolo_trt.py, no PyTorch) checks the camera a few
times a second. When a person holds a tool (its box touches their person or face box,
widened - see ANCHORS) for --confirm-seconds,
KOMA speaks that tool's line from tool_prompts/<tool>.txt word for word through Piper -
the VLM and persona are not involved. Each tool triggers once per run. A person alone does
nothing, and the detector pauses while KOMA is speaking or answering. --no-detect turns the detector off.
"""
import argparse
import importlib.util
import re
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from brain import SENTENCE_LIMIT, Brain, classify
from koma import (CSI_PIPELINE, FALLBACK_VOICE, MAX_TOKENS, ROOT, STT_PYTHON, Speaker,
                  apply_color_lut, encode_frame, log, make_color_lut, pick_file, stream_vlm)

DEFAULT_PACK = ROOT / "KomaDemoRag-v4"
TOOL_MODEL = ROOT / "YOLO26toolmodel/toolppefinetune.engine"
# Classes that stand for a visitor, with the margins (side, up, down) their box is widened by,
# as fractions of its width/height. A face box is small and a held tool sits well below it.
ANCHORS = {"person": (0.25, 0.25, 0.25), "face": (1.5, 0.25, 3.0)}
TOOL_PROMPTS = ROOT / "tool_prompts"
WELCOME = ROOT / "welcome.txt"
QR_CHECK_SECONDS = 0.2  # how often the camera thread looks for the ID card before it is accepted
DETECT_GAP = 0.5  # seconds a tool may drop out of detection without restarting its count
# Looser than the pack's own check, so a misheard "...do you see?" still skips lore.
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
LOOK_REQUEST = re.compile(r"\b(?:you see|can see|you looking at|in front of you|describe (?:what|the|this))\b", re.I)


class PackBrain:
    """KOMA's persona and lore from a KomaDemoRag pack, with the same interface as brain.Brain."""

    def __init__(self, folder):
        spec = importlib.util.spec_from_file_location("koma_demo", folder / "koma_demo.py")
        self.demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.demo)
        self.kb = self.demo.KomaKnowledge()
        self.lore = type("Lore", (), {"records": self.kb.all_records})()
        self.system_prompt = (folder / "koma_system_prompt.txt").read_text(encoding="utf-8")
        self.handover_script = (folder / "handover.txt").read_text(encoding="utf-8").strip()
        self.runtime_facts = "prototype passages"  # the pack's prototype_rag.jsonl covers these
        self.recent_lore = []
        self.last_topic = ""  # lets "why?" follow on from the previous answer

    def wants_image(self, event, question, labels):
        return (Brain.wants_image(event, question, labels) or self.demo.is_visual_question(question)
                or bool(LOOK_REQUEST.search(question)))

    def build_messages(self, event, *, labels=(), question="", image_b64=None, history=()):
        if event == "question" and LOOK_REQUEST.search(question):
            event = "visual"  # describe the frame only; no lore to steer the answer
        prompt = self.demo.build_prompt(event, detector_labels=list(labels), question=question,
                                        knowledge=self.kb, last_topic=self.last_topic)
        # build_prompt starts with the system prompt; send that as the system message so
        # llama.cpp reuses its cached prefix, and the rest as this turn.
        text = prompt[len(self.system_prompt):].strip()
        text += f"\n\nReply in at most {SENTENCE_LIMIT.get(event, 2)} short spoken sentences."
        query = question or " ".join(labels)
        hits = [] if event in ("safety", "visual") else self.kb.retrieve(query, last_topic=self.last_topic)
        story = [h for h in hits if h.get("kind") != "definition"]  # a definition is never the follow-up topic
        if story:
            self.last_topic = f"{story[0]['title']} {story[0]['keywords']}"
        content = [{"type": "text", "text": text}]
        if image_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        messages = [{"role": "system", "content": self.system_prompt}, *history,
                    {"role": "user", "content": content}]
        return messages, [r["id"] for r in hits]


def near(box, anchor, margins):
    """True if box overlaps the anchor's box widened by margins (a held tool sits at its edge)."""
    x1, y1, x2, y2 = anchor
    side, up, down = margins
    w, h = x2 - x1, y2 - y1
    return box[0] < x2 + w * side and box[2] > x1 - w * side and box[1] < y2 + h * down and box[3] > y1 - h * up


def load_lines(path):
    """A file's text without its '#' note lines."""
    return "\n".join(l for l in path.read_text(encoding="utf-8").splitlines()
                     if not l.lstrip().startswith("#")).strip()


def load_tool_prompts(folder):
    """{tool class (lowercase): prompt} from folder/<tool>.txt; '#' lines are notes, not prompt."""
    prompts = {}
    for path in sorted(folder.glob("*.txt")):
        text = load_lines(path)
        if text:
            prompts[path.stem.lower()] = text
        else:
            log(f"Tool prompt {path.name} is empty - {path.stem} will not trigger KOMA.")
    return prompts


class SilentSpeaker:
    """Stands in for Speaker with --mute: Piper is never started and replies are only logged."""

    def __init__(self):
        self.idle = threading.Event()
        self.idle.set()

    def say(self, text):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--voice", help="Piper .onnx (default: first file in voices/)")
    p.add_argument("--voice-speed", type=float, default=1.2,
                   help="Piper length scale: 1.0 is normal, higher is slower (default 1.2)")
    p.add_argument("--voice-pitch", type=float, default=0.0,
                   help="Pitch shift in semitones (default 0 = the voice's normal pitch)")
    p.add_argument("--voice-volume", type=float, default=0.9,
                   help="Voice gain; the KOMA voice peaks near clipping at 1.0 (default 0.9)")
    p.add_argument("--vlm-url", default="http://127.0.0.1:8080/v1/chat/completions")
    p.add_argument("--camera", default="csi:0", help="'csi:N' for nvargus sensor N, or a V4L2 index/path")
    p.add_argument("--no-camera", action="store_true", help="Text-only: never send a frame to Qwen")
    p.add_argument("--no-vision", action="store_true",
                   help="Keep the camera and detector but never send a frame (text-only models)")
    p.add_argument("--type", action="store_true",
                   help="Type requests in this terminal instead of speaking (no microphone or STT)")
    p.add_argument("--mute", action="store_true",
                   help="Don't start Piper; replies only appear in this terminal (pair with --type)")
    p.add_argument("--rotate", type=int, choices=sorted(ROTATIONS), default=0,
                   help="Rotate camera frames clockwise by this many degrees (180 = upside down)")
    p.add_argument("--stream", type=int, default=0, metavar="PORT",
                   help="Serve the live camera view at http://<jetson>:PORT in a browser (0 = off)")
    p.add_argument("--image-width", type=int, default=640, help="Resize frames to this width for the VLM")
    p.add_argument("--memory-turns", type=int, default=3, help="Exchanges kept as recent dialogue")
    p.add_argument("--pack", type=Path, default=DEFAULT_PACK, help="KomaDemoRag folder with koma_demo.py")
    p.add_argument("--old-lore", action="store_true", help="Use brain.py's HELIOS 2100 lore instead of the pack")
    p.add_argument("--handover-on-start", action="store_true",
                   help="Speak KOMA's handover welcome at startup")
    p.add_argument("--greeting", help='Say this at startup instead of "KOMA is awake."; "" starts silently (qr_start.py already greeted)')
    p.add_argument("--no-color-fix", action="store_true", help="Skip the IMX219 lens color correction")
    p.add_argument("--no-detect", action="store_true", help="Don't run the tool detector")
    p.add_argument("--tool-model", type=Path, default=TOOL_MODEL,
                   help="TensorRT engine with tool classes and a Person or Face class")
    p.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    p.add_argument("--detect-fps", type=float, default=5, help="Detector checks per second")
    p.add_argument("--confirm-seconds", type=float, default=1,
                   help="Seconds a tool must stay next to a person before KOMA comments (default 1)")
    p.add_argument("--tool-prompts", type=Path, default=TOOL_PROMPTS,
                   help="Folder of <tool>.txt lines KOMA speaks word for word; a tool triggers only if its file has text")
    p.add_argument("--qr-code", help="Load everything, then wait until the camera reads this QR code "
                                     "before speaking --welcome and listening (qr_start.py sets it)")
    p.add_argument("--welcome", type=Path, default=WELCOME,
                   help="What KOMA says word for word when the --qr-code card is accepted ('#' lines are notes)")
    args = p.parse_args()

    voice_path = pick_file(args.voice, ROOT / "voices", ["*.onnx"], FALLBACK_VOICE)
    if args.mute:
        log("Muted: Piper not started, replies are text only.")
    else:
        log(f"Piper voice: {voice_path}{'  (fallback - copy yours to voices/)' if voice_path == FALLBACK_VOICE else ''}"
            f"  [speed {args.voice_speed}, pitch {args.voice_pitch:+g} st, volume {args.voice_volume}]")

    brain = Brain() if args.old_lore else PackBrain(args.pack)
    log(f"Lore source: {'Koma-storyrag (old)' if args.old_lore else args.pack}")
    log(f"Lore: {len(brain.lore.records)} records; runtime facts {'loaded' if brain.runtime_facts else 'MISSING'}")

    # Only the USB cam is switched to a sharper mode for reading the ID card.
    qr_mode = bool(args.qr_code) and not args.camera.startswith("csi:")

    def open_camera(for_qr=False):
        if args.camera.startswith("csi:"):
            return cv2.VideoCapture(CSI_PIPELINE.format(sensor=args.camera[4:]), cv2.CAP_GSTREAMER)
        source = int(args.camera) if args.camera.isdigit() else args.camera
        if for_qr:
            # At the usual 640x480 a held-up card is too few pixels on the wide 16MP USB cam.
            # Only the V4L2 backend honours MJPG (GStreamer falls back to YUYV, 2 fps at 1080p);
            # 1080p MJPG runs ~12 fps, so it is used only until the card is accepted.
            cam = cv2.VideoCapture(source, cv2.CAP_V4L2)
            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        else:
            cam = cv2.VideoCapture(source)  # 640x480 at ~30 fps
        # V4L2 queues 4 frames by default, which go stale while nothing reads the camera.
        # With one buffer, snapshot()'s single grab() is enough to get a live frame.
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cam

    camera, lut = None, None
    if not args.no_camera:
        camera = open_camera(for_qr=qr_mode)
        if not camera.isOpened():
            raise SystemExit(f"Could not open camera {args.camera} (use --no-camera to run without it)")
        lut = None if args.no_color_fix else make_color_lut()
    if args.qr_code and camera is None:
        raise SystemExit("--qr-code needs the camera to read the ID card")
    welcome_text = load_lines(args.welcome) if args.qr_code else ""
    # Set once the ID card is accepted; until then voice requests are ignored and the
    # camera thread looks for the QR code instead of tools.
    accepted = threading.Event()
    if not args.qr_code:
        accepted.set()

    detector = None
    if camera is not None and not args.no_detect:
        from yolo_trt import TrtDetector
        detector = TrtDetector(args.tool_model)
        log(f"Detector: {args.tool_model.name} at {args.detect_fps:g} fps")
    tool_prompts = load_tool_prompts(args.tool_prompts) if detector else {}
    if detector:
        classes = {name.lower() for name in detector.names.values()}
        log(f"Tool prompts: {', '.join(sorted(tool_prompts)) or 'none'} "
            f"(held {args.confirm_seconds:g} s, once per run)")
        for tool in sorted(set(tool_prompts) - classes):
            log(f"Tool prompt {tool}.txt has no matching class in {args.tool_model.name}.")
        if not classes & ANCHORS.keys():
            log(f"{args.tool_model.name} has no person or face class, so no tool will trigger KOMA.")
    latest = {"frame": None, "found": []}  # found: the detector's newest boxes, drawn on --stream
    frame_lock = threading.Lock()

    def orient(frame):
        return cv2.rotate(frame, ROTATIONS[args.rotate]) if args.rotate else frame

    def snapshot():
        if camera is None:
            return None
        if detector or args.stream:  # the camera thread is reading continuously; use its newest frame
            with frame_lock:
                frame = latest["frame"]
            return None if frame is None else apply_color_lut(frame.copy(), lut)
        camera.grab()  # drop the buffered frame so the VLM sees the scene as it is now
        ok, frame = camera.read()
        return apply_color_lut(orient(frame), lut) if ok else None

    listener = None

    def tell_listener(speaking):
        if listener is None:
            return
        try:
            listener.stdin.write("SPEAKING\n" if speaking else "QUIET\n")
            listener.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass

    speaker = SilentSpeaker() if args.mute else Speaker(
        voice_path, length_scale=args.voice_speed, pitch=args.voice_pitch,
        volume=args.voice_volume, on_change=tell_listener)
    requests = queue.Queue()
    cancel = threading.Event()  # set when a wake word should abort current VLM output
    history = deque(maxlen=2 * args.memory_turns)  # this visitor's session only
    session = {"handover_done": False}
    fired = set()  # tools that have already triggered KOMA this run
    busy = threading.Event()  # set while a request is being answered

    def say_tool_line(tool):
        """A detected tool's scripted line goes straight to Piper, word for word."""
        line = tool_prompts[tool]
        speaker.say(line)
        log(f"KOMA [tool {tool}, scripted]: {line}")
        # Kept in the dialogue so the VLM knows what KOMA just said if asked about it.
        history.append({"role": "user", "content": f"(The visitor held up a {tool}.)"})
        history.append({"role": "assistant", "content": line})

    def ask(event, question, frame=None, labels=()):
        if args.no_vision:
            frame = None
        elif frame is None and brain.wants_image(event, question, labels):
            frame = snapshot()
        messages, lore_ids = brain.build_messages(
            event, labels=labels, question=question, history=tuple(history),
            image_b64=encode_frame(frame, args.image_width) if frame is not None else None)
        started = time.monotonic()
        reply = []
        for sentence in stream_vlm(args.vlm_url, messages, MAX_TOKENS[event], cancel):
            reply.append(sentence)
            speaker.say(sentence)
            if len(reply) >= SENTENCE_LIMIT[event]:
                break  # closing the stream stops generation on the server
        answer = " ".join(reply)
        if answer and not cancel.is_set():
            log(f"KOMA [{event}{', image' if frame is not None else ''}"
                f"{', lore ' + ','.join(lore_ids) if lore_ids else ''}] "
                f"({time.monotonic() - started:.1f}s): {answer}")
            history.append({"role": "user", "content": question or
                            f"(The visitor held up something the camera took for: {', '.join(labels)}.)"})
            history.append({"role": "assistant", "content": answer})

    def warm_up():
        """Have the VLM read KOMA's system prompt now, so the first answer after the card is quick."""
        if not hasattr(brain, "system_prompt"):
            return
        started = time.monotonic()
        messages = [{"role": "system", "content": brain.system_prompt}, {"role": "user", "content": "Hello."}]
        try:
            for _ in stream_vlm(args.vlm_url, messages, 1, threading.Event()):
                pass
            log(f"VLM warmed up: system prompt cached ({time.monotonic() - started:.1f}s).")
        except Exception as error:
            log(f"VLM warm-up failed (the first answer will be slower): {error}")

    def welcome():
        speaker.say(welcome_text)
        log(f"KOMA [welcome]: {welcome_text}")
        history.append({"role": "user", "content": "(The TradesMaster's ID card was accepted.)"})
        history.append({"role": "assistant", "content": welcome_text})

    def handover():
        speaker.say(brain.handover_script)
        history.append({"role": "user", "content": "(A new visitor receives KOMA.)"})
        history.append({"role": "assistant", "content": "(KOMA gave its handover welcome.)"})
        session["handover_done"] = True
        log("KOMA [handover script]")

    def handle(text):
        event = classify(text)
        if event == "new_visitor":
            history.clear()
            brain.recent_lore.clear()
            brain.last_topic = ""
            log("New visitor session started.")
            handover()
            return
        if event == "handover" and not session["handover_done"]:
            handover()
            return
        if event == "handover":
            event = "question"  # already introduced; answer it as a normal question
        ask(event, text)

    if not args.type:
        listener = subprocess.Popen([str(STT_PYTHON), str(ROOT / "wake_listener.py")],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def type_worker():
        log('Type a request and press Enter ("new visitor" resets; Ctrl+D stops).')
        for line in sys.stdin:
            if line.strip():
                requests.put(line.strip())
        requests.put(None)

    def listen_worker():
        for line in listener.stdout:
            line = line.strip()
            if line == "READY":
                log('Say "Koma" to talk to me ("Hey Koma" to interrupt).')
            elif not accepted.is_set():
                continue  # nobody has shown the ID card yet
            elif line == "WAKE":
                cancel.set()
                speaker.flush()
                log("Wake phrase heard - listening...")
            elif line.startswith("REQUEST "):
                log(f"Visitor: {line[8:]}")
                requests.put(line[8:])
            elif line == "TIMEOUT":
                log("Didn't catch a request.")
        requests.put(None)

    def camera_worker():
        nonlocal camera
        held_since, last_seen, next_check, last_names = {}, {}, 0.0, None
        qr, last_wrong = cv2.QRCodeDetector(), None
        while True:
            ok, raw = camera.read()  # keep reading so frames never go stale
            if not ok:
                log("Camera read failed; tool detection stopped.")
                return
            raw = orient(raw)
            with frame_lock:
                latest["frame"] = raw
            if not accepted.is_set():
                now = time.monotonic()
                if now >= next_check:
                    next_check = now + QR_CHECK_SECONDS
                    text = qr.detectAndDecode(raw)[0]
                    if text == args.qr_code:
                        log(f"ID card accepted ({args.qr_code}).")
                        accepted.set()
                        if qr_mode:  # back to the usual fast 640x480 for YOLO, the VLM and --stream
                            camera.release()
                            camera = open_camera()
                    elif text and text != last_wrong:
                        log(f"QR code {text!r} is not the start card - ignored.")
                        last_wrong = text
                continue
            if detector is None:
                continue  # only reading for --stream
            now = time.monotonic()
            if now < next_check:
                continue
            next_check = now + 1 / args.detect_fps
            idle = not busy.is_set() and speaker.idle.is_set() and requests.empty()
            if not idle or fired.issuperset(tool_prompts):
                held_since.clear()  # YOLO pauses while KOMA answers; counting starts again after
                latest["found"] = []
                continue
            found = detector.detect(apply_color_lut(raw.copy(), lut), args.conf)
            latest["found"] = found
            names = sorted({name for name, _, _ in found})
            if names != last_names:
                log(f"YOLO sees: {', '.join(names) or 'nothing'}")
                last_names = names
            anchors = [(box, ANCHORS[name.lower()]) for name, _, box in found if name.lower() in ANCHORS]
            for name, _, box in found:
                tool = name.lower()
                if tool in tool_prompts and tool not in fired and any(near(box, *a) for a in anchors):
                    held_since.setdefault(tool, now)
                    last_seen[tool] = now
            for tool in [t for t in held_since if now - last_seen[t] > DETECT_GAP]:
                del held_since[tool]  # put down or out of view: its count starts again
            ready = [t for t, since in held_since.items() if now - since >= args.confirm_seconds]
            if ready:
                tool = min(ready, key=held_since.get)  # the one held longest
                fired.add(tool)
                held_since.clear()
                log(f"Detected: {tool} next to a person for {args.confirm_seconds:g} s "
                    "- speaking its line (once per run).")
                requests.put(("tool", tool))

    threading.Thread(target=type_worker if args.type else listen_worker, daemon=True).start()
    if detector or args.qr_code or (camera is not None and args.stream):
        threading.Thread(target=camera_worker, daemon=True).start()
    if camera is not None and args.stream:
        class StreamHandler(BaseHTTPRequestHandler):
            # multipart/x-mixed-replace: the browser swaps in each JPEG as it arrives.
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        frame = snapshot()
                        if frame is not None:
                            for name, conf, (x1, y1, x2, y2) in latest["found"]:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 3)
                                cv2.putText(frame, f"{name} {conf:.0%}", (x1, max(y1 - 10, 25)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                            jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                        time.sleep(0.1)  # ~10 fps keeps the CPU free for STT and Piper
                except (BrokenPipeError, ConnectionResetError):
                    pass  # browser tab closed

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("0.0.0.0", args.stream), StreamHandler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        log(f"Live camera: http://localhost:{args.stream}  (or http://<jetson-ip>:{args.stream})")
    if args.greeting:
        speaker.say(args.greeting)
        log(f"KOMA [greeting]: {args.greeting}")
    if args.handover_on_start:
        handover()
    elif args.greeting is None:
        speaker.say("KOMA is awake.")
    if args.qr_code:
        warm_up()
    log("Running. Ctrl+C to stop.")
    try:
        if args.qr_code:
            log(f"Everything is loaded. Show the ID card to the camera to start ({args.qr_code}).")
            accepted.wait()
            welcome()
        while (item := requests.get()) is not None:
            cancel.clear()
            busy.set()
            try:
                if isinstance(item, tuple):
                    say_tool_line(item[1])
                else:
                    handle(item)
            except Exception as error:
                log(f"VLM request failed: {error}")
            finally:
                busy.clear()
        log("Input closed." if args.type else "Wake listener stopped; voice requests unavailable.")
    except KeyboardInterrupt:
        pass
    finally:
        cancel.set()
        speaker.close()
        if listener:
            listener.terminate()
        if camera is not None:
            camera.release()
        log("Stopped.")


if __name__ == "__main__":
    main()
