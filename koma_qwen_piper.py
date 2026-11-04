#!/home/jasonreiber/ultralytics-env/bin/python
"""KOMA's Qwen camera and Piper voice runner, without a YOLO model.

Uses the existing character, HELIOS lore, dialogue memory and Hey Koma listener.
The normal-pitch KOMA Piper voice runs on CPU. Camera observations are made only
while idle; one worker serializes observations and visitor questions.
"""
import argparse
import math
import os
import queue
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import cv2

from brain import Brain, SENTENCE_LIMIT, classify
from koma import (CSI_PIPELINE, MAX_TOKENS, ROOT, STT_PYTHON, TTS_PYTHON,
                  Speaker as BaseSpeaker, encode_frame, log, stream_vlm)

DEFAULT_VOICE = ROOT / "voices/koma_lessac_high.onnx"
OBSERVATION = (
    "Look at the current camera image. Briefly say what you can actually see, "
    "in one or two short spoken sentences. If a tool is clearly visible, say "
    "what it is and what it is for. Stay in KOMA's character; do not invent "
    "detector results, actions, hazards, or objects outside the image."
)


def stop_owned_process(proc):
    """Stop this child's private process group, including its audio subprocess."""
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=3)
    # The worker may exit before a child player or recorder does.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class Speaker(BaseSpeaker):
    """Reuse Piper's worker/protocol, owning playback and startup cleanup too."""

    def __init__(self, voice, length_scale, pitch, volume, on_change):
        self.on_change = on_change
        self.idle = threading.Event()
        self.idle.set()
        self.write_lock = threading.RLock()
        self.proc = subprocess.Popen(
            [str(TTS_PYTHON), str(ROOT / "tts_worker.py"), "--voice", str(voice),
             "--length-scale", str(length_scale), "--pitch", str(pitch),
             "--volume", str(volume)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
            start_new_session=True,
        )
        ready = threading.Event()
        loaded = threading.Event()

        def watch():
            for line in self.proc.stdout:
                line = line.strip()
                if line == "READY":
                    loaded.set()
                    ready.set()
                elif line == "IDLE":
                    self.idle.set()
                    self.on_change(False)
                elif line:
                    log(f"TTS: {line}")
            ready.set()

        threading.Thread(target=watch, daemon=True).start()
        try:
            if not ready.wait(60) or not loaded.is_set() or self.proc.poll() is not None:
                raise RuntimeError("Piper worker failed to become ready")
        except BaseException:
            self.close()
            raise

    def _send(self, message):
        with self.write_lock:
            if self.proc.poll() is None and not self.proc.stdin.closed:
                super()._send(message)

    def close(self):
        if self.proc.poll() is None:
            try:
                self.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass
            # EOF lets the worker remove its temporary playback WAVs normally.
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        stop_owned_process(self.proc)
        if self.proc.stdin:
            self.proc.stdin.close()
        if self.proc.stdout:
            self.proc.stdout.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="/dev/video1",
                        help="USB device path/index (default /dev/video1), or csi:N")
    parser.add_argument("--interval", type=float, default=20,
                        help="Seconds between idle camera observations; 0 disables them")
    parser.add_argument("--voice", type=Path, default=DEFAULT_VOICE)
    parser.add_argument("--voice-pitch", type=float, default=0,
                        help="Semitones: 0 keeps the voice's normal pitch (default)")
    parser.add_argument("--voice-speed", type=float, default=1,
                        help="Piper length scale: 1 is normal, higher is slower")
    parser.add_argument("--voice-volume", type=float, default=0.9)
    parser.add_argument("--vlm-url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--memory-turns", type=int, default=3)
    parser.add_argument("--handover-on-start", action="store_true")
    parser.add_argument("--preview", action="store_true", help="Show a local camera window")
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable the microphone/wake listener; Piper still speaks")
    parser.add_argument("--stt", choices=("vosk", "whisper"), default="vosk",
                        help="Request recognizer; vosk saves the extra Whisper model memory")
    parser.add_argument("--ask", help="Ask this question on startup")
    parser.add_argument("--once", action="store_true",
                        help="Speak one --ask answer or one camera observation, then stop")
    return parser


def validate_args(parser, args):
    for name in ("interval", "voice_pitch", "voice_speed", "voice_volume"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if args.interval < 0 or args.voice_speed <= 0 or not 0 < args.voice_volume <= 2:
        parser.error("interval must be >= 0, voice speed > 0, and voice volume in (0, 2]")
    if not -12 <= args.voice_pitch <= 12:
        parser.error("--voice-pitch must be between -12 and 12 semitones")
    if not 1 <= args.image_width <= 4096 or not 0 <= args.memory_turns <= 20:
        parser.error("image width must be 1..4096 and memory turns 0..20")
    if args.ask is not None and not args.ask.strip():
        parser.error("--ask needs a nonempty question")
    args.voice = args.voice.expanduser().resolve()
    required = [args.voice, Path(str(args.voice) + ".json"), TTS_PYTHON, ROOT / "tts_worker.py"]
    if not args.no_voice and not args.once:
        required += [STT_PYTHON, ROOT / "wake_listener.py"]
    for path in required:
        if not path.is_file():
            parser.error(f"Required file is missing: {path}")
    if args.preview and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        parser.error("--preview requires a graphical desktop; omit it over SSH")
    if args.camera.startswith("csi:"):
        if not args.camera[4:].isdigit():
            parser.error("CSI camera must be written as csi:N, for example csi:0")
    elif not args.camera.isdigit() and not Path(args.camera).exists():
        parser.error(f"Camera device does not exist: {args.camera}")


def camera_pipeline(camera):
    if camera.startswith("csi:"):
        return CSI_PIPELINE.format(sensor=camera[4:])
    device = "/dev/video" + camera if camera.isdigit() else camera
    escaped = device.replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'v4l2src device="{escaped}" ! '
        "image/jpeg,width=1280,height=720,framerate=30/1 ! "
        "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def runtime_facts(brain, camera, stt, listener_enabled=True):
    """Correct this run's facts in memory; leave the shared facts file intact."""
    camera_name = "a USB camera" if not camera.startswith("csi:") else "a CSI camera"
    existing = brain.runtime_facts.splitlines()
    retained = [line for line in existing
                if "IMX219" not in line and "YOLO" not in line
                and "Camera frames and audio" not in line]
    recognition = (
        f"offline Vosk wake recognition for Hey KOMA and {stt} request recognition"
        if listener_enabled else "no microphone or wake listener in this run"
    )
    recording = (
        "The current wake listener saves the latest spoken request to logs/last_request.wav. "
        if listener_enabled else "This run does not record microphone audio. "
    )
    brain.runtime_facts = (
        "This exhibition prototype is a real NVIDIA Jetson Orin Nano computer inside the bag. "
        f"This runner uses {camera_name}, a local Qwen vision-language model, "
        f"{recognition}, "
        "and the KOMA Piper voice on CPU through the Jabra speakerphone. "
        "YOLO is disabled: there is no separate object detector in this runner. "
        "Camera frames are processed in memory and are not saved. "
        f"{recording}Piper uses temporary WAV files for playback. Dialogue is kept only for the last few "
        "exchanges of this session and cleared for a new visitor or when the runner stops.\n"
        + "\n".join(retained)
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    # A shell may start this runner in the background with SIGINT ignored.
    # Restore Ctrl+C and route the launcher's TERM through the same cleanup.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    brain = Brain()
    runtime_facts(brain, args.camera, args.stt, not args.no_voice and not args.once)
    log(f"Qwen + Piper; YOLO disabled. Lore: {len(brain.lore.records)} records.")
    log(f"Piper voice: {args.voice} [pitch {args.voice_pitch:+g} st, "
        f"length scale {args.voice_speed:g}, volume {args.voice_volume:g}]")
    stopping, cancel, done = threading.Event(), threading.Event(), threading.Event()
    listening, busy = threading.Event(), threading.Event()
    requests = queue.Queue(maxsize=1)
    lock = threading.RLock()
    history = deque(maxlen=2 * args.memory_turns)
    state = {"frame": None, "frame_time": 0, "epoch": 0, "handover_done": False,
             "status": "watching", "last_activity": time.monotonic(), "failed": False}
    camera = speaker = listener = None
    worker = None

    def tell_listener(speaking):
        if listener is not None and listener.poll() is None:
            try:
                listener.stdin.write("SPEAKING\n" if speaking else "QUIET\n")
                listener.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass

    def submit(kind, text="", interrupt=False):
        with lock:
            if interrupt:
                state["epoch"] += 1
                cancel.set()
                speaker.flush()
            while not requests.empty():
                try:
                    requests.get_nowait()
                except queue.Empty:
                    break
            requests.put_nowait((state["epoch"], kind, text))

    def handover():
        speaker.say(brain.handover_script)
        history.extend([{"role": "user", "content": "(A new visitor receives KOMA.)"},
                        {"role": "assistant", "content": "(KOMA gave its handover welcome.)"}])
        state["handover_done"] = True
        log("KOMA [handover script]")

    def answer(kind, text):
        observation = kind == "observation"
        event = "question" if observation else classify(text)
        if event == "new_visitor":
            history.clear()
            brain.recent_lore.clear()
            handover()
            return
        if event == "handover":
            if not state["handover_done"]:
                handover()
                return
            event = "question"
        question = OBSERVATION if observation else text
        with lock:
            frame = state["frame"]
            fresh = time.monotonic() - state["frame_time"] <= 2
        wants_image = brain.wants_image(event, question, ())
        if wants_image and (frame is None or not fresh):
            raise RuntimeError("No fresh camera frame; skipping this request")
        messages, lore_ids = brain.build_messages(
            event, question=question, history=tuple(history),
            image_b64=encode_frame(frame, args.image_width) if wants_image else None,
        )
        # These facts also clarify that an image-only observation has no detector trigger.
        limit = 2 if observation else SENTENCE_LIMIT[event]
        if observation:
            messages[-1]["content"][0]["text"] += "\nFor this observation, use at most two short sentences."
        reply = []
        stream = stream_vlm(args.vlm_url, messages, 90 if observation else MAX_TOKENS[event], cancel)
        try:
            for sentence in stream:
                with lock:
                    if stopping.is_set() or cancel.is_set():
                        break
                    reply.append(sentence)
                    speaker.say(sentence)
                if len(reply) >= limit:
                    break
        finally:
            stream.close()
        if reply and not cancel.is_set():
            response = " ".join(reply)
            history.extend([{"role": "user", "content": question},
                            {"role": "assistant", "content": response}])
            log(f"KOMA [{kind}, {event}, lore {','.join(lore_ids) or 'none'}]: {response}")

    def vlm_worker():
        while not stopping.is_set():
            try:
                epoch, kind, text = requests.get(timeout=0.1)
            except queue.Empty:
                with lock:
                    idle = (not args.once and args.interval > 0 and not listening.is_set()
                            and speaker.idle.is_set()
                            and time.monotonic() - state["last_activity"] >= args.interval)
                    if not idle:
                        continue
                    epoch, kind, text = state["epoch"], "observation", ""
            with lock:
                if epoch != state["epoch"] or listening.is_set():
                    continue
                cancel.clear()
                busy.set()
                state["status"] = "looking" if kind == "observation" else "answering"
            try:
                answer(kind, text)
            except Exception as error:
                state["failed"] = True
                log(f"VLM request failed: {error}")
            finally:
                busy.clear()
                state["last_activity"] = time.monotonic()
                if not listening.is_set():
                    state["status"] = "watching"
                if args.once:
                    done.set()
                    return

    def listen_worker():
        for line in listener.stdout:
            if stopping.is_set():
                break
            line = line.strip()
            if line == "READY":
                log('Say "Hey Koma" to talk to me.')
            elif line == "WAKE":
                with lock:
                    state["epoch"] += 1
                    cancel.set()
                    listening.set()
                    state["status"] = "listening"
                    speaker.flush()
                log("Wake phrase heard - listening...")
            elif line.startswith("REQUEST "):
                text = line[8:].strip()
                log(f"Visitor: {text}")
                with lock:
                    listening.clear()
                    submit("voice", text)
            elif line == "TIMEOUT":
                listening.clear()
                state["last_activity"] = time.monotonic()
                state["status"] = "watching"
                log("Didn't catch a request.")
        if not stopping.is_set():
            listening.clear()
            log("Wake listener stopped; use Ctrl+C to stop and check the Jabra connection.")

    try:
        camera = cv2.VideoCapture(camera_pipeline(args.camera), cv2.CAP_GSTREAMER)
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {args.camera}; stop any other camera preview first")
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("Camera did not provide a frame")
        state["frame"], state["frame_time"] = frame, time.monotonic()
        speaker = Speaker(args.voice, args.voice_speed, args.voice_pitch, args.voice_volume, tell_listener)
        if not args.no_voice and not args.once:
            listener = subprocess.Popen(
                [str(STT_PYTHON), str(ROOT / "wake_listener.py"), "--stt", args.stt],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                start_new_session=True,
            )
            threading.Thread(target=listen_worker, daemon=True).start()
        if args.ask:
            submit("voice", args.ask)
        elif args.once:
            submit("observation")
        elif args.handover_on_start:
            handover()
        else:
            speaker.say("KOMA is awake.")
        # Piper startup can take longer than the allowed frame age.
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("Camera read failed after Piper startup")
        state["frame"], state["frame_time"] = frame, time.monotonic()
        worker = threading.Thread(target=vlm_worker, daemon=True)
        worker.start()
        log("Running Qwen + Piper without YOLO. Ctrl+C to stop.")
        while not stopping.is_set():
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera read failed; check the USB connection")
            with lock:
                state["frame"], state["frame_time"] = frame, time.monotonic()
            if args.once and done.is_set() and speaker.idle.is_set():
                break
            if speaker.proc.poll() is not None:
                raise RuntimeError("Piper worker stopped unexpectedly")
            if args.preview:
                view = frame.copy()
                cv2.putText(view, f"KOMA: {state['status']} | YOLO off", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("KOMA - Qwen + Piper", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        stopping.set()
        cancel.set()
        stop_owned_process(listener)
        if speaker is not None:
            speaker.close()
        if camera is not None:
            camera.release()
        if args.preview:
            cv2.destroyAllWindows()
        if worker is not None:
            worker.join(timeout=1)
        log("Stopped.")
    return 1 if args.once and state["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
