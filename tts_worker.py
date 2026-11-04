#!/home/jasonreiber/tts-env/bin/python
"""Persistent Piper TTS worker: keeps the voice loaded and plays on the Jabra.

Reads JSON lines on stdin:
  {"text": "..."}   queue a sentence to speak
  {"cmd": "flush"}  stop current playback and drop anything queued
Prints READY once loaded, and IDLE whenever the queue empties.
"""
import argparse
import json
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path

from piper import PiperVoice, SynthesisConfig


def find_jabra_sink():
    nodes = json.loads(subprocess.check_output(["pw-dump"], text=True))
    for node in nodes:
        props = node.get("info", {}).get("props", {})
        if props.get("media.class") == "Audio/Sink" and "Jabra" in props.get("node.name", ""):
            return props["node.name"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", type=Path, required=True, help="Piper .onnx (with .onnx.json beside it)")
    parser.add_argument("--length-scale", type=float, default=1.0,
                        help="Speaking pace: 1.0 is the voice's normal speed, higher is slower")
    parser.add_argument("--pitch", type=float, default=0.0,
                        help="Pitch shift in semitones via sox (speed unchanged), e.g. 2")
    parser.add_argument("--volume", type=float, default=1.0, help="Output gain; below 1.0 avoids clipping")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # koma.py handles Ctrl+C and stops us

    voice = PiperVoice.load(str(args.voice))
    config = SynthesisConfig(length_scale=args.length_scale, volume=args.volume)
    target = find_jabra_sink()
    if target is None:
        print("WARN Jabra output not found; using default PipeWire sink", flush=True)

    # Two stages so the next sentence is synthesized while the current one plays:
    # texts -> synthesizer thread -> clips -> player (main thread).
    texts = queue.Queue()
    clips = queue.Queue()
    lock = threading.Lock()
    # generation increments on flush so stale sentences are dropped at every stage;
    # pending counts sentences of the current generation not yet finished playing.
    state = {"player": None, "generation": 0, "pending": 0}

    def reader():
        for line in sys.stdin:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("cmd") == "flush":
                with lock:
                    state["generation"] += 1
                    state["pending"] = 0
                    if state["player"] and state["player"].poll() is None:
                        state["player"].terminate()
            elif msg.get("text", "").strip():
                with lock:
                    state["pending"] += 1
                    texts.put((state["generation"], msg["text"]))
        texts.put(None)

    def synthesizer(temp):
        count = 0
        while True:
            item = texts.get()
            if item is None:
                clips.put(None)
                return
            generation, text = item
            if generation != state["generation"]:
                continue
            raw = Path(temp) / f"{count}.wav"
            count += 1
            with wave.open(str(raw), "wb") as wav:
                voice.synthesize_wav(text, wav, syn_config=config)
            clip = raw
            if args.pitch:
                # sox measures pitch in cents (100 per semitone); tempo is preserved.
                clip = raw.with_suffix(".pitched.wav")
                subprocess.run(["sox", "-q", str(raw), str(clip), "pitch",
                                str(round(args.pitch * 100))], check=True)
                raw.unlink()
            clips.put((generation, clip))

    threading.Thread(target=reader, daemon=True).start()
    print("READY", flush=True)

    with tempfile.TemporaryDirectory(prefix="koma-tts-") as temp:
        threading.Thread(target=synthesizer, args=(temp,), daemon=True).start()
        while (item := clips.get()) is not None:
            generation, clip = item
            command = ["pw-play"] + (["--target", target] if target else []) + [str(clip)]
            with lock:
                stale = generation != state["generation"]
                if not stale:
                    state["player"] = subprocess.Popen(command)
            if not stale:
                state["player"].wait()
            clip.unlink(missing_ok=True)
            with lock:
                if not stale and generation == state["generation"]:
                    state["pending"] -= 1
                    if state["pending"] == 0:
                        print("IDLE", flush=True)


if __name__ == "__main__":
    main()
