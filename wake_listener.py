#!/home/jasonreiber/stt-env/bin/python
"""Offline 'Koma' wake word + request transcription from the Jabra mic.

Vosk spots the wake phrase. The request is then recorded until Silero VAD hears
--end-silence seconds without speech, and Whisper transcribes it (much more
accurate than the small Vosk model). Use --stt vosk to skip Whisper.

Prints one line per event on stdout for koma.py to read:
  READY            listener is waiting for the wake phrase
  WAKE             wake phrase heard; now transcribing the request
  REQUEST <text>   the transcribed request
  TIMEOUT          wake phrase heard but no request recognized

koma.py writes SPEAKING / QUIET on stdin. While KOMA is speaking, only a complete,
high-confidence "Hey Koma" interrupts it, so KOMA's own voice ("I'm KOMA") doesn't
wake it; while quiet, "Koma" alone works and partial results trigger for a faster response.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import wave

import re

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps
from vosk import KaldiRecognizer, Model, SetLogLevel

RATE = 16000
LAST_REQUEST = Path(__file__).resolve().parent / "logs/last_request.wav"
DEFAULT_MODEL = Path.home() / "voiceAssistant/stt/models/vosk-model-small-en-us-0.15"
# Vosk spells the name several ways depending on accent; accept all of them.
# "Koma" alone wakes KOMA while it is quiet ("hey koma" contains it, so that works too).
WAKE_PHRASES = ["koma", "coma", "comma"]
# KOMA says its own name ("I'm KOMA"), so while it speaks only "hey/okay koma" interrupts it.
SPEAKING_PHRASES = ["hey koma", "hey coma", "hey comma", "okay koma"]
# Whisper also hears the wake phrase in the pre-roll audio. It must appear near the start of
# the transcript (otherwise background chatter fooled Vosk), and is stripped from the request.
WAKE_NAME = re.compile(r"(?:(?:hey|hi|okay|ok)\W+)?(?:koma|coma|como|comma|karma|kuma)\b\W*", re.I)
WAKE_SEARCH_CHARS = 60  # how far into the transcript the wake phrase may start
PREROLL_CHUNKS = 12  # 0.1 s chunks kept from before the wake phrase, enough for Whisper to hear all of it
# A real wake phrase stays in Vosk's partial result for a second or more; chatter that the
# small grammar bends into "hey koma" usually shows up for a single chunk.
WAKE_PARTIAL_CHUNKS = 3
# Vosk only proposes a wake. Whisper must also hear the name in the pre-roll plus this many
# 0.1 s chunks before WAKE is sent, so distant voices neither interrupt KOMA nor block the mic.
VERIFY_CHUNKS = 4
# Biases Whisper toward the names visitors are likely to say.
WHISPER_PROMPT = "KOMA. Hey KOMA. HELIOS, PPES, TradesMaster."


def emit(line):
    print(line, flush=True)


def debug(line):
    print(f"[stt] {line}", file=sys.stderr, flush=True)


def find_jabra_source():
    """PipeWire node name of the Jabra mic, so capture shares the device with playback."""
    nodes = json.loads(subprocess.check_output(["pw-dump"], text=True))
    for node in nodes:
        props = node.get("info", {}).get("props", {})
        if props.get("media.class") == "Audio/Source" and "Jabra" in props.get("node.name", ""):
            return props["node.name"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="Capture straight from this ALSA device (e.g. plughw:CARD=USB,DEV=0) "
                                         "instead of through PipeWire")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=15,
                        help="Max seconds to record the request after the wake phrase")
    parser.add_argument("--end-silence", type=float, default=0.8,
                        help="Seconds of silence after speech that end the request")
    parser.add_argument("--start-timeout", type=float, default=5,
                        help="Give up if no speech starts within this many seconds of the wake phrase")
    parser.add_argument("--preroll", type=float, default=PREROLL_CHUNKS / 10,
                        help="Seconds of audio from before the wake phrase given to Whisper")
    parser.add_argument("--stt", choices=["whisper", "vosk"], default="whisper",
                        help="Recognizer for the request (the wake phrase always uses Vosk)")
    parser.add_argument("--whisper-model", default="base.en",
                        help="faster-whisper model: tiny.en is faster, small.en more accurate but slow")
    parser.add_argument("--speaking-conf", type=float, default=0.3,
                        help="Min word confidence for the wake phrase while KOMA is speaking")
    parser.add_argument("--wav", type=Path, help="Test with a mono 16-bit 16 kHz WAV instead of the mic")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # koma.py handles Ctrl+C and stops us

    SetLogLevel(-1)
    model = Model(str(args.model))
    wake = KaldiRecognizer(model, RATE, json.dumps(SPEAKING_PHRASES + WAKE_PHRASES + ["[unk]"]))
    wake.SetWords(True)
    whisper = None
    if args.stt == "whisper":
        from faster_whisper import WhisperModel
        whisper = WhisperModel(args.whisper_model, device="cpu", compute_type="int8", cpu_threads=4)

    # Strict options decide that the visitor has started talking (ignoring clicks and
    # the tail of the wake phrase); sensitive ones notice when they resume after a pause.
    vad_options = VadOptions(min_speech_duration_ms=250, min_silence_duration_ms=300, speech_pad_ms=100)
    vad_sensitive = VadOptions(min_silence_duration_ms=300, speech_pad_ms=100)

    def transcribe(pcm):
        if whisper is None:
            vosk = KaldiRecognizer(model, RATE)
            vosk.AcceptWaveform(pcm)
            return json.loads(vosk.FinalResult()).get("text", "")
        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
        # Whisper invents sentences from silence, so check there is real speech first,
        # and let vad_filter cut the silent parts out. temperature=0 skips the slow
        # re-decode retries on unclear audio.
        if not get_speech_timestamps(audio, vad_options):
            return ""
        segments, _ = whisper.transcribe(audio, language="en", beam_size=1, temperature=0.0,
                                         condition_on_previous_text=False,
                                         initial_prompt=WHISPER_PROMPT, vad_filter=True,
                                         vad_parameters={"min_silence_duration_ms": 500})
        texts = []
        for segment in segments:
            text = segment.text.strip()
            if text and text not in texts[-1:]:  # drop hallucinated repeats
                texts.append(text)
        return " ".join(texts)

    def wake_said(pcm):
        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
        # No initial_prompt here: it would nudge Whisper toward writing "KOMA".
        segments, _ = whisper.transcribe(audio, language="en", beam_size=1, temperature=0.0,
                                         condition_on_previous_text=False, without_timestamps=True)
        return " ".join(segment.text.strip() for segment in segments)

    command = None
    pending = None  # audio since Vosk proposed a wake, waiting for Whisper to confirm it
    speaking = threading.Event()

    def read_control():
        for line in sys.stdin:
            (speaking.set if line.strip() == "SPEAKING" else speaking.clear)()

    if not sys.stdin.isatty():
        threading.Thread(target=read_control, daemon=True).start()

    if args.wav:
        source = wave.open(str(args.wav), "rb")
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, RATE):
            parser.error("WAV must be mono, 16-bit PCM, 16000 Hz")
        read = lambda: source.readframes(1600)
        recorder = None
    else:
        # Capturing through PipeWire (not raw ALSA) leaves the Jabra free for speech playback.
        if args.device:
            capture = ["arecord", "-q", "-D", args.device, "-t", "raw",
                       "-f", "S16_LE", "-r", str(RATE), "-c", "1"]
        else:
            source = find_jabra_source()
            if source is None:
                raise SystemExit("Jabra microphone not found in PipeWire; check its USB connection.")
            capture = ["pw-record", "--target", source, "--rate", str(RATE),
                       "--channels", "1", "--format", "s16", "-"]
        recorder = subprocess.Popen(capture, stdout=subprocess.PIPE)
        read = lambda: recorder.stdout.read(3200)

    partial_hits = 0  # consecutive chunks whose partial result held a wake phrase

    def heard_wake(data):
        nonlocal partial_hits
        if wake.AcceptWaveform(data):
            partial_hits = 0
            result = json.loads(wake.Result())
            words = result.get("result", [])
            if not any(p in result.get("text", "") for p in WAKE_PHRASES):
                return False
            if speaking.is_set():
                if not any(p in result.get("text", "") for p in SPEAKING_PHRASES):
                    return False
                confs = [w["conf"] for w in words if w["word"] != "[unk]"]
                return bool(confs) and min(confs) >= args.speaking_conf
            return True
        if speaking.is_set():
            return False  # partials have no confidence; wait for the full result
        partial = json.loads(wake.PartialResult()).get("partial", "")
        partial_hits = partial_hits + 1 if any(p in partial for p in WAKE_PHRASES) else 0
        return partial_hits >= WAKE_PARTIAL_CHUNKS

    # Vosk recognises the wake phrase a few tenths of a second late, so a request said in
    # the same breath ("Hey Koma, what do you see?") would lose its first word.
    recent_chunks = deque(maxlen=max(1, round(args.preroll * 10)))
    emit("READY")
    while data := read():
        if command is None:
            if pending is not None:
                pending += data
                verify_left -= 1
                if verify_left:
                    continue
                said = wake_said(bytes(pending))
                name = WAKE_NAME.search(said)
                if not name:
                    debug(f"Vosk proposed a wake but Whisper heard {said!r}; ignoring")
                    pending = None
                    continue
                preroll, pending = bytes(pending), None
                # "Koma, what is this?" in one breath: the request has already begun.
                already_spoke = bool(said[name.end():].strip())
            else:
                recent_chunks.append(data)
                if not heard_wake(data):
                    continue
                wake.Reset()
                partial_hits = 0
                preroll = b"".join(recent_chunks)
                recent_chunks.clear()
                already_spoke = False
                if whisper is not None:
                    pending, verify_left = bytearray(preroll), VERIFY_CHUNKS
                    continue
            command = recording = bytearray()
            spoke, chunks = already_spoke, 0
            emit("WAKE")
            continue

        # Record until the visitor stops talking. Short pauses mid-sentence are fine:
        # only --end-silence seconds with no speech at all ends the request.
        recording += data
        chunks += 1
        elapsed = len(recording) / (RATE * 2)
        if chunks % 2 and elapsed < args.timeout:
            continue  # check the VAD every 0.2 s
        # Check the whole request: Silero misses speech when its audio starts mid-word, so a
        # sliding window cut people off while they were still talking.
        audio = np.frombuffer(recording, np.int16).astype(np.float32) / 32768
        speech = get_speech_timestamps(audio, vad_sensitive)
        silent_for = (len(audio) - speech[-1]["end"]) / RATE if speech else elapsed
        # The tail of the wake phrase sits at the very start; only later or longer speech
        # means the visitor has started their request.
        spoke = spoke or any(seg["start"] >= 0.3 * RATE or seg["end"] - seg["start"] >= 0.6 * RATE
                             for seg in get_speech_timestamps(audio, vad_options))
        finished = spoke and silent_for >= args.end_silence
        if finished or elapsed >= args.timeout or (not spoke and elapsed >= args.start_timeout):
            started = time.monotonic()
            # The VAD only judges audio after the wake phrase; Whisper also gets the pre-roll.
            text = transcribe(preroll + bytes(recording)).replace("\n", " ").strip() if spoke else ""
            reason = "silence" if finished else "no speech" if not spoke else "max length"
            name = WAKE_NAME.search(text[:WAKE_SEARCH_CHARS])
            if name:
                text = text[name.end():].strip()
            elif whisper is not None and text:
                debug(f"no wake phrase in {text[:WAKE_SEARCH_CHARS]!r}; ignoring background speech")
                text, reason = "", "false wake"
            text = text[:1].upper() + text[1:]
            debug(f"{elapsed:.1f}s recorded, ended by {reason}, "
                  f"transcribed in {time.monotonic() - started:.1f}s: {text!r}")
            LAST_REQUEST.parent.mkdir(exist_ok=True)
            with wave.open(str(LAST_REQUEST), "wb") as out:
                out.setnchannels(1); out.setsampwidth(2); out.setframerate(RATE)
                out.writeframes(preroll + bytes(recording))  # includes the pre-roll
            emit(f"REQUEST {text}" if text else "TIMEOUT")
            command = None
    if recorder:
        recorder.terminate()
        raise SystemExit("Microphone capture stopped; check the Jabra USB connection.")


if __name__ == "__main__":
    main()
