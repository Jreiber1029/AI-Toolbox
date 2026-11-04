# KOMA: Qwen vision and Piper

Start from a Jetson terminal:

```bash
bash ~/YOLO-VLM/run_qwen_piper.sh
```

Stop an older KOMA pipeline or camera preview with Ctrl-C in its terminal first.
The launcher starts its own Qwen3.5-2B server on port 8080 and refuses to start
if that port is occupied. Ctrl-C stops the app, its audio workers, and its server.

Defaults:

- USB camera `/dev/video1`, MJPEG 1280x720 at 30 FPS, with neutral colour gains.
- Existing Piper voice `voices/koma_lessac_high.onnx` on CPU, **0 semitone pitch
  shift**, length scale **1.0**, and volume **0.9**. Normal pitch means the model's
  native voice, with no extra pitch processing.
- Offline "Hey Koma" wake requests, using Vosk for request recognition to avoid
  loading an additional Whisper model. Use `--stt whisper` to select Whisper.
- A camera observation after 20 seconds of idle time. Requests are serialized;
  observations wait for visitor requests and speech to finish. These are timed
  observations, rather than object-detector triggers.
- No YOLO model or PyTorch import in this app. The Python executable comes from
  the existing `ultralytics-env` because it supplies the working OpenCV build.

The new files are `run_qwen_piper.sh` and `koma_qwen_piper.py`. They reuse the
existing Piper worker, wake listener, and KOMA helpers. The original `run.sh`
and `koma.py` remain available as the older detector-based application.

## Existing persona and RAG

The app uses the same `Brain` from `brain.py`, with the same system prompt and
handover from `Koma-storyrag/koma_exhibition_prompts.py` and the same
`Koma-storyrag/HELIOS_knowledge.jsonl` index. The current index has 97 records:
26 lore, 58 visitor Q&A, one tone-example record, and 12 story scenes.

World-guide chunks follow headings; long story sections are split on paragraph
boundaries at about 2,000 characters. BM25 retrieval boosts titles, questions,
and keywords. Question turns retrieve up to two records and limit each supplied
passage to about 700 characters. The new app does not rebuild or change chunks.
It preserves visitor-session history, "new visitor", handover, and safety routing.
Timed observations use a question with an actual image, not a detector prompt.

This app corrects its runtime facts in memory to describe the selected camera,
disabled detector, and enabled microphone mode. The shared `runtime_facts.txt`
is not edited. The existing wake listener saves the latest spoken request as
`logs/last_request.wav`; Piper creates temporary WAV files for playback.

## Checks and options

Check assets and Python imports without opening the camera, microphone, model
server, or speaker:

```bash
bash ~/YOLO-VLM/run_qwen_piper.sh --check
```

For one camera response, then shutdown:

```bash
bash ~/YOLO-VLM/run_qwen_piper.sh --once
```

For a single spoken question:

```bash
bash ~/YOLO-VLM/run_qwen_piper.sh --ask "Who made your bag?" --once
```

Disable automatic observations and wait for "Hey Koma":

```bash
bash ~/YOLO-VLM/run_qwen_piper.sh --interval 0
```

If camera numbering changes, use `v4l2-ctl --list-devices` to locate the USB
capture device, then pass its path with `--camera /dev/videoN`. The launcher
checks that path before loading Qwen. The USB camera must be connected first.
`--preview` opens a window on the Jetson's graphical desktop; omit it over SSH.
Server output is written to `logs/qwen-piper-server.log`.

CPU preflight and mocked request/cleanup tests were used during creation.
The camera, microphone, live Qwen inference, and speaker must be checked when
the operator starts the launcher.
