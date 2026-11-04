#!/home/jasonreiber/ultralytics-env/bin/python
"""Ask each VLM the same KOMA test cases and record speed, memory and answers.

Models run one at a time (8 GB can't hold two): serve_model.sh starts each on :8080,
and every case goes through koma_voice's PackBrain and koma's stream_vlm, so the model
gets exactly the messages, sampling and sentence cutoff of the live pipeline.

Cases: the pack's regression_queries.json (lore retrieval questions), tests/model_bench.json
(story, persona, safety, follow-ups), and for each photo in tests/bench_images/ a tool
comment plus "What do you see?". Name photos after the tools in them: drill.jpg,
hammer_2.jpg, drill+wrench.jpg.

Results: bench/<timestamp>/<model>.jsonl and <model>.meta.json. Score them with
compare_models.py.
"""
import argparse
import json
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2

from brain import SENTENCE_LIMIT, classify
from koma import MAX_TOKENS, ROOT, encode_frame, log, stream_vlm
from koma_voice import DEFAULT_PACK, PackBrain

MODELS = ("qwen", "gemma", "ministral", "nemotron")
TEXT_ONLY = {"nemotron"}  # no mmproj: image cases are asked without the photo
SERVER = "http://127.0.0.1:8080"
CASE_FILE = ROOT / "tests/model_bench.json"
IMAGE_DIR = ROOT / "tests/bench_images"


def load_cases(pack):
    cases = []
    for i, q in enumerate(json.loads((pack / "regression_queries.json").read_text())):
        kind = "prototype" if q["expected_first"].startswith("prototype_") else "lore"
        cases.append({"id": f"rq{i:02d}", "kind": kind, "turns": [q["question"]],
                      "expected_chunk": q["expected_first"]})
    cases += json.loads(CASE_FILE.read_text())["cases"]
    for path in sorted(IMAGE_DIR.glob("*.jpg")):
        labels = [w.capitalize() for w in path.stem.split("_")[0].split("+")]
        cases.append({"id": f"tool_{path.stem}", "kind": "tool", "turns": [""],
                      "labels": labels, "image": str(path)})
        cases.append({"id": f"visual_{path.stem}", "kind": "visual",
                      "turns": ["What do you see?"], "image": str(path)})
    return cases


class MemoryPeak:
    """Samples used RAM (MemTotal - MemAvailable, in MB) twice a second in the background."""

    def __init__(self):
        self.peak = self.used()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def used():
        info = dict(line.split(":") for line in Path("/proc/meminfo").read_text().splitlines())
        kb = lambda key: int(info[key].split()[0])
        return (kb("MemTotal") - kb("MemAvailable")) // 1024

    def _loop(self):
        while self.running:
            self.peak = max(self.peak, self.used())
            time.sleep(0.5)

    def stop(self):
        self.running = False
        return self.peak


def healthy():
    try:
        urllib.request.urlopen(f"{SERVER}/health", timeout=2)
        return True
    except OSError:
        return False


def start_server(model, log_path):
    if healthy():
        raise SystemExit("Something is already serving :8080 - stop it so the benchmark controls the model.")
    started = time.monotonic()
    proc = subprocess.Popen([str(ROOT / "serve_model.sh"), model],
                            stdout=log_path.open("w"), stderr=subprocess.STDOUT)
    while time.monotonic() - started < 300:
        if proc.poll() is not None:
            log(f"{model} server exited:\n" + "\n".join(log_path.read_text().splitlines()[-15:]))
            return None, 0
        if healthy():
            return proc, time.monotonic() - started
        time.sleep(1)
    stop_server(proc)
    log(f"{model} server did not become ready in 300 s; see {log_path}")
    return None, 0


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()  # llama-server can hang while freeing GPU memory
        proc.wait()


def ask(brain, event, question, labels, history, frame, seed):
    """One pipeline turn, as koma_voice.ask does it, with timings."""
    image = encode_frame(frame, 640) if frame is not None else None
    messages, lore_ids = brain.build_messages(event, labels=labels, question=question,
                                              history=tuple(history), image_b64=image)
    marks = {"first_token": None, "tokens": 0}

    def on_delta(_):
        marks["tokens"] += 1  # llama-server streams one token per chunk
        marks["first_token"] = marks["first_token"] or time.monotonic()

    started = time.monotonic()
    first_sentence = None
    reply = []
    for sentence in stream_vlm(f"{SERVER}/v1/chat/completions", messages, MAX_TOKENS[event],
                               threading.Event(), on_delta=on_delta, seed=seed):
        first_sentence = first_sentence or time.monotonic()
        reply.append(sentence)
        if len(reply) >= SENTENCE_LIMIT[event]:
            break
    ended = time.monotonic()
    first = marks["first_token"] or ended
    gen_time = ended - first
    return {
        "event": event, "question": question, "labels": list(labels), "image": image is not None,
        "lore_ids": lore_ids, "sentences": reply, "answer": " ".join(reply),
        "first_token_s": round(first - started, 2),
        "first_sentence_s": round((first_sentence or ended) - started, 2),
        "total_s": round(ended - started, 2), "tokens": marks["tokens"],
        "tok_per_s": round((marks["tokens"] - 1) / gen_time, 1) if gen_time > 0 and marks["tokens"] > 1 else None,
    }


def run_case(brain, case, seed, vision=True):
    brain.last_topic = ""  # every case is a fresh visitor
    history, turns = [], []
    frame = cv2.imread(case["image"]) if vision and case.get("image") else None
    labels = case.get("labels", [])
    for question in case["turns"]:
        event = "tool" if case["kind"] == "tool" else classify(question)
        if event in ("handover", "new_visitor"):
            event = "question"
        send = frame if frame is not None and (event == "tool" or brain.wants_image(event, question, labels)) else None
        turn = ask(brain, event, question, labels, history, send, seed)
        turns.append(turn)
        history.append({"role": "user", "content": question or
                        f"(The visitor held up something the camera took for: {', '.join(labels)}.)"})
        history.append({"role": "assistant", "content": turn["answer"]})
    return turns


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", default=list(MODELS), help=f"Any of {', '.join(MODELS)} (default: all)")
    p.add_argument("--kinds", help="Only these case kinds, comma separated "
                   "(lore, prototype, story, persona, safety, followup, tool, visual)")
    p.add_argument("--limit", type=int, help="Only the first N cases (quick smoke test)")
    p.add_argument("--repeats", type=int, default=1, help="Ask each case N times with different seeds")
    p.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    p.add_argument("--out", type=Path, help="Results folder (default: bench/<timestamp>)")
    args = p.parse_args()

    unknown = set(args.models) - set(MODELS)
    if unknown:
        p.error(f"unknown model(s): {', '.join(sorted(unknown))}")
    brain = PackBrain(args.pack)
    cases = load_cases(args.pack)
    if args.kinds:
        cases = [c for c in cases if c["kind"] in args.kinds.split(",")]
    cases = cases[:args.limit]
    out = args.out or ROOT / "bench" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    log(f"{len(cases)} cases x {args.repeats} repeat(s) x {len(args.models)} model(s) -> {out}")

    for model in args.models:
        baseline = MemoryPeak.used()
        memory = MemoryPeak()
        proc, load_s = start_server(model, out / f"{model}.server.log")
        if proc is None:
            memory.stop()
            continue
        try:
            props = json.load(urllib.request.urlopen(f"{SERVER}/props", timeout=5))
            log(f"{model} ready in {load_s:.0f} s: {Path(props.get('model_path', '?')).name}")
            # Untimed warm-up: cache the system prompt and initialise the vision encoder,
            # as they would be after KOMA's first visitor.
            warm = None if model in TEXT_ONLY else next((c for c in cases if c.get("image")), None)
            ask(brain, "question", "Hello.", [], [], cv2.imread(warm["image"]) if warm else None, 0)
            with (out / f"{model}.jsonl").open("w") as results:
                for repeat in range(args.repeats):
                    for n, case in enumerate(cases, 1):
                        turns = run_case(brain, case, seed=1000 + repeat, vision=model not in TEXT_ONLY)
                        last = turns[-1]
                        log(f"{model} {n}/{len(cases)} {case['id']} ({last['total_s']:.1f}s): {last['answer'][:90]}")
                        row = {"model": model, "repeat": repeat, **{k: v for k, v in case.items() if k != "turns"},
                               "turns": turns}
                        results.write(json.dumps(row) + "\n")
                        results.flush()
        finally:
            stop_server(proc)
            peak = memory.stop()
        meta = {"model": model, "model_file": props.get("model_path"), "load_s": round(load_s, 1),
                "ram_baseline_mb": baseline, "ram_peak_mb": peak, "ram_model_mb": peak - baseline}
        (out / f"{model}.meta.json").write_text(json.dumps(meta, indent=2))
        log(f"{model} done: peak RAM {peak} MB ({peak - baseline:+} MB over baseline)")
        time.sleep(3)  # let the GPU memory settle before the next model
    log(f"Results in {out}. Next: ~/ultralytics-env/bin/python compare_models.py {out}")


if __name__ == "__main__":
    main()
