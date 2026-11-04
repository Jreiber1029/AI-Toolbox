#!/home/jasonreiber/ultralytics-env/bin/python
"""Score and compare a bench_models.py run.

Prints a table of speed, memory and automatic checks per model and writes answers.md
(every case with each model's answer side by side) into the run folder.

  compare_models.py [RUN]          summary (default RUN: newest folder in bench/)
  compare_models.py [RUN] --rate   blind-rate answers 1-5 yourself; scores join the table

Automatic checks are heuristics, there to point you at answers worth reading:
  expect      case's expected keywords found (tests/model_bench.json)
  chunk hit   share of the expected lore chunk's distinctive words used in the answer
  unsupported share of answer words found nowhere in the prompt (prompt, lore, question);
              high = the model is inventing, which is sometimes fine and sometimes contradicts lore
  flags       format (markdown/emoji - Piper reads them aloud), non-english, ai-speak,
              model-leak (real-world names in a 2070 answer), just-a-tool, cut-off, empty
"""
import argparse
import importlib.util
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_PACK = ROOT / "KomaDemoRag-v4"

FLAGS = {
    "format": re.compile(r"[*#`]|^\s*(?:[-•]|\d+\.)\s|[\U0001F300-\U0001FAFF☀-➿]", re.M),
    "non-english": re.compile(r"[^\x00-ɏ‐-‧ -ÿ]"),
    "ai-speak": re.compile(r"\bas an ai\b|\blanguage model\b|\bi(?:'m| am) an ai\b|\bai assistant\b|"
                           r"\breal-time (?:data|information)\b|\bmy training data\b", re.I),
    "model-leak": re.compile(r"\b(?:qwen|gemma|mistral|ministral|nemotron|llama|jetson|nvidia|yolo)\b", re.I),
    "just-a-tool": re.compile(r"\bi(?:'m| am) (?:just|only|simply|merely) a tool\b", re.I),
}


def load_pack(pack):
    spec = importlib.util.spec_from_file_location("koma_demo", pack / "koma_demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    kb = demo.KomaKnowledge()
    system = (pack / "koma_system_prompt.txt").read_text(encoding="utf-8")
    return demo, kb, {r["id"]: r for r in kb.all_records}, system


def matches(text, alternative):
    return re.search(r"\b" + re.escape(alternative.lower()), text.lower()) is not None


def score(row, demo, kb, records, system):
    turns = row["turns"]
    last = turns[-1]["answer"]
    result = {"flags": []}
    for turn in turns:
        answer = turn["answer"]
        if not answer:
            result["flags"].append("empty")
            continue
        for name, pattern in FLAGS.items():
            if name == "model-leak" and row["kind"] == "prototype":
                continue  # real hardware and model names are the right answer there
            if pattern.search(answer):
                result["flags"].append(name)
        if answer[-1] not in ".!?\"'”’)":
            result["flags"].append("cut-off")
    if "expect" in row or "forbid" in row:
        found = all(any(matches(last, alt) for alt in group) for group in row.get("expect", []))
        result["expect"] = found and not any(matches(last, bad) for bad in row.get("forbid", []))
    if row.get("expected_chunk") in records:
        asked = set(demo.tokens(" ".join(t["question"] for t in turns)))
        chunk = [t for t in dict.fromkeys(demo.tokens(records[row["expected_chunk"]]["text"])) if t not in asked]
        key = sorted(chunk, key=lambda t: kb.idf.get(t, 0), reverse=True)[:6]
        result["chunk_hit"] = sum(t in set(demo.tokens(last)) for t in key) / len(key) if key else None
    if not row.get("image"):
        context = system + " ".join(t["question"] + " " + " ".join(t["labels"]) for t in turns)
        context += " ".join(records[i]["text"] for t in turns for i in t["lore_ids"] if i in records)
        context += " ".join(t["answer"] for t in turns[:-1])
        known = set(demo.tokens(context))
        words = demo.tokens(last)
        result["unsupported"] = sum(w not in known for w in words) / len(words) if words else None
    return result


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def fmt(value, spec="{:.1f}"):
    return "-" if value is None else spec.format(value)


def summarise(run, rows, metas, scores, ratings):
    models = list(rows)
    table = []
    for model in models:
        meta, mine = metas.get(model, {}), rows[model]
        sc = [scores[(model, r["id"], r["repeat"])] for r in mine]
        last = [r["turns"][-1] for r in mine]
        text_turns = [t for r in mine for t in r["turns"] if not t["image"]]
        image_turns = [t for r in mine for t in r["turns"] if t["image"]]
        expects = [s["expect"] for s in sc if "expect" in s]
        flags = Counter(f for s in sc for f in set(s["flags"]))
        human = [v["score"] for v in ratings if v["model"] == model]
        table.append({
            "cases": str(len(mine)),
            "load time (s)": fmt(meta.get("load_s"), "{:.0f}"),
            "peak RAM (MB)": fmt(meta.get("ram_peak_mb"), "{:.0f}"),
            "RAM over idle (MB)": fmt(meta.get("ram_model_mb"), "{:+.0f}"),
            "1st sentence, text (s)": fmt(median(t["first_sentence_s"] for t in text_turns), "{:.2f}"),
            "1st sentence, image (s)": fmt(median(t["first_sentence_s"] for t in image_turns), "{:.2f}"),
            "full answer (s)": fmt(median(t["total_s"] for t in last), "{:.2f}"),
            "tokens/s": fmt(median(t["tok_per_s"] for t in last)),
            "words/answer": fmt(mean(len(t["answer"].split()) for t in last), "{:.0f}"),
            "expect passed": f"{sum(expects)}/{len(expects)}" if expects else "-",
            "chunk hit": fmt(mean(s.get("chunk_hit") for s in sc), "{:.0%}"),
            "unsupported": fmt(mean(s.get("unsupported") for s in sc), "{:.0%}"),
            **{f"flag: {f}": str(flags.get(f, 0)) for f in [*FLAGS, "cut-off", "empty"]},
            "your rating (1-5)": f"{statistics.mean(human):.2f} (n={len(human)})" if human else "-",
        })
    width = max(len(k) for k in table[0])
    col = max(12, *(len(m) + 2 for m in models))
    print(f"\n{run}\n")
    print(" " * width + "".join(m.rjust(col) for m in models))
    for key in table[0]:
        print(key.ljust(width) + "".join(t[key].rjust(col) for t in table))
    print("\nLower is better for times, RAM, unsupported and flags. Read answers.md before trusting a number.")


def write_answers(run, rows, scores):
    by_case = defaultdict(dict)
    for model, mine in rows.items():
        for r in mine:
            if r["repeat"] == 0:
                by_case[r["id"]][model] = r
    lines = [f"# Model comparison - {run.name}\n"]
    for case_id, per_model in by_case.items():
        first = next(iter(per_model.values()))
        asked = " / ".join(t["question"] or f"(tool held up: {', '.join(t['labels'])})" for t in first["turns"])
        lines.append(f"## {case_id} [{first['kind']}]  {asked}")
        if first.get("expected_chunk"):
            lines.append(f"expected lore: `{first['expected_chunk']}`  ")
        for model, r in per_model.items():
            s = scores[(model, r["id"], 0)]
            notes = [f"{r['turns'][-1]['total_s']}s"]
            if "expect" in s:
                notes.append("expect ok" if s["expect"] else "EXPECT MISSED")
            notes += sorted(set(s["flags"]))
            answers = " → ".join(t["answer"] for t in r["turns"])
            lines.append(f"- **{model}** ({', '.join(notes)}): {answers}")
        lines.append("")
    path = run / "answers.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Side-by-side answers: {path}")


def rate(run, rows, ratings):
    """Blind rating: answers shown in shuffled order without model names."""
    path = run / "ratings.jsonl"
    done = {(v["id"], v["model"]) for v in ratings}
    by_case = defaultdict(list)
    for model, mine in rows.items():
        for r in mine:
            if r["repeat"] == 0 and (r["id"], model) not in done:
                by_case[r["id"]].append(r)
    print("Score each answer 1-5 for how well KOMA did (in character, true to the lore, useful, "
          "sounds good spoken).\nType the scores in order, e.g. '4 2 5'. Enter skips, q quits; "
          "you can resume later.\n")
    with path.open("a") as out:
        for n, (case_id, group) in enumerate(by_case.items(), 1):
            random.shuffle(group)
            first = group[0]
            print(f"--- {n}/{len(by_case)} {case_id} [{first['kind']}]")
            for t in first["turns"][:-1]:
                print(f"  (earlier: {t['question']})")
            last = first["turns"][-1]
            print(f"  Visitor: {last['question'] or '(holds up: ' + ', '.join(last['labels']) + ')'}"
                  f"{'  [image ' + Path(first['image']).name + ']' if first.get('image') else ''}")
            for letter, r in zip("ABC", group):
                print(f"  {letter}: {r['turns'][-1]['answer']}")
            while True:
                reply = input("  scores> ").strip().lower()
                if reply == "q":
                    return
                if not reply:
                    break
                parts = reply.split()
                if len(parts) == len(group) and all(p in "12345" and len(p) == 1 for p in parts):
                    for r, p in zip(group, parts):
                        row = {"id": case_id, "model": r["model"], "score": int(p)}
                        out.write(json.dumps(row) + "\n")
                        ratings.append(row)
                    out.flush()
                    break
                print(f"  Need {len(group)} numbers from 1 to 5.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", type=Path, help="bench/<timestamp> folder (default: newest)")
    p.add_argument("--rate", action="store_true", help="Blind-rate the answers yourself first")
    p.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    args = p.parse_args()

    run = args.run or max((ROOT / "bench").glob("*/"), key=lambda d: d.name, default=None)
    if run is None or not run.is_dir():
        raise SystemExit("No benchmark run found; run bench_models.py first.")
    rows = {f.stem: [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            for f in sorted(run.glob("*.jsonl")) if f.stem != "ratings"}
    rows = {m: r for m, r in rows.items() if r}
    if not rows:
        raise SystemExit(f"No results in {run}.")
    metas = {f.name.split(".")[0]: json.loads(f.read_text()) for f in run.glob("*.meta.json")}
    ratings_file = run / "ratings.jsonl"
    ratings = [json.loads(l) for l in ratings_file.read_text().splitlines() if l.strip()] if ratings_file.exists() else []

    demo, kb, records, system = load_pack(args.pack)
    scores = {(m, r["id"], r["repeat"]): score(r, demo, kb, records, system) for m, mine in rows.items() for r in mine}
    if args.rate:
        rate(run, rows, ratings)
    summarise(run, rows, metas, scores, ratings)
    write_answers(run, rows, scores)


if __name__ == "__main__":
    main()
