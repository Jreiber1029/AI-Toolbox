#!/home/jasonreiber/ultralytics-env/bin/python
"""Compare YOLO TensorRT engines on the same images, through the same TrtDetector KOMA uses.

  compare_yolo.py OLD.engine NEW.engine [...]   [--images captures] [--conf 0.5] [--diff DIR]

Images are labelled by folder: captures/Hammer/*.jpg should contain a hammer. Folder names
are matched loosely to class names (pliers -> plier). For each engine it reports:
  load MB      GPU memory taken by loading the engine
  ms / frame   full detect() time (letterbox + copy + inference + NMS), median and p95
  hit rate     per folder: share of images where the expected tool was detected
  conf         mean confidence of the expected tool when it was found
  other        detections of a different tool class per image (likely false positives)
--diff DIR saves side-by-side images where the engines disagree on the expected tool.
"""
import argparse
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from yolo_trt import TrtDetector, _check, cudart

ROOT = Path(__file__).resolve().parent
NON_TOOLS = {"face", "person"}


def gpu_used_mb():
    free, total = _check(cudart.cudaMemGetInfo())
    return (total - free) / 2**20


def expected_class(folder, names):
    f = folder.lower().rstrip("s")
    for n in names:
        if n.lower().rstrip("s") == f:
            return n
    return None


def run(engine, images, conf, warmup=20):
    before = gpu_used_mb()
    det = TrtDetector(str(engine))
    load_mb = gpu_used_mb() - before
    frames = [(p, cv2.imread(str(p))) for p in images]
    for _, img in frames[:warmup]:
        det.detect(img, conf)
    times, results = [], {}
    for p, img in frames:
        t = time.perf_counter()
        results[p] = det.detect(img, conf)
        times.append((time.perf_counter() - t) * 1000)
    return det.names, load_mb, times, results


def draw(img, dets, label):
    out = img.copy()
    for name, c, (x1, y1, x2, y2) in dets:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{name} {c:.2f}", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(out, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("engines", nargs="+", type=Path)
    p.add_argument("--images", type=Path, default=ROOT / "captures")
    p.add_argument("--conf", type=float, default=0.5, help="same default as KOMA's detect()")
    p.add_argument("--diff", type=Path, help="save images where engines disagree here")
    args = p.parse_args()
    _check(cudart.cudaFree(0))  # create the CUDA context first so it isn't billed to engine 1

    images = sorted(x for x in args.images.rglob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"no images under {args.images}")
    folders = sorted({x.parent.name for x in images})
    print(f"{len(images)} images in {len(folders)} folders: {', '.join(folders)}  (conf >= {args.conf})\n")

    report = {}
    for engine in args.engines:
        print(f"running {engine.name} ...", flush=True)
        names, load_mb, times, results = run(engine, images, args.conf)
        tool_names = [n for n in names.values() if n.lower() not in NON_TOOLS]
        per = defaultdict(lambda: {"n": 0, "hit": 0, "conf": [], "other": 0})
        found = {}
        for img, dets in results.items():
            want = expected_class(img.parent.name, names.values())
            s = per[img.parent.name]
            s["n"] += 1
            best = max((c for n, c, _ in dets if n == want), default=None)
            found[img] = best is not None
            if best is not None:
                s["hit"] += 1
                s["conf"].append(best)
            s["other"] += sum(1 for n, _, _ in dets if n != want and n in tool_names)
        report[engine] = dict(names=names, load_mb=load_mb, times=times, per=per, found=found,
                              results=results, classes=Counter(n for d in results.values() for n, _, _ in d))

    w = max(len(e.name) for e in args.engines) + 2
    q = lambda xs, f: sorted(xs)[int(f * (len(xs) - 1))]
    print(f"\n{'engine':<{w}}{'size MB':>8}{'load MB':>9}{'median ms':>11}{'p95 ms':>8}{'FPS':>7}")
    for e, r in report.items():
        med = statistics.median(r["times"])
        print(f"{e.name:<{w}}{e.stat().st_size / 2**20:>8.1f}{r['load_mb']:>9.0f}{med:>11.1f}"
              f"{q(r['times'], .95):>8.1f}{1000 / med:>7.1f}")

    print(f"\nhit rate / mean conf of expected tool / other-tool detections per image")
    print(f"{'folder':<14}" + "".join(f"{e.name:>{w + 12}}" for e in report))
    for f in folders:
        row = f"{f:<14}"
        for r in report.values():
            s = r["per"][f]
            if expected_class(f, r["names"].values()) is None:
                row += f"{'(no class)':>{w + 12}}"
                continue
            cf = statistics.mean(s["conf"]) if s["conf"] else 0
            row += f"{s['hit'] / s['n']:>{w - 2}.0%} {cf:>5.2f} {s['other'] / s['n']:>5.2f}"
        print(row)
    for e, r in report.items():
        print(f"\n{e.name} classes: {r['names']}\n  detections at conf>={args.conf}: {dict(r['classes'])}")

    if args.diff and len(report) >= 2:
        args.diff.mkdir(parents=True, exist_ok=True)
        rs = list(report.items())
        n = 0
        for img in images:
            if len({r["found"][img] for _, r in rs}) > 1:
                frame = cv2.imread(str(img))
                tiles = [draw(frame, r["results"][img], e.stem) for e, r in rs]
                cv2.imwrite(str(args.diff / f"{img.parent.name}_{img.name}"), cv2.hconcat(tiles))
                n += 1
        print(f"\n{n} disagreeing images saved to {args.diff}")


if __name__ == "__main__":
    main()
