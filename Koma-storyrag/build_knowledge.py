#!/usr/bin/env python3
"""Rebuild HELIOS_knowledge.jsonl from HELIOS_world_guide.md and HELIOS_story.md.

Edit the markdown, then run:  python3 build_knowledge.py
Every ## or ### section becomes one retrieval record. Numbered ### sections under
"Visitor questions and KOMA answers" become visitor_qa records; story sections are
split into scenes of about 2,000 characters.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUIDE = HERE / "HELIOS_world_guide.md"
STORY = HERE / "HELIOS_story.md"
OUT = HERE / "HELIOS_knowledge.jsonl"
VERSION = "1.1"
SCENE_CHARS = 2000

KEYWORDS = ["HELIOS", "KOMA", "TradesMaster", "local", "PPES", "toilet", "cloud", "Mara",
            "privacy", "models", "Nell", "training", "2100", "exposure", "licensing", "purpose",
            "luxury", "2071", "safety", "edge", "domain shift", "open source", "2068",
            "consciousness", "Quiet Libraries", "Open Hand", "retirement", "artisans",
            "United States", "purge"]

PREFIX = {
    "lore": "Setting facts for the fictional HELIOS world of 2100.",
    "visitor_qa": "Example KOMA answer in the fictional HELIOS world of 2100.",
    "tone_examples": "Voice examples for KOMA in the fictional HELIOS setting of 2100.",
    "story_scene": ("Illustrative fictional scene from A Private Life, set in 2100. Mara is a new "
                    "TradesMaster, Nell is a retired maintainer, and this story's KOMA belongs to "
                    "Mara. This is not a personal memory of the exhibition unit or visitor."),
}


def sections(markdown):
    """Yield (level, heading, parent_heading, body) for each heading with body text."""
    parent = None
    parts = re.split(r"(?m)^(#{1,3}) (.+)$", markdown)
    for i in range(1, len(parts), 3):
        level, heading, body = len(parts[i]), parts[i + 1].strip(), parts[i + 2].strip()
        if level <= 2:
            parent = heading
        if body:
            yield level, heading, parent, body


def keywords(body):
    return [k for k in KEYWORDS if re.search(rf"\b{re.escape(k)}\b", body, re.IGNORECASE)]


def record(n, title, rtype, source_file, section, body, questions=(), answer=None):
    rec = {
        "id": f"helios2100_{n:03d}", "version": VERSION, "title": title, "record_type": rtype,
        "fictional": True, "setting_year": 2100,
        "canon_status": "illustrative_example" if rtype == "story_scene" else "proposed_setting_canon",
        "source_file": source_file, "source_section": section, "keywords": keywords(body),
        "questions": list(questions), "text": f"{PREFIX[rtype]}\n\n{title}\n\n{body}",
    }
    if answer is not None:
        rec["answer"] = answer
    return rec


def scenes(body):
    """Split a long story section on paragraph boundaries."""
    chunk = ""
    for para in body.split("\n\n"):
        if chunk and len(chunk) + len(para) > SCENE_CHARS:
            yield chunk
            chunk = ""
        chunk = f"{chunk}\n\n{para}" if chunk else para
    if chunk:
        yield chunk


def main():
    records = []
    for level, heading, parent, body in sections(GUIDE.read_text()):
        n = len(records) + 1
        qa = re.match(r"\d+\s+(.+)", heading)
        if parent == "Visitor questions and KOMA answers" and level == 3 and qa:
            question = qa.group(1).rstrip("?") + "?"
            records.append(record(n, question, "visitor_qa", GUIDE.name, parent, body,
                                  questions=[question], answer=body))
        else:
            rtype = "tone_examples" if heading.startswith("Short phrases") else "lore"
            records.append(record(n, heading, rtype, GUIDE.name, heading, body))
    for level, heading, parent, body in sections(STORY.read_text()):
        if level == 1:
            continue  # the story's title line
        for part, chunk in enumerate(scenes(body), start=1):
            title = heading if part == 1 else f"{heading} continued {part}"
            records.append(record(len(records) + 1, title, "story_scene", STORY.name, heading, chunk))

    with OUT.open("w") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    counts = {}
    for rec in records:
        counts[rec["record_type"]] = counts.get(rec["record_type"], 0) + 1
    print(f"Wrote {len(records)} records to {OUT.name}: {counts}")


if __name__ == "__main__":
    main()
