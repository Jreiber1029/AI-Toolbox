"""Rebuild the definition chunks from ~/YOLO-VLM/glossary_draft.md: python build_glossary.py, then python validate_pack.py.
A new heading in the draft also needs a line in TERMS below (its id, question wording and aliases)."""
import json
import re
import sys
from pathlib import Path

DRAFT = Path.home() / "YOLO-VLM/glossary_draft.md"
PACK = Path.home() / "YOLO-VLM/KomaDemoRag-v4"

# draft heading -> (id suffix, how questions name it, plural?, aliases)
TERMS = {
    "AI model": ("ai_model", "an AI model", False, ["ai model", "ai models", "model", "models"]),
    "Distillation": ("distillation", "distillation", False, ["distillation", "distilling", "distil", "distill", "distilled", "model distillation"]),
    "Fine-tuning": ("fine_tuning", "fine-tuning", False, ["fine tuning", "fine tune", "finetuning", "finetune", "fine tuned"]),
    "Training": ("training", "AI training", False, ["ai training", "model training", "training a model", "training an ai"]),
    "Model weights": ("model_weights", "model weights", True, ["model weights", "weights", "model weight"]),
    "Open model": ("open_model", "an open model", False, ["open model", "open models", "open source model", "open source models", "open source ai", "open ai model"]),
    "Frontier model": ("frontier_model", "a frontier model", False, ["frontier model", "frontier models", "frontier ai", "frontier", "frontier cloud service", "frontier cloud services"]),
    "Derived model": ("derived_model", "a derived model", False, ["derived model", "derived models", "derived capabilities"]),
    "Capable AI": ("capable_ai", "capable AI", False, ["capable ai", "capable local ai", "capable model", "capable models"]),
    "Sleep mode": ("sleep_mode", "sleep mode", False, ["sleep mode"]),
    "Vision model / vision-language model (VLM)": ("vision_model", "a vision model", False, ["vision model", "vision models", "vision language model", "vision language models", "vlm"]),
    "Cloud AI": ("cloud_ai", "cloud AI", False, ["cloud ai", "cloud", "cloud service", "cloud services", "cloud computing"]),
    "Local AI / edge AI": ("local_ai", "local AI", False, ["local ai", "edge ai", "local intelligence", "edge computing", "local model", "local models"]),
    "Data centre": ("data_centre", "a data centre", False, ["data centre", "data center", "data centres", "data centers"]),
    "Compute": ("compute", "compute", False, ["compute", "computing power"]),
    "Hosting / model repository": ("hosting", "a model repository", False, ["hosting", "model repository", "model repositories", "repository", "hosting site", "hosting sites"]),
    "Technology ecosystem": ("ecosystem", "a technology ecosystem", False, ["technology ecosystem", "technology ecosystems", "tech ecosystem", "ecosystem", "ecosystems", "corporate ecosystem"]),
    "Automation": ("automation", "automation", False, ["automation"]),
    "Intelligence ownership": ("intelligence_ownership", "intelligence ownership", False, ["intelligence ownership", "intelligence ownership claims", "intelligence ownership law", "intelligence ownership laws"]),
    "Licensed provider": ("licensed_provider", "a licensed provider", False, ["licensed provider", "licensed providers", "government licensed provider", "provider", "providers", "ai provider"]),
    "Professional exemption": ("professional_exemption", "a professional exemption", False, ["professional exemption", "exemption", "exempt"]),
    "Ownership claim": ("ownership_claim", "an ownership claim", False, ["ownership claim", "ownership claims"]),
    "Precedent": ("precedent", "a precedent", False, ["precedent", "legal precedent"]),
    "Lobbying": ("lobbying", "lobbying", False, ["lobbying", "lobby", "lobbied", "lobbyist", "lobbyists"]),
    "Regulator": ("regulator", "a regulator", False, ["regulator", "regulators"]),
    "Takedown claim": ("takedown_claim", "a takedown claim", False, ["takedown claim", "takedown claims", "takedown", "takedowns", "take down"]),
    "Blocking order": ("blocking_order", "a blocking order", False, ["blocking order", "blocking orders"]),
    "Trade pressure": ("trade_pressure", "trade pressure", False, ["trade pressure"]),
    "The crackdown": ("crackdown", "the crackdown", False, ["crackdown"]),
    "The Pirate Bay": ("pirate_bay", "the Pirate Bay", False, ["pirate bay"]),
    "File-sharing": ("file_sharing", "file-sharing", False, ["file sharing", "filesharing"]),
    "Pirate Party": ("pirate_party", "the Pirate Party", False, ["pirate party"]),
    "ACTA": ("acta", "ACTA", False, ["acta", "anti counterfeiting trade agreement"]),
    "Subscription": ("subscription", "a subscription", False, ["subscription", "subscriptions", "subscription service"]),
    "Quota": ("quota", "a quota", False, ["quota"]),
    "Cohort": ("cohort", "a cohort", False, ["cohort"]),
    "Purchasing power": ("purchasing_power", "purchasing power", False, ["purchasing power"]),
    "PPE": ("ppe", "PPE", False, ["ppe", "p p e", "personal protective equipment"]),
    "Microexposure": ("microexposure", "a microexposure", False, ["microexposure", "microexposures", "micro exposure", "micro exposures"]),
    "PM2.5": ("pm25", "PM2.5", False, ["pm2 5", "pm 2 5", "pm25", "pm two point five", "p m two point five", "p m 2 5"]),
    "Volatile organic compounds (VOCs)": ("vocs", "VOCs", True, ["volatile organic compounds", "volatile organic compound", "voc", "vocs", "v o c", "v o cs"]),
    "Nitrogen oxides": ("nitrogen_oxides", "nitrogen oxides", True, ["nitrogen oxides", "nitrogen oxide", "nox"]),
    "Respirator / N95 / cartridges": ("respirator", "a respirator", False, ["respirator", "respirators", "n95", "n95 mask", "n 95", "cartridge", "cartridges", "organic vapour cartridge", "organic vapour cartridges", "organic vapor cartridges"]),
    "Asbestos": ("asbestos", "asbestos", False, ["asbestos"]),
    "Silica dust": ("silica_dust", "silica dust", False, ["silica dust", "silica"]),
    "Solvents": ("solvents", "solvents", True, ["solvents", "solvent"]),
    "Threshold": ("threshold", "a threshold", False, ["threshold"]),
    # Real prototype: prototype_rag.jsonl
    "Jetson Orin Nano Super": ("jetson", "the Jetson Orin Nano Super", False, ["jetson orin nano super", "jetson orin nano", "jetson", "orin nano"]),
    "YOLO": ("yolo", "YOLO", False, ["yolo"]),
    "Arduino UNO Q": ("arduino", "the Arduino UNO Q", False, ["arduino uno q", "arduino uno", "arduino"]),
    "SEN66": ("sen66", "the SEN66", False, ["sen66", "sen 66", "sen sixty six"]),
    "Edge Impulse": ("edge_impulse", "Edge Impulse", False, ["edge impulse"]),
    "Image classifier": ("image_classifier", "an image classifier", False, ["image classifier", "image classifiers", "classifier"]),
    "Jabra": ("jabra", "the Jabra", False, ["jabra", "jabra unit", "jabra speaker"]),
    "Piezo buzzer / LED matrix": ("piezo_led", "a piezo buzzer", False, ["piezo buzzer", "piezo", "buzzer", "led matrix", "led display"]),
}


def signature(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def parse(draft):
    """Yield (section number, heading, text) for each **entry** in the draft."""
    section = 0
    for block in re.split(r"\n\s*\n", draft):
        lines = block.strip().splitlines()
        if lines and (m := re.match(r"## (\d+)\.", lines[0])):
            section = int(m.group(1))
            continue
        if lines and (m := re.fullmatch(r"\*\*(?:★ )?(.+?)\*\*", lines[0].strip())):
            yield section, m.group(1), " ".join(l.strip() for l in lines[1:])


def record(heading, text, domain):
    suffix, name, plural, aliases = TERMS[heading]
    bare = re.sub(r"^(?:a|an|the) ", "", name)
    questions = [f"What {'are' if plural else 'is'} {name}?", f"What {'do' if plural else 'does'} {bare} mean?"]
    return {"id": f"term_{suffix}", "title": f"What {bare} means", "kind": "definition",
            "source_questions": [], "domain": domain, "status": "glossary",
            "keywords": " ".join(dict.fromkeys(" ".join(aliases).split())),
            "aliases": [signature(a) for a in aliases], "text": text, "questions": questions,
            "word_count": len(text.split()), "chunking": {"method": "one_term_per_chunk", "overlap_words": 0},
            "corpus_version": "4"}


def main():
    entries = list(parse(DRAFT.read_text(encoding="utf-8")))
    missing = set(TERMS) - {h for _, h, _ in entries}
    unknown = {h for _, h, _ in entries} - set(TERMS)
    if missing or unknown:
        sys.exit(f"draft/TERMS mismatch: missing {missing}, unknown {unknown}")
    world = [record(h, t, "world") for s, h, t in entries if s != 6]
    proto = [record(h, t, "prototype") for s, h, t in entries if s == 6]

    (PACK / "glossary_chunks").mkdir(exist_ok=True)
    for r in world + proto:
        folder = "glossary_chunks" if r["domain"] == "world" else "prototype_chunks"
        (PACK / folder / f"{r['id']}.md").write_text(f"# {r['title']}\n\n{r['text']}\n", encoding="utf-8")
    (PACK / "glossary_rag.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in world), encoding="utf-8")

    # Replace earlier definitions in prototype_rag.jsonl so the script can be rerun after editing the draft.
    path = PACK / "prototype_rag.jsonl"
    kept = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l).get("kind") != "definition"]
    path.write_text("".join(l + "\n" for l in kept) + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in proto), encoding="utf-8")
    print(f"{len(world)} world definitions -> glossary_rag.jsonl, {len(proto)} prototype definitions -> prototype_rag.jsonl")
    print("longest:", max(r["word_count"] for r in world + proto), "words")


main()
