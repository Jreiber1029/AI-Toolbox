"""KOMA's character and memory: prompt assembly + HELIOS lore retrieval.

The persona comes straight from Koma-storyrag/koma_exhibition_prompts.py (the
koma_exhibition_prompts module), so edits there take effect on the next start.
Lore comes from Koma-storyrag/HELIOS_knowledge.jsonl via a small BM25 index:
no embedding model, no extra memory, millisecond lookups.
"""
import importlib.machinery
import importlib.util
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STORY_DIR = ROOT / "Koma-storyrag"
PROMPT_FILE = STORY_DIR / "koma_exhibition_prompts.py"
KNOWLEDGE_FILE = STORY_DIR / "HELIOS_knowledge.jsonl"
RUNTIME_FACTS_FILE = ROOT / "runtime_facts.txt"


def _load_prompt_module():
    loader = importlib.machinery.SourceFileLoader("koma_exhibition_prompts", str(PROMPT_FILE))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


prompts = _load_prompt_module()

STOPWORDS = set("""a an and are as at be but by can did do does for from has have how i if in
is it its me my of on or so that the their them there this to was we what when where which
who why will with you your yours am im i'm be been being would could should about into""".split())


IRREGULAR = {"made": "make", "making": "make", "makes": "make", "built": "build",
             "cant": "cannot", "can't": "cannot", "doesnt": "not", "dont": "not",
             "everyone": "people", "citizens": "people", "citizen": "people",
             "ordinary": "people", "robots": "automation", "robot": "automation",
             "machines": "automation", "machine": "automation"}


def stem(word):
    word = IRREGULAR.get(word, word)
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def tokenize(text):
    words = re.findall(r"[a-z0-9']+", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


class LoreIndex:
    """BM25 over the knowledge records. Titles, questions and keywords count extra."""

    def __init__(self, path=KNOWLEDGE_FILE, k1=1.4, b=0.6):
        self.records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.k1, self.b = k1, b
        self.docs = []
        self.question_terms = []
        for r in self.records:
            boosted = " ".join([r["title"]] * 2 + r.get("questions", []) * 3 + r.get("keywords", []) * 2)
            self.docs.append(Counter(tokenize(boosted + " " + r["text"])))
            self.question_terms.append(set(tokenize(" ".join([r["title"]] + r.get("questions", [])))))
        self.avg_len = sum(sum(d.values()) for d in self.docs) / len(self.docs)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query, k=2, exclude=(), types=None, min_score=1.0):
        terms = tokenize(query)
        scored = []
        for i, (rec, doc) in enumerate(zip(self.records, self.docs)):
            if rec["id"] in exclude or (types and rec["record_type"] not in types):
                continue
            length = sum(doc.values())
            score = sum(
                self.idf.get(t, 0) * doc[t] * (self.k1 + 1)
                / (doc[t] + self.k1 * (1 - self.b + self.b * length / self.avg_len))
                for t in terms if t in doc)
            # A visitor question that closely matches a record's own question is the
            # strongest signal we have; long story scenes shouldn't outrank it.
            q = self.question_terms[i]
            if q and terms:
                overlap = len(q & set(terms)) / len(q | set(terms))
                score += 12 * overlap
            if score >= min_score:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [self.records[i] for _, i in scored[:k]]


def passage(record, max_chars=700):
    """Compact text for the prompt: Q/A pairs use the short answer, lore is trimmed."""
    if record.get("answer"):
        body = f"Q: {record['title']}\nA: {record['answer']}"
    else:
        # Drop the boilerplate first line ("Setting facts for the fictional ...").
        text = record["text"].split("\n\n", 1)[-1]
        body = f"{record['title']}: {text}"
    return body if len(body) <= max_chars else body[:max_chars].rsplit(" ", 1)[0] + " ..."


HANDOVER_PATTERNS = re.compile(
    r"\b(what are you|who are you|what is koma|what's koma|introduce yourself)\b")
# Questions about the real exhibit get the operator's runtime facts.
PROTOTYPE_PATTERNS = re.compile(
    r"\b(real|really|today|prototype|actually|record\w*|camera|connected|privacy|private|data|"
    r"delete\w*|jetson|computer|cloud|listening|hear\w*|how do you work|q4|sensor\w*|turn you off)\b")
# Questions that need the camera frame; others are answered from lore alone.
VISUAL_PATTERNS = re.compile(
    r"\b(this|that|these|those|it|holding|hold|hand|see|look\w*|show\w*|colou?r|tool|object|here)\b")
STORY_PATTERNS = re.compile(r"\b(mara|nell|story|shop\w*|grandmother|first job|first commission)\b")
SENTENCE_LIMIT = {"tool": 2, "question": 4, "safety": 3}
NEW_VISITOR_PATTERNS = re.compile(r"\b(new visitor|new session)\b")
SAFETY_PATTERNS = re.compile(
    r"\b(safe|safety|danger|dangerous|hurt|injur\w*|bleed\w*|burn\w*|fire|smoke|shock|"
    r"electric\w*|live wire|gas|fume\w*|dust|asbestos|emergency|first aid|cut myself)\b")


def classify(text):
    """Pick the exhibition event for a spoken request."""
    t = text.lower()
    if NEW_VISITOR_PATTERNS.search(t):
        return "new_visitor"
    if HANDOVER_PATTERNS.search(t):
        return "handover"
    if SAFETY_PATTERNS.search(t):
        return "safety"
    return "question"


class Brain:
    def __init__(self, tool_lore=True):
        self.lore = LoreIndex()
        self.tool_lore = tool_lore
        self.recent_lore = []  # ids used recently, so tool comments don't repeat themselves
        self.reload_runtime_facts()

    def reload_runtime_facts(self):
        self.runtime_facts = (RUNTIME_FACTS_FILE.read_text().strip()
                              if RUNTIME_FACTS_FILE.exists() else "")

    @property
    def system_prompt(self):
        return prompts.KOMA_SYSTEM_PROMPT

    @property
    def handover_script(self):
        return prompts.KOMA_HANDOVER_SCRIPT

    def _remember_lore(self, records):
        self.recent_lore = ([r["id"] for r in records] + self.recent_lore)[:6]

    @staticmethod
    def wants_image(event, question, labels):
        """Attach the frame for tool turns, or questions that refer to the scene."""
        return event == "tool" or bool(labels) or bool(VISUAL_PATTERNS.search(question.lower()))

    def user_prompt(self, event, *, labels=(), question="", image=False):
        """Return (prompt_text, lore_ids) for one turn."""
        lore = []
        if event == "question":
            types = None if STORY_PATTERNS.search(question.lower()) else {"lore", "visitor_qa", "tone_examples"}
            lore = self.lore.search(question, k=2, types=types)
        elif event == "tool" and self.tool_lore:
            # One small world detail per tool, rotated so the same joke doesn't recur.
            query = " ".join(labels) + " tool repair training automation toilet"
            lore = self.lore.search(query, k=1, exclude=self.recent_lore,
                                    types={"lore", "visitor_qa"}, min_score=0)
        self._remember_lore(lore)
        about_prototype = event == "safety" or bool(PROTOTYPE_PATTERNS.search(question.lower()))
        text = prompts.build_koma_user_prompt(
            event,
            detector_labels=list(labels),
            visitor_question=question,
            image_attached=image,
            retrieved_lore="\n\n".join(passage(r) for r in lore),
            runtime_facts=self.runtime_facts if about_prototype else "",
        )
        # Small models follow a limit stated last far better than one buried in the system prompt.
        limit = SENTENCE_LIMIT[event]
        text += (f"\n\nReply in at most {limit} short spoken sentences. Do not repeat earlier "
                 "replies. Do not describe the image unless it helps answer.")
        return text, [r["id"] for r in lore]

    def build_messages(self, event, *, labels=(), question="", image_b64=None, history=()):
        """Chat messages for one turn: system, recent text-only turns, then this turn."""
        text, lore_ids = self.user_prompt(event, labels=labels, question=question,
                                          image=image_b64 is not None)
        content = [{"type": "text", "text": text}]
        if image_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        # The system prompt is identical every turn, so llama.cpp reuses its cached prefix.
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += list(history)
        messages.append({"role": "user", "content": content})
        return messages, lore_ids
