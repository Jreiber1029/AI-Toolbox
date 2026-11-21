# KOMA demo pack revision 4

This revision addresses the Claude Code review. It supports your current camera, Qwen3.5 VLM 2B and Jabra setup without assuming a YOLO detector. It prepares prompts for your existing application; it does not replace that application's camera or audio code.

## Replace the old corpus

Extract this ZIP into a fresh folder. Replace your application’s imported helper and active lore collection, then restart its process or rebuild its index. Do not merge old helios2100 passages with this corpus. Removing files from a folder does not remove records already loaded into a process or vector index.

The revised corpus contains 58 world passages in koma_rag.jsonl and six actual-prototype passages in prototype_rag.jsonl. Style and interaction rules are no longer searchable lore. The helper keeps the two domains separate and selects at most two relevant chunks.

## Integration

Keep the complete extracted KOMA_2070_demo folder together, beside your application script:

```python
from KOMA_2070_demo.koma_demo import KomaKnowledge, build_prompt

# Once at startup:
koma_knowledge = KomaKnowledge()

# At an existing question request:
prompt = build_prompt(
    mode="question",
    question=transcribed_question,
    knowledge=koma_knowledge,
    history=recent_session_text,  # optional; use "" if unavailable
    last_topic=previous_topic,   # optional; caller-managed
)
# Pass prompt to your existing VLM call.
# Attach a fresh image only when the question needs vision.
# Use your existing speech output for the returned text.
```

The example variables belong to your application; replace them with your actual names. No model, embeddings, API key or network is required for this retriever. It uses Python's standard library.

“What do you see?” and similar visual requests skip lore. The helper provides a visual task but cannot capture or refresh the image itself. Your caller must attach the current image. Tool questions can also use mode="tool"; no detector is required. The legacy detector_labels argument remains accepted for compatibility, but it is not a requirement or evidence of active YOLO use.

## World and actual-prototype questions

Basic routing is implemented with conservative phrase rules. Ordinary world questions use only world records. Explicit questions about actual hardware, models or recording operations use prototype records. A plain visual request uses neither collection.

Routing remains heuristic. If your application knows the intent, pass domain="world" or domain="prototype" explicitly. Use mode="visual" for a description request. Use mode="safety" for a real safety question; this pack does not replace a separate trustworthy technical-safety knowledge base.

In particular, “Are you spying on me?” retrieves local_privacy, while “Are you recording me?” uses prototype_recording. The former explains the ownership principle without claiming actual recording settings. The latter admits that the settings cannot be verified unless supplied. A wearable question retrieves Q4 knowledge but does not establish that an unfamiliar device in the image is Q4.

## Existing RAG systems

Import koma_rag.jsonl as the world collection and prototype_rag.jsonl as a separate collection for actual-device questions. Use text as the body; preserve title, keywords, questions, domain and source_questions as metadata. Each line is already one chunk.

Alternative loaders may use chunks/ for lore and prototype_chunks/ for actual facts, with chunk_index.json as metadata. koma_rag_passages.md is another lore-only alternative. Ingest one representation of each collection, not all duplicates.

Do not recursively index the entire ZIP. Exclude completed questionnaire files, author notes, operator_reference.json, README, test cases, trigger guidance, persona and narrative. These are review or operating material, not independent factual evidence.

## Retrieval changes

- Conservative word normalisation makes robot/robots, model/models, spy/spying, watch/watching and listen/listening match.
- Titles, keywords, example questions and body text are indexed.
- Conversational aliases improve matching without putting search keywords into KOMA's spoken facts.
- Weak second matches are discarded rather than padding every answer with two passages.
- Context contains titles and facts only; editorial statuses and record IDs are kept out of generation input.
- Follow-up expansion only applies to actual follow-ups such as “Why?”, not every short new question.

This remains a small lexical retriever, not a semantic model. It will not understand every paraphrase, and the test results are not a guarantee of generated-answer quality.

## Checks

```bash
python validate_pack.py
python koma_demo.py --question "Are you spying on me?" --retrieve-only
python koma_demo.py --question "What model are you running?" --retrieve-only
python koma_demo.py --question "What do you see?" --retrieve-only
```

The last command should return no reference. Check regression_queries.json for the questions and expected first-ranked passages. review_results.json contains the actual local validation results. These test retrieval and routing, not VLM performance or audio controls.

## Useful files

- world_introduction.md and world_guide.md: human-readable story references.
- koma_system_prompt.txt: identity and behaviour, including the rules moved out of retrieval.
- handover.txt: a fixed welcome you can send directly to your working TTS.
- visitor_answers.md: example responses for rehearsal.
- CHUNKING.md: corpus layout and context budgeting.
- REVIEW.md: findings, changes and limits of this review.
- completed_questionnaire.md: all responses with origin labels for your review only.

Original questionnaire references to YOLO describe an earlier setup. They are preserved as historical author material, not active implementation facts. No live Q4 connection, automatic training, logging setting or speech control is assumed merely because the world describes it.
