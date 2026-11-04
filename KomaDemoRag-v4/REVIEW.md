# Review of Claude Code findings

Both categories are valid. Some quoted wording, including law_enforcement_open and the old Q4 description, comes from earlier versions. The current version had already replaced those exact passages, but it still mixed operator instructions, style and real-device limitations into ordinary lore. The earlier validation was insufficient because it checked a few forbidden phrases and a small question set.

## Corpus repair

All 60 former world passages were reviewed. KOMA's references to his own body, abilities and relationship with the worker now use first-person phrasing where appropriate. The law and history passages give the established account rather than discussing draft decisions. Prototype facts are separate from fiction and written as things KOMA can say when asked about the actual device.

Two style records, koma_voice and tone_satire, are no longer retrievable. Four interaction records have also been taken out of the factual index. Their content is in the system rules, helper routing and operator reference, where appropriate. This produces 58 world passages and six separately routed prototype passages.

The prototype record explicitly states that YOLO is no longer used. The helper does not require detections; its legacy label parameter remains optional for existing callers. Current recording settings and live sensor connections are not invented.

There is an important distinction: a fictional data-ownership principle should not make KOMA promise that the actual installation records nothing. “Are you spying on me?” retrieves the privacy principle; explicit recording questions retrieve the actual-device limitation. Neither path asserts unverified deletion or network settings.

## Retrieval repair

The six reported phrases now rank the intended record first:

| Question | First record |
|---|---|
| Are you spying on me? | local_privacy |
| What is this thing on my wrist? | q4_role |
| How long do I have to work? | work_conditions |
| What tools do I need for plumbing? | work_examples |
| Why are you different from ChatGPT? | local_definition |
| So what are you exactly? | koma_body |

Changes include vocabulary and conversational aliases, conservative singular/plural and word-form normalisation, privacy-idiom expansion, and suppression of weak second matches. This is not a full English stemmer or semantic search system. For example, it handles robot/robots and spy/spying without adding an embedding model.

The exact Q4 question retrieves the explanation, but does not identify an unknown wrist device as Q4. The visual task still requires image evidence. Search relevance and object identification are separate decisions.

## Validation and remaining limits

The local regression suite passes 33 of 33 first-ranked questions, including the six reports and additional phrasings. Additional checks cover morphology, source IDs, individual chunk files, context budgets, new-topic versus follow-up handling, world/prototype isolation and visual questions that should retrieve no lore. These are recorded in review_results.json.

The original 25-question Claude suite was not supplied in full, so this is not a claim to have rerun that exact suite. No Qwen model was run and no Jetson performance was measured. Aliases and heuristics improve these cases without guaranteeing every paraphrase; the application can explicitly select world, prototype or visual intent.

## Applying the fix

Extract into a fresh directory and replace the active helper and collection. Rebuild or reload any existing index. Do not leave old helios2100 records or deleted style records in a persistent collection. corpus_manifest.json gives the file hashes for this revision so the operator can identify which corpus is actually loaded.
