# Chunked retrieval

There are 58 short in-world passages and six separate real-prototype passages. Their facts are written as things KOMA can explain, with first-person wording when discussing himself. They do not contain instructions to the author, drafting status or style rules. The system prompt controls behaviour.

For a question, the helper now sends up to four chunks (MAX_CHUNKS in koma_demo.py) and keeps any passage scoring at least 25 percent of the best match (RELATIVE_CUTOFF; it was 35 percent, which usually left one or two). build_prompt allows 4,000 characters of reference text, preserving whole passages. Titles and passage text are sent to the VLM; record IDs, source metadata and completion status remain available for logging without appearing in the reference text.

Use one format per collection: JSONL, individual Markdown files, or the combined lore Markdown. Do not index the entire ZIP or duplicate the same passages across these formats. The VLM does not browse the files itself; the Python application selects evidence and attaches it to the current request.

The retriever handles common singular/plural and selected word forms. It uses conversational aliases and modest phrase expansion for privacy idioms. This is deliberately lightweight and dependency-free; it is not a full linguistic stemmer or semantic embedding model. Morphology and aliases improve recall but cannot guarantee the correct result for every phrase.

Ordinary world queries search only lore. Questions about real recording, model or hardware use the prototype collection. Pure scene descriptions skip retrieval. The caller can override world/prototype routing or request visual mode explicitly. An image-dependent answer still requires a fresh image supplied by the caller.

For a short follow-up, pass last_topic and recent conversation. This module does not itself store conversation state. Metadata tags, source origins and the completed questionnaire exist for the author and operator; KOMA does not need to explain them.

The package does not set your Gemma runtime's thinking mode, context allocation or image-token budget. Check those in the existing application. No Jetson latency or generation-quality benchmark is claimed by the retrieval tests.


## Passage budget and links (13 November 2026)

retrieve() now decides how many passages a question gets: 3 for quick questions, 4 normally and 6 when the visitor asks for more ("go into", "tell me more", "explain how", "step by step"). MAX_CHUNKS (6) is the ceiling. A passage is kept only if it scores at least 30 percent of the best match (RELATIVE_CUTOFF), so the budget is a maximum, not a quota. Results are ordered best match first. For elaboration requests, the best passage's "related" links (set in koma_rag.jsonl) are added straight after it, before keyword neighbours, so the model gets the next part of the story. The application gets this automatically if it calls kb.retrieve(question, last_topic=...) or build_prompt(); passing top_k larger than 6 has no effect, and passing a smaller top_k lowers the ceiling.
