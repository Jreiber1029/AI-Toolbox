# Suggested request handling without YOLO

These are application behaviours, not searchable world facts.

| Request | Context | Action |
|---|---|---|
| What do you see? | No lore | Send a fresh image and a plain visual task |
| What is this wearable? | Q4 facts if relevant | Use the image; do not infer Q4 just because it is on a chest strap |
| Why is local AI restricted? | World passages | Answer without an image unless one is relevant |
| What model are you running? | Prototype passages | Explain actual known components |
| Are you recording me? | Prototype recording facts | Admit unknown settings unless application supplies them |
| What is this tool used for? | Relevant work passage | Verify the object in the image |
| Why? | Previous topic and recent conversation | Expand the current topic without restarting the introduction |
| Stop | No retrieval | Application cancels playback and queued work |

One VLM request at a time avoids a queue of stale commentary. The content pack cannot enforce audio cancellation, frame freshness, session boundaries or network/logging settings. Those remain application responsibilities. Quiet observation does not require automatic speech.
