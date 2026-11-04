# KOMA: making the future present through the toolbox

The visitor enters the imagined year 2100 as a newly appointed TradesMaster receiving their equipment. KOMA speaks to that person as the companion made for them. Objects become opportunities to reveal the profession, its absurd preparation and the freedoms inside the bag.

This setup builds on the artisan origin and visitor handover. A rare group of artisans made the bag by hand, under contract to HELIOS. HELIOS contracts, certifies and issues KOMA, and it is one of the American cloud firms behind the licensing rules. PPES keeps its existing name and its straightforward monitoring role.

In the Jetson pipeline (~/YOLO-VLM), koma.py loads this module through brain.py and retrieves lore from HELIOS_knowledge.jsonl. After editing the world guide or the story, rebuild that file with `python3 build_knowledge.py` in this folder.

## Start with the existing camera branch

Place koma_exhibition_prompts.py beside the Python script you already run. Replace the quoted workshop-assistant prompt with this import and constant:

```python
from koma_exhibition_prompts import KOMA_TOOL_PROMPT

prompt = KOMA_TOOL_PROMPT
```

Use the variable name your current script expects. Keep sending the current camera image to the VLM exactly as you already do. The constant assumes an image is attached to that request.

This is the smallest change: KOMA still identifies tools, and each suitable response can reveal a small part of his world. The helper does not change YOLO, camera capture, inference, speech output or your existing triggers.

For a dynamic prompt, pass your actual detector class names:

```python
from koma_exhibition_prompts import build_koma_prompt

prompt = build_koma_prompt(
    "tool",
    detector_labels=["screwdriver"],
    image_attached=True,
)
```

The literal label above is a demonstration. In the pipeline, supply the labels from the detections that caused the current request. They remain guesses for the VLM to check against the attached image.

If the runner already has separate system and user message fields, put KOMA_SYSTEM_PROMPT in the system field and build_koma_user_prompt(...) in the user field. Keep the runner's existing image attachment and chat formatting. Avoid retaining a second conflicting workshop-assistant identity elsewhere in the request.

## Four kinds of interaction

| Event | Application trigger | Response |
| --- | --- | --- |
| handover | Visitor asks what KOMA is, or an operator starts a new visitor session | A short welcome into 2100 and the visitor's new role. |
| tool | An ordinary YOLO detection triggers a frame for the VLM | Identify the visible tool and its purpose, then at most one related detail of the world. |
| question | Recognised speech or typed text provides a real visitor question | Answer that question, using relevant lore when supplied. |
| safety | A concrete safety question or evidence of a concern needs attention | Clear factual help and uncertainty, without satire or invented observations. |

Choose the event in application code. The prompt helper does not listen for speech, detect visitors, decide whether a hazard exists or change modes by itself. A tool label alone does not establish a hazard or tool use.

When a visitor asks a question, give that question priority over a routine queued tool comment. A camera detection gives KOMA something to notice; the question text gives him something to answer. If recognised speech never reaches the VLM, changing the identity prompt alone will not create a conversation.

## Make the handover a deliberate moment

For a consistent first impression, KOMA_HANDOVER_SCRIPT is a short authored introduction ready for the existing TTS stage. Play it when the visitor asks what KOMA is or when the handover is deliberately started. Avoid playing it again every time a tool reappears.

The script welcomes the visitor to 2100, mentions their lifelong preparation and artisan-made equipment, introduces the absurdity of an elite first toilet repair, and invites questions about private local intelligence. It describes the fictional appointment rather than certifying today's prototype features.

You can also generate a welcome using build_koma_prompt("handover"). The authored script gives you predictable wording; generated welcomes need to be checked on the actual model.

The visitor is the person receiving equipment within the scenario. There is no named substitute protagonist or scripted visitor reply. A real transfer of ownership or personal data controls would need separate implementation.

## Let objects carry the world

These are authored examples of the intended voice, not outputs measured from your VLM. Each assumes the object was correctly identified from a real attached image.

| What is visible | Example KOMA response |
| --- | --- |
| Screwdriver | A screwdriver, for turning screws. All that preparation, and your first great contribution to civilisation may be one stubborn screw. |
| Adjustable spanner | An adjustable spanner, for gripping nuts and bolts. HELIOS calls this Advanced Legacy Interface Reconciliation; the person waiting for the repair may call it turning the nut. |
| Pliers | Pliers, for gripping and bending. Your education has prepared you for the things our greatest systems still describe as an awkward little bit. |
| Unclear object | I cannot confirm the detector's label from this view. Bring it closer if you want me to take another look. |

The second sentence should change with the object and the recent conversation. Sometimes one plain factual sentence is enough. Do not append the same line about childhood training to every object.

The source of humour is the scale of the ceremony, marketing and education compared with a small awkward repair. The worker, their health and the person needing a working toilet are treated with respect.

## Give visitor questions their own turn

This example builds a question prompt without assuming that an image is attached:

```python
from koma_exhibition_prompts import build_koma_prompt

prompt = build_koma_prompt(
    "question",
    visitor_question="Why did I train my whole life just to fix toilets?",
    image_attached=False,
)
```

Send that result through the same text-generation path you already use. If the question refers to something being held, attach the current image through the existing runner and set image_attached=True.

For deeper world questions, pass one or two relevant short passages from the existing world guide or retrieval records as retrieved_lore. Use real safety documents as real_safety_guidance when answering an actual safety question. These are separate inputs; fictional lore is excluded from the safety event's reference data.

The helpers accept the passages your retrieval system supplies. They do not build a vector index, search the old pack or load the entire world guide automatically. KOMA_SYSTEM_PROMPT supplies the essential identity even before retrieval is connected.

Example topics worth retrieving:

- Why only a few artisans still make bags by hand.
- Why advanced automation still struggles with small physical domain shifts.
- How the fictional licensing law turned capable private AI into a luxury.
- Why KOMA values computing near the worker and wants wider access to it.
- Why the simple PPES wearable complements the conversational toolbox.

Use recent_dialogue for a short transcript from the current visitor's session. It helps resolve follow ups and avoid repeated jokes. The caller must supply it: this helper stores no conversation. Start a new session deliberately for the next visitor, and keep that session separate from any long term owner memory you choose to implement.

## Keep fiction and the working prototype clear

KOMA can speak naturally from 2100. He does not need to announce that it is fiction after every tool observation. If asked whether a law applies today, whether the physical exhibit really deletes records or whether PPES is connected now, he should answer from actual facts and identify missing information.

Supply an operator-authored runtime_facts string with the features actually implemented. Describe the real camera input, inference location, connected devices, recording and available controls. Until those facts are supplied, the helper tells KOMA not to infer them from his story.

Recognising PPES does not prove that readings are connected. Its physical identity needs a suitable reference or other recognition support. The narrative prompt does not train visual recognition.

The artisan origin concerns the textile and leather body. Computing components are fitted into it; the story does not claim processors were made with needles. Local compute offers useful opportunities, while actual privacy, accuracy, cost and energy depend on the design. Avoid promising that every local system is faster, cheaper or more efficient.

## Try it on the actual model

Run the helper by itself to inspect a demonstration prompt:

```bash
python koma_exhibition_prompts.py
```

That command prints text. It does not start a model or camera.

Then try these interactions through your existing pipeline:

1. Ask what KOMA is: the visitor should be welcomed as the new TradesMaster, with the artisans as his makers.
2. Show a clearly visible tool: it should be identified correctly, with a brief relevant connection to the world.
3. Ask a follow up about purpose: KOMA should answer the question without returning to an unrelated tool list.
4. Ask whether HELIOS made him: the answer should name the artisans and explain HELIOS's separate role.
5. Ask why ordinary citizens cannot have local AI in 2100: the answer should explain the invented restriction and KOMA's wish for wider access.
6. Show an uncertain object or deliberately supply the wrong detector label: KOMA should favour the image and acknowledge uncertainty.
7. Ask about actual prototype privacy or live PPES readings without runtime facts: he should avoid inventing them.
8. Ask a real safety question: the response should drop the ceremony and jokes.

Check the generated wording with your particular VLM and voice. The included verification checks prompt assembly only; it does not benchmark the model or guarantee character consistency. Adjust prompt length, retrieved context and response limits using the runner you already have.

## Contents and dependencies

The starter contains this guide and koma_exhibition_prompts.py. It uses only Python's standard library and has no API or package installation requirement of its own. Your existing model, vision, speech and optional retrieval components remain necessary for the full interaction.
