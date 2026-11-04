#!/usr/bin/env python3
"""Continuously inspect IMX219 frames using SmolVLM and safety RAG."""

import argparse
import base64
import json
import queue
import re
import threading
import time
import urllib.error
import urllib.request

import cv2


SMOLVLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
SAFETY_RAG_URL = "http://127.0.0.1:8090/retrieve"
MODEL = "ggml-org/SmolVLM2-2.2B-Instruct-GGUF:Q4_K_M"
PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)


def inspect_frame(frame, prompt):
    """Run structured visual triage on one camera frame."""
    ok, encoded = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
    )
    if not ok:
        raise RuntimeError("Could not encode the camera frame")

    image = base64.b64encode(encoded).decode("ascii")
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the visual triage layer for a workplace safety system. "
                    "Use only visible evidence and do not claim regulatory compliance. "
                    "Fill every JSON field with a concrete, short assessment of this "
                    "specific image. Never repeat field instructions or placeholder "
                    "phrases. Use NONE when no work activity, equipment, people, or "
                    "potential hazard is visible. Use UNKNOWN only when material cannot "
                    "be identified. Set rag_required true when a potential workplace "
                    "hazard is visible or a work-related safety uncertainty needs rules; "
                    "otherwise set it false."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    },
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 160,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "workplace_scene_assessment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "work_activity": {"type": "string"},
                        "tools_equipment": {"type": "string"},
                        "materials": {"type": "string"},
                        "people": {"type": "string"},
                        "potential_hazards": {"type": "string"},
                        "rag_required": {"type": "boolean"},
                    },
                    "required": [
                        "work_activity",
                        "tools_equipment",
                        "materials",
                        "people",
                        "potential_hazards",
                        "rag_required",
                    ],
                    "additionalProperties": False,
                },
            },
        },
    }
    request = urllib.request.Request(
        SMOLVLM_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    raw_assessment = result["choices"][0]["message"]["content"].strip()
    assessment = json.loads(raw_assessment)

    def clean_field(name, fallback):
        value = str(assessment.get(name, "")).strip()
        return value if value else fallback

    work_activity = clean_field("work_activity", "NONE")
    tools_equipment = clean_field("tools_equipment", "NONE")
    materials = clean_field("materials", "UNKNOWN")
    people = clean_field("people", "NONE")
    potential_hazards = clean_field("potential_hazards", "NONE")

    no_hazard_values = {"NONE", "NO HAZARD", "NO HAZARDS"}
    rag_required = potential_hazards.upper() not in no_hazard_values

    return "\n".join(
        [
            f"WORK_ACTIVITY: {work_activity}",
            f"TOOLS_EQUIPMENT: {tools_equipment}",
            f"MATERIALS: {materials}",
            f"PEOPLE: {people}",
            f"POTENTIAL_HAZARDS: {potential_hazards}",
            f"RAG_REQUIRED: {'YES' if rag_required else 'NO'}",
        ]
    )


def should_query_rag(scene_assessment):
    """Read RAG_REQUIRED; fail safely to retrieval if output is malformed."""
    match = re.search(
        r"^RAG_REQUIRED:\s*\[?\s*(YES|NO)\s*\]?\s*$",
        scene_assessment,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if match:
        return match.group(1).upper() == "YES"

    hazards = re.search(
        r"^POTENTIAL_HAZARDS:\s*\[?\s*([^\]\r\n]+)\s*\]?\s*$",
        scene_assessment,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if hazards and hazards.group(1).strip().upper() == "NONE":
        print("RAG_REQUIRED was malformed, but POTENTIAL_HAZARDS is NONE; skipping RAG.")
        return False

    print("Triage decision was malformed or missing; using RAG as a safe fallback.")
    return True


def retrieve_safety_context(observation):
    """Retrieve safety references relevant to a visual observation."""
    payload = {"observation": observation}
    request = urllib.request.Request(
        SAFETY_RAG_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)

    references = []
    for item in result.get("references", []):
        source = item.get("source", "Unknown source")
        page = item.get("page")
        text = item.get("text", "").strip()
        citation = f"{source}, page {page}" if page else source
        if text:
            references.append(f"Source: {citation}\n{text}")

    if not references:
        return "No relevant safety references were found."
    return "\n\n".join(references)


def recommend_from_observation(observation, safety_context, prompt):
    """Analyze an observation using the retrieved safety references."""
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a health and safety expert. Analyze the camera "
                    "observation using only the supplied safety references. Do not "
                    "invent requirements or citations. Respond in no more than "
                    "three short sentences and 80 words. Identify the main hazard, "
                    "assess the precautions, and recommend the most important action."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Structured camera assessment:\n{observation}\n\n"
                    f"Retrieved safety references:\n{safety_context}\n\n"
                    f"Analysis task:\n{prompt}"
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 100,
        "stream": False,
    }
    request = urllib.request.Request(
        SMOLVLM_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    return result["choices"][0]["message"]["content"].strip()


def run_analysis(frame, observation_prompt, recommendation_prompt):
    """Run triage, then conditionally retrieve and recommend."""
    print("Pass 1: running structured workplace-hazard triage...")
    observation = inspect_frame(frame, observation_prompt)
    print(f"Scene assessment:\n{observation}")

    if not should_query_rag(observation):
        print("RAG_REQUIRED is NO; skipping safety retrieval and second pass.")
        return "No observable potential workplace hazard identified; RAG was not queried."

    print("Retrieving relevant safety rules...")
    safety_context = retrieve_safety_context(observation)

    print("Pass 2: analyzing observation with safety rules...")
    recommendation = recommend_from_observation(
        observation,
        safety_context,
        recommendation_prompt,
    )
    return recommendation


def main():
    parser = argparse.ArgumentParser(
        description="Inspect IMX219 frames with SmolVLM and safety RAG"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="seconds between inspection attempts (default: 5)",
    )
    parser.add_argument(
        "--prompt",
        default="Inspect the frame for workplace activity and potential hazards.",
        help="visual triage instruction",
    )
    parser.add_argument(
        "--recommendation-prompt",
        default=(
            "Determine whether appropriate health and safety precautions appear "
            "to have been taken and recommend the most important action."
        ),
        help="task for the grounded safety analysis",
    )
    parser.add_argument(
        "--once", action="store_true", help="inspect one frame and exit"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="run without opening a camera preview window",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    camera = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
    if not camera.isOpened():
        raise RuntimeError("Could not open IMX219 camera with nvarguscamerasrc")

    if args.once:
        try:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Failed to read an IMX219 frame")
            recommendation = run_analysis(
                frame, args.prompt, args.recommendation_prompt
            )
            print(f"Recommendation: {recommendation}")
        finally:
            camera.release()
        return

    jobs = queue.Queue(maxsize=1)
    state = {"text": "Waiting for first inspection...", "busy": False}
    state_lock = threading.Lock()

    def worker():
        while True:
            frame = jobs.get()
            try:
                with state_lock:
                    state["busy"] = True
                recommendation = run_analysis(
                    frame, args.prompt, args.recommendation_prompt
                )
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] Recommendation: {recommendation}\n")
                with state_lock:
                    state["text"] = recommendation
            except (OSError, KeyError, ValueError, urllib.error.URLError) as error:
                print(f"Analysis request failed: {error}")
                with state_lock:
                    state["text"] = f"Request failed: {error}"
            finally:
                with state_lock:
                    state["busy"] = False
                jobs.task_done()

    threading.Thread(target=worker, daemon=True).start()
    next_inspection = 0.0
    print("Camera inspection is running. Press q in the preview or Ctrl+C to stop.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Failed to read an IMX219 frame")

            now = time.monotonic()
            if now >= next_inspection:
                try:
                    jobs.put_nowait(frame.copy())
                    next_inspection = now + args.interval
                except queue.Full:
                    pass

            if not args.no_preview:
                with state_lock:
                    text = state["text"]
                    busy = state["busy"]
                status = "SmolVLM inspecting..." if busy else "SmolVLM ready"
                cv2.putText(
                    frame,
                    status,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    text[:100],
                    (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow("IMX219 + SmolVLM Safety RAG", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
