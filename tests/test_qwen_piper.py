"""CPU-only checks for the separate Qwen/Piper runner.

Run from YOLO-VLM with:
    ~/ultralytics-env/bin/python -B -m unittest discover -s tests -v

Camera, speaker, model responses, subprocesses and network access are replaced.
The actual Brain and HELIOS index are used to verify character/lore routing.
"""
import importlib.util
import json
import signal
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("koma_qwen_piper_tested", ROOT / "koma_qwen_piper.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def run_once(self, *extra, response_error=False, camera_error=False):
        """Exercise main's real dispatch/threading/cleanup with in-memory devices."""
        calls = {"requests": [], "spoken": [], "closed": [], "speaker_args": []}
        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        self.addCleanup(lambda: [signal.signal(sig, handler)
                                 for sig, handler in old_handlers.items()])

        class Camera:
            def __init__(self, *args):
                self.reads = 0
                self.deadline = time.monotonic() + 5

            def isOpened(self):
                return True

            def read(self):
                self.reads += 1
                if camera_error and self.reads == 2:
                    return False, None
                if time.monotonic() > self.deadline:
                    raise AssertionError("Single-request runner did not finish")
                # Yield to the request thread as a real blocking camera read would.
                time.sleep(0.002)
                return True, np.zeros((48, 64, 3), dtype=np.uint8)

            def release(self):
                calls["closed"].append("camera")

        class Process:
            def poll(self):
                return None

        class Speaker:
            def __init__(self, *args):
                calls["speaker_args"].append(args)
                self.idle = threading.Event()
                self.idle.set()
                self.proc = Process()

            def say(self, text):
                calls["spoken"].append(text)

            def flush(self):
                pass

            def close(self):
                calls["closed"].append("speaker")

        def response(url, messages, max_tokens, cancel):
            calls["requests"].append((messages, max_tokens))
            if response_error:
                raise RuntimeError("Simulated Qwen failure")
            yield "I can see a screwdriver."

        # Never open hardware or launch any real worker, even if dispatch changes.
        with patch.object(runner.cv2, "VideoCapture", Camera), \
                patch.object(runner, "Speaker", Speaker), \
                patch.object(runner, "stream_vlm", response), \
                patch.object(runner, "validate_args"), \
                patch.object(runner, "log"), \
                patch.object(runner.subprocess, "Popen", side_effect=AssertionError("Real subprocess forbidden")), \
                patch("urllib.request.urlopen", side_effect=AssertionError("Real network forbidden")):
            args = ["--camera", "/dev/null", "--once", "--no-voice", *extra]
            if camera_error:
                with self.assertRaisesRegex(RuntimeError, "Camera read failed"):
                    runner.main(args)
                calls["result"] = None
            else:
                calls["result"] = runner.main(args)
        return calls

    @staticmethod
    def turn_data(messages):
        text = messages[-1]["content"][0]["text"]
        data, _ = json.JSONDecoder().raw_decode(text[text.index("\n{") + 1:])
        return text, data

    def assert_closed(self, calls):
        self.assertEqual(calls["closed"].count("speaker"), 1)
        self.assertEqual(calls["closed"].count("camera"), 1)

    def test_native_voice_without_yolo_or_torch_import(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("ultralytics", sys.modules)
        calls = self.run_once()
        voice, length_scale, pitch, volume, _ = calls["speaker_args"][0]
        self.assertEqual(voice, ROOT / "voices/koma_lessac_high.onnx")
        self.assertEqual(length_scale, 1)
        self.assertEqual(pitch, 0)
        self.assertEqual(volume, 0.9)
        self.assertEqual(calls["result"], 0)
        self.assert_closed(calls)

    def test_observation_attaches_image_and_bounds_generation(self):
        calls = self.run_once()
        messages, tokens = calls["requests"][0]
        text, data = self.turn_data(messages)
        self.assertEqual(tokens, 90)
        self.assertTrue(data["image_attached"])
        self.assertTrue(any(item["type"] == "image_url" for item in messages[-1]["content"]))
        self.assertNotIn("A detector triggered", text)
        self.assertNotIn("detector_guesses_not_verified_objects", data)
        self.assertIn("two short sentences", text)

    def test_world_question_retrieves_lore_without_camera_claim(self):
        calls = self.run_once("--ask", "Why was local AI banned?")
        messages, tokens = calls["requests"][0]
        _, data = self.turn_data(messages)
        self.assertEqual(tokens, 170)
        self.assertFalse(data["image_attached"])
        self.assertIn("TradesMaster professional exemption", data["fictional_world_reference"])
        self.assertEqual(messages[0]["content"], runner.Brain().system_prompt)

    def test_safety_turn_excludes_fictional_lore(self):
        calls = self.run_once("--ask", "Is this tool safe?")
        messages, tokens = calls["requests"][0]
        text, data = self.turn_data(messages)
        self.assertEqual(tokens, 120)
        self.assertIn("CURRENT TASK: Address the concrete safety", text)
        self.assertNotIn("fictional_world_reference", data)
        self.assertTrue(data["image_attached"])
        self.assertIn("YOLO is disabled", data["operator_supplied_prototype_facts"])

    def test_new_visitor_uses_authored_handover(self):
        calls = self.run_once("--ask", "new visitor")
        self.assertEqual(calls["requests"], [])
        self.assertEqual(calls["spoken"], [runner.Brain().handover_script])
        self.assert_closed(calls)

    def test_model_failure_returns_failure_and_releases_resources(self):
        calls = self.run_once(response_error=True)
        self.assertEqual(calls["result"], 1)
        self.assert_closed(calls)

    def test_startup_camera_failure_releases_resources(self):
        calls = self.run_once(camera_error=True)
        self.assertEqual(calls["requests"], [])
        self.assert_closed(calls)


if __name__ == "__main__":
    unittest.main()
