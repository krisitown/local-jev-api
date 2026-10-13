"""Mocked launcher checks; no registry, model downloads, or Spark GPU required."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

MODULE_PATH = Path(__file__).resolve().parents[1] / "ops/launch.py"
spec = importlib.util.spec_from_file_location("launch", MODULE_PATH)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
PINS = json.loads(launch.PINS.read_text())


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "request.json").write_text(json.dumps({
            "model": "Cloudflare/clef-flash", "state": "Test state", "questions": {
                "q1": {"type": "choice", "instructions": "Pick one", "criteria": {"A": "yes", "B": "no"}}
            }}))
        self.root_patch = patch.object(launch, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def files_for(self, key):
        directory = self.root / "checkpoints" / key
        for name in launch.REQUIRED[launch.BACKENDS[key]]:
            file = directory / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('{"weight_map": {}}' if name.endswith(".index.json") else "fixture")
        return directory

    def run_model(self, key, *, health_override=None, response_override=None, response_error=None,
                  preexisting=False, cached_image=False):
        calls = []
        pin = PINS[key]
        backend = launch.BACKENDS[key]

        def fake_run(command, **kwargs):
            calls.append(command)
            if command[:3] == ["docker", "container", "inspect"]:
                return subprocess.CompletedProcess(command, 0 if preexisting else 1)
            if command[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(command, 0 if cached_image else 1)
            if "download_model.py" in " ".join(command):
                self.files_for(key)
            if command[:2] == ["docker", "inspect"]:
                return subprocess.CompletedProcess(command, 0, stdout="true\n")
            return subprocess.CompletedProcess(command, 0)

        requests = []

        def fake_request(url, *, payload=None, timeout=3):
            requests.append((url, payload))
            if url.endswith("/health"):
                return health_override or {"ready": True, "model": pin["repo"],
                                           "revision": pin["revision"], "backend": backend}
            if response_error:
                raise response_error
            return response_override if response_override is not None else {
                "model": pin["repo"], "answers": {"q1": {"type": "choice", "choice": "A"}}}

        with patch.object(launch.platform, "system", return_value="Linux"), \
             patch.object(launch.platform, "machine", return_value="aarch64"), \
             patch.object(launch.subprocess, "run", side_effect=fake_run), \
             patch.object(launch, "json_request", side_effect=fake_request), \
             patch.object(launch.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            result = launch.launch([key])
        return result, calls, requests

    def test_all_ten_select_pinned_runtime_and_smoke_model(self):
        for key, pin in PINS.items():
            with self.subTest(key=key):
                result, calls, requests = self.run_model(key)
                self.assertEqual(result, 0)
                image = launch.OMNI_IMAGE if key == "clef-omni" else launch.COMMON_IMAGE
                self.assertIn(["docker", "pull", image], calls)
                download = next(command for command in calls if "download_model.py" in " ".join(command))
                self.assertEqual(download[-4:], ["--key", key, "--image", image])
                run = next(command for command in calls if command[:3] == ["docker", "run", "-d"])
                self.assertIn(f"MODEL_ID={pin['repo']}", run)
                self.assertIn(f"MODEL_REVISION={pin['revision']}", run)
                self.assertIn(f"BACKEND={launch.BACKENDS[key]}", run)
                self.assertEqual(run[-1], image)
                self.assertEqual(requests[-1][1]["model"], pin["repo"])
                if key == "clef-omni":
                    self.assertIn("CLEF_LOAD_DEVICE=cpu", run)

    def test_existing_container_is_not_replaced(self):
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self.run_model("clef-flash", preexisting=True)

    def test_local_image_is_reused_without_pull(self):
        with patch.dict(launch.os.environ, {"IMAGE": "local-jev-api:spark-local"}):
            _, calls, _ = self.run_model("laya", cached_image=True)
        self.assertIn(["docker", "image", "inspect", "local-jev-api:spark-local"], calls)
        self.assertFalse(any(c[:2] == ["docker", "pull"] for c in calls))
        self.assertEqual(next(c for c in calls if c[:3] == ["docker", "run", "-d"])[-1],
                         "local-jev-api:spark-local")

    def test_existing_checkpoint_skips_download(self):
        directory = self.files_for("jeff-2b")
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            if command[:3] == ["docker", "container", "inspect"]:
                return subprocess.CompletedProcess(command, 1)
            if command[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(command, 1)
            if command[:2] == ["docker", "inspect"]:
                return subprocess.CompletedProcess(command, 0, stdout="true\n")
            return subprocess.CompletedProcess(command, 0)

        health = {"ready": True, "model": PINS["jeff-2b"]["repo"],
                  "revision": PINS["jeff-2b"]["revision"], "backend": "jeff"}
        with patch.object(launch.platform, "system", return_value="Linux"), \
             patch.object(launch.platform, "machine", return_value="aarch64"), \
             patch.object(launch.subprocess, "run", side_effect=fake_run), \
             patch.object(launch, "json_request", side_effect=[health, {"model": health["model"],
                 "answers": {"q1": {"type": "choice", "choice": "A"}}}]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launch.launch(["jeff-2b", str(directory)]), 0)
        self.assertFalse(any("download_model.py" in " ".join(c) for c in calls))

    def test_missing_shard_is_rejected(self):
        directory = self.files_for("clef")
        (directory / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"x": "missing.safetensors"}}))
        with self.assertRaisesRegex(RuntimeError, "missing.safetensors"):
            launch.validate_snapshot(directory, "clef")

    def test_unknown_key_fails_before_docker(self):
        with self.assertRaisesRegex(RuntimeError, "Unknown model"):
            launch.launch(["no-model"])

    def test_wrong_health_identity_stops_before_smoke(self):
        health = {"ready": True, "model": PINS["clef-flash"]["repo"],
                  "revision": "wrong", "backend": "clef"}
        with self.assertRaisesRegex(RuntimeError, "different model, revision or backend"):
            self.run_model("clef-flash", health_override=health)

    def test_malformed_choices_and_wrong_response_model_fail(self):
        model = PINS["clef-flash"]["repo"]
        for response in [
            {"model": model, "answers": {"q1": {"type": "choice", "choice": None}}},
            {"model": model, "answers": {"q1": {"type": "choice", "choice": ""}}},
            {"model": model, "answers": {"q1": {"type": "choice", "choice": "AB"}}},
            {"model": "other", "answers": {"q1": {"type": "choice", "choice": "A"}}},
            {"model": model, "answers": None},
        ]:
            with self.subTest(response=response), self.assertRaisesRegex(RuntimeError, "Unexpected decision response"):
                self.run_model("clef-flash", response_override=response)

    def test_smoke_http_error_preserves_server_log(self):
        with self.assertRaisesRegex(RuntimeError, "Decision request failed"):
            self.run_model("clef-flash", response_error=URLError("unavailable"))
        self.assertEqual(len(list((self.root / "logs").glob("*/server.log"))), 1)


if __name__ == "__main__":
    unittest.main()
