"""Check the tensor handoff contract without importing the simulator."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

TASK = Path(__file__).resolve().parents[1] / "tasks" / "hopper-sensor-dropout"
spec = importlib.util.spec_from_file_location("hopper_verify", TASK / "verify.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)
generator_spec = importlib.util.spec_from_file_location("hopper_generate", TASK / "generate.py")
generator = importlib.util.module_from_spec(generator_spec)
generator_spec.loader.exec_module(generator)


class PatchTests(unittest.TestCase):
    def test_docker_native_offline_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator.generate(tmp, limit=1, offline_compose=1)
            task = Path(tmp) / "hopper-sensor-dropout-1"
            config = (task / "task.toml").read_text()
            self.assertIn('environment_mode = "separate"', config)
            self.assertNotIn('network_mode = "no-network"', config)
            self.assertFalse((task / "environment" / "verify.py").exists())
            for stage in ("environment", "tests"):
                self.assertIn("network_mode: none", (task / stage / "docker-compose.yaml").read_text())
            generator.generate(tmp, limit=1)
            self.assertEqual((task / "task.toml").read_text().count('network_mode = "no-network"'), 2)
            for stage in ("environment", "tests"):
                self.assertFalse((task / stage / "docker-compose.yaml").exists())

    def test_patch_validation(self):
        payload = {"weight": [[0.] * 64 for _ in range(64)], "bias": [0.] * 64}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patch.json"
            path.write_text(json.dumps(payload), encoding="ascii")
            w, b = verify.read_patch(path)
            self.assertEqual((w.shape, b.shape), ((64, 64), (64,)))
            for bad in (
                {**payload, "bias": [float("inf")] * 64},
                {**payload, "bias": [0.] * 63},
                {**payload, "weight": [["0"] * 64] * 64},
                {**payload, "executable": "echo hello"},
            ):
                path.write_text(json.dumps(bad), encoding="ascii")
                with self.assertRaises(ValueError):
                    verify.read_patch(path)
            for text in ('{"bias":0,"bias":0}', " " * 131_073):
                path.write_text(text, encoding="ascii")
                with self.assertRaises(ValueError):
                    verify.read_patch(path)
