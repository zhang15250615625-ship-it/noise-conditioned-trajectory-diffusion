"""Small end-to-end checks, including a separate-process resume comparison."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from nctd.data import load_training_data
from nctd.model import make_model
from nctd.runtime import load_checkpoint, save_new
from nctd.train import initialize_from_parent

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_tree_equal(self, a, b):
        self.assertEqual(type(a), type(b))
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def command(self, directory, module, *arguments, success=True):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT/"src")
        env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run([sys.executable, "-m", module, *map(str, arguments)],
                                cwd=directory, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_training_resume_sampling_and_evaluation(self):
        with tempfile.TemporaryDirectory(prefix="nctd-test-") as tmp:
            tmp = Path(tmp)
            config = tmp/"config.json"
            config.write_text(json.dumps({"variant": "noise", "steps": 4, "batch": 2,
                "aux_batch": 1, "trajectory_weight": .1, "trajectory_span": 8,
                "trajectory_start_min": 8, "checkpoint_every": 2}))
            self.command(tmp, "nctd.train", "--config", config, "--synthetic", "--output", tmp/"full")
            self.command(tmp, "nctd.train", "--resume", tmp/"full/checkpoint_000002.pt",
                         "--steps", 4, "--output", tmp/"resumed")
            full = load_checkpoint(tmp/"full/checkpoint_000004.pt")
            resumed = load_checkpoint(tmp/"resumed/checkpoint_000004.pt")
            for key in ("model_state", "ema_state", "optimizer_state", "teacher_state", "rng",
                        "data_fingerprint", "ema_updates", "step"):
                self.assert_tree_equal(full[key], resumed[key])
            self.assertEqual(full["step"], 4)
            self.assertIsNotNone(full["teacher_state"])
            self.command(tmp, "nctd.sample", "--checkpoint", tmp/"resumed/checkpoint_000004.pt",
                         "--output", tmp/"samples", "--count", 2, "--steps", 4)
            samples = np.load(tmp/"samples/raw_samples.npy", allow_pickle=False)
            self.assertEqual(samples.shape, (2, 3, 32, 32))
            self.assertTrue(np.isfinite(samples).all())
            with Image.open(tmp/"samples/grid.png") as preview:
                self.assertEqual(preview.size, (64, 32))
            receipt = json.loads((tmp/"samples/sampling.json").read_text())
            self.assertTrue(receipt["not_formal_result"])
            np.save(tmp/"reference.npy", np.clip(samples, -1, 1))
            self.command(tmp, "nctd.evaluate", "--generated", tmp/"samples/raw_samples.npy",
                         "--reference", tmp/"reference.npy", "--output", tmp/"evaluation.json")
            metrics = json.loads((tmp/"evaluation.json").read_text())
            self.assertEqual(metrics["clamped"]["lowpass_swd"], 0.)
            self.assertIsNone(metrics["kid"])
            self.command(tmp, "nctd.train", "--synthetic", "--output", tmp/"full", success=False)
            changed = self.command(tmp, "nctd.train", "--resume", tmp/"full/checkpoint_000002.pt",
                                   "--batch", 1, "--steps", 4, "--output", tmp/"invalid", success=False)
            self.assertIn("Resume cannot change", changed.stderr)
            self.assertFalse((tmp/"invalid").exists())

    def test_parent_initialization_preserves_dense_weights_and_zero_delta(self):
        parent = make_model("dense", 21)
        target = make_model("noise", 20)
        initialize_from_parent(target, {"variant": "noise"},
                               {"config": {"variant": "dense"}, "ema_state": parent.state_dict()})
        state = target.backbone.state_dict()
        for key, value in parent.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]))
        for key, value in state.items():
            if key.endswith(("delta_weight", "delta_bias")):
                self.assertEqual(value.count_nonzero().item(), 0)

    def test_checksum_before_unpickling_and_no_clobber(self):
        with tempfile.TemporaryDirectory(prefix="nctd-test-") as tmp:
            tmp = Path(tmp)
            (tmp/"data_batch_1").write_bytes(b"not-a-verified-CIFAR-pickle")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_training_data(tmp)
            destination = tmp/"saved.json"
            save_new(destination, {"value": 1})
            with self.assertRaises(FileExistsError):
                save_new(destination, {"value": 2})
            self.assertEqual(json.loads(destination.read_text()), {"value": 1})

    def test_real_trajectory_rejects_missing_parent_before_data_access(self):
        with tempfile.TemporaryDirectory(prefix="nctd-test-") as tmp:
            tmp = Path(tmp)
            result = self.command(tmp, "nctd.train", "--config", ROOT/"configs/trajectory_span4.json",
                                  "--output", tmp/"invalid", success=False)
            self.assertIn("Real trajectory training requires", result.stderr)
            self.assertFalse((tmp/"invalid").exists())


if __name__ == "__main__":
    unittest.main()
