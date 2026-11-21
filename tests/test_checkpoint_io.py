"""CPU-only safety and crash-consistency checks for checkpoint I/O."""
from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import checkpoint_io  # noqa: E402


class UnsupportedLegacyObject:
    """Not part of the explicit tensor/NumPy-scalar checkpoint contract."""
    pass


class CheckpointIOTests(unittest.TestCase):
    def test_new_checkpoint_uses_primitive_metadata_and_weights_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "last_ckpt.pt"
            payload = {
                "epoch_id": 12,
                "best_epoch_id": 11,
                "best_val_acc": float(np.float64(0.913)),
                "model_G_state_dict": {"weight": torch.ones(3)},
                "optimizer_G_state_dict": {},
                "exp_lr_scheduler_G_state_dict": {},
            }
            checkpoint_io.save_checkpoint(path, payload)
            result = checkpoint_io.load_checkpoint(path, map_location="cpu")
            self.assertEqual(type(result["best_val_acc"]), float)
            self.assertEqual(result["epoch_id"], 12)
            self.assertTrue(torch.equal(result["model_G_state_dict"]["weight"], torch.ones(3)))
            self.assertEqual([x.name for x in Path(folder).iterdir()], ["last_ckpt.pt"])

    def test_legacy_numpy_float64_metric_has_narrow_safe_compatibility(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.pt"
            torch.save(
                {"best_val_acc": np.float64(0.807), "model_G_state_dict": {"w": torch.ones(1)}},
                path,
            )
            loaded = checkpoint_io.load_checkpoint(path, map_location="cpu")
            self.assertAlmostEqual(float(loaded["best_val_acc"]), 0.807)
            self.assertTrue(torch.equal(loaded["model_G_state_dict"]["w"], torch.ones(1)))

    def test_unexpected_pickled_class_never_triggers_unsafe_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "untrusted.pt"
            torch.save({"arbitrary_object": UnsupportedLegacyObject()}, path)
            with self.assertRaises(pickle.UnpicklingError):
                checkpoint_io.load_checkpoint(path, map_location="cpu")

    def test_replace_failure_preserves_previous_checkpoint_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best_ckpt.pt"
            checkpoint_io.save_checkpoint(path, {"epoch_id": 1})
            previous = path.read_bytes()
            with patch.object(checkpoint_io.os, "replace", side_effect=OSError("disk rename failed")):
                with self.assertRaises(OSError):
                    checkpoint_io.save_checkpoint(path, {"epoch_id": 2})
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_serialization_failure_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best_ckpt.pt"
            checkpoint_io.save_checkpoint(path, {"epoch_id": 1})
            previous = path.read_bytes()

            def fail_after_partial_write(_obj, stream):
                stream.write(b"partial")
                raise OSError("serialization interrupted")

            with patch.object(checkpoint_io.torch, "save", side_effect=fail_after_partial_write):
                with self.assertRaises(OSError):
                    checkpoint_io.save_checkpoint(path, {"epoch_id": 2})
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(folder).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
