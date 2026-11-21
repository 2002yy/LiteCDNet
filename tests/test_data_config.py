"""Pure path-resolution contracts; no torch, GPU, or dataset download required."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_config import DATA_ROOT, REPO_ROOT, DataConfig


class DataConfigTest(unittest.TestCase):
    def test_defaults_keep_dataset_specific_layout(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(DataConfig().get_data_config("LEVIR").root_dir, str(DATA_ROOT / "LEVIR"))
            self.assertEqual(DataConfig().get_data_config("quick_start").root_dir, str(REPO_ROOT / "samples"))

    def test_global_data_root_is_parent_of_named_datasets(self):
        with tempfile.TemporaryDirectory() as base:
            with patch.dict(os.environ, {"LITECDNET_DATA_ROOT": base}, clear=True):
                for name, folder in (("LEVIR", "LEVIR"), ("DSIFN", "DSIFN_256"),
                                     ("LEVIR+", "LEVIR-CD+_256"), ("WHU-CD", "WHU-CUT")):
                    with self.subTest(name=name):
                        self.assertEqual(DataConfig().get_data_config(name).root_dir, str(Path(base) / folder))

    def test_specific_root_precedence_does_not_append_dataset_name(self):
        with tempfile.TemporaryDirectory() as base:
            specific = str(Path(base) / "direct" / "LEVIR")
            with patch.dict(os.environ, {"LITECDNET_DATA_ROOT": base, "LITECDNET_LEVIR_ROOT": specific}, clear=True):
                self.assertEqual(DataConfig().get_data_config("LEVIR").root_dir, specific)
                self.assertEqual(DataConfig().get_data_config("LEVIR", root_dir=base).root_dir, base)

    def test_unknown_dataset_remains_rejected(self):
        with self.assertRaises(TypeError):
            DataConfig().get_data_config("unknown")


if __name__ == "__main__":
    unittest.main()
