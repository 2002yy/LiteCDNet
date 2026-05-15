from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

DATASET_PATHS = {
    "LEVIR": "LEVIR",
    "DSIFN": "DSIFN_256",
    "SYSU-CD": "SYSU-CD",
    "LEVIR+": "LEVIR-CD+_256",
    "BBCD": "Big_Building_ChangeDetection",
    "GZ_CD": "GZ",
    "WHU-CD": "WHU-CUT",
    "test": "att_test_whu",
    "quick_start": "samples",
}


def _normalize_dataset_name(data_name: str) -> str:
    normalized = []
    for char in data_name.upper():
        normalized.append(char if char.isalnum() else "_")
    return "".join(normalized)


def _default_dataset_root(data_name: str) -> Path:
    relative_path = DATASET_PATHS[data_name]
    if data_name == "quick_start":
        return REPO_ROOT / relative_path
    return DATA_ROOT / relative_path


class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"

    def get_data_config(self, data_name: str, root_dir: str | None = None):
        self.data_name = data_name
        self.label_transform = "norm"

        if data_name not in DATASET_PATHS:
            raise TypeError(f"{data_name} has not defined")

        env_name = f"LITECDNET_{_normalize_dataset_name(data_name)}_ROOT"
        resolved_root = (
            root_dir
            or os.getenv(env_name)
            or os.getenv("LITECDNET_DATA_ROOT")
            or str(_default_dataset_root(data_name))
        )
        self.root_dir = str(Path(resolved_root).expanduser())
        return self


if __name__ == "__main__":
    data = DataConfig().get_data_config(data_name="LEVIR")
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)
