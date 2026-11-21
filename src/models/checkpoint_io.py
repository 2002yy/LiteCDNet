"""Checkpoint I/O contract shared by training resume and evaluation.

Only tensor/primitive state plus the historical NumPy scalar used for mF1
is supported. Never fall back to arbitrary pickle execution.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch


def load_checkpoint(path: str | Path, *, map_location):
    """Load a checkpoint with PyTorch's weights-only guard.

    Older LiteCDNet checkpoints may contain a NumPy scalar for best_val_acc.
    Allow only the three NumPy dtype helpers needed to reconstruct that scalar,
    rather than switching the entire file to weights_only=False.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as first_error:
        # PyTorch <2.6 may lack this API; don't silently deserialize arbitrary
        # classes when the compatibility guard is unavailable.
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is None:
            raise ValueError(
                "checkpoint cannot be loaded with weights_only=True; "
                "convert trusted legacy checkpoints in an isolated environment"
            ) from first_error
        np_scalar = np.core.multiarray.scalar if int(np.__version__.split(".")[0]) < 2 else np._core.multiarray.scalar
        with safe_globals([np_scalar, np.dtype, type(np.dtype(np.float64))]):
            return torch.load(path, map_location=map_location, weights_only=True)


def save_checkpoint(path: str | Path, payload: dict) -> None:
    """Keep the previous complete checkpoint when serialization fails.

    Stage bytes in the target directory, sync them, then atomically replace
    the canonical checkpoint. This is single-writer crash safety, not a lock
    for concurrent trainers sharing one checkpoint directory.
    """
    destination = Path(path)
    temporary_path = None
    fd, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    temporary_path = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
