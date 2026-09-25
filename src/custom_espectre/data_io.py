"""Data loading helpers for the custom prototype."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import LABEL_TO_IDX


def load_window(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    label = str(data["label"])
    return {
        "amplitudes": data["amplitudes"].astype(np.float32),
        "phases": data["phases"].astype(np.float32),
        "label": label,
        "label_idx": LABEL_TO_IDX.get(label, -1),
    }


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    npz_files = sorted(data_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {data_dir}")

    amplitudes, phases, labels, paths = [], [], [], []
    for p in npz_files:
        sample = load_window(p)
        if sample["label_idx"] < 0:
            continue
        amplitudes.append(sample["amplitudes"])
        phases.append(sample["phases"])
        labels.append(sample["label_idx"])
        paths.append(str(p))

    if len(amplitudes) == 0:
        raise ValueError(f"No valid labelled windows found under {data_dir}")

    return (
        np.stack(amplitudes, axis=0),
        np.stack(phases, axis=0),
        np.array(labels, dtype=np.int32),
        paths,
    )
