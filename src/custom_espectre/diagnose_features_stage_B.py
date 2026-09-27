"""Diagnose Stage-B walking/running features."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import extract_features


LABELS = ["walking", "running"]

FEATURE_NAMES = [
    "dominant_freq_hz",
    "spectral_energy_ratio",
    "amp_variance",
    "rate_of_change",
    "zero_crossing_rate",
    "phase_spread",
    "max_slope",
]


def load_features(data_dir: Path):
    all_features = {}
    all_features["walking"] = []
    all_features["running"] = []

    for label in LABELS:
        files = sorted((data_dir / label).glob("*.npz"))

        for path in files:
            data = np.load(path, allow_pickle=False)

            amplitudes = data["amplitudes"].astype(np.float32)
            phases = data["phases"].astype(np.float32)

            features = extract_features(amplitudes, phases)

            all_features[label].append(features)

        all_features[label] = np.stack(all_features[label])

    return all_features


def main():
    data_dir = Path("data/raw/stage_b")

    data = load_features(data_dir)

    walking = data["walking"]
    running = data["running"]

    print("=" * 70)
    print("STAGE-B FEATURE DIAGNOSTIC")
    print("=" * 70)

    print(f"Walking samples : {len(walking)}")
    print(f"Running samples : {len(running)}")

    print()
    print(f"{'Feature':25s} {'Walking mean':>15s} {'Running mean':>15s} {'Effect':>12s}")
    print("-" * 70)

    for i, name in enumerate(FEATURE_NAMES):
        w = walking[:, i]
        r = running[:, i]

        w_mean = np.mean(w)
        r_mean = np.mean(r)

        pooled_std = np.sqrt(
            (np.var(w) + np.var(r)) / 2
        )

        if pooled_std < 1e-8:
            effect = 0.0
        else:
            effect = abs(w_mean - r_mean) / pooled_std

        print(
            f"{name:25s} "
            f"{w_mean:15.4f} "
            f"{r_mean:15.4f} "
            f"{effect:12.3f}"
        )

    print()
    print("=" * 70)
    print("FEATURE RANGES")
    print("=" * 70)

    for i, name in enumerate(FEATURE_NAMES):
        w = walking[:, i]
        r = running[:, i]

        print()
        print(name)

        print(
            f"  Walking: "
            f"min={np.min(w):.4f}, "
            f"median={np.median(w):.4f}, "
            f"max={np.max(w):.4f}"
        )

        print(
            f"  Running: "
            f"min={np.min(r):.4f}, "
            f"median={np.median(r):.4f}, "
            f"max={np.max(r):.4f}"
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()