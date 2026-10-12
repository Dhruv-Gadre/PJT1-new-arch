"""Calibrate the mathematics-based Stage-A detector from idle windows.

This script reads previously collected idle .npz windows, computes
a quiet-room baseline with MathDetector.calibrate(), and saves the
resulting thresholds as a small JSON file (no large pickle or model
weights — just numbers).

Usage::

    python -m custom_espectre.calibrate_math \\
        --data-dir data/raw/idle \\
        --out models/math_baseline.json

After calibration, run predict_math.py to use it live.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .math_detector import MathDetector


def load_idle_windows(data_dir: Path) -> list[np.ndarray]:
    """Load all .npz amplitude arrays from *data_dir*."""
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {data_dir}")

    windows = []
    for path in files:
        data = np.load(path, allow_pickle=False)
        windows.append(data["amplitudes"].astype(np.float32))

    return windows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate the math Stage-A detector from idle CSI windows"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/idle"),
        help="Directory containing idle .npz windows",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/math_baseline.json"),
        help="Output path for the calibration JSON",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    print("=" * 60)
    print("MATH DETECTOR CALIBRATION")
    print("=" * 60)
    print(f"Loading idle windows from: {args.data_dir}")

    windows = load_idle_windows(args.data_dir)
    print(f"Loaded {len(windows)} idle windows.")

    detector = MathDetector()
    detector.calibrate(windows)

    detector.save(args.out)

    print()
    print("Calibration complete.")
    print(f"  MVS threshold : {detector.mvs_threshold:.6f}")
    print(f"  NBVI threshold: {detector.nbvi_threshold:.4f}")
    print(f"  Saved       -> {args.out}")
    print()
    print("Now run predict_math.py --baseline models/math_baseline.json")


if __name__ == "__main__":
    main()
