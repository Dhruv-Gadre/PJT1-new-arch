"""Live motion/idle prediction using the mathematics-based Stage-A detector.

No trained model is needed.  If a calibration JSON produced by
calibrate_math.py is available, it is used to normalise scores
against the quiet-room baseline.  If not, uncalibrated defaults
are used (still useful for a quick sanity check).

Usage::

    # Without calibration (defaults):
    python -m custom_espectre.predict_math --count 20

    # With calibration:
    python -m custom_espectre.predict_math \\
        --baseline models/math_baseline.json \\
        --count 20 --port COM8

    # Math detector in parallel with the ML model (compare outputs):
    python -m custom_espectre.predict_math \\
        --baseline models/math_baseline.json \\
        --ml-model models/motion_idle_model.pkl \\
        --count 20
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from .collect import read_serial_windows
from .features import extract_features
from .math_detector import MathDetector


def apply_scaler(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (X - mean) / scale


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live motion/idle prediction — mathematics-based Stage-A detector"
    )
    parser.add_argument(
        "--baseline",
        "--calibration",
        dest="baseline",
        type=Path,
        default=None,
        help="Path to calibration JSON from calibrate_math.py (optional)",
    )
    parser.add_argument(
        "--ml-model",
        type=Path,
        default=None,
        help="Optional: also run the LogisticRegression Stage-A model for comparison",
    )
    parser.add_argument("--port", type=str, default="COM8")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--count", type=int, default=20)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    print("=" * 65)
    print("MATH STAGE-A LIVE PREDICTION")
    print("=" * 65)

    # ----------------------------------------------------------
    # Load math detector
    # ----------------------------------------------------------
    if args.baseline is not None:
        detector = MathDetector.load(args.baseline)
        print(f"Baseline loaded  : {args.baseline}")
        print(f"MVS threshold    : {detector.mvs_threshold:.6f}")
        print(f"NBVI threshold   : {detector.nbvi_threshold:.4f}")
    else:
        detector = MathDetector()
        print("Baseline         : none (using defaults — run calibrate_math.py first)")

    # ----------------------------------------------------------
    # Optionally load the ML model for side-by-side comparison
    # ----------------------------------------------------------
    ml_bundle = None
    if args.ml_model is not None:
        with open(args.ml_model, "rb") as fh:
            ml_bundle = pickle.load(fh)
        print(f"ML model loaded  : {args.ml_model}")

    print(f"Serial port      : {args.port}")
    print(f"Window count     : {args.count}")
    print()
    print("Waiting for CSI windows...")
    print()

    # Header
    if ml_bundle is not None:
        print(f"{'Win':>3s}  {'Math':8s}  {'MVS':10s}  {'NBVI':8s}  {'ML':8s}  {'ML-conf':7s}")
        print("-" * 60)
    else:
        print(f"{'Win':>3s}  {'Label':8s}  {'MVS':10s}  {'NBVI':8s}")
        print("-" * 45)

    for idx, (amp, pha) in enumerate(
        read_serial_windows(args.port, args.baud, count=args.count), start=1
    ):
        # Math detection
        math_label, scores = detector.predict(amp, pha)
        mvs = scores["mvs"]
        nbvi = scores["nbvi"]

        if ml_bundle is not None:
            # ML detection
            feat = extract_features(amp, pha).reshape(1, -1)
            feat_z = apply_scaler(feat, ml_bundle["mean"], ml_bundle["scale"])
            ml_pred = ml_bundle["clf"].predict(feat_z)[0]
            ml_conf = ml_bundle["clf"].predict_proba(feat_z)[0][ml_pred]
            ml_label = ml_bundle["labels"][ml_pred]
            print(
                f"{idx:3d}  {math_label:8s}  {mvs:10.6f}  {nbvi:8.4f}"
                f"  {ml_label:8s}  {ml_conf:7.3f}"
            )
        else:
            print(f"{idx:3d}  {math_label:8s}  {mvs:10.6f}  {nbvi:8.4f}")

    print()
    print("Prediction complete.")


if __name__ == "__main__":
    main()
