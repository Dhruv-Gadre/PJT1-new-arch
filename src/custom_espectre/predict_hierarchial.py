"""Live hierarchical CSI activity prediction.

Stage A:
    IDLE / MOTION

Stage B:
    WALKING / RUNNING

Stage B is only executed when Stage A detects MOTION.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from .collect import read_serial_windows
from .features import (
    extract_features,
    extract_activity_features,
)


def apply_scaler(
    X: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Apply the Stage-A scaler."""
    return (X - mean) / scale


def build_parser():
    parser = argparse.ArgumentParser(
        description="Hierarchical IDLE/MOTION/WALKING/RUNNING predictor"
    )

    parser.add_argument(
        "--stage-a-model",
        type=Path,
        default=Path(
            "models/motion_idle_model.pkl"
        ),
    )

    parser.add_argument(
        "--stage-b-model",
        type=Path,
        default=Path(
            "models/activity_model.pkl"
        ),
    )

    parser.add_argument(
        "--port",
        default="COM8",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=921600,
    )

    parser.add_argument(
        "--count",
        type=int,
        default=20,
    )

    return parser


def load_model(path: Path):
    """Load a pickle model bundle."""

    with open(path, "rb") as f:
        return pickle.load(f)


def predict_stage_a(
    amplitudes,
    phases,
    bundle,
):
    """Run Stage-A IDLE/MOTION prediction."""

    clf = bundle["clf"]
    mean = bundle["mean"]
    scale = bundle["scale"]
    labels = bundle["labels"]

    features = extract_features(
        amplitudes,
        phases,
    )

    features = features.reshape(
        1,
        -1,
    )

    features_scaled = apply_scaler(
        features,
        mean,
        scale,
    )

    prediction = clf.predict(
        features_scaled
    )[0]

    probabilities = clf.predict_proba(
        features_scaled
    )[0]

    label = labels[prediction]

    confidence = probabilities[
        prediction
    ]

    return label, confidence


def predict_stage_b(
    amplitudes,
    phases,
    bundle,
):
    """Run Stage-B WALKING/RUNNING prediction."""

    clf = bundle["clf"]
    scaler = bundle["scaler"]
    labels = bundle["labels"]

    features = extract_activity_features(
        amplitudes,
        phases,
    )

    features = features.reshape(
        1,
        -1,
    )

    features_scaled = scaler.transform(
        features
    )

    prediction = clf.predict(
        features_scaled
    )[0]

    probabilities = clf.predict_proba(
        features_scaled
    )[0]

    label = labels[prediction]

    confidence = probabilities[
        prediction
    ]

    return label, confidence


def main(argv=None):

    args = build_parser().parse_args(argv)

    print("=" * 65)
    print("HIERARCHICAL CSI ACTIVITY PREDICTION")
    print("=" * 65)

    print(
        f"Stage-A model : {args.stage_a_model}"
    )

    print(
        f"Stage-B model : {args.stage_b_model}"
    )

    print(
        f"Serial port   : {args.port}"
    )

    print(
        f"Window count  : {args.count}"
    )

    print()

    # --------------------------------------------------------
    # Load both models.
    # --------------------------------------------------------

    stage_a = load_model(
        args.stage_a_model
    )

    stage_b = load_model(
        args.stage_b_model
    )

    print("Stage-A model loaded.")
    print("Stage-B model loaded.")
    print()

    print(
        "Waiting for CSI windows..."
    )

    print()

    # --------------------------------------------------------
    # Process live CSI windows.
    # --------------------------------------------------------

    for idx, (
        amplitudes,
        phases,
    ) in enumerate(
        read_serial_windows(
            args.port,
            args.baud,
            count=args.count,
        ),
        start=1,
    ):

        # ====================================================
        # STAGE A
        # ====================================================

        stage_a_label, stage_a_conf = (
            predict_stage_a(
                amplitudes,
                phases,
                stage_a,
            )
        )

        # ----------------------------------------------------
        # IDLE
        # ----------------------------------------------------

        if stage_a_label == "idle":

            print(
                f"Window {idx:02d}: "
                f"IDLE "
                f"(Stage-A conf={stage_a_conf:.3f})"
            )

            continue

        # ====================================================
        # STAGE B
        # ====================================================

        stage_b_label, stage_b_conf = (
            predict_stage_b(
                amplitudes,
                phases,
                stage_b,
            )
        )

        print(
            f"Window {idx:02d}: "
            f"{stage_b_label.upper():8s} "
            f"(Stage-A MOTION={stage_a_conf:.3f}, "
            f"Stage-B={stage_b_conf:.3f})"
        )

    print()
    print("Prediction complete.")


if __name__ == "__main__":
    main()