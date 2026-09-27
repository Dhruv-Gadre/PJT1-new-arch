"""Run live Stage-B walking/running prediction."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from .collect_activity import read_serial_windows
from .features import extract_activity_features


def build_parser():
    parser = argparse.ArgumentParser(
        description="Live Stage-B walking/running prediction"
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/activity_model.pkl"),
        help="Path to trained Stage-B model",
    )

    parser.add_argument(
        "--port",
        default="COM8",
        help="ESP32 serial port",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=921600,
        help="Serial baud rate",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of live windows to classify",
    )

    return parser


def main(argv=None):

    args = build_parser().parse_args(argv)

    print("=" * 55)
    print("STAGE-B LIVE WALKING/RUNNING PREDICTION")
    print("=" * 55)

    print(f"Model : {args.model}")
    print(f"Port  : {args.port}")
    print(f"Baud  : {args.baud}")
    print()

    # Load trained model.
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)

    clf = bundle["clf"]
    scaler = bundle["scaler"]
    labels = bundle["labels"]

    print("Model loaded.")
    print("Waiting for CSI windows...")
    print()

    for idx, (amplitudes, phases) in enumerate(
        read_serial_windows(
            args.port,
            args.baud,
            count=args.count,
        ),
        start=1,
    ):

        # IMPORTANT:
        # This must be the same feature extractor used during training.
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

        print(
            f"Window {idx:02d}: "
            f"{label.upper():8s} "
            f"(confidence={confidence:.3f})"
        )

    print()
    print("Prediction complete.")


if __name__ == "__main__":
    main()