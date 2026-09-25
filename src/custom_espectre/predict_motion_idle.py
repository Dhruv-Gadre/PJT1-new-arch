"""Run live motion/idle prediction on new CSI windows."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

from .collect import read_serial_windows
from .features import extract_features


def apply_scaler(X: np.ndarray, mean: np.ndarray, scale: np.ndarray):
    return (X - mean) / scale


def build_parser():
    parser = argparse.ArgumentParser(description="Predict motion vs idle from CSI windows")
    parser.add_argument("--model", type=Path, default=Path("models/motion_idle_model.pkl"), help="Saved model bundle")
    parser.add_argument("--port", type=str, default="COM8", help="Serial port for ESP32-S3")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate")
    parser.add_argument("--count", type=int, default=5, help="Windows to process")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)

    clf = bundle["clf"]
    mean = bundle["mean"]
    scale = bundle["scale"]
    labels = bundle["labels"]

    for idx, (amp, pha) in enumerate(read_serial_windows(args.port, args.baud, count=args.count), start=1):
        feat = extract_features(amp, pha).reshape(1, -1)
        feat_z = apply_scaler(feat, mean, scale)
        pred = clf.predict(feat_z)[0]
        conf = clf.predict_proba(feat_z)[0][pred]
        label = labels[pred]
        print(f"window {idx}: {label} (conf={conf:.3f})")


if __name__ == "__main__":
    main()
