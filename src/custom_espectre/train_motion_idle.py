"""Train a motion/idle classifier from CSI windows.

This is the first milestone in the project and is intentionally small.
Later, the same structure can be extended for running/walking/sitting.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from .config import FEATURE_NAMES, LABELS, LABEL_TO_IDX
from .data_io import load_dataset
from .features import extract_features, synth_motion_idle_window


def make_synthetic_dataset(n_per_class: int = 200):
    X, y = [], []
    for label in LABELS:
        for _ in range(n_per_class):
            amp, pha = synth_motion_idle_window(label)
            X.append(extract_features(amp, pha))
            y.append(LABEL_TO_IDX[label])
    return np.stack(X, axis=0), np.array(y, dtype=np.int32)


def compute_scaler(X: np.ndarray):
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def apply_scaler(X: np.ndarray, mean: np.ndarray, scale: np.ndarray):
    return (X - mean) / scale


def build_feature_matrix_from_dir(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    windows, phases, labels, _ = load_dataset(data_dir)
    X, y = [], []
    for amp, pha, label_idx in zip(windows, phases, labels):
        X.append(extract_features(amp, pha))
        y.append(label_idx)
    return np.stack(X, axis=0), np.array(y, dtype=np.int32)


def train_model(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    mean, scale = compute_scaler(X_train)
    X_train_z = apply_scaler(X_train, mean, scale)
    X_test_z = apply_scaler(X_test, mean, scale)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_z, y_train)
    pred = clf.predict(X_test_z)

    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print(classification_report(y_test, pred, target_names=LABELS))

    return clf, mean, scale


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a motion/idle classifier")
    parser.add_argument("--synthetic", action="store_true", help="Train on synthetic CSI-like data")
    parser.add_argument("--n-per-class", type=int, default=200, help="Synthetic samples per class")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Directory containing labelled .npz windows")
    parser.add_argument("--model-out", type=Path, default=Path("models/motion_idle_model.pkl"), help="Output model path")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.synthetic:
        X, y = make_synthetic_dataset(n_per_class=args.n_per_class)
    else:
        X, y = build_feature_matrix_from_dir(args.data_dir)

    clf, mean, scale = train_model(X, y)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump({"clf": clf, "mean": mean, "scale": scale, "labels": LABELS, "feature_names": FEATURE_NAMES}, f)

    print(f"Saved model -> {args.model_out}")


if __name__ == "__main__":
    main()
