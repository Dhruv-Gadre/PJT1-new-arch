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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .config import FEATURE_NAMES, LABELS, LABEL_TO_IDX
from .data_io import load_dataset
from .features import extract_features, synth_motion_idle_window


STAGE_A_SOURCE_LABELS = ["idle", "motion"]
STAGE_B_SOURCE_LABELS = ["walking", "running"]


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


def load_stage_a_windows(data_dir: Path, include_stage_b: bool):
    source_labels = list(STAGE_A_SOURCE_LABELS)
    if include_stage_b:
        source_labels.extend(STAGE_B_SOURCE_LABELS)

    amplitudes, phases, labels, sources, paths = [], [], [], [], []
    for source_label in source_labels:
        source_dir = data_dir / source_label
        if source_label in STAGE_B_SOURCE_LABELS:
            source_dir = data_dir / "stage_b" / source_label
        if not source_dir.exists():
            raise FileNotFoundError(f"Missing directory: {source_dir}")

        target_label = "idle" if source_label == "idle" else "motion"
        for path in sorted(source_dir.glob("*.npz")):
            data = np.load(path, allow_pickle=False)
            amplitudes.append(data["amplitudes"].astype(np.float32))
            phases.append(data["phases"].astype(np.float32))
            labels.append(LABEL_TO_IDX[target_label])
            sources.append(source_label)
            paths.append(str(path))

        print(f"{source_label}: {len(list(source_dir.glob('*.npz')))} windows")

    if not amplitudes:
        raise ValueError(f"No labelled windows found under {data_dir}")

    return (
        np.stack(amplitudes, axis=0),
        np.stack(phases, axis=0),
        np.array(labels, dtype=np.int32),
        np.array(sources),
        paths,
    )


def build_feature_matrix_from_dir(
    data_dir: Path,
    include_stage_b: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows, phases, labels, sources, _ = load_stage_a_windows(
        data_dir,
        include_stage_b=include_stage_b,
    )
    X, y = [], []
    for amp, pha, label_idx in zip(windows, phases, labels):
        X.append(extract_features(amp, pha))
        y.append(label_idx)
    return np.stack(X, axis=0), np.array(y, dtype=np.int32), sources


def train_model(X: np.ndarray, y: np.ndarray, sources: np.ndarray):
    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    test_sources = sources[test_indices]
    mean, scale = compute_scaler(X_train)
    X_train_z = apply_scaler(X_train, mean, scale)
    X_test_z = apply_scaler(X_test, mean, scale)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_z, y_train)
    pred = clf.predict(X_test_z)

    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print(classification_report(y_test, pred, target_names=LABELS))

    print("Held-out source evaluation:")
    for source in STAGE_A_SOURCE_LABELS + STAGE_B_SOURCE_LABELS:
        mask = test_sources == source
        if not np.any(mask):
            continue
        expected_label = "idle" if source == "idle" else "motion"
        source_pred = pred[mask]
        source_expected = LABEL_TO_IDX[expected_label]
        source_accuracy = np.mean(source_pred == source_expected)
        print(
            f"  {source:8s}: {int(np.sum(mask)):3d} samples, "
            f"{source_accuracy:.3f} passed as {expected_label}"
        )

    print("Held-out confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, pred, labels=[0, 1]))

    return clf, mean, scale


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a motion/idle classifier")
    parser.add_argument("--synthetic", action="store_true", help="Train on synthetic CSI-like data")
    parser.add_argument("--n-per-class", type=int, default=200, help="Synthetic samples per class")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Directory containing labelled .npz windows")
    parser.add_argument(
        "--include-stage-b",
        action="store_true",
        help="Include stage_b walking/running windows as motion samples",
    )
    parser.add_argument("--model-out", type=Path, default=Path("models/motion_idle_model.pkl"), help="Output model path")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.synthetic:
        X, y = make_synthetic_dataset(n_per_class=args.n_per_class)
        sources = np.array([LABELS[label_idx] for label_idx in y])
    else:
        X, y, sources = build_feature_matrix_from_dir(
            args.data_dir,
            include_stage_b=args.include_stage_b,
        )

    clf, mean, scale = train_model(X, y, sources)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump(
            {
                "clf": clf,
                "mean": mean,
                "scale": scale,
                "labels": LABELS,
                "feature_names": FEATURE_NAMES,
                "training_sources": sorted(set(sources.tolist())),
                "include_stage_b": args.include_stage_b,
            },
            f,
        )

    print(f"Saved model -> {args.model_out}")


if __name__ == "__main__":
    main()
