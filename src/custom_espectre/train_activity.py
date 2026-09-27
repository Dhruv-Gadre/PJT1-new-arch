"""Train the Stage-B walking/running classifier."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from .features import extract_activity_features

LABELS = ["walking", "running"]
LABEL_TO_IDX = {"walking": 0, "running": 1}


def load_dataset(data_dir: Path):
    X = []
    y = []

    for label in LABELS:
        label_dir = data_dir / label

        if not label_dir.exists():
            raise FileNotFoundError(f"Missing directory: {label_dir}")

        files = sorted(label_dir.glob("*.npz"))

        print(f"{label}: {len(files)} windows")

        for path in files:
            data = np.load(path, allow_pickle=False)

            amplitudes = data["amplitudes"].astype(np.float32)
            phases = data["phases"].astype(np.float32)

            features = extract_activity_features(amplitudes, phases)

            X.append(features)
            y.append(LABEL_TO_IDX[label])

    if not X:
        raise RuntimeError("No Stage-B data found.")

    return np.stack(X), np.array(y, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(
        description="Train Stage-B walking/running classifier"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/stage_b"),
    )

    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/activity_model.pkl"),
    )

    args = parser.parse_args()

    print("=" * 50)
    print("STAGE-B WALKING/RUNNING TRAINING")
    print("=" * 50)

    X, y = load_dataset(args.data_dir)

    print()
    print(f"Total samples : {len(y)}")
    print(f"Walking       : {np.sum(y == 0)}")
    print(f"Running       : {np.sum(y == 1)}")
    print(f"Features      : {X.shape[1]}")

    # Standardize features.
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # 5-fold stratified validation.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    predictions = cross_val_predict(
        clf,
        X_scaled,
        y,
        cv=cv,
    )

    accuracy = accuracy_score(y, predictions)

    print()
    print("=" * 50)
    print("5-FOLD VALIDATION")
    print("=" * 50)

    print(f"Accuracy: {accuracy:.3f}")

    print()
    print("Class-level metrics:")
    print(
        classification_report(
            y,
            predictions,
            target_names=LABELS,
            digits=3,
        )
    )

    print("Confusion matrix:")
    print()
    print("                 Predicted")
    print("              Walking Running")
    cm = confusion_matrix(y, predictions)

    print(f"Actual Walking   {cm[0, 0]:4d}   {cm[0, 1]:4d}")
    print(f"Actual Running   {cm[1, 0]:4d}   {cm[1, 1]:4d}")

    # Train final model on all available data.
    clf.fit(X_scaled, y)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.model_out, "wb") as f:
        pickle.dump(
            {
                "clf": clf,
                "scaler": scaler,
                "labels": LABELS,
            },
            f,
        )

    print()
    print("=" * 50)
    print("FINAL MODEL")
    print("=" * 50)
    print(f"Saved model -> {args.model_out}")
    print()
    print("Stage-B training complete.")


if __name__ == "__main__":
    main()