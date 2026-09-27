"""Diagnose subcarrier-level features for Stage-B walking/running."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data/raw/stage_b")

LABELS = ["walking", "running"]


def load_window(path):
    data = np.load(path, allow_pickle=False)

    amplitudes = data["amplitudes"].astype(np.float32)

    return amplitudes


def extract_subcarrier_features(amplitudes):
    """
    Extract features that preserve information across subcarriers.

    The result contains:
      1. Mean temporal variance across subcarriers
      2. Mean temporal standard deviation
      3. Mean maximum slope
      4. Mean temporal range
      5. Spatial standard deviation of subcarrier means
      6. Spatial range of subcarrier means
    """

    # Temporal statistics for every subcarrier.
    variance = np.var(amplitudes, axis=0)
    std = np.std(amplitudes, axis=0)

    differences = np.abs(np.diff(amplitudes, axis=0))
    max_slope = np.max(differences, axis=0)

    temporal_range = (
        np.max(amplitudes, axis=0)
        - np.min(amplitudes, axis=0)
    )

    # Statistics across subcarriers.
    subcarrier_means = np.mean(amplitudes, axis=0)

    features = np.array(
        [
            np.mean(variance),
            np.mean(std),
            np.mean(max_slope),
            np.mean(temporal_range),
            np.std(subcarrier_means),
            np.ptp(subcarrier_means),
        ],
        dtype=np.float32,
    )

    return features


def load_dataset():
    X = []
    y = []

    for label_idx, label in enumerate(LABELS):

        files = sorted(
            (DATA_DIR / label).glob("*.npz")
        )

        for path in files:

            amplitudes = load_window(path)

            features = extract_subcarrier_features(
                amplitudes
            )

            X.append(features)
            y.append(label_idx)

    return np.asarray(X), np.asarray(y)


def main():

    print("=" * 75)
    print("STAGE-B SUBCARRIER FEATURE DIAGNOSTIC")
    print("=" * 75)

    X, y = load_dataset()

    print()
    print(f"Total samples : {len(y)}")
    print(f"Walking       : {np.sum(y == 0)}")
    print(f"Running       : {np.sum(y == 1)}")
    print(f"Features      : {X.shape[-1]}")

    feature_names = [
        "mean_subcarrier_variance",
        "mean_subcarrier_std",
        "mean_max_slope",
        "mean_temporal_range",
        "subcarrier_mean_std",
        "subcarrier_mean_range",
    ]

    print()
    print(
        f"{'Feature':30s}"
        f"{'Walking':>15s}"
        f"{'Running':>15s}"
        f"{'Effect':>12s}"
    )

    print("-" * 75)

    for i, name in enumerate(feature_names):

        walking = X[y == 0, i]
        running = X[y == 1, i]

        walking_mean = np.mean(walking)
        running_mean = np.mean(running)

        pooled_std = np.sqrt(
            (np.var(walking) + np.var(running)) / 2
        )

        if pooled_std < 1e-8:
            effect = 0.0
        else:
            effect = (
                abs(walking_mean - running_mean)
                / pooled_std
            )

        print(
            f"{name:30s}"
            f"{walking_mean:15.4f}"
            f"{running_mean:15.4f}"
            f"{effect:12.3f}"
        )

    # Evaluate all subcarrier features together.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    print()
    print("=" * 75)
    print("5-FOLD CLASSIFICATION")
    print("=" * 75)

    for i, score in enumerate(scores, start=1):
        print(f"Fold {i}: {score:.3f}")

    print()
    print(f"Mean accuracy: {scores.mean():.3f}")
    print(f"Std accuracy : {scores.std():.3f}")

    print()
    print("=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    main()