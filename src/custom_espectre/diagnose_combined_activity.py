"""Compare original, subcarrier, and combined Stage-B features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import extract_features


DATA_DIR = Path("data/raw/stage_b")

LABELS = ["walking", "running"]


SUBCARRIER_NAMES = [
    "mean_subcarrier_variance",
    "mean_subcarrier_std",
    "mean_max_slope",
    "mean_temporal_range",
    "subcarrier_mean_std",
    "subcarrier_mean_range",
]


def extract_subcarrier_features(amplitudes):
    variance = np.var(amplitudes, axis=0)
    std = np.std(amplitudes, axis=0)

    differences = np.abs(np.diff(amplitudes, axis=0))
    max_slope = np.max(differences, axis=0)

    temporal_range = (
        np.max(amplitudes, axis=0)
        - np.min(amplitudes, axis=0)
    )

    subcarrier_means = np.mean(amplitudes, axis=0)

    return np.array(
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


def load_dataset():
    original_features = []
    subcarrier_features = []
    labels = []

    for label_idx, label in enumerate(LABELS):

        files = sorted(
            (DATA_DIR / label).glob("*.npz")
        )

        for path in files:

            data = np.load(
                path,
                allow_pickle=False,
            )

            amplitudes = data["amplitudes"].astype(
                np.float32
            )

            phases = data["phases"].astype(
                np.float32
            )

            original = extract_features(
                amplitudes,
                phases,
            )

            subcarrier = extract_subcarrier_features(
                amplitudes
            )

            original_features.append(original)
            subcarrier_features.append(subcarrier)
            labels.append(label_idx)

    return (
        np.asarray(original_features),
        np.asarray(subcarrier_features),
        np.asarray(labels),
    )


def evaluate(name, X, y, cv):

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    print()
    print(name)

    for i, score in enumerate(scores, start=1):
        print(f"  Fold {i}: {score:.3f}")

    print(f"  Mean: {scores.mean():.3f}")
    print(f"  Std : {scores.std():.3f}")

    return scores


def main():

    print("=" * 75)
    print("STAGE-B COMBINED FEATURE EXPERIMENT")
    print("=" * 75)

    original, subcarrier, y = load_dataset()

    print()
    print(f"Total samples : {len(y)}")
    print(f"Walking       : {np.sum(y == 0)}")
    print(f"Running       : {np.sum(y == 1)}")

    print()
    print(f"Original features   : {original.shape[1]}")
    print(f"Subcarrier features: {subcarrier.shape[1]}")
    print(f"Combined features  : {original.shape[1] + subcarrier.shape[1]}")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # Experiment 1:
    # Existing seven features.
    evaluate(
        "A) CURRENT 7 FEATURES",
        original,
        y,
        cv,
    )

    # Experiment 2:
    # Six subcarrier features.
    evaluate(
        "B) SUBCARRIER FEATURES",
        subcarrier,
        y,
        cv,
    )

    # Experiment 3:
    # Seven original + six subcarrier features.
    combined = np.hstack(
        [
            original,
            subcarrier,
        ]
    )

    evaluate(
        "C) COMBINED FEATURES",
        combined,
        y,
        cv,
    )

    print()
    print("=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    main()