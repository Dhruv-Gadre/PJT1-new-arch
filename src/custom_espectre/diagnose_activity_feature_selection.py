"""Stage-B feature-selection experiment.

Compares different subsets of the 13 available features
for walking vs running classification.
"""

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


ORIGINAL_FEATURE_NAMES = [
    "dominant_freq_hz",
    "spectral_energy_ratio",
    "amp_variance",
    "rate_of_change",
    "zero_crossing_rate",
    "phase_spread",
    "max_slope",
]


SUBCARRIER_FEATURE_NAMES = [
    "mean_subcarrier_variance",
    "mean_subcarrier_std",
    "mean_max_slope",
    "mean_temporal_range",
    "subcarrier_mean_std",
    "subcarrier_mean_range",
]


ALL_FEATURE_NAMES = (
    ORIGINAL_FEATURE_NAMES
    + SUBCARRIER_FEATURE_NAMES
)


def extract_subcarrier_features(amplitudes):
    """Extract six subcarrier-level features."""

    variance = np.var(amplitudes, axis=0)

    std = np.std(amplitudes, axis=0)

    differences = np.abs(
        np.diff(amplitudes, axis=0)
    )

    max_slope = np.max(
        differences,
        axis=0,
    )

    temporal_range = (
        np.max(amplitudes, axis=0)
        - np.min(amplitudes, axis=0)
    )

    subcarrier_means = np.mean(
        amplitudes,
        axis=0,
    )

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
    """Load all walking/running windows."""

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

            amplitudes = data[
                "amplitudes"
            ].astype(np.float32)

            phases = data[
                "phases"
            ].astype(np.float32)

            original = extract_features(
                amplitudes,
                phases,
            )

            subcarrier = (
                extract_subcarrier_features(
                    amplitudes
                )
            )

            original_features.append(
                original
            )

            subcarrier_features.append(
                subcarrier
            )

            labels.append(label_idx)

    return (
        np.asarray(original_features),
        np.asarray(subcarrier_features),
        np.asarray(labels),
    )


def evaluate_feature_set(
    name,
    X,
    y,
    indices,
    cv,
):
    """Evaluate one feature subset."""

    X_selected = X[:, indices]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
    )

    scores = cross_val_score(
        model,
        X_selected,
        y,
        cv=cv,
        scoring="accuracy",
    )

    print()
    print(name)

    print(
        "  Features: "
        + ", ".join(
            ALL_FEATURE_NAMES[i]
            for i in indices
        )
    )

    for fold, score in enumerate(
        scores,
        start=1,
    ):
        print(
            f"  Fold {fold}: {score:.3f}"
        )

    print(
        f"  Mean: {scores.mean():.3f}"
    )

    print(
        f"  Std : {scores.std():.3f}"
    )

    return scores.mean()


def main():

    print("=" * 80)
    print("STAGE-B FEATURE SELECTION EXPERIMENT")
    print("=" * 80)

    original, subcarrier, y = (
        load_dataset()
    )

    X = np.hstack(
        [
            original,
            subcarrier,
        ]
    )

    print()
    print(f"Total samples : {len(y)}")
    print(
        f"Walking       : {np.sum(y == 0)}"
    )
    print(
        f"Running       : {np.sum(y == 1)}"
    )
    print(
        f"Total features: {X.shape[1]}"
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = {}

    # --------------------------------------------------
    # A. Current seven features
    # --------------------------------------------------

    results["Current 7"] = evaluate_feature_set(
        "A) CURRENT 7 FEATURES",
        X,
        y,
        list(range(7)),
        cv,
    )

    # --------------------------------------------------
    # B. Mean temporal range only
    # --------------------------------------------------

    results["Temporal range"] = evaluate_feature_set(
        "B) MEAN TEMPORAL RANGE ONLY",
        X,
        y,
        [10],
        cv,
    )

    # --------------------------------------------------
    # C. Mean max slope only
    # --------------------------------------------------

    results["Subcarrier max slope"] = evaluate_feature_set(
        "C) MEAN MAX SLOPE ONLY",
        X,
        y,
        [9],
        cv,
    )

    # --------------------------------------------------
    # D. Temporal range + max slope
    # --------------------------------------------------

    results["Range + slope"] = evaluate_feature_set(
        "D) TEMPORAL RANGE + MAX SLOPE",
        X,
        y,
        [9, 10],
        cv,
    )

    # --------------------------------------------------
    # E. All six subcarrier features
    # --------------------------------------------------

    results["Subcarrier 6"] = evaluate_feature_set(
        "E) ALL 6 SUBCARRIER FEATURES",
        X,
        y,
        list(range(7, 13)),
        cv,
    )

    # --------------------------------------------------
    # F. Original + two strongest subcarrier features
    # --------------------------------------------------

    results["Original + range + slope"] = (
        evaluate_feature_set(
            "F) ORIGINAL 7 + RANGE + SLOPE",
            X,
            y,
            list(range(7))
            + [9, 10],
            cv,
        )
    )

    # --------------------------------------------------
    # G. All 13 features
    # --------------------------------------------------

    results["All 13"] = evaluate_feature_set(
        "G) ALL 13 FEATURES",
        X,
        y,
        list(range(13)),
        cv,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    sorted_results = sorted(
        results.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for name, score in sorted_results:
        print(
            f"{name:35s} "
            f"{score:.3f}"
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()