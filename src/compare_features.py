import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from custom_espectre.config import LABELS
from custom_espectre.data_io import load_dataset
from custom_espectre.features import extract_features
from custom_espectre.train_motion_idle import compute_scaler, apply_scaler


DATA_DIR = Path("data")


def main():
    print("=" * 40)
    print("6 vs 7 FEATURE DIRECT COMPARISON")
    print("=" * 40)

    amplitudes, phases, y, paths = load_dataset(DATA_DIR)

    # Build the original 6-feature representation.
    X7 = np.stack([
        extract_features(amplitudes[i], phases[i])
        for i in range(len(y))
    ])

    # Build the 7-feature representation.
    X6 = X7[:, :-1]  # Remove the last feature (max_slope) to get 6 features.

    print(f"Total samples : {len(y)}")
    print(f"6 features    : {X6.shape[1]}")
    print(f"7 features    : {X7.shape[1]}")

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results6 = []
    results7 = []

    print()
    print("Same folds are used for both models.")
    print()

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X6, y), start=1
    ):
        # -------------------------
        # 6-feature model
        # -------------------------
        X6_train = X6[train_idx]
        X6_test = X6[test_idx]

        mean6, scale6 = compute_scaler(X6_train)

        X6_train_z = apply_scaler(X6_train, mean6, scale6)
        X6_test_z = apply_scaler(X6_test, mean6, scale6)

        clf6 = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        clf6.fit(X6_train_z, y[train_idx])
        pred6 = clf6.predict(X6_test_z)

        acc6 = accuracy_score(y[test_idx], pred6)

        # -------------------------
        # 7-feature model
        # -------------------------
        X7_train = X7[train_idx]
        X7_test = X7[test_idx]

        mean7, scale7 = compute_scaler(X7_train)

        X7_train_z = apply_scaler(X7_train, mean7, scale7)
        X7_test_z = apply_scaler(X7_test, mean7, scale7)

        clf7 = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        clf7.fit(X7_train_z, y[train_idx])
        pred7 = clf7.predict(X7_test_z)

        acc7 = accuracy_score(y[test_idx], pred7)

        results6.append(acc6)
        results7.append(acc7)

        print(f"Fold {fold}")
        print(f"  6 features       : {acc6:.3f}")
        print(f"  7 features       : {acc7:.3f}")
        print(f"  Improvement      : {acc7 - acc6:+.3f}")
        print()

    results6 = np.array(results6)
    results7 = np.array(results7)

    print("=" * 40)
    print("FINAL COMPARISON")
    print("=" * 40)

    print(
        f"6 features mean : "
        f"{results6.mean():.3f}"
    )

    print(
        f"6 features std  : "
        f"{results6.std():.3f}"
    )

    print()

    print(
        f"7 features mean : "
        f"{results7.mean():.3f}"
    )

    print(
        f"7 features std  : "
        f"{results7.std():.3f}"
    )

    print()

    print(
        f"Mean improvement: "
        f"{results7.mean() - results6.mean():+.3f}"
    )

    print("=" * 40)


if __name__ == "__main__":
    main()