import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from custom_espectre.data_io import load_dataset
from custom_espectre.features import extract_features
from custom_espectre.train_motion_idle import compute_scaler, apply_scaler


DATA_DIR = Path("data")

def main():
    print("=" * 45)
    print("PHASE FEATURE DIRECT COMPARISON")
    print("=" * 45)

    amplitudes, phases, y, paths = load_dataset(DATA_DIR)

    # Current pipeline:
    # [dominant_freq, spectral_ratio, amp_variance,
    #  rate_of_change, zero_crossing, phase_spread, max_slope]
    X7 = np.stack([
        extract_features(amplitudes[i], phases[i])
        for i in range(len(y))
    ])

    # Remove phase_spread.
    # Feature index 5 = phase_spread.
    X6 = np.delete(X7, 5, axis=1)

    print(f"Total samples : {len(y)}")
    print(f"Without phase : {X6.shape[1]}")
    print(f"With phase    : {X7.shape[1]}")
    print()
    print("Same folds are used for both models.")
    print()

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results_without = []
    results_with = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X7, y), start=1
    ):
        # -------------------------
        # Without phase_spread
        # -------------------------
        X_train = X6[train_idx]
        X_test = X6[test_idx]

        mean, scale = compute_scaler(X_train)

        X_train_z = apply_scaler(X_train, mean, scale)
        X_test_z = apply_scaler(X_test, mean, scale)

        clf = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        clf.fit(X_train_z, y[train_idx])
        pred = clf.predict(X_test_z)

        acc_without = accuracy_score(y[test_idx], pred)

        # -------------------------
        # With phase_spread
        # -------------------------
        X_train = X7[train_idx]
        X_test = X7[test_idx]

        mean, scale = compute_scaler(X_train)

        X_train_z = apply_scaler(X_train, mean, scale)
        X_test_z = apply_scaler(X_test, mean, scale)

        clf = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        clf.fit(X_train_z, y[train_idx])
        pred = clf.predict(X_test_z)

        acc_with = accuracy_score(y[test_idx], pred)

        results_without.append(acc_without)
        results_with.append(acc_with)

        print(f"Fold {fold}")
        print(f"  Without phase : {acc_without:.3f}")
        print(f"  With phase    : {acc_with:.3f}")
        print(f"  Improvement   : {acc_with - acc_without:+.3f}")
        print()

    results_without = np.array(results_without)
    results_with = np.array(results_with)

    print("=" * 45)
    print("FINAL COMPARISON")
    print("=" * 45)

    print(
        f"Without phase mean : "
        f"{results_without.mean():.3f}"
    )
    print(
        f"Without phase std  : "
        f"{results_without.std():.3f}"
    )

    print()

    print(
        f"With phase mean    : "
        f"{results_with.mean():.3f}"
    )
    print(
        f"With phase std     : "
        f"{results_with.std():.3f}"
    )

    print()

    print(
        f"Mean improvement   : "
        f"{results_with.mean() - results_without.mean():+.3f}"
    )

    print("=" * 45)


if __name__ == "__main__":
    main()