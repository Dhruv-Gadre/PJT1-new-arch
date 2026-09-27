"""Test frequency-band features for Stage-B walking/running."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


DATA_DIR = Path("data/raw/stage_b")

LABELS = ["walking", "running"]

BANDS = [
    (0.5, 2.0),
    (0.5, 3.0),
    (0.5, 4.0),
    (0.5, 5.0),
    (1.0, 3.0),
    (1.0, 4.0),
    (1.0, 5.0),
]


def load_signal(path):
    data = np.load(path, allow_pickle=False)
    amplitudes = data["amplitudes"].astype(np.float32)

    # Same amplitude envelope used by the current feature pipeline.
    return amplitudes.mean(axis=1)


def spectral_features(signal, sample_rate, low, high):
    centered = signal - np.mean(signal)

    n_fft = 128

    fft_vals = np.abs(np.fft.rfft(centered, n=n_fft))
    freqs = np.fft.rfftfreq(
        n_fft,
        d=1.0 / sample_rate,
    )

    # Ignore DC.
    total_energy = float(np.sum(fft_vals[1:] ** 2))

    band_mask = (
        (freqs >= low)
        & (freqs <= high)
    )

    band_energy = float(
        np.sum(fft_vals[band_mask] ** 2)
    )

    if total_energy < 1e-12:
        ratio = 0.0
    else:
        ratio = band_energy / total_energy

    # Dominant frequency inside the selected band.
    if not np.any(band_mask):
        dominant = 0.0
    else:
        band_values = fft_vals[band_mask]
        band_freqs = freqs[band_mask]

        dominant = float(
            band_freqs[np.argmax(band_values)]
        )

    return dominant, ratio


def build_dataset(low, high):
    X = []
    y = []

    for label_idx, label in enumerate(LABELS):
        files = sorted(
            (DATA_DIR / label).glob("*.npz")
        )

        for path in files:
            signal = load_signal(path)

            dominant, ratio = spectral_features(
                signal,
                sample_rate=61.0,
                low=low,
                high=high,
            )

            X.append([dominant, ratio])
            y.append(label_idx)

    return np.asarray(X), np.asarray(y)


def main():
    print("=" * 75)
    print("STAGE-B FREQUENCY BAND DIAGNOSTIC")
    print("=" * 75)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    print()
    print(
        f"{'Band (Hz)':15s}"
        f"{'Accuracy':>12s}"
        f"{'Std':>12s}"
        f"{'Walking ratio':>18s}"
        f"{'Running ratio':>18s}"
    )

    print("-" * 75)

    for low, high in BANDS:

        X, y = build_dataset(low, high)

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

        walking_ratio = X[y == 0, 1].mean()
        running_ratio = X[y == 1, 1].mean()

        print(
            f"{low:.1f}-{high:.1f} Hz"
            f"{scores.mean():15.3f}"
            f"{scores.std():12.3f}"
            f"{walking_ratio:18.4f}"
            f"{running_ratio:18.4f}"
        )

    print()
    print("=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    main()