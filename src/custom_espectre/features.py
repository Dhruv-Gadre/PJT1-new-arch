"""Feature extraction for CSI windows.

This module contains two feature pipelines:

Stage A:
    IDLE / MOTION
    Uses the original 7 features.

Stage B:
    WALKING / RUNNING
    Uses the original 7 features plus 6 subcarrier-level features.

Keeping the two extractors separate prevents changes to Stage B
from affecting the Stage-A model.
"""

from __future__ import annotations

import numpy as np

from .config import NUM_SUBCARRIERS, SAMPLE_RATE_HZ, WINDOW_SIZE


# ============================================================
# COMMON / STAGE-A FEATURES
# ============================================================


def amplitude_envelope(amplitudes: np.ndarray) -> np.ndarray:
    """Return the per-packet mean amplitude across subcarriers."""
    return amplitudes.mean(axis=1)


def amp_variance(signal: np.ndarray) -> float:
    """Return normalized amplitude variance."""
    mean_val = float(np.mean(signal))

    if abs(mean_val) < 1e-8:
        return 0.0

    return float(np.var(signal) / (mean_val ** 2))


def max_slope(signal: np.ndarray) -> float:
    """Maximum normalized packet-to-packet amplitude change."""
    mean_val = float(np.mean(signal))

    if abs(mean_val) < 1e-8:
        return 0.0

    slope = np.abs(np.diff(signal))

    return float(np.max(slope) / mean_val)


def dominant_frequency(signal: np.ndarray) -> float:
    """Peak frequency in the human-motion band."""
    if amp_variance(signal) < 0.01:
        return 0.0

    centered = signal - np.mean(signal)

    fft_vals = np.abs(
        np.fft.rfft(centered, n=128)
    )

    freqs = np.fft.rfftfreq(
        128,
        d=1.0 / SAMPLE_RATE_HZ,
    )

    mask = (
        (freqs >= 0.5)
        & (freqs <= 10.0)
    )

    if not np.any(mask):
        return 0.0

    peak_idx = np.argmax(
        fft_vals[mask]
    )

    return float(
        freqs[mask][peak_idx]
    )


def spectral_energy_ratio(signal: np.ndarray) -> float:
    """Energy in the motion band divided by total spectral energy."""

    if amp_variance(signal) < 0.01:
        return 0.0

    centered = signal - np.mean(signal)

    fft_vals = np.abs(
        np.fft.rfft(centered, n=128)
    )

    freqs = np.fft.rfftfreq(
        128,
        d=1.0 / SAMPLE_RATE_HZ,
    )

    total_energy = float(
        np.sum(fft_vals[1:] ** 2)
    )

    if total_energy < 1e-12:
        return 0.0

    motion_mask = (
        (freqs >= 0.5)
        & (freqs <= 4.0)
    )

    motion_energy = float(
        np.sum(
            fft_vals[motion_mask] ** 2
        )
    )

    return float(
        motion_energy / total_energy
    )


def rate_of_change(signal: np.ndarray) -> float:
    """Mean normalized packet-to-packet amplitude change."""

    mean_val = float(np.mean(signal))

    if abs(mean_val) < 1e-8:
        return 0.0

    diffs = np.abs(
        np.diff(signal)
    )

    return float(
        np.mean(diffs) / mean_val
    )


def zero_crossing_rate(signal: np.ndarray) -> float:
    """Rate at which the centered signal crosses zero."""

    centered = signal - np.mean(signal)

    signs = np.signbit(centered)

    crossings = np.count_nonzero(
        np.diff(signs)
    )

    return float(
        crossings
        / max(len(signal) - 1, 1)
    )


def phase_spread(phases: np.ndarray) -> float:
    """Measure phase variation across adjacent subcarriers."""

    if phases.shape[1] < 2:
        return 0.0

    diffs = np.diff(
        phases,
        axis=1,
    )

    return float(
        diffs.std(axis=1).mean()
    )


# ============================================================
# STAGE A
# ============================================================


def extract_features(
    amplitudes: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    """Return the 7-feature Stage-A IDLE/MOTION vector.

    Feature order:

        1. dominant_freq_hz
        2. spectral_energy_ratio
        3. amp_variance
        4. rate_of_change
        5. zero_crossing_rate
        6. phase_spread
        7. max_slope
    """

    amplitude_signal = amplitude_envelope(
        amplitudes
    )

    return np.array(
        [
            dominant_frequency(
                amplitude_signal
            ),
            spectral_energy_ratio(
                amplitude_signal
            ),
            amp_variance(
                amplitude_signal
            ),
            rate_of_change(
                amplitude_signal
            ),
            zero_crossing_rate(
                amplitude_signal
            ),
            phase_spread(
                phases
            ),
            max_slope(
                amplitude_signal
            ),
        ],
        dtype=np.float32,
    )


# ============================================================
# STAGE B SUBCARRIER FEATURES
# ============================================================


def mean_subcarrier_variance(
    amplitudes: np.ndarray,
) -> float:
    """Mean temporal variance across all subcarriers."""

    variance = np.var(
        amplitudes,
        axis=0,
    )

    return float(
        np.mean(variance)
    )


def mean_subcarrier_std(
    amplitudes: np.ndarray,
) -> float:
    """Mean temporal standard deviation across subcarriers."""

    std = np.std(
        amplitudes,
        axis=0,
    )

    return float(
        np.mean(std)
    )


def mean_subcarrier_max_slope(
    amplitudes: np.ndarray,
) -> float:
    """Mean maximum temporal slope across subcarriers."""

    differences = np.abs(
        np.diff(
            amplitudes,
            axis=0,
        )
    )

    max_slopes = np.max(
        differences,
        axis=0,
    )

    return float(
        np.mean(max_slopes)
    )


def mean_temporal_range(
    amplitudes: np.ndarray,
) -> float:
    """Mean temporal amplitude range across subcarriers."""

    temporal_range = (
        np.max(
            amplitudes,
            axis=0,
        )
        - np.min(
            amplitudes,
            axis=0,
        )
    )

    return float(
        np.mean(temporal_range)
    )


def subcarrier_mean_std(
    amplitudes: np.ndarray,
) -> float:
    """Standard deviation of the mean amplitude across subcarriers."""

    subcarrier_means = np.mean(
        amplitudes,
        axis=0,
    )

    return float(
        np.std(subcarrier_means)
    )


def subcarrier_mean_range(
    amplitudes: np.ndarray,
) -> float:
    """Range of mean amplitudes across subcarriers."""

    subcarrier_means = np.mean(
        amplitudes,
        axis=0,
    )

    return float(
        np.ptp(subcarrier_means)
    )


# ============================================================
# STAGE B
# ============================================================


def extract_activity_features(
    amplitudes: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    """Return the 13-feature Stage-B WALKING/RUNNING vector.

    The first 7 features are the same features used by Stage A.

    The final 6 features preserve subcarrier-level information.

    Feature order:

        1.  dominant_freq_hz
        2.  spectral_energy_ratio
        3.  amp_variance
        4.  rate_of_change
        5.  zero_crossing_rate
        6.  phase_spread
        7.  max_slope
        8.  mean_subcarrier_variance
        9.  mean_subcarrier_std
        10. mean_max_slope
        11. mean_temporal_range
        12. subcarrier_mean_std
        13. subcarrier_mean_range
    """

    # Keep the original seven features.
    original_features = extract_features(
        amplitudes,
        phases,
    )

    # Add the six subcarrier-level features.
    subcarrier_features = np.array(
        [
            mean_subcarrier_variance(
                amplitudes
            ),
            mean_subcarrier_std(
                amplitudes
            ),
            mean_subcarrier_max_slope(
                amplitudes
            ),
            mean_temporal_range(
                amplitudes
            ),
            subcarrier_mean_std(
                amplitudes
            ),
            subcarrier_mean_range(
                amplitudes
            ),
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            original_features,
            subcarrier_features,
        ]
    ).astype(np.float32)


# ============================================================
# SYNTHETIC DATA FOR STAGE-A SMOKE TESTS
# ============================================================


def synth_motion_idle_window(
    label: str,
    noise: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate simple synthetic CSI-like data.

    This is only for smoke tests and should not be confused
    with real CSI data.
    """

    rng = np.random.default_rng(42)

    t = np.linspace(
        0.0,
        WINDOW_SIZE / SAMPLE_RATE_HZ,
        WINDOW_SIZE,
    )

    if label == "motion":
        base = (
            0.5
            + 0.4
            * np.sin(
                2 * np.pi * 2.0 * t
            )
        )
    else:
        base = (
            0.5
            + 0.03
            * np.sin(
                2 * np.pi * 0.2 * t
            )
        )

    amplitudes = (
        base[:, None]
        + rng.normal(
            0.0,
            noise,
            (
                WINDOW_SIZE,
                NUM_SUBCARRIERS,
            ),
        )
    ).astype(np.float32)

    phases = rng.uniform(
        -np.pi,
        np.pi,
        (
            WINDOW_SIZE,
            NUM_SUBCARRIERS,
        ),
    ).astype(np.float32)

    return amplitudes, phases