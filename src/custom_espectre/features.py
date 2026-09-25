"""Feature extraction for CSI windows.

This file is intentionally small and modular. It contains the core signal
features used for motion/idle detection and can later be extended to
running/walking/sitting with the same window-based workflow.
"""

from __future__ import annotations

import numpy as np

from .config import NUM_SUBCARRIERS, SAMPLE_RATE_HZ, WINDOW_SIZE


def amplitude_envelope(amplitudes: np.ndarray) -> np.ndarray:
    """Return the per-packet mean amplitude across subcarriers."""
    return amplitudes.mean(axis=1)


def amp_variance(signal: np.ndarray) -> float:
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
    """Peak frequency in the human-motion band, or 0 when signal is flat."""
    if amp_variance(signal) < 0.01:
        return 0.0

    centered = signal - np.mean(signal)
    fft_vals = np.abs(np.fft.rfft(centered, n=128))
    freqs = np.fft.rfftfreq(128, d=1.0 / SAMPLE_RATE_HZ)
    mask = (freqs >= 0.5) & (freqs <= 10.0)
    if not np.any(mask):
        return 0.0

    peak_idx = np.argmax(fft_vals[mask])
    return float(freqs[mask][peak_idx])


def spectral_energy_ratio(signal: np.ndarray) -> float:
    """Energy in motion band divided by total spectral energy."""
    if amp_variance(signal) < 0.01:
        return 0.0

    centered = signal - np.mean(signal)
    fft_vals = np.abs(np.fft.rfft(centered, n=128))
    freqs = np.fft.rfftfreq(128, d=1.0 / SAMPLE_RATE_HZ)

    total_energy = float(np.sum(fft_vals[1:] ** 2))
    if total_energy < 1e-12:
        return 0.0

    motion_mask = (freqs >= 0.5) & (freqs <= 4.0)
    motion_energy = float(np.sum(fft_vals[motion_mask] ** 2))
    return float(motion_energy / total_energy)


def rate_of_change(signal: np.ndarray) -> float:
    mean_val = float(np.mean(signal))
    if abs(mean_val) < 1e-8:
        return 0.0
    diffs = np.abs(np.diff(signal))
    return float(np.mean(diffs) / mean_val)


def zero_crossing_rate(signal: np.ndarray) -> float:
    centered = signal - np.mean(signal)
    signs = np.signbit(centered)
    crossings = np.count_nonzero(np.diff(signs))
    return float(crossings / max(len(signal) - 1, 1))


def phase_spread(phases: np.ndarray) -> float:
    if phases.shape[1] < 2:
        return 0.0
    diffs = np.diff(phases, axis=1)
    return float(diffs.std(axis=1).mean())


def extract_features(amplitudes: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Return one 6-feature vector for a single CSI window."""
    amplitude_signal = amplitude_envelope(amplitudes)
    return np.array(
        [
            dominant_frequency(amplitude_signal),
            spectral_energy_ratio(amplitude_signal),
            amp_variance(amplitude_signal),
            rate_of_change(amplitude_signal),
            zero_crossing_rate(amplitude_signal),
            phase_spread(phases),
            max_slope(amplitude_signal),
        ],
        dtype=np.float32,
    )


def synth_motion_idle_window(label: str, noise: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple synthetic CSI-like window.

    This is only for smoke tests and should not be confused with real CSI data.
    It gives us a working training loop before hardware is connected.
    """
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, WINDOW_SIZE / SAMPLE_RATE_HZ, WINDOW_SIZE)

    if label == "motion":
        base = 0.5 + 0.4 * np.sin(2 * np.pi * 2.0 * t)
    else:
        base = 0.5 + 0.03 * np.sin(2 * np.pi * 0.2 * t)

    amplitudes = (base[:, None] + rng.normal(0.0, noise, (WINDOW_SIZE, NUM_SUBCARRIERS))).astype(np.float32)
    phases = rng.uniform(-np.pi, np.pi, (WINDOW_SIZE, NUM_SUBCARRIERS)).astype(np.float32)
    return amplitudes, phases
