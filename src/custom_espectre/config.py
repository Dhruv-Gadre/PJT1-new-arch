"""Global configuration for the custom ESPectre prototype."""

from __future__ import annotations

WINDOW_SIZE = 100
# Default shape for synthetic windows. Real collection uses each window's
# largest runtime CSI shape instead of forcing this value.
NUM_SUBCARRIERS = 64
SAMPLE_RATE_HZ = 61.0
LABELS = ["idle", "motion"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}

FEATURE_NAMES = [
    "dominant_freq_hz",
    "spectral_energy_ratio",
    "amp_variance",
    "rate_of_change",
    "zero_crossing_rate",
    "phase_spread",
    "max_slope",
]