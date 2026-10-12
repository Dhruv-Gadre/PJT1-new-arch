"""Mathematics-based Stage-A motion detector.

This replaces the LogisticRegression Stage-A model with a purely
signal-processing approach, inspired by the ESPectre project
(https://github.com/francescopace/espectre).

Two complementary metrics are used:

    MVS – Moving Variance Score
        Captures how much the mean amplitude envelope fluctuates
        over time.  A quiet room has very low variance; a person
        moving causes the envelope to swing, raising MVS sharply.

    NBVI – Narrowband Variance Indicator
        Compares the temporal variance of each subcarrier against a
        quiet-room baseline.  When the baseline is available it gives
        a normalised score; without a baseline it falls back to the
        raw mean per-subcarrier variance.

A window is classified as MOTION when either MVS or NBVI exceeds
its respective threshold.  The thresholds have sensible defaults
but can be calibrated from a set of idle windows using
`MathDetector.calibrate()`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ============================================================
# DEFAULT THRESHOLDS
# These are starting points.  Run calibrate_math.py on real
# idle data to get room-specific values.
# ============================================================

DEFAULT_MVS_THRESHOLD = 0.01
DEFAULT_NBVI_THRESHOLD = 2.5   # ratio vs baseline
WINDOW_SIZE_MS = 50            # sliding sub-window in packets


# ============================================================
# CORE METRICS
# ============================================================


def moving_variance_score(signal: np.ndarray, win: int = WINDOW_SIZE_MS) -> float:
    """Return the mean variance of sliding sub-windows of *signal*.

    Each sub-window of length *win* is computed with stride 1.
    The mean over all sub-windows gives a single scalar that is
    large when the signal is actively changing and tiny when the
    room is quiet.

    Args:
        signal: 1-D amplitude envelope, shape (T,).
        win: sub-window length in samples.

    Returns:
        Scalar moving-variance score (always >= 0).
    """
    if len(signal) < win:
        return float(np.var(signal))

    variances = []
    for start in range(0, len(signal) - win + 1):
        variances.append(float(np.var(signal[start : start + win])))

    return float(np.mean(variances))


def nbvi_score(
    amplitudes: np.ndarray,
    quiet_variances: np.ndarray | None = None,
) -> float:
    """Narrowband Variance Indicator.

    Computes the temporal variance of each subcarrier and compares
    it against a quiet-room reference (``quiet_variances``).  When
    no reference is available the raw mean subcarrier variance is
    returned instead.

    Args:
        amplitudes: 2-D array, shape (T, subcarriers).
        quiet_variances: Optional 1-D array of per-subcarrier
            variances measured during calibration (idle room).

    Returns:
        Scalar NBVI score.  Higher means more change relative to
        quiet baseline.
    """
    # Temporal variance per subcarrier.
    live_var = np.var(amplitudes, axis=0)  # shape (subcarriers,)

    if quiet_variances is None:
        return float(np.mean(live_var))

    # Avoid division by zero for extremely quiet subcarriers.
    denom = np.where(quiet_variances < 1e-8, 1e-8, quiet_variances)
    ratios = live_var / denom
    return float(np.mean(ratios))


# ============================================================
# STATELESS SINGLE-WINDOW HELPER
# ============================================================


def detect_motion(
    amplitudes: np.ndarray,
    phases: np.ndarray,
    mvs_threshold: float = DEFAULT_MVS_THRESHOLD,
    nbvi_threshold: float = DEFAULT_NBVI_THRESHOLD,
    quiet_variances: np.ndarray | None = None,
) -> tuple[str, dict]:
    """Classify a single CSI window using only mathematics.

    No trained model is required.  Pass quiet_variances from a
    prior calibration to enable the NBVI baseline comparison.

    Args:
        amplitudes: shape (T, subcarriers).
        phases:     shape (T, subcarriers). Not currently used for
                    the score, reserved for future phase metrics.
        mvs_threshold: Windows with MVS above this are MOTION.
        nbvi_threshold: Windows with NBVI above this are MOTION.
        quiet_variances: Optional per-subcarrier idle baseline.

    Returns:
        (label, scores) where label is "motion" or "idle" and scores
        is a dict with "mvs" and "nbvi" for diagnostics.
    """
    # Mean amplitude envelope across subcarriers.
    envelope = amplitudes.mean(axis=1)

    mvs = moving_variance_score(envelope)
    nbvi = nbvi_score(amplitudes, quiet_variances)

    is_motion = (mvs > mvs_threshold) or (
        quiet_variances is not None and nbvi > nbvi_threshold
    )

    label = "motion" if is_motion else "idle"
    scores = {"mvs": mvs, "nbvi": nbvi}
    return label, scores


# ============================================================
# STATEFUL DETECTOR WITH CALIBRATION
# ============================================================


class MathDetector:
    """Stateful Stage-A detector that uses a calibrated idle baseline.

    Workflow::

        detector = MathDetector()
        detector.calibrate(idle_amplitudes_list)   # list of (T, C) arrays
        label, scores = detector.predict(amp, pha)

    You can also save and load the calibration as a JSON file so
    you do not need to re-run calibration every session.
    """

    def __init__(
        self,
        mvs_threshold: float = DEFAULT_MVS_THRESHOLD,
        nbvi_threshold: float = DEFAULT_NBVI_THRESHOLD,
    ) -> None:
        self.mvs_threshold = mvs_threshold
        self.nbvi_threshold = nbvi_threshold
        self.quiet_variances: np.ndarray | None = None
        self._calibrated = False

    # ----------------------------------------------------------
    # CALIBRATION
    # ----------------------------------------------------------

    def calibrate(self, idle_windows: list[np.ndarray]) -> None:
        """Fit quiet baseline from a list of idle amplitude windows.

        Also auto-tunes MVS threshold to mean_idle_mvs * 3 so that
        the detector is room-specific rather than relying on a fixed
        default.

        Args:
            idle_windows: list of (T, subcarriers) amplitude arrays
                measured while the room is empty.
        """
        if not idle_windows:
            raise ValueError("Need at least one idle window to calibrate.")

        # Per-subcarrier variance averaged across idle windows.
        all_var = np.stack(
            [np.var(w, axis=0) for w in idle_windows], axis=0
        )
        self.quiet_variances = all_var.mean(axis=0)

        # Auto-tune MVS threshold at 3× the idle level.
        envelopes = [w.mean(axis=1) for w in idle_windows]
        idle_mvs = np.mean([moving_variance_score(e) for e in envelopes])
        self.mvs_threshold = max(idle_mvs * 3.0, DEFAULT_MVS_THRESHOLD)

        self._calibrated = True

    # ----------------------------------------------------------
    # PREDICTION
    # ----------------------------------------------------------

    def predict(
        self,
        amplitudes: np.ndarray,
        phases: np.ndarray,
    ) -> tuple[str, dict]:
        """Classify a single window.

        Args:
            amplitudes: shape (T, subcarriers).
            phases:     shape (T, subcarriers).

        Returns:
            (label, scores) — same contract as detect_motion().
        """
        return detect_motion(
            amplitudes,
            phases,
            mvs_threshold=self.mvs_threshold,
            nbvi_threshold=self.nbvi_threshold,
            quiet_variances=self.quiet_variances,
        )

    # ----------------------------------------------------------
    # PERSISTENCE
    # ----------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save calibration to a JSON file.

        Args:
            path: destination .json path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mvs_threshold": float(self.mvs_threshold),
            "nbvi_threshold": float(self.nbvi_threshold),
            "calibrated": self._calibrated,
        }
        if self.quiet_variances is not None:
            payload["quiet_variances"] = self.quiet_variances.tolist()
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: Path) -> "MathDetector":
        """Load a previously saved calibration.

        Args:
            path: path to a JSON file written by ``save()``.

        Returns:
            MathDetector instance ready for predict().
        """
        with open(path) as fh:
            payload = json.load(fh)
        detector = cls(
            mvs_threshold=payload["mvs_threshold"],
            nbvi_threshold=payload["nbvi_threshold"],
        )
        qv = payload.get("quiet_variances")
        if qv is not None:
            detector.quiet_variances = np.array(qv, dtype=np.float32)
        detector._calibrated = payload.get("calibrated", False)
        return detector
