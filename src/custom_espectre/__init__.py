"""Minimal custom CSI motion detection prototype."""

# Math-based Stage-A detector (no training required).
from .math_detector import MathDetector, detect_motion, moving_variance_score, nbvi_score

__all__ = [
    "MathDetector",
    "detect_motion",
    "moving_variance_score",
    "nbvi_score",
]
