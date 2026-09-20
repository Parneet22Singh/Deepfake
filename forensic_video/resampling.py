"""Interpolation, resampling, and duplicate-frame residual diagnostics.

These measurements are deliberately kept separate from manipulation evidence:
ordinary frame-rate conversion and a static shot can produce the same residual
patterns.  The branch therefore exposes a diagnostic score and locations but
does not contribute a manipulation score to fusion.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .frames import read_sampled_frames
from .periodicity import frame_signature, _longest_run
from .stats import clipped, mad, median


def analyze_resampling(path: str, max_frames: int = 48,
                       indices: Optional[List[int]] = None) -> Dict[str, Any]:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return _unavailable("opencv is not installed")
    selected, warning = read_sampled_frames(path, max_frames, indices)
    if len(selected) < 5:
        return _unavailable(warning or "fewer than five decodable frames")
    signatures = [frame_signature(frame) for _, frame in selected]
    distances = np.asarray([
        float(np.mean(np.abs(left - right)))
        for left, right in zip(signatures, signatures[1:])
    ], dtype=np.float32)
    duplicate_flags = distances <= 0.65
    duplicate_runs = _longest_run(duplicate_flags)
    residuals = []
    motion_scales = []
    residual_indices = []
    for offset in range(1, len(signatures) - 1):
        expected = (signatures[offset - 1] + signatures[offset + 1]) * 0.5
        residual = float(np.mean(np.abs(signatures[offset] - expected)))
        scale = float(np.mean(np.abs(signatures[offset + 1] -
                                     signatures[offset - 1]))) + 0.05
        residuals.append(residual / scale)
        motion_scales.append(scale)
        if residual / scale > 2.5:
            residual_indices.append(int(selected[offset][0]))
    center = float(median(residuals) or 0.0)
    spread = float(mad(residuals) or 0.0) + 1e-6
    # A generated intermediate frame has unusually small second difference
    # while its neighbors still move.  Keep this separate from large residual
    # glitches, which are more naturally codec/dropout diagnostics.
    low_residual_floor = center - max(2.0 * spread, 0.12)
    interpolation_events = [
        int(selected[offset + 1][0]) for offset, value in enumerate(residuals)
        if value < low_residual_floor and motion_scales[offset] > 1.0
    ]
    residual_signal = float(clipped(
        (float(np.median(residuals)) - 1.0) / 2.0) or 0.0)
    event_signal = float(clipped(
        len(interpolation_events) / max(2.0, len(residuals) * 0.20)) or 0.0)
    duplicate_signal = float(clipped(
        (duplicate_runs - 1.0) / max(2.0, len(distances) * 0.30)) or 0.0)
    diagnostic_score = float(clipped(
        0.45 * event_signal + 0.35 * residual_signal + 0.20 * duplicate_signal
    ) or 0.0)
    return {
        "status": "ok",
        "score": None,
        "confidence": min(1.0, len(signatures) / 16.0),
        "metrics": {
            "frames": len(signatures),
            "adjacent_signature_median": float(median(distances.tolist()) or 0.0),
            "adjacent_signature_mad": float(mad(distances.tolist()) or 0.0),
            "duplicate_run": int(duplicate_runs),
            "interpolation_residual_median": center,
            "interpolation_residual_mad": float(mad(residuals) or 0.0),
            "interpolation_event_count": len(interpolation_events),
            "interpolation_event_indices": interpolation_events,
            "duplicate_residual_indices": residual_indices,
            "diagnostic_score": diagnostic_score,
            "fusion_eligible": False,
        },
        "findings": (["interpolation/resampling residual pattern observed"]
                     if diagnostic_score >= 0.35 else []),
        "warnings": ([warning] if warning else []) + [
            "resampling and duplicate residuals are diagnostics only; "
            "they are not manipulation claims"
        ],
    }


def _unavailable(message: str) -> Dict[str, Any]:
    return {"status": "unavailable", "score": None, "confidence": 0.0,
            "metrics": {}, "findings": [], "warnings": [message]}
