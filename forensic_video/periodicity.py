"""Deterministic duplicate-frame and periodicity diagnostics."""

from typing import Any, Dict, List, Optional

import numpy as np

from .frames import read_sampled_frames
from .stats import clipped, median


def frame_signature(frame):
    import cv2  # type: ignore
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (24, 14), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32)


def analyze_periodicity(path: str, max_frames: int = 48,
                        indices: Optional[List[int]] = None) -> Dict[str, Any]:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return _unavailable("opencv is not installed")
    selected, warning = read_sampled_frames(path, max_frames, indices)
    if len(selected) < 4:
        return _unavailable(warning or "fewer than four decodable frames")
    signatures = [frame_signature(frame) for _, frame in selected]
    adjacent = [float(np.mean(np.abs(a - b))) for a, b in zip(signatures, signatures[1:])]
    duplicate_flags = [value <= 0.65 for value in adjacent]
    duplicate_rate = float(np.mean(duplicate_flags)) if duplicate_flags else 0.0
    longest_run = _longest_run(duplicate_flags)
    periodic_matches = []
    for lag in range(2, max(3, min(12, len(signatures) // 2 + 1))):
        distances = [float(np.mean(np.abs(signatures[i] - signatures[i - lag])))
                     for i in range(lag, len(signatures))]
        if distances:
            periodic_matches.append(float(np.mean(np.asarray(distances) <= 1.0)))
    periodic_rate = max(periodic_matches) if periodic_matches else 0.0
    motion = float(median(adjacent) or 0.0)
    # A duplicate burst surrounded by changing content is more informative
    # than a naturally static shot or an ordinary low-FPS encode.
    burst_support = clipped((longest_run - 1.0) / max(2.0, len(adjacent) * 0.35)) or 0.0
    motion_support = clipped((motion - 0.65) / 3.0) or 0.0
    score = float(clipped(0.55 * burst_support + 0.45 * periodic_rate * motion_support) or 0.0)
    supported = longest_run >= 3 and motion_support >= 0.2 and score >= 0.25
    return {
        "status": "ok",
        "score": score if supported else None,
        "confidence": min(1.0, len(signatures) / 12.0) if supported else 0.0,
        "metrics": {
            "frames": len(signatures),
            "median_adjacent_signature_distance": motion,
            "duplicate_rate": duplicate_rate,
            "longest_duplicate_run": longest_run,
            "periodic_match_rate": periodic_rate,
            "periodic_lag_count": len(periodic_matches),
        },
        "findings": (["duplicate/periodic frame pattern detected"]
                     if supported else []),
        "warnings": [warning] if warning else [],
    }


def _longest_run(flags: List[bool]) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def _unavailable(message: str) -> Dict[str, Any]:
    return {"status": "unavailable", "score": None, "confidence": 0.0,
            "metrics": {}, "findings": [], "warnings": [message]}
