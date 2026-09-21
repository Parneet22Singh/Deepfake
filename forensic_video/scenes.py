"""Histogram-based scene boundary detection."""

from typing import Any, Dict, List
import numpy as np

from .stats import median, robust_z


def detect_scene_cuts(path: str, threshold: float = 0.35) -> Dict[str, Any]:
    metrics: List[float] = []
    indices: List[int] = []
    warnings: List[str] = []
    try:
        import cv2  # type: ignore
    except ImportError:
        return {"status": "unavailable", "cuts": [], "metrics": {}, "warnings": ["opencv is not installed"]}
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return {"status": "unavailable", "cuts": [], "metrics": {}, "warnings": ["opencv could not open input"]}
    previous = None
    index = -1
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, frame_count // 120)
    while True:
        index += 1
        if index % stride:
            ok = capture.grab()
            frame = None
        else:
            ok, frame = capture.read()
        if not ok:
            break
        if frame is None:
            continue
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if previous is not None:
            metrics.append(float(cv2.compareHist(previous, hist, cv2.HISTCMP_BHATTACHARYYA)))
            indices.append(index)
        previous = hist
    capture.release()
    if not metrics:
        return {"status": "ok", "cuts": [], "metrics": {"comparisons": 0}, "warnings": warnings}
    cut_positions = [indices[i] for i, value in enumerate(metrics) if value >= threshold]
    # Robust outliers catch cuts even if global contrast makes a fixed threshold poor.
    center = median(metrics)
    robust_positions = [indices[i] for i, value in enumerate(metrics)
                        if (robust_z(value, metrics) or 0.0) >= 6.0]
    cuts = sorted(set(cut_positions + robust_positions))
    return {
        "status": "ok",
        "cuts": cuts,
        "metrics": {
            "comparisons": len(metrics),
            "median_histogram_distance": center,
            "max_histogram_distance": max(metrics),
            "cut_rate": len(cuts) / max(1, len(metrics)),
        },
        "warnings": warnings,
    }
