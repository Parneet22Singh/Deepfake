"""Model-free Haar subband and localized temporal instability diagnostics.

The transform is intentionally small and explicit: one level of a separable
Haar transform is enough to expose changes in coarse structure and horizontal,
vertical, and diagonal detail without a learned representation.  This branch
describes an anomaly signal only; it is not a manipulation classifier.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .frames import read_sampled_frames
from .stats import clipped, mad, median


def _haar(gray: np.ndarray) -> Dict[str, np.ndarray]:
    """Return normalized one-level 2-D Haar subbands."""
    height = gray.shape[0] - gray.shape[0] % 2
    width = gray.shape[1] - gray.shape[1] % 2
    image = gray[:height, :width].astype(np.float32) / 255.0
    return _haar_unit(image)


def _haar_unit(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Haar transform for an already normalized array."""
    height = image.shape[0] - image.shape[0] % 2
    width = image.shape[1] - image.shape[1] % 2
    image = image[:height, :width].astype(np.float32)
    a, b = image[0::2, 0::2], image[0::2, 1::2]
    c, d = image[1::2, 0::2], image[1::2, 1::2]
    return {
        "ll": (a + b + c + d) * 0.5,
        "lh": (a - b + c - d) * 0.5,
        "hl": (a + b - c - d) * 0.5,
        "hh": (a - b - c + d) * 0.5,
    }


def analyze_wavelet(path: str, max_frames: int = 32,
                    indices: Optional[List[int]] = None,
                    cut_indices: Optional[List[int]] = None) -> Dict[str, Any]:
    """Measure Haar energy changes and record localized detail events."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return _unavailable("opencv is not installed")
    selected, warning = read_sampled_frames(path, max_frames, indices)
    if len(selected) < 4:
        return _unavailable(warning or "fewer than four decodable frames")

    records = []
    for index, frame in selected:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 72), interpolation=cv2.INTER_AREA)
        bands = _haar(gray)
        # Recurse once on LL to retain a genuinely multiscale coarse/detail
        # summary while keeping the computation bounded and transparent.
        bands.update({("level2_" + name): value
                      for name, value in _haar_unit(bands["ll"]).items()})
        scale = float(np.mean(np.abs(bands["ll"])) + 1e-6)
        energies = {name: float(np.mean(np.abs(value)) / scale)
                    for name, value in bands.items()}
        detail = np.abs(bands["hh"])
        records.append((int(index), energies, detail))

    names = ("ll", "lh", "hl", "hh")
    vectors = np.asarray([[item[1][name] for name in names] for item in records])
    deltas = np.mean(np.abs(np.diff(vectors, axis=0)), axis=1)
    cuts = set(int(value) for value in (cut_indices or []))
    usable = [
        float(value) for offset, value in enumerate(deltas, start=1)
        if not any(abs(records[offset][0] - cut) <= 1 or
                   abs(records[offset - 1][0] - cut) <= 1 for cut in cuts)
    ]
    if not usable:
        usable = [float(value) for value in deltas]
    baseline = float(median(usable) or 0.0)
    spread = float(mad(usable) or 0.0) + 1e-6
    event_indices = [
        records[offset][0] for offset, value in enumerate(deltas, start=1)
        if value > baseline + 4.0 * spread and value > max(0.02, baseline * 2.5)
    ]
    # A localized detail change is stronger than a global brightness change.
    local_ratios = []
    for previous, current in zip(records, records[1:]):
        delta = np.abs(current[2] - previous[2])
        tiles = _tile_means(delta, 6, 8)
        center = float(np.median(tiles))
        local_ratios.append(float(np.percentile(tiles, 90) - center) /
                           (float(np.median(np.abs(tiles - center))) + 1e-6))
    local_event_count = sum(value > 4.0 for value in local_ratios)
    normalized_instability = float(clipped(
        (float(mad(usable) or 0.0) / (baseline + 0.01)) * 2.0) or 0.0)
    burst_signal = float(clipped(len(event_indices) / max(2.0, len(usable) * 0.25)) or 0.0)
    local_signal = float(clipped(local_event_count / max(2.0, len(local_ratios) * 0.25)) or 0.0)
    diagnostic_score = float(clipped(
        0.45 * normalized_instability + 0.30 * burst_signal + 0.25 * local_signal
    ) or 0.0)
    support = min(1.0, len(usable) / 8.0)
    supported = len(usable) >= 4 and support >= 0.5
    return {
        "status": "ok",
        "score": diagnostic_score if supported and diagnostic_score >= 0.35 else None,
        "confidence": min(1.0, support * (1.0 if len(event_indices) else 0.6)),
        "metrics": {
            "frames": len(records),
            "pairs": len(usable),
            "subband_energy_median": {
                name: round(float(np.median(vectors[:, pos])), 6)
                for pos, name in enumerate(names)
            },
            "subband_energy_mad": {
                name: round(float(np.median(np.abs(vectors[:, pos] -
                                                   np.median(vectors[:, pos])))), 6)
                for pos, name in enumerate(names)
            },
            "multiscale_levels": 2,
            "level2_subband_energy_median": {
                name: round(float(np.median(
                    [item[1]["level2_" + name] for item in records])), 6)
                for name in names
            },
            "haar_temporal_instability": normalized_instability,
            "localized_detail_instability": local_signal,
            "detail_event_count": len(event_indices),
            "instability_indices": event_indices,
            "median_subband_delta": baseline,
            "cut_excluded_pairs": len(deltas) - len(usable),
        },
        "findings": (["Haar subband energy changed with localized temporal instability"]
                     if diagnostic_score >= 0.35 and supported else []),
        "warnings": ([warning] if warning else []) + [
            "Haar diagnostics are signal inconsistencies, not manipulation proof"
        ],
    }


def _tile_means(image: np.ndarray, rows: int, cols: int) -> np.ndarray:
    height, width = image.shape[:2]
    values = []
    for row in range(rows):
        y0, y1 = row * height // rows, (row + 1) * height // rows
        for col in range(cols):
            x0, x1 = col * width // cols, (col + 1) * width // cols
            values.append(float(np.mean(image[y0:y1, x0:x1])))
    return np.asarray(values, dtype=np.float32)


def _unavailable(message: str) -> Dict[str, Any]:
    return {"status": "unavailable", "score": None, "confidence": 0.0,
            "metrics": {}, "findings": [], "warnings": [message]}
