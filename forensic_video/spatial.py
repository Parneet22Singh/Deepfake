"""Model-free region-level spatial inconsistency diagnostics."""

from typing import Any, Dict, List, Optional

import numpy as np

from .frames import read_sampled_frames
from .stats import clipped, mad, median, outlier_fraction


def analyze_spatial(path: str, max_frames: int = 32,
                    indices: Optional[List[int]] = None) -> Dict[str, Any]:
    """Measure localized residual energy after removing each frame's global quality.

    It is intentionally not a face detector.  A score requires a persistent
    localized hotspot and is normalized by the frame median, so low bitrate or
    globally sharpened videos are reported as confounders rather than evidence.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        return _unavailable("opencv is not installed")
    selected, warning = read_sampled_frames(path, max_frames, indices)
    if not selected:
        return _unavailable(warning)
    concentration: List[float] = []
    hotspot_positions: List[int] = []
    global_noise: List[float] = []
    for _, frame in selected:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.resize(gray, (256, 144), interpolation=cv2.INTER_AREA)
        residual = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.1))
        tiles = _tile_means(residual, 8, 8)
        baseline = float(np.median(tiles))
        spread = float(np.median(np.abs(tiles - baseline))) + 1e-5
        normalized = (tiles - baseline) / spread
        global_noise.append(baseline / 255.0)
        # Top tile energy above the robust frame baseline is the localized
        # component; a global codec/noise lift affects all tiles similarly.
        top = float(np.mean(np.sort(normalized.ravel())[-max(1, normalized.size // 10):]))
        concentration.append(max(0.0, top))
        hotspot_positions.append(int(np.argmax(normalized)))
    if len(concentration) < 4:
        return {"status": "ok", "score": None, "confidence": 0.0,
                "metrics": {"frames": len(concentration)},
                "findings": [], "warnings": ["insufficient frames for spatial scoring"]}
    median_concentration = float(median(concentration) or 0.0)
    concentration_variation = float((mad(concentration) or 0.0) /
                                    (median_concentration + 1e-3))
    # Persistent same-area evidence is more useful than a moving edge.
    dominant_position = max(set(hotspot_positions), key=hotspot_positions.count)
    persistence = hotspot_positions.count(dominant_position) / len(hotspot_positions)
    localized_signal = clipped((median_concentration - 2.0) / 8.0) or 0.0
    temporal_support = clipped((persistence - 0.15) / 0.55) or 0.0
    instability_support = clipped(concentration_variation / 2.0) or 0.0
    global_quality = float(median(global_noise) or 0.0)
    confounder = global_quality >= 0.12
    score = float(clipped(0.55 * localized_signal * temporal_support +
                          0.45 * localized_signal * instability_support) or 0.0)
    confidence = min(1.0, len(concentration) / 8.0)
    supported = (score >= 0.25 and localized_signal >= 0.45 and
                 persistence >= 0.75 and not confounder)
    return {
        "status": "ok",
        "score": score if supported else None,
        "confidence": confidence if supported else 0.0,
        "metrics": {
            "frames": len(concentration),
            "median_localized_residual": median_concentration,
            "mad_localized_residual": float(mad(concentration) or 0.0),
            "localized_residual_signal": localized_signal,
            "hotspot_persistence": persistence,
            "hotspot_instability": concentration_variation,
            "median_global_noise": global_quality,
            "global_quality_confounder": confounder,
            "hotspot_tile": dominant_position,
            "outlier_fraction": outlier_fraction(concentration),
        },
        "findings": (["persistent localized spatial inconsistency detected"]
                     if supported else []),
        "warnings": ([warning] if warning else []) +
                    (["global noise/compression dominates spatial residuals"]
                     if confounder else []),
    }


def _tile_means(image: np.ndarray, rows: int, cols: int) -> np.ndarray:
    height, width = image.shape
    tile_height, tile_width = height // rows, width // cols
    return np.asarray([
        np.mean(image[r * tile_height:(r + 1) * tile_height,
                      c * tile_width:(c + 1) * tile_width])
        for r in range(rows) for c in range(cols)
    ], dtype=np.float32)


def _unavailable(message: str) -> Dict[str, Any]:
    return {"status": "unavailable", "score": None, "confidence": 0.0,
            "metrics": {}, "findings": [], "warnings": [message]}
