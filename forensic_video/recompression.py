"""Controlled JPEG recompression residual analysis."""

from typing import Any, Dict, List, Optional
import numpy as np

from .frames import read_sampled_frames
from .stats import clipped, mad, median, outlier_fraction


def analyze_recompression(path: str, quality: int = 75, max_frames: int = 32,
                          indices: Optional[List[int]] = None) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return _unavailable("opencv is not installed")
    selected, warning = read_sampled_frames(path, max_frames, indices)
    if not selected:
        return _unavailable(warning)
    residuals: List[float] = []
    blockiness: List[float] = []
    localized: List[float] = []
    used = 0
    for _, frame in selected:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            continue
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        residual_image = cv2.absdiff(frame, decoded)
        residuals.append(float(np.mean(residual_image)) / 255.0)
        gray_residual = cv2.cvtColor(residual_image, cv2.COLOR_BGR2GRAY)
        tile_values = _tile_means(gray_residual, 8, 8)
        localized.append(float(np.percentile(tile_values, 90)) /
                         (float(np.median(tile_values)) + 1.0))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        # A high difference between adjacent 8-pixel boundary gradients is a
        # reproducible JPEG-blocking proxy, not a claim of manipulation.
        vertical = np.abs(np.diff(gray, axis=1))
        horizontal = np.abs(np.diff(gray, axis=0))
        boundary = np.mean(vertical[:, 7::8]) + np.mean(horizontal[7::8, :])
        interior = np.mean(vertical[:, 3::8]) + np.mean(horizontal[3::8, :])
        blockiness.append(float(boundary / (interior + 1e-6)))
        used += 1
    if not residuals:
        return _unavailable("no decodable frames")
    median_residual = float(median(residuals) or 0.0)
    residual_inconsistency = clipped((mad(residuals) or 0.0) /
                                     (median_residual + 0.01) * 4.0) or 0.0
    block_inconsistency = clipped((mad(blockiness) or 0.0) /
                                  (float(median(blockiness) or 0.0) + 0.01) * 3.0) or 0.0
    localized_signal = clipped(((median(localized) or 1.0) - 1.0) / 3.0) or 0.0
    localized_variation = clipped((mad(localized) or 0.0) /
                                   (float(median(localized) or 1.0) + 0.1) * 4.0) or 0.0
    anomaly = float(clipped(0.20 * residual_inconsistency +
                            0.20 * block_inconsistency +
                            0.25 * localized_variation +
                            0.20 * outlier_fraction(localized) +
                            0.15 * localized_signal) or 0.0)
    return {
        "status": "ok",
        "score": anomaly if used >= 4 else None,
        "confidence": min(1.0, used / 8.0) if used >= 4 else 0.0,
        "metrics": {
            "frames": used,
            "jpeg_quality": quality,
            "median_residual": median_residual,
            "p95_residual": float(np.percentile(residuals, 95)),
            "median_blockiness_ratio": float(median(blockiness) or 0.0),
            "mad_residual": float(mad(residuals) or 0.0),
            "residual_inconsistency": residual_inconsistency,
            "blockiness_inconsistency": block_inconsistency,
            "residual_outlier_fraction": outlier_fraction(residuals),
            "median_localized_residual_ratio": median(localized),
            "localized_residual_signal": localized_signal,
            "localized_residual_variation": localized_variation,
            "global_compression_signal": clipped(median_residual * 2.0) or 0.0,
        },
        "findings": (["within-video recompression inconsistency detected"]
                     if used >= 4 and anomaly >= 0.35 else []),
        "warnings": [warning] if warning else [],
    }


def _unavailable(message: str) -> Dict[str, Any]:
    return {"status": "unavailable", "score": None, "metrics": {}, "findings": [], "warnings": [message]}


def _tile_means(image, rows: int, cols: int):
    height, width = image.shape[:2]
    tile_height, tile_width = height // rows, width // cols
    return np.asarray([
        np.mean(image[r * tile_height:(r + 1) * tile_height,
                      c * tile_width:(c + 1) * tile_width])
        for r in range(rows) for c in range(cols)
    ], dtype=np.float32)
