"""Multiscale frequency and sensor-noise diagnostics."""

from typing import Any, Dict, List, Optional
import numpy as np

from .frames import read_sampled_frames
from .stats import clipped, mad, median, outlier_fraction


def analyze_frequency(path: str, max_frames: int = 32,
                      indices: Optional[List[int]] = None) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return _unavailable("opencv is not installed")
    selected, warning = read_sampled_frames(path, max_frames, indices)
    if not selected:
        return _unavailable(warning)
    high: List[float] = []
    noise: List[float] = []
    localized: List[float] = []
    used = 0
    for _, frame in selected:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.resize(gray, (256, 144), interpolation=cv2.INTER_AREA)
        spectrum = np.fft.fftshift(np.abs(np.fft.fft2(gray - gray.mean())))
        h, w = spectrum.shape
        yy, xx = np.ogrid[:h, :w]
        radius = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
        total = float(np.mean(spectrum) + 1e-9)
        high.append(float(np.mean(spectrum[radius > 0.55]) / total))
        blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
        residual = gray - blur
        noise.append(float(np.std(residual) / 255.0))
        tiles = _tile_means(np.abs(residual), 8, 8)
        localized.append(float(np.percentile(tiles, 90)) /
                         (float(np.median(tiles)) + 1e-3))
        used += 1
    if not high:
        return _unavailable("no decodable frames")
    median_noise = float(median(noise) or 0.0)
    noise_mad = float(mad(noise) or 0.0)
    noise_inconsistency = clipped(noise_mad / (median_noise + 0.005) * 5.0) or 0.0
    high_inconsistency = clipped((mad(high) or 0.0) /
                                 (float(median(high) or 0.0) + 0.01) * 3.0) or 0.0
    localized_signal = clipped(((median(localized) or 1.0) - 1.0) / 3.0) or 0.0
    localized_variation = clipped((mad(localized) or 0.0) /
                                  (float(median(localized) or 1.0) + 0.1) * 4.0) or 0.0
    anomaly = float(clipped(0.15 * noise_inconsistency +
                            0.20 * high_inconsistency +
                            0.25 * localized_variation +
                            0.20 * outlier_fraction(localized) +
                            0.20 * localized_signal) or 0.0)
    confidence = min(1.0, used / 8.0)
    return {
        "status": "ok",
        "score": anomaly if used >= 4 else None,
        "confidence": confidence if used >= 4 else 0.0,
        "metrics": {
            "frames": used,
            "median_high_frequency_ratio": float(np.median(high)),
            "mad_high_frequency_ratio": float(np.median(np.abs(high - np.median(high)))),
            "median_noise_sigma": median_noise,
            "mad_noise_sigma": noise_mad,
            "noise_p95": float(np.percentile(noise, 95)),
            "noise_inconsistency": noise_inconsistency,
            "high_frequency_inconsistency": high_inconsistency,
            "noise_outlier_fraction": outlier_fraction(noise),
            "median_localized_noise_ratio": median(localized),
            "localized_noise_signal": localized_signal,
            "localized_noise_variation": localized_variation,
            "global_noise_confounder": median_noise >= 0.12,
        },
        "findings": (["within-video high-frequency/noise inconsistency detected"]
                     if used >= 4 and anomaly >= 0.35 else []),
        "warnings": [warning] if warning else [],
    }


def _unavailable(message: str) -> Dict[str, Any]:
    return {"status": "unavailable", "score": None, "metrics": {}, "findings": [], "warnings": [message]}


def _tile_means(image, rows: int, cols: int):
    height, width = image.shape
    tile_height, tile_width = height // rows, width // cols
    return np.asarray([
        np.mean(image[r * tile_height:(r + 1) * tile_height,
                      c * tile_width:(c + 1) * tile_width])
        for r in range(rows) for c in range(cols)
    ], dtype=np.float32)
