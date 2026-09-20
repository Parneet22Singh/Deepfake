"""Codec-aware, transform-invariant frame consistency diagnostics.

This branch deliberately combines measurements that are useful only as
within-video diagnostics.  It normalizes letterbox/crop geometry before
estimating flow, and reports the leave-one-tile-out stability so a single
border, subtitle, or compression block cannot dominate the result.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .frames import read_sampled_frames
from .stats import clipped, mad, median, outlier_fraction


def normalize_content(frame, target: Tuple[int, int] = (256, 144)):
    """Remove obvious black borders, center-crop to a stable aspect, resize."""
    import cv2  # type: ignore
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    if float(np.median(border)) < 12.0 and float(np.median(gray)) > float(np.median(border)) + 5:
        mask = gray > max(8.0, float(np.percentile(border, 90)) + 3.0)
        rows = np.flatnonzero(np.mean(mask, axis=1) > 0.08)
        cols = np.flatnonzero(np.mean(mask, axis=0) > 0.08)
        if len(rows) >= 8 and len(cols) >= 8:
            frame = frame[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    height, width = frame.shape[:2]
    target_width, target_height = target
    desired = target_width / float(target_height)
    actual = width / float(max(1, height))
    if actual > desired:
        new_width = max(1, int(round(height * desired)))
        left = (width - new_width) // 2
        frame = frame[:, left:left + new_width]
    elif actual < desired:
        new_height = max(1, int(round(width / desired)))
        top = (height - new_height) // 2
        frame = frame[top:top + new_height, :]
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def analyze_integrity(path: str, max_frames: int = 32,
                      indices: Optional[List[int]] = None) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return _unavailable("opencv is not installed")
    selected, warning = read_sampled_frames(path, max_frames, indices)
    if len(selected) < 2:
        return _unavailable(warning or "fewer than two decodable frames")
    normalized = [normalize_content(frame) for _, frame in selected]
    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
             for frame in normalized]
    residuals: List[float] = []
    divergence: List[float] = []
    occlusion: List[float] = []
    local_ratio: List[float] = []
    loo_ratio: List[float] = []
    dct_energy: List[float] = []
    for previous, current in zip(grays, grays[1:]):
        forward = cv2.calcOpticalFlowFarneback(
            previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        backward = cv2.calcOpticalFlowFarneback(
            current, previous, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        warped = _warp(previous, forward)
        residual = cv2.absdiff(warped, current)
        residuals.append(float(np.median(residual)) / 255.0)
        dx = cv2.Sobel(forward[..., 0], cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(forward[..., 1], cv2.CV_32F, 0, 1, ksize=3)
        divergence.append(float(np.median(np.abs(dx + dy))))
        reverse = _warp(backward, forward)
        consistency = np.linalg.norm(forward + reverse, axis=2)
        occlusion.append(float(np.mean(consistency > 1.5)))
        tiles = _tiles(residual, 8, 8)
        center = float(np.median(tiles))
        spread = float(np.median(np.abs(tiles - center))) + 1e-3
        local_ratio.append(float(np.percentile(tiles, 90) - center) / spread)
        without_hotspot = np.delete(tiles, int(np.argmax(tiles)))
        loo_center = float(np.median(without_hotspot))
        loo_spread = float(np.median(np.abs(without_hotspot - loo_center))) + 1e-3
        loo_ratio.append(float(np.percentile(without_hotspot, 90) - loo_center) / loo_spread)
    for gray in grays:
        blocks = []
        for y in range(0, gray.shape[0] - 7, 8):
            for x in range(0, gray.shape[1] - 7, 8):
                block = gray[y:y + 8, x:x + 8] - float(np.mean(gray[y:y + 8, x:x + 8]))
                coeff = cv2.dct(block.astype(np.float32))
                blocks.append(float(np.mean(np.abs(coeff[2:, 2:]))))
        dct_energy.append(float(np.median(blocks)) if blocks else 0.0)
    residual_outliers = outlier_fraction(residuals)
    divergence_signal = clipped((median(divergence) or 0.0) / 2.5) or 0.0
    occlusion_signal = clipped(((median(occlusion) or 0.0) - 0.04) * 3.0) or 0.0
    residual_signal = clipped(max(0.0, residual_outliers - 0.15) * 3.5) or 0.0
    local_signal = clipped(max(0.0, outlier_fraction(local_ratio) - 0.15) * 3.5) or 0.0
    dct_signal = clipped((mad(dct_energy) or 0.0) /
                         (float(median(dct_energy) or 0.0) + 1e-3) * 4.0) or 0.0
    raw = float(clipped(0.25 * residual_signal + 0.20 * divergence_signal +
                        0.20 * occlusion_signal + 0.20 * local_signal +
                        0.15 * dct_signal) or 0.0)
    # Leave-one-region-out: retain the signal only when removing the most
    # suspicious tile does not erase essentially all pairwise evidence.
    loo_support = clipped((median(loo_ratio) or 0.0) /
                          (median(local_ratio) or 1.0)) or 0.0
    score = float(raw * min(1.0, 0.5 + loo_support))
    return {
        "status": "ok",
        "score": score if len(residuals) >= 4 else None,
        "confidence": min(1.0, len(residuals) / 8.0) if len(residuals) >= 4 else 0.0,
        "metrics": {
            "pairs": len(residuals),
            "normalization": "border-crop-center-aspect-resize-256x144",
            "median_warp_residual": median(residuals),
            "residual_outlier_fraction": residual_outliers,
            "median_flow_divergence": median(divergence),
            "median_occlusion_fraction": median(occlusion),
            "flow_divergence_signal": divergence_signal,
            "occlusion_signal": occlusion_signal,
            "local_residual_outlier_fraction": outlier_fraction(local_ratio),
            "leave_one_region_out_support": loo_support,
            "leave_one_region_out_ratio": median(loo_ratio),
            "median_dct_high_frequency_energy": median(dct_energy),
            "dct_energy_inconsistency": dct_signal,
        },
        "findings": (["transform-invariant residual/flow inconsistency detected"]
                     if score >= 0.35 and len(residuals) >= 4 else []),
        "warnings": [warning] if warning else [],
    }


def _warp(image, flow):
    import cv2  # type: ignore
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(image, (xx + flow[..., 0]).astype(np.float32),
                     (yy + flow[..., 1]).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def _tiles(image, rows: int, cols: int):
    height, width = image.shape[:2]
    tile_height, tile_width = height // rows, width // cols
    return np.asarray([
        np.mean(image[r * tile_height:(r + 1) * tile_height,
                      c * tile_width:(c + 1) * tile_width])
        for r in range(rows) for c in range(cols)
    ], dtype=np.float32)


def _unavailable(message: str) -> Dict[str, Any]:
    return {"status": "unavailable", "score": None, "confidence": 0.0,
            "metrics": {}, "findings": [], "warnings": [message]}
