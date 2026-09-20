"""Deterministic synthetic video fixtures for tests and examples."""

from pathlib import Path
from typing import Optional

import numpy as np


def write_synthetic_video(path: str, frames: int = 24, width: int = 96, height: int = 64,
                          fps: float = 12.0, scene_cut_at: int = 12,
                          manipulation: Optional[str] = None) -> str:
    """Write a small deterministic fixture, optionally with a duplicate burst.

    ``manipulation="duplicate-burst"`` repeats one decoded frame for a short
    interval while retaining changing content before and after it.  This is a
    deliberately simple temporal manipulation fixture for regression tests;
    it is not representative of the full range of real-world forgeries.
    """
    import cv2  # type: ignore

    if manipulation not in {None, "duplicate-burst", "interpolation"}:
        raise ValueError("unsupported synthetic manipulation")
    if frames < 1 or width < 1 or height < 1:
        raise ValueError("synthetic video dimensions and frame count must be positive")

    target = Path(path)
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV MJPG writer unavailable")

    generated = []
    for i in range(frames):
        frame = np.zeros((height, width, 3), np.uint8)
        base = 30 if i < scene_cut_at else 180
        x = (6 * i) % max(1, width - 16)
        frame[:, :] = (base, base // 2, min(255, base + 20))
        frame[height // 3:height // 3 + 12, x:x + 16] = (255, 255, 255)
        generated.append(frame)

    if manipulation == "duplicate-burst" and frames >= 9:
        start = max(1, min(frames - 8, scene_cut_at // 2))
        source = generated[start - 1]
        for index in range(start, min(frames, start + 8)):
            generated[index] = source.copy()
    elif manipulation == "interpolation" and frames >= 7:
        # A deterministic frame-rate-conversion fixture: replace interior
        # frames with a linear blend, while keeping changing neighbors.
        for index in range(max(1, scene_cut_at // 2),
                           min(frames - 1, scene_cut_at // 2 + 3)):
            generated[index] = cv2.addWeighted(
                generated[index - 1], 0.5, generated[index + 1], 0.5, 0.0
            )

    for frame in generated:
        writer.write(frame)
    writer.release()
    return str(target)
