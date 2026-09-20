"""Deterministic, full-duration frame access shared by diagnostic branches."""

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

_SAMPLED_CACHE = {}


def frame_indices(capture, limit: int) -> List[int]:
    """Return a bounded, endpoint-preserving set of decoded-frame indices."""
    count = int(round(capture.get(7) or 0))  # CAP_PROP_FRAME_COUNT
    limit = max(1, int(limit))
    if count <= 0:
        return []
    return np.linspace(0, count - 1, min(limit, count), dtype=int).tolist()


def read_sampled_frames(path: str, limit: int,
                        indices: Optional[List[int]] = None) -> Tuple[List[Tuple[int, object]], str]:
    """Read frames across the whole file, rather than silently using its prefix.

    Bounded random access avoids decoding a long 4K prefix for every branch.
    The requested frame numbers and decoder are fixed, so this remains
    reproducible for a given input file; the returned index is recorded.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        return [], "opencv is not installed"
    # Include file identity so deterministic fixtures or callers that replace
    # a path do not receive frames decoded from an older file.
    try:
        stat = Path(path).stat()
        identity = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        identity = None
    cache_key = (str(path), identity,
                 tuple(sorted(set(indices))) if indices is not None else None)
    if cache_key in _SAMPLED_CACHE:
        cached = _SAMPLED_CACHE[cache_key]
        return _limit_frames(cached, limit), ""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return [], "opencv could not open input"
    wanted = sorted(set(indices if indices is not None else frame_indices(capture, limit)))
    if len(wanted) > limit:
        wanted = [wanted[i] for i in np.linspace(0, len(wanted) - 1, limit, dtype=int)]
        wanted = sorted(set(wanted))
    result: List[Tuple[int, object]] = []
    for requested_index in wanted:
        capture.set(cv2.CAP_PROP_POS_FRAMES, requested_index)
        ok, frame = capture.read()
        if not ok:
            continue
        actual_index = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        result.append((max(0, actual_index), frame))
    capture.release()
    if result and indices is not None:
        # All branches in one analysis share a deterministic decode, while
        # each branch still receives its own bounded subset below.
        _SAMPLED_CACHE[cache_key] = result
        result = _limit_frames(result, limit)
    return result, "" if result else "no decodable frames"


def _limit_frames(frames: List[Tuple[int, object]], limit: int):
    if len(frames) <= max(1, int(limit)):
        return frames
    selected = np.linspace(0, len(frames) - 1, max(1, int(limit)), dtype=int).tolist()
    return [frames[index] for index in sorted(set(selected))]
