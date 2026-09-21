"""Full-video, deterministic adaptive frame sampling."""

from typing import List, Tuple
import numpy as np

from .models import SamplingResult, VideoMetadata


def sample_video(path: str, metadata: VideoMetadata, requested: int = 48) -> SamplingResult:
    requested = max(1, int(requested))
    count = metadata.frame_count or 0
    fps = metadata.fps or 1.0
    if count <= 0:
        return SamplingResult(requested, [], [], "unavailable", 0.0, ["frame count unavailable"])
    uniform = np.linspace(0, max(0, count - 1), min(requested, count), dtype=int).tolist()
    indices = list(dict.fromkeys(uniform))
    warnings: List[str] = []
    method = "uniform"
    try:
        import cv2  # type: ignore
        capture = cv2.VideoCapture(path)
        if capture.isOpened() and len(indices) > 2:
            candidates = _motion_candidates(capture, count)
            capture.release()
            if candidates:
                # Replace at most a third with high-change positions while retaining endpoints.
                ranks = sorted(candidates, key=lambda pair: pair[1], reverse=True)
                additions = max(0, min(len(indices) // 3, len(ranks)))
                indices = sorted(set(indices + [p[0] for p in ranks[:additions]]))
                indices = _downselect(indices, requested, count)
                method = "uniform+adaptive-motion"
            else:
                capture.release()
        else:
            capture.release()
            warnings.append("opencv could not open input; using metadata-only sampling")
    except ImportError:
        warnings.append("opencv is not installed; using uniform metadata-only sampling")
    timestamps = [round(i / fps, 6) for i in indices]
    # Endpoints are retained, so this measures temporal span rather than frame count.
    coverage = ((indices[-1] - indices[0]) / float(max(1, count - 1))) if indices else 0.0
    return SamplingResult(requested, indices, timestamps, method, coverage, warnings)


def _motion_candidates(capture, count: int) -> List[Tuple[int, float]]:
    import cv2  # type: ignore
    previous = None
    result: List[Tuple[int, float]] = []
    stride = max(1, count // 120)
    for index in range(count):
        if index % stride:
            ok = capture.grab()
            frame = None
        else:
            ok, frame = capture.read()
        if not ok:
            break
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
        if previous is not None:
            result.append((index, float(np.mean(cv2.absdiff(gray, previous)))))
        previous = gray
    return result


def _downselect(indices: List[int], requested: int, count: int) -> List[int]:
    if requested <= 1:
        return [indices[0] if indices else 0]
    if len(indices) <= requested:
        return indices
    selected = np.linspace(0, len(indices) - 1, requested, dtype=int).tolist()
    answer = sorted({indices[i] for i in selected} | {0, max(0, count - 1)})
    if len(answer) > requested:
        answer = [answer[i] for i in np.linspace(0, len(answer) - 1, requested, dtype=int)]
    return sorted(set(answer) | {0, max(0, count - 1)})
