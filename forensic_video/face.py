"""Conservative model-free face-region geometry and tracking diagnostics.

The branch deliberately does not call a face detector or assign an authenticity
score to a single skin-colored blob.  A score is emitted only when a region is
observed repeatedly and its geometry is internally inconsistent.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .frames import read_sampled_frames
from .stats import clipped, mad, median, outlier_fraction


def analyze_faces(path: str, max_frames: int = 24,
                  indices: Optional[List[int]] = None) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return _unavailable("opencv is not installed")
    selected, warning = read_sampled_frames(path, max_frames, indices)
    if not selected:
        return _unavailable(warning)

    observations: List[Tuple[int, float, float, float, float]] = []
    gray_frames = {}
    detections_per_frame: List[int] = []
    for index, frame in selected:
        height, width = frame.shape[:2]
        gray_frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 25, 45), (25, 220, 255))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        candidates = []
        for component in range(1, components):
            x, y, box_width, box_height, area = stats[component]
            ratio = box_width / float(max(1, box_height))
            fill = area / float(max(1, box_width * box_height))
            # Face-like geometry is intentionally broad, but rejects long
            # background regions and tiny compression-colored components.
            if (area >= max(120, width * height * 0.004) and
                    0.45 <= ratio <= 1.8 and box_width >= 16 and box_height >= 16 and
                    fill >= 0.18):
                candidates.append((area, component, box_width, box_height, fill))
        # One best region per frame avoids counting a background as a track.
        detections_per_frame.append(len(candidates))
        if candidates:
            _, component, box_width, box_height, fill = max(candidates)
            cx, cy = centroids[component]
            observations.append((index, cx / width, cy / height,
                                 box_width / width, box_height / height))

    frame_count = len(selected)
    detection_frames = len(observations)
    result: Dict[str, Any] = {
        "status": "ok",
        "score": None,
        "confidence": 0.0,
        "metrics": {
            "frames": frame_count,
            "detection_frames": detection_frames,
            "detections": int(sum(detections_per_frame)),
            "detection_rate": detection_frames / float(max(1, frame_count)),
            "median_candidates_per_frame": float(np.median(detections_per_frame)),
            "observation_indices": [int(item[0]) for item in observations],
        },
        "findings": [],
        "warnings": ([warning] if warning else []) + [
            "skin/shape regions are not face identities or authenticity evidence"
        ],
    }
    if detection_frames < 4:
        result["warnings"].append("insufficient repeated regions for face geometry scoring")
        return result

    # Track nearest observations in time.  Gaps and abrupt shape changes are
    # stronger evidence than the mere presence of a skin-colored component.
    jumps: List[float] = []
    size_jumps: List[float] = []
    for previous, current in zip(observations, observations[1:]):
        dt = max(1, current[0] - previous[0])
        jumps.append(float(np.hypot(current[1] - previous[1], current[2] - previous[2])) / dt)
        previous_size = np.array(previous[3:5])
        current_size = np.array(current[3:5])
        size_jumps.append(float(np.linalg.norm(current_size - previous_size)) / dt)
    position_mad = (mad(jumps) or 0.0) / (median(jumps) or 0.02)
    size_mad = (mad(size_jumps) or 0.0) / (median(size_jumps) or 0.01)
    geometry = [item[3] / max(0.001, item[4]) for item in observations]
    geometry_cv = float(np.std(geometry) / (np.mean(geometry) + 1e-6))
    # Compare the observed region displacement with the local Farneback flow.
    # This rejects a geometry jump that is fully explained by camera/object
    # motion while retaining a deterministic, model-free support measure.
    flow_residuals: List[float] = []
    flow_supported_pairs = 0
    for previous, current in zip(observations, observations[1:]):
        previous_gray = gray_frames.get(previous[0])
        current_gray = gray_frames.get(current[0])
        if previous_gray is None or current_gray is None:
            continue
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray, current_gray, None, 0.5, 2, 15, 3, 5, 1.2, 0
        )
        px = int(np.clip(round(previous[1] * (previous_gray.shape[1] - 1)),
                         0, previous_gray.shape[1] - 1))
        py = int(np.clip(round(previous[2] * (previous_gray.shape[0] - 1)),
                         0, previous_gray.shape[0] - 1))
        x0, x1 = max(0, px - 3), min(flow.shape[1], px + 4)
        y0, y1 = max(0, py - 3), min(flow.shape[0], py + 4)
        local_flow = np.median(flow[y0:y1, x0:x1], axis=(0, 1))
        observed_flow = np.asarray([
            (current[1] - previous[1]) * previous_gray.shape[1],
            (current[2] - previous[2]) * previous_gray.shape[0],
        ], dtype=np.float32)
        flow_residuals.append(float(np.linalg.norm(observed_flow - local_flow)) /
                             max(1.0, np.linalg.norm(observed_flow) + 1.0))
        flow_supported_pairs += 1
    flow_outliers = outlier_fraction(flow_residuals)
    flow_support = min(1.0, flow_supported_pairs / 6.0)
    # Require both continuity and a geometry inconsistency signal.  This keeps
    # isolated skin-colored scenery from contributing to fusion.
    instability = float(clipped(0.35 * position_mad + 0.35 * size_mad +
                                0.30 * geometry_cv * 3.0) or 0.0)
    continuity = min(1.0, detection_frames / float(max(1, frame_count)))
    confidence = min(1.0, detection_frames / 12.0) * continuity
    # Geometry variation is required; tiny centroid jitter alone is expected
    # from skin segmentation and codec noise.
    median_candidates = float(np.median(detections_per_frame))
    supported_geometry = (geometry_cv >= 0.10 and instability >= 0.25 and
                         median_candidates <= 3.0 and flow_supported_pairs >= 3)
    result["score"] = instability if confidence >= 0.3 and supported_geometry else None
    result["confidence"] = confidence if result["score"] is not None else 0.0
    result["metrics"].update({
        "track_count": 1,
        "candidate_ambiguity": median_candidates,
        "track_continuity": continuity,
        "median_position_step": median(jumps),
        "median_size_step": median(size_jumps),
        "position_instability": position_mad,
        "size_instability": size_mad,
        "geometry_cv": geometry_cv,
        "flow_supported_pairs": flow_supported_pairs,
        "flow_support": flow_support,
        "flow_residual_median": median(flow_residuals),
        "flow_residual_outlier_fraction": flow_outliers,
    })
    if result["score"] is not None:
        result["findings"].append("repeated face-like region has inconsistent geometry")
    else:
        result["warnings"].append("face-like geometry was stable or weak; score withheld")
    return result


def _unavailable(message: str) -> Dict[str, Any]:
    return {"status": "unavailable", "score": None, "confidence": 0.0,
            "metrics": {}, "findings": [], "warnings": [message]}
