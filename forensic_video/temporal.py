"""Classical feature tracking and optical-flow temporal consistency."""

from typing import Any, Dict, List, Optional
import numpy as np

from .frames import read_sampled_frames
from .stats import clipped, mad, median, outlier_fraction, robust_z


def analyze_temporal(path: str, max_pairs: int = 120,
                     indices: Optional[List[int]] = None,
                     cut_indices: Optional[List[int]] = None) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return _unavailable("opencv is not installed")
    selected, warning = read_sampled_frames(path, max_pairs + 1, indices)
    if not selected:
        return _unavailable(warning)
    prev = None
    errors: List[float] = []
    magnitudes: List[float] = []
    feature_errors: List[float] = []
    frame_gaps: List[int] = []
    local_ratios: List[float] = []
    excluded_cut_pairs = 0
    cuts = cut_indices or []
    tracked_features = 0
    abrupt: List[int] = []
    for index, frame in selected:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
        if prev is not None:
            if any(abs(index - cut) <= 1 or abs(previous_index - cut) <= 1
                   for cut in cuts):
                excluded_cut_pairs += 1
                prev = gray
                previous_index = index
                continue
            frame_gaps.append(max(1, index - previous_index))
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(float(np.median(mag)))
            warped = _warp_previous(prev, flow)
            residual_image = cv2.absdiff(warped, gray)
            errors.append(float(np.mean(residual_image)) / 255.0)
            # Compare the upper tail of local residual tiles to each pair's
            # own baseline. Global camera/codec residuals therefore contribute
            # little unless they are spatially concentrated.
            tile_values = _tile_means(residual_image.astype(np.float32), 8, 8)
            tile_center = float(np.median(tile_values))
            tile_spread = float(np.median(np.abs(tile_values - tile_center))) + 1.0
            local_ratios.append(float(np.percentile(tile_values, 90) - tile_center) /
                               tile_spread)
            points = cv2.goodFeaturesToTrack(
                prev, maxCorners=120, qualityLevel=0.01, minDistance=5
            )
            if points is not None and len(points) >= 2:
                forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
                    prev, gray, points, None
                )
                if forward is not None and forward_status is not None:
                    valid = forward_status.ravel().astype(bool)
                    if int(np.count_nonzero(valid)) >= 2:
                        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                            gray, prev, forward[valid], None
                        )
                        if backward is not None and backward_status is not None:
                            backward_valid = backward_status.ravel().astype(bool)
                            original = points[valid][backward_valid]
                            round_trip = backward[backward_valid]
                            if len(original) >= 2:
                                feature_errors.append(
                                    float(np.mean(np.linalg.norm(original - round_trip, axis=2)))
                                )
                                tracked_features += int(len(original))
            if len(errors) > 5 and (robust_z(errors[-1], errors[:-1]) or 0) > 6:
                abrupt.append(index)
        prev = gray
        previous_index = index
    if not errors:
        return {"status": "ok", "score": None, "confidence": 0.0,
                "metrics": {"pairs": 0}, "findings": [],
                "warnings": ["fewer than two sampled frames"]}
    median_error = median(errors) or 0.0
    median_motion = median(magnitudes) or 0.0
    # Motion-compensated residuals prevent ordinary camera motion from being
    # treated as manipulation.  Outlier rate captures within-video glitches.
    motion_ratio = median_error / (0.02 + median_motion / 18.0)
    residual_signal = clipped(motion_ratio / 2.5) or 0.0
    instability = clipped((mad(errors) or 0.0) / (median_error + 0.01) * 3.0) or 0.0
    outliers = outlier_fraction(errors)
    round_trip = median(feature_errors)
    feature_outliers = outlier_fraction(feature_errors)
    feature_support = clipped((len(feature_errors) - 8.0) / 20.0) or 0.0
    feature_signal = clipped(max(0.0, feature_outliers - 0.25) * 4.0) or 0.0
    feature_signal *= feature_support
    local_outliers = outlier_fraction(local_ratios)
    local_signal = clipped(max(0.0, local_outliers - 0.15) * 4.0) or 0.0
    instability_signal = clipped(max(0.0, instability - 0.75) * 2.0) or 0.0
    residual_signal = clipped(max(0.0, outliers - 0.20) * 4.0) or 0.0
    anomaly = float(clipped(0.25 * instability_signal +
                            0.20 * residual_signal +
                            0.35 * feature_signal +
                            0.20 * local_signal) or 0.0)
    consistency = 1.0 - anomaly
    # Optical flow is meaningful only over a bounded temporal interval.  A
    # sparse caller therefore gets an explicit abstention rather than a
    # large false residual from unrelated frames.
    median_gap = median(frame_gaps) or 1.0
    capture_fps = 30.0
    gap_supported = median_gap <= 2.0 * capture_fps
    confidence = min(1.0, len(errors) / 10.0) if gap_supported else 0.0
    if feature_errors:
        confidence *= min(1.0, tracked_features / 20.0)
    return {
        "status": "ok",
        "score": anomaly if len(errors) >= 3 and gap_supported else None,
        "confidence": confidence if len(errors) >= 3 and gap_supported else 0.0,
        "metrics": {
            "pairs": len(errors),
            "excluded_scene_cut_pairs": excluded_cut_pairs,
            "sampled_frame_count": len(selected),
            "median_frame_gap": median_gap,
            "temporal_interval_supported": gap_supported,
            "temporal_consistency": consistency,
            "anomaly_residual": anomaly,
            "median_warp_error": median_error,
            "mad_warp_error": mad(errors),
            "motion_compensated_residual": motion_ratio,
            "residual_outlier_fraction": outliers,
            "feature_round_trip_outlier_fraction": feature_outliers,
            "local_residual_outlier_fraction": local_outliers,
            "median_local_residual_ratio": median(local_ratios),
            "local_residual_inconsistency": (clipped(
                ((median(local_ratios) or 0.0) - 1.0) / 3.0) or 0.0),
            "feature_tracking_pairs": len(feature_errors),
            "median_feature_round_trip_error": round_trip,
            "tracked_features": tracked_features,
            "median_motion_magnitude": median_motion,
            "abrupt_transition_count": len(abrupt),
            "abrupt_transition_indices": abrupt,
        },
        "findings": (["within-video temporal residual inconsistency detected"]
                     if len(errors) >= 3 and gap_supported and anomaly >= 0.35 else []),
        "warnings": ([warning] if warning else []) +
                    ([] if gap_supported else ["sample spacing is too sparse for temporal scoring"]),
    }


def _warp_previous(previous, flow):
    import cv2  # type: ignore
    h, w = previous.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xx + flow[..., 0]).astype(np.float32)
    map_y = (yy + flow[..., 1]).astype(np.float32)
    return cv2.remap(previous, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _tile_means(image, rows: int, cols: int):
    height, width = image.shape[:2]
    tile_height, tile_width = height // rows, width // cols
    return np.asarray([
        np.mean(image[r * tile_height:(r + 1) * tile_height,
                      c * tile_width:(c + 1) * tile_width])
        for r in range(rows) for c in range(cols)
    ], dtype=np.float32)


def _unavailable(message: str) -> Dict[str, Any]:
    return {"status": "unavailable", "score": None, "metrics": {}, "findings": [], "warnings": [message]}
