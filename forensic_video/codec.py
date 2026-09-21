"""Container-level keyframe/GOP irregularity diagnostics via ffprobe."""

import json
import shutil
import subprocess
from typing import Any, Dict, List

import numpy as np

from .stats import clipped, mad, median


def analyze_codec(path: str) -> Dict[str, Any]:
    if not shutil.which("ffprobe"):
        return _unavailable("ffprobe is not installed")
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=key_frame,best_effort_timestamp_time,pict_type",
        "-of", "json", path,
    ]
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True,
                                   check=False, timeout=5)
        payload = json.loads(completed.stdout) if completed.returncode == 0 else {}
        frames = payload.get("frames", [])
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        frames = []
    if len(frames) < 8:
        return _unavailable("ffprobe returned too few video frames")
    key_indices = [i for i, frame in enumerate(frames)
                   if str(frame.get("key_frame", "0")) == "1"]
    intervals = np.diff(key_indices).astype(float).tolist() if len(key_indices) > 1 else []
    timestamps = []
    for frame in frames:
        try:
            timestamps.append(float(frame["best_effort_timestamp_time"]))
        except (KeyError, TypeError, ValueError):
            pass
    deltas = np.diff(timestamps).astype(float).tolist() if len(timestamps) > 1 else []
    interval_cv = float((mad(intervals) or 0.0) / (median(intervals) or 1.0))
    delta_jitter = float((mad(deltas) or 0.0) / (median(deltas) or 1.0))
    gop_signal = clipped(interval_cv * 4.0) or 0.0
    timing_signal = clipped(delta_jitter * 8.0) or 0.0
    score = float(clipped(0.7 * gop_signal + 0.3 * timing_signal) or 0.0)
    return {
        "status": "ok",
        "score": score if intervals and len(frames) >= 12 else None,
        "confidence": min(1.0, len(frames) / 60.0) if intervals else 0.0,
        "metrics": {
            "frames": len(frames),
            "keyframe_count": len(key_indices),
            "keyframe_indices": key_indices,
            "gop_intervals": intervals,
            "median_gop": median(intervals),
            "gop_interval_mad": mad(intervals),
            "gop_interval_cv": interval_cv,
            "timestamp_delta_jitter": delta_jitter,
            "gop_irregularity_signal": gop_signal,
            "timestamp_irregularity_signal": timing_signal,
        },
        "findings": (["irregular keyframe/GOP cadence detected"] if score >= 0.35 else []),
        "warnings": ["GOP cadence is codec/container evidence, not proof of editing"],
    }


def _unavailable(message: str) -> Dict[str, Any]:
    return {"status": "unavailable", "score": None, "confidence": 0.0,
            "metrics": {}, "findings": [], "warnings": [message]}
