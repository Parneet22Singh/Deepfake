"""Conservative audio/video stream timing checks using ffprobe metadata."""

import json
import shutil
import subprocess
from typing import Any, Dict

from .stats import clipped


def analyze_avsync(path: str) -> Dict[str, Any]:
    if not shutil.which("ffprobe"):
        return _unavailable("ffprobe is not installed")
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,start_time,duration,time_base", "-of", "json", path,
    ]
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True,
                                   check=False, timeout=15)
        payload = json.loads(completed.stdout) if completed.returncode == 0 else {}
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        payload = {}
    streams = payload.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video or not audio:
        return _unavailable("both reliable audio and video stream metadata are required")
    try:
        video_start, audio_start = float(video["start_time"]), float(audio["start_time"])
        video_duration, audio_duration = float(video["duration"]), float(audio["duration"])
    except (KeyError, TypeError, ValueError):
        return _unavailable("audio/video timestamps or durations are unavailable")
    offset = abs(video_start - audio_start)
    duration_delta = abs(video_duration - audio_duration)
    # Start-time offsets below 350 ms are common muxing/encoder behavior.  A
    # score is emitted only when both independent timing checks are large.
    signal = float(clipped((offset - 0.35) / 1.5) or 0.0)
    duration_signal = float(clipped((duration_delta - 0.5) / 3.0) or 0.0)
    score = float(clipped(0.65 * signal + 0.35 * duration_signal) or 0.0)
    supported = offset >= 0.35 and duration_delta >= 0.5
    return {
        "status": "ok",
        "score": score if supported else None,
        "confidence": 0.7 if supported else 0.0,
        "metrics": {
            "video_start_time": video_start,
            "audio_start_time": audio_start,
            "stream_start_offset": offset,
            "video_duration": video_duration,
            "audio_duration": audio_duration,
            "duration_delta": duration_delta,
            "timing_support": score,
        },
        "findings": ["audio/video timing mismatch detected"] if supported else [],
        "warnings": (["audio/video timing is within normal muxing tolerance"]
                     if not supported else
                     ["timing mismatch is a triage signal, not authenticity proof"]),
    }


def _unavailable(message: str) -> Dict[str, Any]:
    return {"status": "unavailable", "score": None, "confidence": 0.0,
            "metrics": {}, "findings": [], "warnings": [message]}
