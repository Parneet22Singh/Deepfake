"""Optional audio extraction and deterministic within-track diagnostics."""

import shutil
import subprocess
import wave
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .stats import clipped, mad, median, outlier_fraction


def analyze_audio(path: str, max_seconds: float = 300.0) -> Dict[str, Any]:
    samples, sample_rate, decoder, warning = _decode_audio(path, max_seconds)
    if samples is None or len(samples) == 0:
        return {
            "status": "unavailable", "score": None, "confidence": 0.0,
            "metrics": {}, "findings": [], "warnings": [warning],
        }
    window = max(1, int(sample_rate * 0.25))
    rms = []
    spectral = []
    clipping = []
    for start in range(0, len(samples) - window + 1, window):
        chunk = samples[start:start + window]
        rms.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
        clipping.append(float(np.mean(np.abs(chunk) >= 0.995)))
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        frequencies = np.fft.rfftfreq(len(chunk), 1.0 / sample_rate)
        spectral.append(float(np.sum(frequencies * spectrum) /
                             (np.sum(spectrum) + 1e-9) / max(1.0, sample_rate)))
    if not rms:
        return {"status": "ok", "score": None, "confidence": 0.0,
                "metrics": {"decoder": decoder, "duration_seconds": len(samples) / sample_rate},
                "findings": [], "warnings": ["audio shorter than diagnostic window"]}
    rms_median = float(median(rms) or 0.0)
    # Speech and music naturally vary in loudness.  Use adjacent, robust
    # discontinuities instead of mistaking ordinary dynamics for editing.
    rms_jumps = np.abs(np.diff(np.log(np.asarray(rms) + 1e-4))).tolist()
    spectral_jumps = np.abs(np.diff(np.asarray(spectral))).tolist()
    rms_inconsistency = clipped(max(0.0, (median(rms_jumps) or 0.0) - 0.15) * 3.0 +
                                (mad(rms_jumps) or 0.0) * 8.0) or 0.0
    spectral_inconsistency = clipped(max(0.0, (median(spectral_jumps) or 0.0) - 0.01) * 20.0 +
                                     (mad(spectral_jumps) or 0.0) * 30.0) or 0.0
    clipping_rate = float(np.mean(clipping))
    rms_jump_outliers = outlier_fraction(rms_jumps)
    spectral_jump_outliers = outlier_fraction(spectral_jumps)
    discontinuity = float(clipped(
        2.0 * max(0.0, rms_jump_outliers - 0.15) +
        2.0 * max(0.0, spectral_jump_outliers - 0.15) +
        max(0.0, clipping_rate - 0.01) * 20.0) or 0.0)
    anomaly = float(clipped(0.35 * rms_inconsistency +
                            0.25 * spectral_inconsistency +
                            0.40 * discontinuity) or 0.0)
    confidence = min(1.0, len(rms) / 8.0)
    return {
        "status": "ok",
        "score": anomaly if len(rms) >= 4 and discontinuity >= 0.25 else None,
        "confidence": confidence if len(rms) >= 4 and discontinuity >= 0.25 else 0.0,
        "metrics": {
            "decoder": decoder,
            "sample_rate": sample_rate,
            "channels": 1,
            "duration_seconds": len(samples) / float(sample_rate),
            "windows": len(rms),
            "median_rms": rms_median,
            "mad_rms": float(mad(rms) or 0.0),
            "median_rms_jump": float(median(rms_jumps) or 0.0),
            "median_spectral_jump": float(median(spectral_jumps) or 0.0),
            "rms_inconsistency": rms_inconsistency,
            "spectral_inconsistency": spectral_inconsistency,
            "rms_jump_outlier_fraction": rms_jump_outliers,
            "spectral_jump_outlier_fraction": spectral_jump_outliers,
            "discontinuity_support": discontinuity,
            "clipping_rate": clipping_rate,
            "rms_outlier_fraction": outlier_fraction(rms),
        },
        "findings": (["within-video audio inconsistency detected"]
                     if anomaly >= 0.35 and discontinuity >= 0.25 else []),
        "warnings": [],
    }


def _decode_audio(path: str, max_seconds: float) -> Tuple[Optional[np.ndarray], int, str, str]:
    """Prefer soundfile for direct files, then use ffmpeg without temp files."""
    try:
        import soundfile as sf  # type: ignore
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        mono = np.mean(data, axis=1)
        return mono[:int(max_seconds * rate)], int(rate), "soundfile", ""
    except (ImportError, OSError, RuntimeError, ValueError):
        pass
    if shutil.which("ffmpeg"):
        command = [
            "ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1",
            "-ar", "16000", "-t", str(float(max_seconds)), "-f", "f32le", "pipe:1",
        ]
        try:
            completed = subprocess.run(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, check=False,
                                       timeout=max(15.0, max_seconds / 2.0))
            if completed.returncode == 0 and completed.stdout:
                return np.frombuffer(completed.stdout, dtype=np.float32), 16000, "ffmpeg", ""
        except (OSError, subprocess.SubprocessError):
            pass
    # A WAV fallback keeps the minimum install useful when ffmpeg is absent.
    try:
        with wave.open(path, "rb") as stream:
            rate = stream.getframerate()
            frames = stream.readframes(int(max_seconds * rate))
            width = stream.getsampwidth()
            channels = stream.getnchannels()
            if width == 2:
                values = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
            elif width == 1:
                values = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
            else:
                return None, 0, "wave", "unsupported WAV sample width"
            values = values.reshape(-1, channels).mean(axis=1)
            return values, int(rate), "wave", ""
    except (wave.Error, OSError, ValueError):
        return None, 0, "none", "no decodable audio stream (install ffmpeg or soundfile)"
