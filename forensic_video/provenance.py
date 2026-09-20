"""File and container provenance inspection (never mutates the input)."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .models import VideoMetadata


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def inspect_provenance(path: str, include_hash: bool = True) -> VideoMetadata:
    target = Path(path)
    result = VideoMetadata(path=str(target.resolve()))
    if not target.exists():
        result.warnings.append("input does not exist")
        return result
    if not target.is_file():
        result.warnings.append("input is not a regular file")
        return result
    result.size_bytes = target.stat().st_size
    result.container = target.suffix.lower().lstrip(".") or None
    if include_hash:
        try:
            result.sha256 = _hash_file(target)
        except OSError as exc:
            result.warnings.append("sha256 unavailable: %s" % exc)
    try:
        import cv2  # type: ignore
    except ImportError:
        result.warnings.append("opencv is not installed; media metadata unavailable")
        return result
    capture = cv2.VideoCapture(str(target))
    if not capture.isOpened():
        result.warnings.append("opencv could not open input")
        return result
    try:
        result.width = _positive_int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        result.height = _positive_int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result.fps = _positive_float(capture.get(cv2.CAP_PROP_FPS))
        result.frame_count = _positive_int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if result.fps and result.frame_count:
            result.duration_seconds = result.frame_count / result.fps
        result.codec = _fourcc(capture.get(cv2.CAP_PROP_FOURCC))
        result.audio_present = _probe_audio(str(target))
        if result.audio_present is None:
            result.warnings.append("audio stream metadata unavailable; audio branch is best-effort")
        result.quality_metrics = _quality_metrics(result, str(target))
    finally:
        capture.release()
    return result


def _probe_audio(path: str) -> Optional[bool]:
    """Use optional ffprobe when present; never make provenance depend on it."""
    if not shutil.which("ffprobe"):
        return None
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            check=False, timeout=10,
        )
        return bool(completed.stdout.strip()) if completed.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _quality_metrics(metadata: VideoMetadata, path: str) -> dict:
    """Normalize codec/container quality confounders without a learned model."""
    metrics = {
        "container": metadata.container,
        "codec": metadata.codec,
        "resolution_pixels": ((metadata.width or 0) * (metadata.height or 0)),
        "fps_valid": bool(metadata.fps and metadata.fps > 0),
        "frame_count_valid": bool(metadata.frame_count and metadata.frame_count > 0),
    }
    if metadata.size_bytes and metadata.frame_count and metrics["resolution_pixels"]:
        bppf = metadata.size_bytes * 8.0 / (metadata.frame_count * metrics["resolution_pixels"])
        metrics["bits_per_pixel_frame"] = round(bppf, 8)
        metrics["quality_tier"] = (
            "low-bitrate" if bppf < 0.035 else
            "moderate-bitrate" if bppf < 0.12 else "high-bitrate"
        )
        metrics["global_compression_confounder"] = bppf < 0.12
    else:
        metrics["bits_per_pixel_frame"] = None
        metrics["quality_tier"] = "unknown"
        metrics["global_compression_confounder"] = True
    probe = _probe_stream(path)
    if probe:
        metrics.update({key: probe[key] for key in
                        ("codec_name", "pix_fmt", "bit_rate", "color_range")
                        if key in probe})
        metrics["ffprobe_available"] = True
    else:
        metrics["ffprobe_available"] = False
    return metrics


def _probe_stream(path: str) -> Optional[dict]:
    if not shutil.which("ffprobe"):
        return None
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt,bit_rate,color_range",
             "-of", "json", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            check=False, timeout=10,
        )
        payload = json.loads(completed.stdout) if completed.returncode == 0 else {}
        streams = payload.get("streams", [])
        return streams[0] if streams else None
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return None


def _positive_int(value: float) -> Optional[int]:
    return int(round(value)) if value and value > 0 else None


def _positive_float(value: float) -> Optional[float]:
    return float(value) if value and value > 0 else None


def _fourcc(value: float) -> Optional[str]:
    if not value:
        return None
    try:
        code = int(value)
        text = "".join(chr((code >> (8 * i)) & 255) for i in range(4))
        return text.strip("\x00 ") or None
    except (ValueError, OverflowError):
        return None
