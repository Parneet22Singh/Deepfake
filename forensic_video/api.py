"""HTTP API for local files and direct YouTube URL analysis."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .analyzer import AnalysisConfig, analyze_video

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as exc:  # pragma: no cover - exercised only without api extra
    raise RuntimeError("Install the api extra to run forensic_video.api") from exc


app = FastAPI(title="Deterministic Video Forensics API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "NEUROFORGE_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080",
        ).split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _router_configuration() -> tuple[str, str, str]:
    """Resolve explicit router paths, then the standard local snapshot layout."""
    snapshot = os.getenv("NEUROFORGE_PRODUCTION_SNAPSHOT_ROOT", "")
    specialist = os.getenv("NEUROFORGE_ROUTER_CHECKPOINT", "")
    binary = os.getenv("NEUROFORGE_GENERAL_CHECKPOINT", "")
    router_flag = os.getenv("NEUROFORGE_ENABLE_ROUTERS", "").lower()
    routers_enabled = router_flag not in {"0", "false", "no", "off"}
    if not snapshot and routers_enabled:
        default_snapshot = (
            Path(__file__).resolve().parents[2]
            / "The-Neuroforge-production-final-year-snapshot-2026-09-10"
        )
        specialist_default = (
            default_snapshot / "training" / "runs"
            / "expanded-efficientnet-controlled-saved" / "router_best.pt"
        )
        binary_default = (
            default_snapshot / "training" / "runs"
            / "binary-authenticity-efficientnet" / "router_best.pt"
        )
        if specialist_default.is_file() and binary_default.is_file():
            snapshot = str(default_snapshot)
    if snapshot:
        root = Path(snapshot)
        if not specialist:
            specialist = str(
                root / "training" / "runs"
                / "expanded-efficientnet-controlled-saved" / "router_best.pt"
            )
        if not binary:
            binary = str(
                root / "training" / "runs"
                / "binary-authenticity-efficientnet" / "router_best.pt"
            )
    return snapshot, specialist, binary


def _is_youtube_url(value: str) -> bool:
    parsed = urlparse(value)
    host = parsed.netloc.lower().split(":", 1)[0]
    return parsed.scheme in {"http", "https"} and (
        host == "youtube.com"
        or host.endswith(".youtube.com")
        or host == "youtu.be"
    )


def _download_youtube(url: str, directory: str) -> Path:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "YouTube analysis requires the API extra: python -m pip install -e '.[api]'"
        ) from exc
    output_template = str(Path(directory) / "%(id)s.%(ext)s")
    options = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": {"node": {}},
        "remote_components": {"ejs": ["github"]},
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            downloaded = Path(downloader.prepare_filename(info))
            candidates = [downloaded, downloaded.with_suffix(".mp4")]
            if info.get("requested_downloads"):
                candidates.extend(
                    Path(item["filepath"])
                    for item in info["requested_downloads"]
                    if item.get("filepath")
                )
            result = next((candidate for candidate in candidates if candidate.is_file()), None)
    except Exception as exc:
        raise RuntimeError(f"YouTube download failed: {exc}") from exc
    if result is None:
        raise RuntimeError("YouTube download completed without a readable media file")
    return result


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": "deterministic"}


@app.post("/analyze")
async def analyze(
    video_path: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    youtube_url: Optional[str] = Form(default=None),
    samples: int = Form(default=8),
    max_frames: int = Form(default=4),
) -> dict:
    provided = [bool(video_path), bool(file), bool(youtube_url)]
    if sum(provided) != 1:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of video_path, file, or youtube_url",
        )
    snapshot, specialist, binary = _router_configuration()
    config = AnalysisConfig(
        samples=max(1, samples),
        max_frames_per_branch=max(1, max_frames),
        production_snapshot_root=snapshot,
        specialist_checkpoint=specialist,
        binary_checkpoint=binary,
    )
    temporary_path: Optional[str] = None
    temporary_directory: Optional[str] = None
    try:
        if video_path:
            path = Path(video_path).expanduser().resolve()
            if not path.is_file():
                raise HTTPException(status_code=404, detail=f"Video does not exist: {path}")
        elif youtube_url:
            if not _is_youtube_url(youtube_url):
                raise HTTPException(status_code=400, detail="Provide a valid YouTube URL")
            temporary_directory = tempfile.mkdtemp(prefix="forensic-youtube-")
            try:
                path = _download_youtube(youtube_url, temporary_directory)
            except RuntimeError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
        else:
            suffix = Path(file.filename or "upload.bin").suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                temporary_path = handle.name
                while chunk := await file.read(1024 * 1024):
                    handle.write(chunk)
            path = Path(temporary_path)
        return analyze_video(str(path), config).to_dict()
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)
        if temporary_directory:
            import shutil
            shutil.rmtree(temporary_directory, ignore_errors=True)
