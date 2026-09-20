"""Optional local-file HTTP API for the production frontend.

The API deliberately accepts local paths or uploaded files only. Remote URL
download remains outside this deterministic package; the CLI is still the
canonical local-file interface.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

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
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": "deterministic"}


@app.post("/analyze")
async def analyze(
    video_path: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    samples: int = Form(default=48),
    max_frames: int = Form(default=32),
) -> dict:
    if bool(video_path) == bool(file):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of video_path or file",
        )
    config = AnalysisConfig(
        samples=max(1, samples),
        max_frames_per_branch=max(1, max_frames),
        production_snapshot_root=os.getenv("NEUROFORGE_PRODUCTION_SNAPSHOT_ROOT", ""),
        specialist_checkpoint=os.getenv("NEUROFORGE_ROUTER_CHECKPOINT", ""),
        binary_checkpoint=os.getenv("NEUROFORGE_GENERAL_CHECKPOINT", ""),
    )
    temporary_path: Optional[str] = None
    try:
        if video_path:
            path = Path(video_path).expanduser().resolve()
            if not path.is_file():
                raise HTTPException(status_code=404, detail=f"Video does not exist: {path}")
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
