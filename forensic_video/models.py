"""Stable JSON-facing data models used by the package."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VideoMetadata:
    path: str
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    container: Optional[str] = None
    codec: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    audio_present: Optional[bool] = None
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SamplingResult:
    requested: int
    indices: List[int]
    timestamps: List[float]
    method: str
    coverage: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class BranchResult:
    name: str
    status: str = "ok"
    score: Optional[float] = None
    confidence: Optional[float] = None
    findings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    schema_version: str
    metadata: VideoMetadata
    sampling: SamplingResult
    branches: Dict[str, BranchResult]
    fusion: Dict[str, Any]
    fusion_sets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    localization: Dict[str, Any] = field(default_factory=dict)
    stability: Dict[str, Any] = field(default_factory=dict)
    transcode_stability: Dict[str, Any] = field(default_factory=dict)
    analysis_outputs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
