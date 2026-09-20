"""Production-facing report views and optional protected-snapshot routers.

The deterministic engine remains authoritative.  Checkpoint-backed outputs are
advisory, disabled unless explicitly configured, and loaded directly from the
protected snapshot without copying or modifying checkpoint files.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .evaluation import classify_report


DEFAULT_SNAPSHOT_ROOT = (
    r"C:\Users\parne\Downloads\The-Neuroforge-production.worktrees"
    r"\The-Neuroforge-production-final-year-snapshot-2026-09-10"
)
ROUTER_MIN_CONFIDENCE = 0.70
ROUTER_MIN_MARGIN = 0.15
_TORCHVISION_COMPAT_LIBRARY: Any = None


def _finite_score(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0.0 <= number <= 1.0 else None


def _mean_scores(branches: Mapping[str, Any], names: Sequence[str]) -> float | None:
    scores = [
        score for name in names
        if (score := _finite_score((branches.get(name) or {}).get("score"))) is not None
    ]
    return round(sum(scores) / len(scores), 6) if scores else None


def _apply_router_policy(output: Mapping[str, Any]) -> dict[str, Any]:
    """Apply this repository's router gate without changing protected files."""
    result = dict(output)
    probabilities = result.get("probabilities")
    if not isinstance(probabilities, Mapping) or not probabilities:
        return result
    numeric = {
        str(label): float(value)
        for label, value in probabilities.items()
        if _finite_score(value) is not None
    }
    if not numeric:
        return result
    ranked = sorted(numeric.items(), key=lambda item: item[1], reverse=True)
    label, confidence = ranked[0]
    margin = confidence - ranked[1][1] if len(ranked) > 1 else confidence
    classified = (
        confidence >= ROUTER_MIN_CONFIDENCE
        and margin >= ROUTER_MIN_MARGIN
    )
    result.update({
        "status": "classified" if classified else "abstain",
        "label": label if classified else "unknown",
        "confidence": round(confidence, 4),
        "margin": round(margin, 4),
        "router_policy": {
            "minimum_confidence": ROUTER_MIN_CONFIDENCE,
            "minimum_margin": ROUTER_MIN_MARGIN,
            "source": "repository_gate; protected checkpoint unchanged",
        },
    })
    return result


def build_five_layer_output(report: Mapping[str, Any]) -> dict[str, Any]:
    """Project deterministic evidence into the protected snapshot's five layers."""
    branches = report.get("branches") or {}
    layers = [
        {
            "id": "layer1_provenance",
            "name": "Provenance and media integrity",
            "score": _mean_scores(branches, ("provenance", "codec", "recompression")),
            "evidence_branches": ["provenance", "codec", "recompression"],
        },
        {
            "id": "layer2_temporal",
            "name": "Temporal and motion consistency",
            "score": _mean_scores(branches, ("temporal", "scene", "periodicity")),
            "evidence_branches": ["temporal", "scene", "periodicity"],
        },
        {
            "id": "layer3_face_integrity",
            "name": "Face geometry and spatial integrity",
            "score": _mean_scores(branches, ("face", "integrity", "spatial")),
            "evidence_branches": ["face", "integrity", "spatial"],
        },
        {
            "id": "layer4_frequency_audio",
            "name": "Frequency, noise, and stream diagnostics",
            "score": _mean_scores(branches, ("frequency", "wavelet", "audio", "avsync")),
            "evidence_branches": ["frequency", "wavelet", "audio", "avsync"],
        },
        {
            "id": "layer5_fusion",
            "name": "Robust evidence fusion",
            "score": _finite_score((report.get("fusion") or {}).get("score")),
            "evidence_branches": ["fusion", "stability", "transcode_stability"],
        },
    ]
    return {
        "status": "available",
        "mode": "production_five_layer_deterministic_projection",
        "authoritative": False,
        "layers": layers,
        "label": (report.get("fusion") or {}).get("label"),
        "limitations": [
            "Five-layer output is an explainable projection of deterministic evidence.",
            "It is not a trained model prediction or calibrated probability.",
        ],
    }


def _safe_checkpoint(checkpoint: Optional[str], snapshot_root: str) -> Optional[str]:
    if not checkpoint:
        return None
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        raise ValueError("Production checkpoint paths must be absolute")
    resolved = path.resolve()
    root = Path(snapshot_root).expanduser()
    if not root.is_absolute():
        raise ValueError("Production snapshot root must be absolute")
    root = root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Production checkpoint must be inside the configured snapshot root") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Production checkpoint does not exist: {resolved}")
    return str(resolved)


def _load_snapshot_router(snapshot_root: str):
    helper_path = Path(snapshot_root) / "backend" / "forensic-service" / "routing_helpers.py"
    if not helper_path.is_file():
        raise FileNotFoundError(f"Protected routing helper does not exist: {helper_path}")
    module_name = "_neuroforge_protected_routing_helpers"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, helper_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load protected routing helper: {helper_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def _prepare_torchvision_compatibility() -> None:
    """Register the optional NMS schema needed by some CPU torchvision builds."""
    global _TORCHVISION_COMPAT_LIBRARY
    try:
        import torch
        try:
            torch.ops.torchvision.nms
        except AttributeError:
            _TORCHVISION_COMPAT_LIBRARY = torch.library.Library("torchvision", "DEF")
            for operator in ("nms", "qnms"):
                _TORCHVISION_COMPAT_LIBRARY.define(
                    f"{operator}(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
                )
    except ImportError:
        return


def run_optional_routers(
    frames: Sequence[Any],
    *,
    snapshot_root: Optional[str] = None,
    specialist_checkpoint: Optional[str] = None,
    binary_checkpoint: Optional[str] = None,
) -> dict[str, Any]:
    """Run configured snapshot routers without affecting deterministic fusion."""
    root = snapshot_root or os.getenv("NEUROFORGE_PRODUCTION_SNAPSHOT_ROOT", "")
    specialist = specialist_checkpoint or os.getenv("NEUROFORGE_ROUTER_CHECKPOINT", "")
    binary = binary_checkpoint or os.getenv("NEUROFORGE_GENERAL_CHECKPOINT", "")
    if not root:
        return {
            "specialist_three_class_router": {"status": "not_configured", "enabled": False},
            "binary_authenticity_router": {"status": "not_configured", "enabled": False},
        }
    if not Path(root).is_absolute():
        raise ValueError("NEUROFORGE_PRODUCTION_SNAPSHOT_ROOT must be absolute")
    _prepare_torchvision_compatibility()
    helper = _load_snapshot_router(root)
    outputs: dict[str, Any] = {}
    if specialist:
        safe_path = _safe_checkpoint(specialist, root)
        outputs["specialist_three_class_router"] = _apply_router_policy(
            helper.classify_with_router(list(frames), safe_path)
        )
    else:
        outputs["specialist_three_class_router"] = {
            "status": "not_configured", "enabled": False
        }
    if binary:
        safe_path = _safe_checkpoint(binary, root)
        outputs["binary_authenticity_router"] = _apply_router_policy(
            helper.classify_with_general_model(list(frames), safe_path)
        )
    else:
        outputs["binary_authenticity_router"] = {
            "status": "not_configured", "enabled": False
        }
    return outputs


def build_analysis_outputs(
    report: Mapping[str, Any],
    *,
    frames: Sequence[Any] = (),
    snapshot_root: Optional[str] = None,
    specialist_checkpoint: Optional[str] = None,
    binary_checkpoint: Optional[str] = None,
) -> dict[str, Any]:
    """Build all four stable report outputs."""
    face_metrics = (report.get("branches", {}).get("face") or {}).get("metrics") or {}
    try:
        face_detection_frames = int(face_metrics.get("detection_frames", 0))
    except (TypeError, ValueError):
        face_detection_frames = 0
    configured_root = snapshot_root or os.getenv("NEUROFORGE_PRODUCTION_SNAPSHOT_ROOT", "")
    if configured_root and face_detection_frames <= 0:
        no_face_router = {
            "status": "abstain",
            "enabled": True,
            "label": "unknown",
            "confidence": None,
            "margin": None,
            "abstention_reason": (
                "No reliable face detections were found in the sampled video; "
                "face-dependent routers are not applicable."
            ),
            "router_policy": {
                "minimum_confidence": ROUTER_MIN_CONFIDENCE,
                "minimum_margin": ROUTER_MIN_MARGIN,
                "face_evidence_required": True,
            },
        }
        router_outputs = {
            "specialist_three_class_router": dict(no_face_router),
            "binary_authenticity_router": dict(no_face_router),
        }
    else:
        try:
            router_outputs = run_optional_routers(
                frames,
                snapshot_root=snapshot_root,
                specialist_checkpoint=specialist_checkpoint,
                binary_checkpoint=binary_checkpoint,
            )
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
            error = {"status": "error", "enabled": False, "error": str(exc)}
            router_outputs = {
                "specialist_three_class_router": dict(error),
                "binary_authenticity_router": dict(error),
            }
    return {
        "deterministic_engine": {
            "status": "available",
            "authoritative": True,
            "schema_version": report.get("schema_version"),
            "fusion": report.get("fusion", {}),
            "branches": report.get("branches", {}),
        },
        "specialist_three_class_router": router_outputs[
            "specialist_three_class_router"
        ],
        "binary_authenticity_router": router_outputs[
            "binary_authenticity_router"
        ],
        "directional_analysis": classify_report(
            report, evidence_set="directional", allow_provisional=False
        ),
        "production_five_layer": build_five_layer_output(report),
    }


def reconcile_analysis_outputs(outputs: Mapping[str, Any]) -> dict[str, Any]:
    """Describe agreement between independent outputs without overriding them."""
    deterministic = outputs.get("deterministic_engine") or {}
    fusion = deterministic.get("fusion") or {}
    score = _finite_score(fusion.get("score"))
    fusion_label = str(fusion.get("label") or "")
    deterministic_signal = (
        "uncertain"
        if fusion_label.startswith("abstain-")
        else (
            "synthetic"
            if fusion_label == "high-anomaly-signal"
            or (score is not None and score >= 0.62)
            else "original"
            if fusion_label == "low-anomaly-signal"
            or (score is not None and score < 0.30)
            else "uncertain"
        )
    )
    sources: dict[str, str] = {"deterministic_engine": deterministic_signal}
    specialist = outputs.get("specialist_three_class_router") or {}
    if specialist.get("status") == "classified":
        label = str(specialist.get("label") or "")
        sources["specialist_three_class_router"] = (
            "synthetic" if label in {"ai_generated", "face_deepfake"} else
            "uncertain"
        )
    binary = outputs.get("binary_authenticity_router") or {}
    if binary.get("status") == "classified":
        label = str(binary.get("label") or "")
        sources["binary_authenticity_router"] = (
            "synthetic" if label == "synthetic" else
            "original" if label == "original" else "uncertain"
        )
    signals = {signal for signal in sources.values() if signal != "uncertain"}
    conflicts = [
        name for name, signal in sources.items()
        if signal != "uncertain" and len(signals) > 1
    ]
    if len(signals) == 1 and not conflicts and deterministic_signal != "uncertain":
        consensus = next(iter(signals))
        status = "agreed"
    elif conflicts:
        consensus = "review"
        status = "conflict"
    else:
        consensus = "review"
        status = "inconclusive"
    decision = consensus if status == "agreed" else "review"
    decision_reason = (
        "All classified outputs support the same direction."
        if status == "agreed"
        else "Conflicting or insufficiently informative outputs require human review."
    )
    return {
        "status": status,
        "consensus": consensus,
        "decision": decision,
        "decision_reason": decision_reason,
        "sources": sources,
        "conflicting_sources": conflicts,
        "policy": (
            "Deterministic fusion remains authoritative; router disagreement is "
            "reported as review and never silently resolved."
        ),
    }


def apply_reconciliation_to_fusion(
    fusion: dict[str, Any],
    reconciliation: Mapping[str, Any],
) -> None:
    """Attach a backend decision without changing the numerical evidence score."""
    fusion["decision"] = reconciliation.get("decision", "review")
    fusion["decision_status"] = reconciliation.get("status", "inconclusive")
    fusion["decision_reason"] = reconciliation.get(
        "decision_reason",
        "Insufficient cross-system agreement for a unified decision.",
    )
    if reconciliation.get("status") == "conflict":
        reason_codes = list(fusion.get("reason_codes") or [])
        if "cross-system-conflict" not in reason_codes:
            reason_codes.append("cross-system-conflict")
        fusion["reason_codes"] = sorted(set(reason_codes))
