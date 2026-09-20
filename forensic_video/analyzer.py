"""End-to-end deterministic orchestration, stability evaluation, and evidence localization."""

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict
import numpy as np

from .audio import analyze_audio
from .avsync import analyze_avsync
from .codec import analyze_codec
from .face import analyze_faces
from .frequency import analyze_frequency
from .fusion import aligned_event_consensus, fuse, fuse_evidence_sets
from .frames import read_sampled_frames
from .integrity import analyze_integrity
from .models import AnalysisResult, BranchResult
from .periodicity import analyze_periodicity
from .provenance import inspect_provenance
from .recompression import analyze_recompression
from .resampling import analyze_resampling
from .sampling import sample_video
from .scenes import detect_scene_cuts
from .spatial import analyze_spatial
from .temporal import analyze_temporal
from .wavelet import analyze_wavelet
from .production import build_analysis_outputs


@dataclass
class AnalysisConfig:
    samples: int = 48
    include_hash: bool = True
    max_frames_per_branch: int = 32
    jpeg_quality: int = 75
    stability: bool = False
    stability_windows: tuple = (10, 20, 30, 40)
    stability_budgets: tuple = (24, 48, 96)
    transcode_check: bool = False
    production_snapshot_root: str = ""
    specialist_checkpoint: str = ""
    binary_checkpoint: str = ""


def analyze_video(path: str, config: AnalysisConfig = None) -> AnalysisResult:
    config = config or AnalysisConfig()
    metadata = inspect_provenance(path, include_hash=config.include_hash)
    sampling = sample_video(path, metadata, requested=config.samples)
    scene = detect_scene_cuts(path)
    raw = _run_branches(path, sampling.indices, config, scene)
    branches = {name: BranchResult(name=name, status=value.get("status", "unavailable"),
                                    score=value.get("score"), confidence=value.get("confidence"),
                                    findings=value.get("findings", []),
                                    metrics=value.get("metrics", {}),
                                    limitations=value.get("warnings", []))
                for name, value in raw.items()}
    warnings = metadata.warnings + sampling.warnings
    for value in raw.values():
        warnings.extend(value.get("warnings", []))
    fusion = fuse(raw, metadata=metadata)
    fusion_sets = fuse_evidence_sets(raw, metadata=metadata)
    localization = _localize(raw, metadata, sampling.indices)
    stability = _evaluate_stability(path, metadata, config, scene)
    transcode_stability = _transcode_check(path, config, fusion)
    _apply_stability_consensus(fusion, stability)
    _apply_robustness_abstention(fusion, stability, transcode_stability)
    if transcode_stability.get("status") == "unstable":
        warnings.append("fusion changed materially after the controlled transcode")
    frame_pairs, frame_warning = read_sampled_frames(
        path, max(1, config.max_frames_per_branch), indices=sampling.indices
    )
    if frame_warning:
        warnings.append(frame_warning)
    report = AnalysisResult(
        schema_version="1.1", metadata=metadata, sampling=sampling,
        branches=branches, fusion=fusion, fusion_sets=fusion_sets,
        warnings=sorted(set(warnings)),
        localization=localization, stability=stability,
        transcode_stability=transcode_stability)
    report_dict = report.to_dict()
    report.analysis_outputs = build_analysis_outputs(
        report_dict,
        frames=[frame for _, frame in frame_pairs],
        snapshot_root=config.production_snapshot_root or None,
        specialist_checkpoint=config.specialist_checkpoint or None,
        binary_checkpoint=config.binary_checkpoint or None,
    )
    return report


def _apply_stability_consensus(fusion: Dict[str, Any],
                               stability: Dict[str, Any]) -> None:
    """Make consensus hints require robust support from the requested windows.

    A window pass can be unstable for a high score while still repeatedly
    observing low scores.  Keep those two cases separate: the former remains
    abstention-worthy, while the latter is useful negative evidence.
    """
    if not stability.get("enabled"):
        return
    support = bool(stability.get("provisional_positive_support"))
    negative_support = bool(stability.get("stable_negative_support"))
    fusion["stability_consensus"] = {
        "required": True,
        "supported": support,
        "stable_negative_support": negative_support,
        "evaluation_fraction": stability.get("consensus_evaluation_fraction", 0.0),
        "score_median": stability.get("consensus_score_median"),
        "score_range": stability.get("consensus_score_range"),
        "negative_score": stability.get("stable_negative_score"),
        "negative_groups": stability.get("stable_negative_groups", []),
        "positive_groups": stability.get("stable_positive_groups", []),
        "reason": stability.get("decision_reason"),
    }
    if fusion.get("provisional_positive") and not support:
        fusion["provisional_positive"] = False
        fusion["provisional_positive_reason"] = "window-consensus-not-stable"
    elif fusion.get("provisional_positive"):
        fusion["provisional_positive_reason"] = (
            "independent-groups-and-windows-stable-consensus"
        )
    if negative_support:
        fusion["reason_codes"] = sorted(set(
            fusion.get("reason_codes", []) + ["stable-negative-evidence"]
        ))


def _apply_robustness_abstention(fusion: Dict[str, Any], stability: Dict[str, Any],
                                 transcode_stability: Dict[str, Any]) -> None:
    """Prevent a fragile score from being presented as a production class."""
    score_range = stability.get("score_range")
    unstable = (stability.get("enabled") and score_range is not None and
                float(score_range) >= 0.35)
    transcode_unstable = transcode_stability.get("status") == "unstable"
    if not (unstable or transcode_unstable):
        return
    # A wide raw score range can be caused by an isolated high window.  Do not
    # discard a repeated, low, independent-group consensus in that case.
    if stability.get("stable_negative_support") and not transcode_unstable:
        fusion["reason_codes"] = sorted(set(
            fusion.get("reason_codes", []) + ["stable-negative-evidence"]
        ))
        return
    reasons = []
    if unstable:
        reasons.append("stability-instability")
    if transcode_unstable:
        reasons.append("transcode-instability")
    fusion["reason_codes"] = sorted(set(fusion.get("reason_codes", []) + reasons))
    # A score that is not robust to a routine window or codec/scale
    # perturbation must not be presented as a usable classification signal.
    if fusion.get("label") != "abstain-insufficient-evidence":
        fusion["label"] = "abstain-unstable-evidence"


def _run_branches(path: str, indices, config: AnalysisConfig, scene: Dict[str, Any],
                  include_stream_branches: bool = True) -> Dict[str, Dict[str, Any]]:
    """Run branches on one deterministic frame set.

    Window stability deliberately omits stream-global audio/GOP diagnostics;
    those cannot be localized to a window and would otherwise be counted as
    repeated independent evidence.
    """
    temporal = analyze_temporal(path, max_pairs=max(1, config.max_frames_per_branch * 4),
                                indices=indices, cut_indices=scene.get("cuts", []))
    recompression = analyze_recompression(path, quality=config.jpeg_quality,
                                          max_frames=config.max_frames_per_branch,
                                          indices=indices)
    frequency = analyze_frequency(path, max_frames=config.max_frames_per_branch,
                                  indices=indices)
    # Face-like geometry is intermittent and can be localized to a short
    # interval. Give it a denser bounded pass than whole-video diagnostics so
    # sparse endpoint sampling does not erase the only observable signal.
    face = analyze_faces(path, max_frames=max(48, config.max_frames_per_branch),
                         indices=indices)
    spatial = analyze_spatial(path, max_frames=config.max_frames_per_branch, indices=indices)
    integrity = analyze_integrity(path, max_frames=config.max_frames_per_branch, indices=indices)
    periodicity = analyze_periodicity(path, max_frames=config.max_frames_per_branch,
                                      indices=indices)
    wavelet = analyze_wavelet(path, max_frames=config.max_frames_per_branch,
                              indices=indices, cut_indices=scene.get("cuts", []))
    resampling = analyze_resampling(path, max_frames=config.max_frames_per_branch,
                                    indices=indices)
    raw: Dict[str, Dict[str, Any]] = {
        "scene": _scene_branch(scene),
        "temporal": temporal, "recompression": recompression,
        "frequency": frequency, "face": face, "spatial": spatial,
        "integrity": integrity, "periodicity": periodicity,
        "wavelet": wavelet, "resampling": resampling,
    }
    if include_stream_branches:
        raw.update({"codec": analyze_codec(path), "avsync": analyze_avsync(path),
                    "audio": analyze_audio(path)})
    for value in raw.values():
        value.setdefault("metrics", {})["sampled_frame_indices"] = list(indices)
    return raw


def _evaluate_stability(path: str, metadata, config: AnalysisConfig,
                        scene: Dict[str, Any]) -> Dict[str, Any]:
    if not config.stability:
        return {"enabled": False, "windows": [], "evaluations": [],
                "reason": "pass --stability to evaluate duration and budget sensitivity"}
    fps = float(metadata.fps or 0.0)
    count = int(metadata.frame_count or 0)
    duration = float(metadata.duration_seconds or 0.0)
    if fps <= 0 or count < 2 or duration <= 0:
        return {"enabled": True, "status": "unavailable", "windows": [],
                "evaluations": [], "reason": "frame timing metadata unavailable"}
    durations = sorted({max(1, int(value)) for value in config.stability_windows})
    budgets = sorted({max(4, int(value)) for value in config.stability_budgets})
    evaluations = []
    available_windows = []
    for seconds in durations:
        if duration + 1e-6 < seconds:
            continue
        # The centered interval avoids giving the first shot or an end slate
        # disproportionate influence, while each duration remains comparable.
        start_seconds = max(0.0, (duration - seconds) / 2.0)
        start = min(count - 1, int(round(start_seconds * fps)))
        end = min(count - 1, int(round((start_seconds + seconds) * fps)) - 1)
        if end <= start:
            continue
        available_windows.append(seconds)
        for budget in budgets:
            indices = np.linspace(start, end, min(budget, end - start + 1),
                                  dtype=int).tolist()
            window_scene = {"cuts": [cut for cut in scene.get("cuts", [])
                                     if start <= cut <= end]}
            raw = _run_branches(path, indices,
                                AnalysisConfig(samples=budget,
                                               max_frames_per_branch=budget,
                                               jpeg_quality=config.jpeg_quality),
                                window_scene, include_stream_branches=False)
            window_fusion = fuse(raw, metadata=metadata)
            score = window_fusion.get("score")
            evaluations.append({
                "window_seconds": seconds, "start_seconds": round(start / fps, 6),
                "end_seconds": round(end / fps, 6), "sample_budget": budget,
                "sample_count": len(indices), "score": score,
                "label": window_fusion.get("label"),
                "evidence_coverage": window_fusion.get("evidence_coverage", 0.0),
                "branch_scores": window_fusion.get("branch_scores", {}),
                "group_scores": window_fusion.get("group_scores", {}),
                "stable_anomaly_score": window_fusion.get("stable_anomaly_score"),
                "instability_score": window_fusion.get("instability_score"),
                "provisional_positive": bool(
                    window_fusion.get("provisional_positive", False)
                ),
            })
    scored_evaluations = [
        item for item in evaluations
        if item.get("score") is not None and _is_finite_number(item.get("score"))
    ]
    scores = [float(item["score"]) for item in scored_evaluations]
    consensus_policy = _stability_consensus_policy(
        len(scored_evaluations), len(available_windows), len(budgets)
    )
    if not scores:
        consensus_policy["evidence_group_count"] = 0
        return {"enabled": True, "status": "unavailable", "windows": available_windows,
                "budgets": budgets, "evaluations": evaluations,
                "reason": "no comparable window produced a supported score",
                "stable_negative_support": False,
                "provisional_positive_support": False,
                "exploratory_positive_support": False,
                "decision_reason": "no-comparable-window-score",
                "consensus_policy": consensus_policy}
    score_center = float(np.median(scores))
    score_mad = float(np.median(np.abs(np.asarray(scores) - score_center)))
    score_range = float(max(scores) - min(scores))
    consensus_scores = [
        float(item["stable_anomaly_score"])
        for item in scored_evaluations
        if item.get("provisional_positive")
        and item.get("stable_anomaly_score") is not None
    ]
    consensus_fraction = (len(consensus_scores) / float(len(scores))
                          if scores else 0.0)
    # Evaluate each evidence group independently before making a clip-wide
    # stability claim.  A single aggregate score can hide an absent group or
    # an isolated high window; group medians/MADs make that distinction
    # explicit and deterministic.
    group_values = {}
    for item in scored_evaluations:
        for group, value in (item.get("group_scores") or {}).items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                group_values.setdefault(group, []).append(numeric)
    consensus_policy["evidence_group_count"] = len(group_values)
    group_consensus = {}
    stable_negative_groups = []
    stable_positive_groups = []
    negative_window_consensus = 0
    positive_window_consensus = 0
    group_tolerance = 0.20
    negative_floor = 0.30
    positive_floor = 0.50
    for group, values in sorted(group_values.items()):
        group_center = float(np.median(values))
        group_mad = float(np.median(np.abs(np.asarray(values) - group_center)))
        group_range = float(max(values) - min(values))
        support_fraction = (
            len(values) / float(len(scored_evaluations))
            if scored_evaluations else 0.0
        )
        stable = bool(
            len(values) >= consensus_policy["required_evaluation_count"] and
            support_fraction >= consensus_policy["required_evaluation_fraction"] and
            group_range <= group_tolerance
        )
        group_consensus[group] = {
            "sample_count": len(values),
            "support_fraction": round(support_fraction, 6),
            "median": round(group_center, 6),
            "mad": round(group_mad, 6),
            "range": round(group_range, 6),
            "stable": stable,
            "negative": bool(stable and group_center <= negative_floor),
            "positive": bool(stable and group_center >= positive_floor),
        }
        if stable and group_center <= negative_floor:
            stable_negative_groups.append(group)
        if stable and group_center >= positive_floor:
            stable_positive_groups.append(group)
    for item in scored_evaluations:
        values = []
        for value in (item.get("group_scores") or {}).values():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                values.append(numeric)
        if len(values) < 2:
            continue
        window_group_center = float(np.median(values))
        window_group_range = max(values) - min(values)
        if window_group_center <= negative_floor and window_group_range <= group_tolerance:
            negative_window_consensus += 1
        if window_group_center >= positive_floor and window_group_range <= group_tolerance:
            positive_window_consensus += 1
    negative_fraction = (
        negative_window_consensus / float(len(scored_evaluations))
        if scored_evaluations else 0.0
    )
    positive_fraction = (
        positive_window_consensus / float(len(scored_evaluations))
        if scored_evaluations else 0.0
    )
    stable_negative_score = (
        float(np.median([
            group_consensus[group]["median"] for group in stable_negative_groups
        ]))
        if stable_negative_groups else None
    )
    stable_positive_score = (
        float(np.median([
            group_consensus[group]["median"] for group in stable_positive_groups
        ]))
        if stable_positive_groups else None
    )
    stable_negative_support = bool(
        len(stable_negative_groups) >= 2 and
        negative_window_consensus >= consensus_policy["required_evaluation_count"] and
        negative_fraction >= consensus_policy["required_evaluation_fraction"] and
        stable_negative_score is not None and stable_negative_score <= negative_floor
    )
    stable_positive_group_support = bool(
        len(stable_positive_groups) >= 2 and
        positive_window_consensus >= consensus_policy["required_evaluation_count"] and
        positive_fraction >= consensus_policy["required_evaluation_fraction"] and
        stable_positive_score is not None and stable_positive_score >= positive_floor
    )
    exploratory_positive_support = bool(
        stable_positive_group_support and score_range < 0.35
    )
    positive_support = bool(
        stable_positive_group_support
        and len(consensus_scores) >= consensus_policy["required_evaluation_count"]
        and consensus_fraction >= consensus_policy["required_evaluation_fraction"]
        and score_range < 0.35
    )
    if stable_negative_support:
        decision_reason = "stable-low-independent-group-consensus"
    elif positive_support:
        decision_reason = "stable-positive-independent-group-consensus"
    elif score_range >= 0.35:
        decision_reason = "window-scores-unstable"
    else:
        decision_reason = "insufficient-window-consensus"
    return {
        "enabled": True, "status": "ok", "windows": available_windows, "budgets": budgets,
        "evaluations": evaluations, "score_median": round(score_center, 6),
        "score_mad": round(score_mad, 6), "score_min": round(min(scores), 6),
        "score_max": round(max(scores), 6), "score_range": round(score_range, 6),
        "stability_index": round(max(0.0, 1.0 - min(1.0, score_range / 0.5)), 6),
        "consensus_evaluation_count": len(consensus_scores),
        "consensus_evaluation_fraction": round(consensus_fraction, 6),
        "consensus_score_median": round(float(np.median(consensus_scores)), 6)
        if consensus_scores else None,
        "consensus_score_range": round(max(consensus_scores) - min(consensus_scores), 6)
        if consensus_scores else None,
        "provisional_positive_support": positive_support,
        "exploratory_positive_support": exploratory_positive_support,
        "stable_negative_support": stable_negative_support,
        "stable_negative_score": (
            round(stable_negative_score, 6)
            if stable_negative_score is not None else None
        ),
        "stable_negative_groups": stable_negative_groups,
        "stable_positive_groups": stable_positive_groups,
        "negative_consensus_evaluation_count": negative_window_consensus,
        "negative_consensus_evaluation_fraction": round(negative_fraction, 6),
        "positive_consensus_evaluation_count": positive_window_consensus,
        "positive_consensus_evaluation_fraction": round(positive_fraction, 6),
        "consensus_policy": consensus_policy,
        "group_consensus": group_consensus,
        "decision_reason": decision_reason,
        "interpretation": (
            "stable low independent-group consensus; isolated windows may differ"
            if stable_negative_support else
            "stable across tested windows/budgets" if score_range < 0.20 else
            "window/budget-sensitive; do not treat as a clip-wide claim"
        ),
    }


def _is_finite_number(value: Any) -> bool:
    """Return whether a value can be represented as a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _stability_consensus_policy(
    scored_evaluation_count: int, feasible_window_count: int,
    budget_count: int = 0,
) -> Dict[str, Any]:
    """Describe the evidence quorum used by the stability pass.

    The old fixed ``>= 2`` observation rule made a valid one-window audit
    impossible, even when two independent groups agreed and the score range
    was exactly zero.  A quorum now scales with the number of scored
    evaluations: a normal pass still requires 60% of its observations
    (rounded up), while a reduced pass requires all of its single observation.
    This changes only evidence-count requirements; anomaly floors, group
    counts, disagreement handling, and transcode safeguards remain unchanged.
    """
    if scored_evaluation_count <= 0:
        required_count = 0
        required_fraction = 0.0
    else:
        required_count = max(1, int(math.ceil(scored_evaluation_count * 0.60)))
        required_fraction = required_count / float(scored_evaluation_count)
    return {
        "scored_evaluation_count": scored_evaluation_count,
        "feasible_window_count": feasible_window_count,
        "requested_budget_count": budget_count,
        "reduced_resource_scope": bool(budget_count and budget_count < 3),
        "required_evaluation_count": required_count,
        "required_evaluation_fraction": round(required_fraction, 6),
        "minimum_independent_groups": 2,
        "basis": (
            "ceil(0.60 * scored evaluations), bounded at one; "
            "at least two independent evidence groups remain required"
        ),
        "limited_window_scope": feasible_window_count < 2,
    }


def _localize(raw: Dict[str, Dict[str, Any]], metadata, indices) -> Dict[str, Any]:
    fps = float(getattr(metadata, "fps", None) or 0.0)
    duration = float(getattr(metadata, "duration_seconds", None) or 0.0)
    events = []
    event_keys = {
        "cut_indices": "scene-cut", "abrupt_transition_indices": "temporal-transition",
        "observation_indices": "face-region-observation",
        "keyframe_indices": "keyframe",
        "instability_indices": "wavelet-instability",
        "interpolation_event_indices": "interpolation-residual",
        "duplicate_residual_indices": "duplicate-residual",
    }
    for branch, value in raw.items():
        metrics = value.get("metrics") or {}
        for key, kind in event_keys.items():
            for frame_index in metrics.get(key, []) or []:
                try:
                    frame_index = int(frame_index)
                except (TypeError, ValueError):
                    continue
                events.append({
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 6) if fps else None,
                    "branch": branch, "kind": kind,
                })
    events.sort(key=lambda item: (item["frame_index"], item["branch"], item["kind"]))
    # Events from one branch are weak localization.  Report deterministic
    # cross-branch support in coarse time bins so nearby independent signals
    # can be audited without pretending that a location proves manipulation.
    bin_seconds = max(1.0, duration / 12.0) if duration else 1.0
    event_bins = {}
    for event in events:
        timestamp = event["timestamp_seconds"]
        bucket = int(timestamp / bin_seconds) if timestamp is not None else event["frame_index"]
        event_bins.setdefault(bucket, set()).add(event["branch"])
    consensus_events = []
    for bucket, branches in sorted(event_bins.items()):
        if len(branches) < 2:
            continue
        consensus_events.append({
            "bucket": bucket,
            "start_seconds": round(bucket * bin_seconds, 6),
            "end_seconds": round((bucket + 1) * bin_seconds, 6),
            "branches": sorted(branches),
            "branch_count": len(branches),
        })
    event_branch_count = len({event["branch"] for event in events})
    localization_consistency = {
        "status": ("cross-branch-supported" if consensus_events
                   else "single-branch-or-no-events"),
        "bin_seconds": round(bin_seconds, 6),
        "consensus_event_count": len(consensus_events),
        "supporting_branch_count": max(
            (item["branch_count"] for item in consensus_events), default=0
        ),
        "event_branch_count": event_branch_count,
        "support_fraction": round(
            (max((item["branch_count"] for item in consensus_events), default=0) /
             float(event_branch_count)) if event_branch_count else 0.0, 6
        ),
        "events": consensus_events,
    }
    segments = []
    for branch, value in sorted(raw.items()):
        if value.get("score") is None:
            continue
        segments.append({
            "branch": branch, "start_seconds": 0.0,
            "end_seconds": round(duration, 6) if duration else None,
            "score": value.get("score"), "confidence": value.get("confidence"),
            "sample_count": len(indices),
        })
    return {
        "events": events, "segments": segments,
        "event_count": len(events),
        "consistency": localization_consistency,
        "time_aligned_evidence": aligned_event_consensus(raw),
        "duration_seconds": round(duration, 6) if duration else None,
        "notes": ["events are diagnostic locations, not proof of manipulation",
                  "a branch score without a localized event is clip-wide evidence"],
    }


def _transcode_check(path: str, config: AnalysisConfig, source_fusion: Dict[str, Any]) -> Dict[str, Any]:
    if not config.transcode_check:
        return {"enabled": False, "status": "not-requested"}
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"enabled": True, "status": "unavailable", "reason": "ffmpeg is not installed"}
    source = Path(path)
    target = source.with_name(".%s.forensic-transcode.mp4" % source.stem)
    try:
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                   "-map", "0:v:0", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-an", str(target)]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, check=False, timeout=180)
        if completed.returncode != 0 or not target.exists():
            return {"enabled": True, "status": "unavailable",
                    "reason": "ffmpeg transcode failed"}
        transcoded = analyze_video(str(target), AnalysisConfig(
            samples=config.samples, max_frames_per_branch=config.max_frames_per_branch,
            jpeg_quality=config.jpeg_quality, include_hash=False))
        source_score = source_fusion.get("score")
        target_score = transcoded.fusion.get("score")
        delta = (abs(float(source_score) - float(target_score))
                 if source_score is not None and target_score is not None else None)
        return {
            "enabled": True, "status": ("stable" if delta is not None and delta < 0.20
                                         else "unstable" if delta is not None else "inconclusive"),
            "method": "ffmpeg-half-resolution-h264-crf28",
            "source_score": source_score, "transcoded_score": target_score,
            "score_delta": round(delta, 6) if delta is not None else None,
            "source_label": source_fusion.get("label"),
            "transcoded_label": transcoded.fusion.get("label"),
        }
    except (OSError, subprocess.SubprocessError):
        return {"enabled": True, "status": "unavailable", "reason": "ffmpeg transcode error"}
    finally:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass


def _scene_branch(data: Dict[str, Any]) -> Dict[str, Any]:
    metrics = data.get("metrics", {})
    cuts = data.get("cuts", [])
    return {"status": data.get("status", "unavailable"),
            # A cut is an edit boundary, not evidence that pixels were forged.
            "score": None,
            "confidence": 0.0,
            "metrics": dict(metrics, cut_indices=cuts),
            "findings": ["scene boundaries detected"] if cuts else [],
            "warnings": data.get("warnings", [])}
