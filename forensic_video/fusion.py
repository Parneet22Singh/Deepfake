"""Transparent evidence fusion with coverage, confidence, and abstention."""

import math
from typing import Any, Dict

from .stats import mad, median, percentile


_WEIGHTS = {"temporal": 1.25, "integrity": 1.15, "recompression": 1.0,
            "frequency": 0.9, "codec": 0.55, "face": 0.7, "spatial": 0.8,
            "periodicity": 0.75, "audio": 0.65, "avsync": 0.45,
            "wavelet": 0.8}
_GROUPS = {
    "temporal": "motion", "integrity": "motion",
    "recompression": "codec", "frequency": "codec", "codec": "codec",
    "face": "spatial", "spatial": "spatial",
    "periodicity": "periodicity", "audio": "audio", "avsync": "audio",
}
_CONSENSUS_SCORE_FLOOR = 0.50
_CONSENSUS_TOLERANCE = 0.20
# Scores below this boundary are treated as non-supportive observations when
# a separate set of independent branches has a narrow positive consensus.
# They remain visible in the ordinary group diagnostics and can still force
# abstention when they materially contradict the supported groups.
_POSITIVE_EVIDENCE_FLOOR = 0.35
# Directional votes are diagnostic only.  They intentionally use wider
# evidence bands than the evaluator's classification thresholds so that a
# middle score remains visibly uncertain rather than becoming a class.
_DIRECTIONAL_ANOMALY_FLOOR = 0.50
_DIRECTIONAL_NEGATIVE_CEILING = 0.30
# Codec-sensitive diagnostics are useful corroboration, but cannot establish
# manipulation on their own because ordinary encoding and scaling can produce
# the same residuals.
_QUALITY_SENSITIVE_GROUPS = frozenset({"codec"})

# Keep the historical evidence and later additions independently auditable.
# ``resampling`` is intentionally included in the new view even though its
# branch is diagnostic-only and cannot affect fusion.
LEGACY_BRANCHES = frozenset({
    "temporal", "recompression", "frequency", "face", "spatial",
    "integrity", "audio",
})
NEW_ADDITION_BRANCHES = frozenset({
    "periodicity", "codec", "avsync", "wavelet", "resampling",
})


def _finite_float(value):
    """Return a finite numeric value, or None for malformed JSON inputs."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def fuse(branches: Dict[str, Dict[str, Any]], metadata: Any = None) -> Dict[str, Any]:
    """Combine comparable anomaly signals, never treating missing evidence as 0."""
    candidates = {}
    reason_codes = []
    rejected_branches = {}
    quality = getattr(metadata, "quality_metrics", None) or (
        metadata.get("quality_metrics", {}) if isinstance(metadata, dict) else {})
    quality_confounder = bool(quality.get("global_compression_confounder"))
    if quality_confounder:
        reason_codes.append("global-compression-confounder")
    for name, value in branches.items():
        score = value.get("score")
        if value.get("status") != "ok" or score is None or name not in _WEIGHTS:
            continue
        try:
            score = float(score)
            if not math.isfinite(score):
                continue
        except (TypeError, ValueError):
            continue
        metrics = value.get("metrics") or {}
        if metrics.get("fusion_eligible") is False:
            continue
        observed = next(
            (metrics[key] for key in ("pairs", "frames", "windows") if key in metrics),
            None,
        )
        # Legacy/custom branches without confidence are admitted conservatively.
        if "confidence" not in value:
            confidence = 0.5
        elif value.get("confidence") is None:
            confidence = 0.0
        else:
            confidence = _finite_float(value.get("confidence"))
            if confidence is None:
                rejected_branches[name] = "invalid-confidence"
                continue
        raw_coverage = value.get("evidence_coverage", 1.0)
        if raw_coverage is None:
            coverage = 1.0
        else:
            coverage = _finite_float(raw_coverage)
            if coverage is None:
                rejected_branches[name] = "invalid-coverage"
                continue
        if observed is not None:
            observed_value = _finite_float(observed)
            if observed_value is None or observed_value < 0.0:
                rejected_branches[name] = "invalid-observation-count"
                continue
            coverage = min(coverage, min(1.0, observed_value / 8.0))
        reliability = max(0.0, min(1.0, confidence)) * max(0.0, min(1.0, coverage))
        if (quality_confounder
                and _GROUPS.get(name, name) in _QUALITY_SENSITIVE_GROUPS):
            reliability *= 0.35
        if reliability <= 0.0:
            continue
        candidates[name] = (max(0.0, min(1.0, score)), reliability)
    if rejected_branches:
        reason_codes.append("invalid-branch-reliability")

    if not candidates:
        return {
            "label": "abstain-insufficient-evidence", "score": None,
            "used_branches": [], "evidence_coverage": 0.0,
            "used_evidence_groups": [], "branch_scores": {},
            "branch_reliability": {}, "group_scores": {},
            "robust_group_median": None, "robust_group_mad": None,
            "group_score_range": None, "group_score_p10": None,
            "group_score_p90": None, "disagreement_penalty": 0.0,
            "positive_evidence_score": None,
            "positive_evidence_groups": [],
            "group_directional_votes": {},
            "group_support": {},
            "directional_vote_summary": {
                "positive_groups": [], "negative_groups": [], "neutral_groups": [],
                "positive_count": 0, "negative_count": 0, "neutral_count": 0,
                "status": "no-supported-groups",
            },
            "quality_sensitive_only": False,
            "positive_evidence_policy": {
                "score_floor": _POSITIVE_EVIDENCE_FLOOR,
                "max_group_range": _CONSENSUS_TOLERANCE,
                "minimum_independent_groups": 2,
                "quality_sensitive_groups": sorted(_QUALITY_SENSITIVE_GROUPS),
                "use": ("preserve supported positive consensus from dilution; "
                        "quality-sensitive groups cannot support it alone and "
                        "disagreement still forces abstention"),
            },
            "stable_anomaly_score": None, "instability_score": None,
            "provisional_positive": False,
            "provisional_positive_reason": "no-supported-groups",
            "group_consensus": {
                "status": "no-supported-groups", "group_count": 0,
                "consensus_group_count": 0, "consensus_fraction": 0.0,
            },
            "rejected_branches": dict(sorted(rejected_branches.items())),
            "time_aligned_evidence": aligned_event_consensus(branches),
            "reason_codes": sorted(set(
                reason_codes + (["invalid-branch-reliability"]
                                if rejected_branches else []) +
                ["no-supported-branch"]
            )),
            "rule": "abstain when no comparable branch has supported evidence",
        }
    # Branches in a group often measure the same pixels (for example JPEG
    # residual, DCT noise, and GOP cadence).  Count the strongest supported
    # member once rather than letting correlated implementations double count.
    grouped = {}
    for name, (value, reliability) in candidates.items():
        group = _GROUPS.get(name, name)
        grouped.setdefault(group, []).append((name, value, reliability))
    representatives = {}
    for group, values in grouped.items():
        representatives[group] = max(
            values, key=lambda item: item[1] * item[2] * _WEIGHTS[item[0]])
    group_directional_votes = {}
    for group in sorted(representatives):
        branch, value, reliability = representatives[group]
        if value >= _DIRECTIONAL_ANOMALY_FLOOR:
            direction, vote = "positive", 1
        elif value <= _DIRECTIONAL_NEGATIVE_CEILING:
            direction, vote = "negative", -1
        else:
            direction, vote = "neutral", 0
        strength = (
            max(0.0, min(1.0, abs(value - _DIRECTIONAL_ANOMALY_FLOOR) * 2.0))
            if direction != "neutral" else 0.0
        )
        group_directional_votes[group] = {
            "representative_branch": branch,
            "score": round(value, 6),
            "reliability": round(reliability, 6),
            "direction": direction,
            "vote": vote,
            "supported": bool(reliability >= 0.30),
            "support": round(reliability * strength, 6),
        }
    positive_vote_groups = [
        group for group, item in group_directional_votes.items()
        if item["direction"] == "positive"
    ]
    negative_vote_groups = [
        group for group, item in group_directional_votes.items()
        if item["direction"] == "negative"
    ]
    neutral_vote_groups = [
        group for group, item in group_directional_votes.items()
        if item["direction"] == "neutral"
    ]
    directional_vote_summary = {
        "positive_groups": positive_vote_groups,
        "negative_groups": negative_vote_groups,
        "neutral_groups": neutral_vote_groups,
        "positive_count": len(positive_vote_groups),
        "negative_count": len(negative_vote_groups),
        "neutral_count": len(neutral_vote_groups),
        "status": (
            "mixed-directional-votes"
            if positive_vote_groups and negative_vote_groups
            else "directionally-consistent"
            if positive_vote_groups or negative_vote_groups
            else "directionally-uncertain"
        ),
    }
    if positive_vote_groups and negative_vote_groups:
        reason_codes.append("mixed-directional-votes")
    denominator = sum(_WEIGHTS[representatives[group][0]] * representatives[group][2]
                     for group in grouped)
    score = sum(_WEIGHTS[name] * reliability * value
                for group in grouped
                for name, value, reliability in [representatives[group]]) / denominator
    total_weight = sum(max(_WEIGHTS.get(name, 0.0)
                           for name in candidates if _GROUPS.get(name, name) == group)
                       for group in grouped)
    coverage = denominator / total_weight
    independent = len(grouped)
    if any(len(values) > 1 for values in grouped.values()):
        reason_codes.append("correlated-branches-collapsed")
    # Two independent supported branches are the minimum for a combined label.
    # A one-branch report keeps its score for auditability but explicitly abstains.
    non_quality_branches = set(grouped) - _QUALITY_SENSITIVE_GROUPS
    if quality_confounder and not non_quality_branches:
        reason_codes.append("only-quality-sensitive-evidence")
    group_values = [representatives[group][1] for group in sorted(representatives)]
    disagreement = float((max(group_values) - min(group_values))
                         if len(group_values) > 1 else 0.0)
    robust_group_median = median(group_values)
    robust_group_mad = mad(group_values)
    consensus_tolerance = 0.35
    consensus_groups = [
        group for group in sorted(representatives)
        if abs(representatives[group][1] - robust_group_median)
        <= consensus_tolerance
    ]
    if len(representatives) < 2:
        consensus_status = "insufficient-independent-groups"
    elif len(consensus_groups) == len(representatives):
        consensus_status = "consistent"
    else:
        consensus_status = "mixed"
    # A weighted mean remains the primary score for auditability, but an
    # unsupervised disagreement penalty prevents one strong, codec-sensitive
    # group from looking like consensus.  No labels or fitted calibration are
    # involved.
    disagreement_penalty = max(0.0, min(1.0, (disagreement - 0.35) / 0.65))
    score *= (1.0 - 0.20 * disagreement_penalty)
    if disagreement >= 0.55:
        reason_codes.append("independent-evidence-disagrees")
    abstain = (independent < 2 or coverage < 0.20 or disagreement >= 0.80 or
               (quality_confounder and not non_quality_branches))
    # This is deliberately separate from the production label and its 0.62
    # anomaly boundary.  It is an analysis-only hint for stable evidence:
    # every independent group must support the same robust median.
    stable_anomaly_score = float(robust_group_median)
    instability_score = float(max(
        0.0,
        min(1.0, 0.60 * disagreement +
            0.40 * min(1.0, float(robust_group_mad) * 4.0)),
    ))
    stable_consensus_groups = [
        group for group in sorted(representatives)
        if abs(representatives[group][1] - stable_anomaly_score)
        <= _CONSENSUS_TOLERANCE
    ]
    # A low anomaly score is absence of support, not affirmative evidence that
    # the supported groups are clean.  Preserve a narrow consensus among
    # multiple supported positive groups instead of averaging it away with
    # unrelated low-signal groups.  This does not alter the disagreement or
    # abstention policy below: a materially contradictory group still keeps
    # the report from becoming a positive classification.
    positive_groups = [
        group for group in sorted(representatives)
        if representatives[group][1] >= _POSITIVE_EVIDENCE_FLOOR
        and representatives[group][2] >= 0.30
    ]
    quality_sensitive_positive_groups = [
        group for group in positive_groups
        if group in _QUALITY_SENSITIVE_GROUPS
    ]
    non_quality_positive_groups = [
        group for group in positive_groups
        if group not in _QUALITY_SENSITIVE_GROUPS
    ]
    quality_sensitive_only = bool(
        quality_sensitive_positive_groups and not non_quality_positive_groups
    )
    if quality_sensitive_only:
        reason_codes.append("quality-sensitive-evidence-only")
    positive_consensus = []
    # A high codec-sensitive score plus low-signal non-codec groups is a
    # routine encoding/scaling artifact, not positive manipulation evidence.
    # Keep all groups in the ordinary score and diagnostics, but only allow
    # quality-sensitive groups to participate in the positive-consensus
    # override when a non-quality group independently clears the floor.
    positive_consensus_candidates = (
        positive_groups if not quality_sensitive_only else non_quality_positive_groups
    )
    if len(positive_consensus_candidates) >= 2:
        positive_values = [
            representatives[group][1] for group in positive_consensus_candidates
        ]
        positive_center = float(median(positive_values))
        positive_consensus = [
            group for group in positive_consensus_candidates
            if abs(representatives[group][1] - positive_center)
            <= _CONSENSUS_TOLERANCE
        ]
    positive_evidence_score = None
    if (len(positive_consensus) >= 2
            and len(positive_consensus) == len(positive_consensus_candidates)):
        positive_denominator = sum(
            _WEIGHTS[representatives[group][0]] * representatives[group][2]
            for group in positive_consensus
        )
        if positive_denominator > 0.0:
            positive_evidence_score = sum(
                _WEIGHTS[representatives[group][0]]
                * representatives[group][2]
                * representatives[group][1]
                for group in positive_consensus
            ) / positive_denominator
            score = max(score, positive_evidence_score)
            reason_codes.append("supported-positive-evidence-preserved")
    provisional_positive = bool(
        not abstain
        and independent >= 2
        and coverage >= 0.40
        and stable_anomaly_score >= _CONSENSUS_SCORE_FLOOR
        and disagreement <= _CONSENSUS_TOLERANCE
        and len(stable_consensus_groups) == independent
        and all(representatives[group][2] >= 0.30 for group in representatives)
    )
    if quality_confounder and not non_quality_branches:
        provisional_positive = False
    if provisional_positive:
        provisional_reason = "independent-groups-stable-consensus"
    elif independent < 2:
        provisional_reason = "fewer-than-two-independent-groups"
    elif quality_confounder and not non_quality_branches:
        provisional_reason = "quality-sensitive-evidence-only"
    elif coverage < 0.40:
        provisional_reason = "insufficient-supported-coverage"
    elif stable_anomaly_score < _CONSENSUS_SCORE_FLOOR:
        provisional_reason = "robust-median-below-consensus-floor"
    elif disagreement > _CONSENSUS_TOLERANCE:
        provisional_reason = "independent-groups-do-not-agree"
    else:
        provisional_reason = "independent-groups-lack-reliability"
    if abstain:
        label = "abstain-insufficient-evidence"
        reason_codes.append("insufficient-independent-evidence")
    elif score < 0.30:
        label = "low-anomaly-signal"
    elif score < 0.62:
        label = "mixed-anomaly-signal"
    else:
        label = "high-anomaly-signal"
    return {
        "label": label,
        "score": round(score, 6),
        "used_branches": sorted(candidates),
        "used_evidence_groups": sorted(grouped),
        "evidence_coverage": round(min(1.0, coverage), 6),
        "branch_scores": {k: round(v[0], 6) for k, v in sorted(candidates.items())},
        "branch_reliability": {k: round(v[1], 6) for k, v in sorted(candidates.items())},
        "group_scores": {
            group: round(representatives[group][1], 6)
            for group in sorted(representatives)
        },
        "group_directional_votes": group_directional_votes,
        "group_support": {
            group: item["support"]
            for group, item in sorted(group_directional_votes.items())
        },
        "directional_vote_summary": directional_vote_summary,
        "robust_group_median": round(float(robust_group_median), 6)
        if robust_group_median is not None else None,
        "robust_group_mad": round(float(robust_group_mad), 6)
        if robust_group_mad is not None else None,
        "group_score_range": round(disagreement, 6),
        "group_score_p10": percentile(group_values, 0.10),
        "group_score_p90": percentile(group_values, 0.90),
        "disagreement_penalty": round(disagreement_penalty, 6),
        "positive_evidence_score": (
            round(float(positive_evidence_score), 6)
            if positive_evidence_score is not None else None
        ),
        "positive_evidence_groups": positive_consensus,
        "positive_evidence_policy": {
            "score_floor": _POSITIVE_EVIDENCE_FLOOR,
            "max_group_range": _CONSENSUS_TOLERANCE,
            "minimum_independent_groups": 2,
            "quality_sensitive_groups": sorted(_QUALITY_SENSITIVE_GROUPS),
            "use": ("preserve supported positive consensus from dilution; "
                    "quality-sensitive groups cannot support it alone and "
                    "disagreement still forces abstention"),
        },
        "quality_sensitive_only": quality_sensitive_only,
        "stable_anomaly_score": round(stable_anomaly_score, 6),
        "instability_score": round(instability_score, 6),
        "provisional_positive": provisional_positive,
        "provisional_positive_reason": provisional_reason,
        "provisional_positive_policy": {
            "use": "analysis-only; production remains an anomaly signal",
            "score_floor": _CONSENSUS_SCORE_FLOOR,
            "max_group_range": _CONSENSUS_TOLERANCE,
            "minimum_independent_groups": 2,
        },
        "group_consensus": {
            "status": consensus_status,
            "group_count": len(representatives),
            "consensus_group_count": len(consensus_groups),
            "consensus_fraction": round(
                len(consensus_groups) / len(representatives), 6
            ),
            "groups": consensus_groups,
            "tolerance": consensus_tolerance,
        },
        "rejected_branches": dict(sorted(rejected_branches.items())),
        "time_aligned_evidence": aligned_event_consensus(branches),
        "reason_codes": sorted(set(reason_codes)),
        "rule": ("weighted mean of supported classical anomaly scores; "
                 "confidence and coverage weight evidence; no ML calibration"),
    }


def aligned_event_consensus(branches: Dict[str, Dict[str, Any]],
                            tolerance_frames: int = 2) -> Dict[str, Any]:
    """Find coincident branch events without turning locations into labels.

    Whole-video branch scores have no temporal support.  This descriptor gives
    coincident, independently observed events more evidentiary weight for
    review while leaving the established production score and thresholds
    unchanged.
    """
    keys = (
        "instability_indices", "interpolation_event_indices",
        "duplicate_residual_indices", "abrupt_transition_indices",
        "observation_indices", "cut_indices",
    )
    events = []
    for branch, value in sorted(branches.items()):
        metrics = value.get("metrics") or {}
        if value.get("status", "ok") != "ok":
            continue
        for key in keys:
            for index in metrics.get(key, []) or []:
                try:
                    events.append((int(index), branch, key))
                except (TypeError, ValueError):
                    continue
    if not events:
        return {
            "status": "no-localized-events", "event_count": 0,
            "coincident_event_count": 0, "supporting_branch_count": 0,
            "score": None, "confidence": 0.0, "events": [],
            "limitations": ["whole-video scores have no temporal alignment"],
        }
    clusters = []
    for index, branch, kind in sorted(events):
        if not clusters or index - clusters[-1]["last"] > tolerance_frames:
            clusters.append({"first": index, "last": index,
                             "branches": {branch}, "events": [(index, branch, kind)]})
        else:
            cluster = clusters[-1]
            cluster["last"] = index
            cluster["branches"].add(branch)
            cluster["events"].append((index, branch, kind))
    coincident = []
    for cluster in clusters:
        if len(cluster["branches"]) < 2:
            continue
        coincident.append({
            "start_frame": cluster["first"], "end_frame": cluster["last"],
            "branches": sorted(cluster["branches"]),
            "branch_count": len(cluster["branches"]),
            "event_count": len(cluster["events"]),
        })
    max_branches = max((item["branch_count"] for item in coincident), default=0)
    score = (min(1.0, max_branches / 3.0) *
             min(1.0, len(coincident) / max(1.0, len(clusters))))
    return {
        "status": "cross-branch-supported" if coincident else "events-not-coincident",
        "event_count": len(events),
        "coincident_event_count": len(coincident),
        "supporting_branch_count": max_branches,
        "score": round(score, 6) if coincident else 0.0,
        "confidence": round(min(1.0, max_branches / 3.0 *
                               min(1.0, len(events) / 8.0)), 6),
        "events": coincident,
        "tolerance_frames": tolerance_frames,
        "limitations": [
            "coincidence supports review priority, not a manipulation claim",
            "events from a single branch are not cross-branch evidence",
        ],
    }


def fuse_evidence_sets(branches: Dict[str, Dict[str, Any]],
                       metadata: Any = None) -> Dict[str, Dict[str, Any]]:
    """Fuse historical and newer branches separately for audit comparison."""
    return {
        "legacy": fuse(
            {name: value for name, value in branches.items()
             if name in LEGACY_BRANCHES},
            metadata=metadata,
        ),
        "new_additions": fuse(
            {name: value for name, value in branches.items()
             if name in NEW_ADDITION_BRANCHES},
            metadata=metadata,
        ),
    }
