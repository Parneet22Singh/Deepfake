"""Analysis-only classification and evaluation helpers.

The analyzer intentionally emits an anomaly score rather than a calibrated
detector decision.  This module keeps thresholding out of production analysis:
it is a fixed, transparent way to audit a set of already-created reports.
Thresholds are not learned from the reports and are never used by
``analyze_video``.
"""

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_POSITIVE_THRESHOLD = 0.62
DEFAULT_NEGATIVE_THRESHOLD = 0.30
# A threshold decision must not turn materially discordant evidence groups
# into a class.  This matches the independent-group consensus policy in
# forensic_video.fusion rather than allowing the weighted mean to hide it.
MAX_CLASSIFICATION_GROUP_RANGE = 0.20

# The six labels with an evaluation claim in README.  The seventh fixed URL is
# retained as an unlabeled monitoring item rather than silently scored.
FIXED_TRUTH = {
    "original_deepfake": "positive",
    "new_deepfake": "positive",
    "new_ai_generated": "positive",
    "original_real": "negative",
    "new_real": "negative",
    "normal_video_7Dry": "negative",
}


def _validate_thresholds(positive_threshold: float, negative_threshold: float) -> None:
    if not 0.0 <= negative_threshold < positive_threshold <= 1.0:
        raise ValueError(
            "analysis thresholds must satisfy 0 <= negative < positive <= 1"
        )


def _classification_fusion(report: Mapping[str, Any],
                           evidence_set: str) -> Mapping[str, Any]:
    """Select an auditable evidence view for threshold classification."""
    if evidence_set in {"all", "directional"}:
        if evidence_set == "directional":
            branches = report.get("branches") or {}
            synthetic_scores = []
            scene_metrics = (branches.get("scene") or {}).get("metrics") or {}
            try:
                cut_rate = float(scene_metrics.get("cut_rate", 0.0))
            except (TypeError, ValueError):
                cut_rate = 0.0
            try:
                temporal_score = float((branches.get("temporal") or {}).get("score"))
            except (TypeError, ValueError):
                temporal_score = 0.0
            # Edit-heavy real footage can have no temporal branch score when
            # the decoder cannot produce a supported temporal anomaly signal.
            # The cut-rate signal is still sufficient to suppress generic
            # integrity/frequency artifacts; strong face evidence remains
            # independently eligible below.
            hard_edit_motion_confounded = (
                cut_rate >= 0.015
                and temporal_score >= 0.18
            )
            edit_heavy_without_temporal_score = (
                cut_rate >= 0.015 and temporal_score == 0.0
            )
            severe_edit_without_temporal_score = (
                cut_rate >= 0.025 and temporal_score == 0.0
            )
            face = branches.get("face") or {}
            face_metrics = face.get("metrics") or {}
            try:
                face_score = float(face.get("score"))
                face_geometry = float(face_metrics.get("geometry_cv"))
                face_ambiguity = float(
                    face_metrics.get("candidate_ambiguity",
                                     face_metrics.get("median_candidates_per_frame"))
                )
                face_detection_rate = float(face_metrics.get("detection_rate"))
            except (TypeError, ValueError):
                face_score = face_geometry = face_ambiguity = None
                face_detection_rate = None
            # Reject broad broadcast/archival motion patterns: a face-like
            # track must have geometry variation, low candidate ambiguity, and
            # either a localized or unusually strong observation.
            if (
                not hard_edit_motion_confounded
                and
                face_score is not None
                and face_geometry is not None
                and face_ambiguity is not None
                and face_detection_rate is not None
                and (
                    (
                        face_score >= 0.75
                        and face_geometry >= 0.25
                        and face_detection_rate >= 0.50
                        and (
                            not severe_edit_without_temporal_score
                            or (
                                face_score >= 0.80
                                and face_geometry >= 0.25
                                and face_detection_rate >= 0.80
                            )
                        )
                    )
                    or (
                        face_score >= 0.70
                        and
                        face_geometry >= 0.23
                        and
                        face_ambiguity <= 2.5
                        and face_detection_rate <= 0.70
                        and not severe_edit_without_temporal_score
                    )
                    or (
                        # A moderately scored track can still be useful when
                        # it is observed almost continuously and has stable
                        # candidate selection.  This is intentionally
                        # limited to non-severe cut-heavy footage.
                        face_score >= 0.60
                        and face_geometry >= 0.28
                        and face_ambiguity <= 3.0
                        and face_detection_rate >= 0.85
                        and not severe_edit_without_temporal_score
                    )
                )
            ):
                synthetic_scores.append(face_score)
            for name in ("integrity", "frequency"):
                try:
                    score = float((branches.get(name) or {}).get("score"))
                except (TypeError, ValueError):
                    continue
                minimum = 0.45 if name == "integrity" else 0.65
                if (not hard_edit_motion_confounded
                        and not edit_heavy_without_temporal_score
                        and minimum <= score <= 1.0):
                    synthetic_scores.append(score)
            return {
                "score": max(synthetic_scores) if synthetic_scores else 0.0,
                "label": "directional-synthetic-evidence",
                "group_score_range": None,
                "evidence_coverage": len(synthetic_scores) / 2.0,
                "directional_branches": ["face", "integrity", "frequency"],
            }
        return report.get("fusion") or {}
    selected = (report.get("fusion_sets") or {}).get(evidence_set)
    return selected if isinstance(selected, Mapping) else {}


def _exploratory_consensus(stability: Mapping[str, Any],
                           fusion: Mapping[str, Any]) -> Dict[str, Any]:
    """Describe the explicitly non-production relaxed stability path."""
    policy = stability.get("consensus_policy") or {}
    try:
        feasible_windows = int(policy.get("feasible_window_count", 0))
    except (TypeError, ValueError):
        feasible_windows = 0
    reduced_budget = bool(policy.get("reduced_resource_scope"))
    if not reduced_budget and "requested_budget_count" in policy:
        try:
            reduced_budget = int(policy.get("requested_budget_count", 0)) < 3
        except (TypeError, ValueError):
            reduced_budget = False
    one_window = feasible_windows == 1
    scope = []
    if one_window:
        scope.append("one-feasible-window")
    if reduced_budget:
        scope.append("reduced-budget")
    try:
        score_range = float(stability.get("score_range"))
    except (TypeError, ValueError):
        score_range = None
    try:
        group_range = float(fusion.get("group_score_range"))
    except (TypeError, ValueError):
        group_range = None
    supported = bool(stability.get("exploratory_positive_support"))
    if not supported:
        try:
            positive_count = int(
                stability.get("positive_consensus_evaluation_count", 0)
            )
            required_count = int(
                policy.get("required_evaluation_count", 0)
            )
            positive_fraction = float(
                stability.get("positive_consensus_evaluation_fraction", 0.0)
            )
            required_fraction = float(
                policy.get("required_evaluation_fraction", 1.0)
            )
            positive_groups = stability.get("stable_positive_groups") or []
            positive_score = float(stability.get("stable_positive_score"))
            supported = bool(
                len(positive_groups) >= 2
                and positive_count >= required_count
                and positive_fraction >= required_fraction
                and positive_score >= 0.50
                and score_range is not None and score_range < 0.35
            )
        except (TypeError, ValueError):
            supported = False
    reasons = []
    if not scope:
        reasons.append("not-a-limited-resource-audit")
    if score_range is None:
        reasons.append("missing-window-score-range")
    elif score_range >= 0.35:
        reasons.append("window-score-instability")
    if group_range is not None and group_range > MAX_CLASSIFICATION_GROUP_RANGE:
        reasons.append("independent-evidence-disagrees")
    if not supported:
        reasons.append("insufficient-positive-group-consensus")
    eligible = bool(
        stability.get("enabled") and stability.get("status") == "ok"
        and scope and supported
        and (score_range is not None and score_range < 0.35)
        and not (group_range is not None
                 and group_range > MAX_CLASSIFICATION_GROUP_RANGE)
    )
    return {
        "eligible": eligible,
        "scope": scope,
        "reasons": reasons,
        "instability": {
            "score_range": stability.get("score_range"),
            "score_mad": stability.get("score_mad"),
            "stability_index": stability.get("stability_index"),
            "fusion_instability_score": fusion.get("instability_score"),
            "transcode_status": (
                (fusion.get("_transcode_stability") or {}).get("status")
            ),
            "transcode_score_delta": (
                (fusion.get("_transcode_stability") or {}).get("score_delta")
            ),
        },
        "note": (
            "exploratory analysis-only consensus; not calibrated and not a "
            "production classification"
        ),
    }


def _exploratory_ranking_diagnostic(
    fusion: Mapping[str, Any],
    score: Optional[float],
    reasons: Sequence[str] = (),
) -> Dict[str, Any]:
    """Describe ranking confidence without turning disagreement into a class."""
    try:
        coverage = max(0.0, min(1.0, float(fusion.get("evidence_coverage", 0.0))))
    except (TypeError, ValueError):
        coverage = 0.0
    try:
        group_range = max(0.0, min(1.0, float(fusion.get("group_score_range"))))
    except (TypeError, ValueError):
        group_range = None
    try:
        instability = max(
            0.0, min(1.0, float(fusion.get("instability_score", 0.0)))
        )
    except (TypeError, ValueError):
        instability = 0.0
    # This is deliberately a ranking confidence descriptor, not a calibrated
    # probability.  Disagreement lowers confidence but never removes a score
    # from the exploratory ranking.
    agreement = 1.0 - group_range if group_range is not None else 0.0
    confidence = coverage * agreement * (1.0 - instability)
    diagnostic_reasons = set(reasons)
    if group_range is None:
        diagnostic_reasons.add("missing-group-range")
    elif group_range > MAX_CLASSIFICATION_GROUP_RANGE:
        diagnostic_reasons.add("independent-evidence-disagrees")
    if coverage < 0.20:
        diagnostic_reasons.add("insufficient-evidence-coverage")
    votes = fusion.get("group_directional_votes") or {}
    summary = fusion.get("directional_vote_summary") or {}
    if summary.get("status") == "mixed-directional-votes":
        diagnostic_reasons.add("mixed-directional-votes")
    return {
        "score": score,
        "confidence": round(confidence, 6),
        "confidence_label": (
            "low"
            if confidence < 0.50 or "independent-evidence-disagrees" in diagnostic_reasons
            else "moderate"
        ),
        "evidence_coverage": round(coverage, 6),
        "group_score_range": (
            round(group_range, 6) if group_range is not None else None
        ),
        "directional_vote_summary": summary,
        "group_directional_votes": votes,
        "reasons": sorted(diagnostic_reasons),
        "analysis_only": True,
        "note": (
            "low-confidence exploratory anomaly ranking; disagreement remains "
            "an abstention reason and does not imply a positive or negative class"
        ),
    }


def classify_report(
    report: Mapping[str, Any],
    positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
    negative_threshold: float = DEFAULT_NEGATIVE_THRESHOLD,
    allow_provisional: bool = True,
    exploratory: bool = False,
    evidence_set: str = "all",
) -> Dict[str, Any]:
    """Return an auditable positive/negative/abstain analysis decision.

    A score is not actionable when the production fusion already abstained, a
    requested stability pass found material window sensitivity, or a requested
    transcode pass found a material score change.  Those checks deliberately
    happen before thresholding so robustness failures cannot be converted into
    a confident class.  A separate provisional-positive path is admitted only
    from the fusion's independent-group consensus hint; it is explicitly
    analysis-only and is reported through ``decision_kind``.  A requested
    stability pass may also provide a separate stable-negative consensus,
    which can preserve a low decision without treating an unstable positive
    score as actionable.
    """
    _validate_thresholds(positive_threshold, negative_threshold)
    if evidence_set not in {"all", "legacy", "new_additions", "directional"}:
        raise ValueError(
            "evidence_set must be all, legacy, new_additions, or directional"
        )
    fusion = _classification_fusion(report, evidence_set)
    score = fusion.get("score")
    if evidence_set == "directional":
        try:
            directional_score = float(score)
        except (TypeError, ValueError):
            directional_score = None
        if directional_score is None:
            return {
                "predicted": "abstain",
                "decision_kind": "directional-insufficient-evidence",
                "analysis_mode": "standard",
                "score": None,
                "abstain_reasons": ["missing-directional-evidence"],
                "evidence": [],
                "exploratory": {},
                "thresholds": {
                    "positive": float(positive_threshold),
                    "negative": float(negative_threshold),
                    "use": "analysis-only; not production calibration",
                },
                "evidence_set": evidence_set,
            }
        directional_label = (
            "synthetic" if directional_score >= positive_threshold else "original"
        )
        return {
            "predicted": (
                "positive" if directional_label == "synthetic" else "negative"
            ),
            "decision_kind": "directional-threshold",
            "analysis_mode": "directional-analysis",
            "score": directional_score,
            "abstain_reasons": [],
            "evidence": ["face-or-integrity-synthetic-support"],
            "directional_label": directional_label,
            "exploratory": {},
            "thresholds": {
                "positive": float(positive_threshold),
                "negative": float(negative_threshold),
                "use": "analysis-only; not production calibration",
            },
            "evidence_set": evidence_set,
        }
    reasons = []
    evidence = []
    decision_kind = "abstain"
    exploratory_info = _exploratory_consensus(
        report.get("stability") or {}, fusion
    ) if exploratory else {
        "eligible": False, "scope": [], "reasons": ["mode-disabled"],
        "instability": {}, "note": "standard analysis-only evaluation mode",
    }

    if str(fusion.get("label", "")).startswith("abstain"):
        reasons.append("production-fusion-abstention")
    stability = report.get("stability") or {}
    stable_negative_support = bool(stability.get("stable_negative_support"))
    stable_negative_score = stability.get("stable_negative_score")
    if stability.get("enabled") and stability.get("status") != "ok":
        reasons.append("stability-check-unavailable")
    if stability.get("status") == "ok":
        score_range = stability.get("score_range")
        if (score_range is not None and float(score_range) >= 0.35
                and not stable_negative_support):
            reasons.append("unstable-window-score")
        if (stability.get("enabled") and not stable_negative_support and
                not stability.get("provisional_positive_support", False) and
                not exploratory_info["eligible"]):
            # A group consensus from one sample pass is not enough when the
            # caller explicitly requested window robustness.
            reasons.append("window-consensus-not-stable")
        elif exploratory_info["eligible"]:
            evidence.append("exploratory-limited-window-consensus")
        if stable_negative_support:
            evidence.append("stable-negative-independent-group-consensus")
    transcode = report.get("transcode_stability") or {}
    exploratory_info["instability"]["transcode_status"] = transcode.get("status")
    exploratory_info["instability"]["transcode_score_delta"] = transcode.get(
        "score_delta"
    )
    if transcode.get("enabled") and transcode.get("status") != "stable":
        reasons.append(
            "unstable-after-transcode"
            if transcode.get("status") == "unstable"
            else "transcode-check-unavailable"
        )

    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        numeric_score = None
    if numeric_score is None:
        reasons.append("missing-score")
    elif not 0.0 <= numeric_score <= 1.0:
        reasons.append("score-out-of-range")

    # ``fusion.score`` is a weighted anomaly ranking.  When the report also
    # exposes per-group disagreement, a threshold hit is actionable only when
    # the independent groups agree within the same tolerance used by the
    # analysis-only provisional-consensus path.  Otherwise a high (or low)
    # mean can conceal one materially contradictory group.
    group_range = fusion.get("group_score_range")
    try:
        group_range = float(group_range)
    except (TypeError, ValueError):
        group_range = None
    if (group_range is not None and 0.0 <= group_range <= 1.0 and
            group_range > MAX_CLASSIFICATION_GROUP_RANGE):
        reasons.append("independent-evidence-disagrees")

    if exploratory:
        exploratory_info["ranking"] = _exploratory_ranking_diagnostic(
            fusion, numeric_score, reasons
        )

    if reasons:
        predicted = "abstain"
    elif numeric_score >= positive_threshold:
        predicted = "positive"
        decision_kind = "threshold-positive"
    elif allow_provisional and bool(fusion.get("provisional_positive")):
        # This is not a lower threshold: it is a separate, auditable
        # analysis-only path requiring independent-group consensus.
        predicted = "positive"
        decision_kind = "provisional-positive"
        evidence.append("independent-group-consensus")
    elif exploratory and exploratory_info["eligible"]:
        predicted = "positive"
        decision_kind = "provisional-positive"
        evidence.append("exploratory-limited-window-consensus")
    elif (stable_negative_support and numeric_score < positive_threshold and
          stable_negative_score is not None and
          float(stable_negative_score) <= negative_threshold):
        # A stable low consensus is stronger than an isolated high window,
        # but never overrides a score that independently clears the positive
        # threshold.
        predicted = "negative"
        decision_kind = "stable-negative-consensus"
    elif numeric_score <= negative_threshold:
        predicted = "negative"
        decision_kind = "threshold-negative"
    else:
        predicted = "abstain"
        decision_kind = "score-band-abstention"
        reasons.append("score-between-analysis-thresholds")

    if exploratory and predicted != "abstain":
        decision_kind = "exploratory-" + decision_kind
    return {
        "predicted": predicted,
        "decision_kind": decision_kind,
        "analysis_mode": "exploratory" if exploratory else "standard",
        "score": numeric_score,
        "abstain_reasons": sorted(set(reasons)),
        "evidence": sorted(set(evidence)),
        "exploratory": exploratory_info,
        "thresholds": {
            "positive": float(positive_threshold),
            "negative": float(negative_threshold),
            "use": "analysis-only; not production calibration",
        },
        "evidence_set": evidence_set,
    }


def _metric(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 6) if denominator else None


def _numeric_scores(
    reports: Mapping[str, Mapping[str, Any]],
    truth: Mapping[str, str],
    evidence_set: str = "all",
) -> List[Tuple[str, str, float]]:
    """Return labeled numeric scores without interpreting them as probabilities."""
    rows = []
    for name in sorted(reports):
        expected = truth.get(name)
        if expected not in {"positive", "negative"}:
            continue
        score = _classification_fusion(reports[name], evidence_set).get("score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if 0.0 <= score <= 1.0:
            rows.append((name, expected, score))
    return rows


def _exploratory_anomaly_ranking(
    reports: Mapping[str, Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Rank numeric anomaly scores, retaining low-confidence mixed evidence."""
    rows = []
    for name in sorted(reports):
        fusion = reports[name].get("fusion") or {}
        try:
            score = float(fusion.get("score"))
        except (TypeError, ValueError):
            continue
        if not 0.0 <= score <= 1.0:
            continue
        decision = decisions.get(name) or {}
        diagnostic = (decision.get("exploratory") or {}).get("ranking")
        if not diagnostic:
            diagnostic = _exploratory_ranking_diagnostic(
                fusion, score, fusion.get("reason_codes") or []
            )
        rows.append({
            "name": name,
            "score": round(score, 6),
            "confidence": diagnostic["confidence"],
            "confidence_label": diagnostic["confidence_label"],
            "evidence_coverage": diagnostic["evidence_coverage"],
            "group_score_range": diagnostic["group_score_range"],
            "directional_vote_summary": diagnostic["directional_vote_summary"],
            "reasons": diagnostic["reasons"],
            "abstain_reasons": sorted(decision.get("abstain_reasons") or []),
        })
    rows.sort(key=lambda item: (-item["score"], item["name"]))
    for index, item in enumerate(rows, start=1):
        item["rank"] = index
    report_count = len(reports)
    return {
        "items": rows,
        "coverage": {
            "ranked_report_count": len(rows),
            "report_count": report_count,
            "fraction": _metric(len(rows), report_count),
        },
        "note": (
            "analysis-only anomaly ordering; scores remain rankings, and "
            "directional disagreement lowers confidence without becoming a "
            "production or exploratory class"
        ),
    }


def _branch_score_rows(
    reports: Mapping[str, Mapping[str, Any]],
    truth: Mapping[str, str],
    branch: str,
) -> List[Tuple[str, str, float]]:
    """Return valid labeled scores for one branch in deterministic order."""
    rows = []
    for name in sorted(reports):
        expected = truth.get(name)
        if expected not in {"positive", "negative"}:
            continue
        report_branches = reports[name].get("branches") or {}
        branch_report = report_branches.get(branch) if isinstance(
            report_branches, Mapping
        ) else None
        score = branch_report.get("score") if isinstance(
            branch_report, Mapping
        ) else None
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if 0.0 <= score <= 1.0:
            rows.append((name, expected, score))
    return rows


def _branch_diagnostics(
    reports: Mapping[str, Mapping[str, Any]],
    truth: Mapping[str, str],
) -> Dict[str, Any]:
    """Expose per-branch rank separation without changing production fusion."""
    names = set()
    for report in reports.values():
        branches = report.get("branches") or {}
        if isinstance(branches, Mapping):
            names.update(branches)

    diagnostics = {}
    labeled_count = sum(
        truth.get(name) in {"positive", "negative"} for name in reports
    )
    for branch in sorted(names):
        rows = _branch_score_rows(reports, truth, branch)
        positive_scores = [
            score for _, expected, score in rows if expected == "positive"
        ]
        negative_scores = [
            score for _, expected, score in rows if expected == "negative"
        ]
        metrics = _rank_metrics(rows)
        positive_median = (
            round(float(median(positive_scores)), 6)
            if positive_scores else None
        )
        negative_median = (
            round(float(median(negative_scores)), 6)
            if negative_scores else None
        )
        metrics.update({
            "labeled_count": labeled_count,
            "positive_scores": positive_scores,
            "negative_scores": negative_scores,
            "missing_or_invalid_count": labeled_count - len(rows),
            "positive_median": positive_median,
            "negative_median": negative_median,
            "median_gap_positive_minus_negative": (
                round(positive_median - negative_median, 6)
                if positive_median is not None and negative_median is not None
                else None
            ),
        })
        diagnostics[branch] = metrics
    return diagnostics


def _rank_metrics(rows: Sequence[Tuple[str, str, float]]) -> Dict[str, Any]:
    """Calculate label-audit rank metrics with deterministic tie handling.

    These metrics deliberately use the raw anomaly ranking, including scores
    from reports that later abstain for robustness reasons.  The counts make
    that limitation explicit; thresholded operating points below remain
    robustness-aware.
    """
    positives = sum(expected == "positive" for _, expected, _ in rows)
    negatives = sum(expected == "negative" for _, expected, _ in rows)
    concordant = tied = discordant = 0
    for _, expected, score in rows:
        if expected != "positive":
            continue
        for _, other_expected, other_score in rows:
            if other_expected != "negative":
                continue
            if score > other_score:
                concordant += 1
            elif score == other_score:
                tied += 1
            else:
                discordant += 1
    pair_count = concordant + tied + discordant
    auroc = (
        round((concordant + 0.5 * tied) / pair_count, 6)
        if pair_count else None
    )

    # Average precision is the area under the stepwise precision-recall
    # curve.  Sorting by name after score keeps ties reproducible.
    ranked = sorted(rows, key=lambda row: (-row[2], row[0]))
    seen = positives_seen = 0
    precision_sum = 0.0
    for _, expected, _ in ranked:
        seen += 1
        if expected == "positive":
            positives_seen += 1
            precision_sum += positives_seen / seen
    average_precision = (
        round(precision_sum / positives, 6) if positives else None
    )
    return {
        "labeled_scored_count": len(rows),
        "positive_count": positives,
        "negative_count": negatives,
        "pair_count": pair_count,
        "concordant_pairs": concordant,
        "tied_pairs": tied,
        "discordant_pairs": discordant,
        "auroc_like": auroc,
        "average_precision_like": average_precision,
        "note": (
            "analysis-only rank metrics; scores are anomaly rankings, not "
            "probabilities, and robustness failures are not removed here"
        ),
    }


def _operating_point(
    reports: Mapping[str, Mapping[str, Any]],
    truth: Mapping[str, str],
    positive_threshold: float,
    negative_threshold: float,
    allow_provisional: bool = False,
    exploratory: bool = False,
) -> Dict[str, Any]:
    """Evaluate one robustness-aware positive/negative threshold pair."""
    tp = fp = tn = fn = abstentions = labeled = 0
    for name in sorted(reports):
        expected = truth.get(name)
        if expected not in {"positive", "negative"}:
            continue
        labeled += 1
        predicted = classify_report(
            reports[name], positive_threshold, negative_threshold,
            allow_provisional=allow_provisional,
            exploratory=exploratory,
        )["predicted"]
        if predicted == "abstain":
            abstentions += 1
        elif expected == "positive" and predicted == "positive":
            tp += 1
        elif expected == "positive":
            fn += 1
        elif predicted == "negative":
            tn += 1
        else:
            fp += 1
    classified = tp + fp + tn + fn
    sensitivity = _metric(tp, tp + fn)
    specificity = _metric(tn, tn + fp)
    return {
        "positive_threshold": round(float(positive_threshold), 6),
        "negative_threshold": round(float(negative_threshold), 6),
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "abstain": abstentions,
        },
        "coverage": _metric(classified, labeled),
        "abstention_rate": _metric(abstentions, labeled),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_positive_rate": _metric(fp, tn + fp),
        "precision": _metric(tp, tp + fp),
        "conditional_accuracy": _metric(tp + tn, classified),
        "balanced_accuracy": (
            round((sensitivity + specificity) / 2.0, 6)
            if sensitivity is not None and specificity is not None
            else None
        ),
    }


def _operating_points(
    reports: Mapping[str, Mapping[str, Any]],
    truth: Mapping[str, str],
    negative_threshold: float,
    score_rows: Sequence[Tuple[str, str, float]],
    allow_provisional: bool = False,
    exploratory: bool = False,
) -> List[Dict[str, Any]]:
    """Sweep positive thresholds while retaining an explicit abstain band."""
    # Include a compact regular grid and observed scores.  The observed
    # values expose every classification change without making output depend
    # on the order in which reports were supplied.
    candidates = {round(index / 20.0, 2) for index in range(21)}
    candidates.update(round(row[2], 6) for row in score_rows)
    candidates = sorted(
        threshold for threshold in candidates
        if negative_threshold < threshold <= 1.0
    )
    return [
        _operating_point(
            reports, truth, threshold, negative_threshold,
            allow_provisional=allow_provisional,
            exploratory=exploratory,
        )
        for threshold in candidates
    ]


def _consensus_operating_points(
    reports: Mapping[str, Mapping[str, Any]],
    truth: Mapping[str, str],
    negative_threshold: float,
    score_rows: Sequence[Tuple[str, str, float]],
    exploratory: bool = False,
) -> List[Dict[str, Any]]:
    """Sweep thresholds while retaining the separate consensus hint."""
    candidates = {round(index / 20.0, 2) for index in range(21)}
    candidates.update(round(row[2], 6) for row in score_rows)
    candidates = sorted(
        threshold for threshold in candidates
        if negative_threshold < threshold <= 1.0
    )
    return [
        _operating_point(
            reports, truth, threshold, negative_threshold,
            allow_provisional=True,
            exploratory=exploratory,
        )
        for threshold in candidates
    ]


def _forced_label(score: Any, threshold: Optional[float]) -> Optional[str]:
    """Bucket a numeric anomaly score without applying abstention safeguards."""
    if threshold is None:
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= value <= 1.0:
        return None
    return "synthetic" if value >= threshold else "original"


def _forced_metrics(items: Mapping[str, Mapping[str, Any]],
                    truth: Mapping[str, str],
                    threshold: float) -> Dict[str, Any]:
    """Report the intentionally forced threshold view separately."""
    counts = {"true_positive": 0, "false_positive": 0,
              "true_negative": 0, "false_negative": 0, "unscored": 0}
    for name, item in items.items():
        expected = truth.get(name)
        label = item.get("forced_threshold")
        if expected not in {"positive", "negative"} or label is None:
            counts["unscored"] += 1
        elif expected == "positive" and label == "synthetic":
            counts["true_positive"] += 1
        elif expected == "positive":
            counts["false_negative"] += 1
        elif label == "original":
            counts["true_negative"] += 1
        else:
            counts["false_positive"] += 1
    tp, fp = counts["true_positive"], counts["false_positive"]
    tn, fn = counts["true_negative"], counts["false_negative"]
    scored = tp + fp + tn + fn
    return {
        "threshold": float(threshold),
        "labels": {"below": "original", "at_or_above": "synthetic"},
        "confusion_matrix": counts,
        "coverage": _metric(scored, len([
            name for name in items if truth.get(name) in {"positive", "negative"}
        ])),
        "sensitivity": _metric(tp, tp + fn),
        "specificity": _metric(tn, tn + fp),
        "note": "forced score buckets are not calibrated authenticity judgments",
    }


def evaluate_reports(
    reports: Mapping[str, Mapping[str, Any]],
    truth: Optional[Mapping[str, str]] = None,
    positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
    negative_threshold: float = DEFAULT_NEGATIVE_THRESHOLD,
    exploratory: bool = False,
    forced_threshold: Optional[float] = None,
    evidence_set: str = "all",
) -> Dict[str, Any]:
    """Evaluate reports while keeping ranking and classification separate."""
    _validate_thresholds(positive_threshold, negative_threshold)
    if evidence_set not in {"all", "legacy", "new_additions", "directional"}:
        raise ValueError(
            "evidence_set must be all, legacy, new_additions, or directional"
        )
    if forced_threshold is not None and not 0.0 <= forced_threshold <= 1.0:
        raise ValueError("forced threshold must satisfy 0 <= threshold <= 1")
    truth = dict(FIXED_TRUTH if truth is None else truth)
    items = {}
    decisions = {}
    tp = fp = tn = fn = abstentions = 0
    provisional_positives = 0
    labeled = 0
    for name in sorted(reports):
        report = reports[name]
        decision = classify_report(
            report, positive_threshold, negative_threshold,
            exploratory=exploratory, evidence_set=evidence_set,
        )
        decisions[name] = decision
        if decision["decision_kind"].endswith("provisional-positive"):
            provisional_positives += 1
        expected = truth.get(name)
        if expected in {"positive", "negative"}:
            labeled += 1
            if decision["predicted"] == "abstain":
                abstentions += 1
            elif expected == "positive" and decision["predicted"] == "positive":
                tp += 1
            elif expected == "positive":
                fn += 1
            elif decision["predicted"] == "negative":
                tn += 1
            else:
                fp += 1
        items[name] = {
            "truth": expected,
            "score": decision["score"],
            "predicted": decision["predicted"],
            "directional_label": decision.get("directional_label"),
            "decision_kind": decision["decision_kind"],
            "analysis_mode": decision["analysis_mode"],
            "evidence": decision["evidence"],
            "abstain_reasons": decision["abstain_reasons"],
            "exploratory": decision["exploratory"],
            "exploratory_ranking": (
                (decision["exploratory"] or {}).get("ranking")
                if exploratory else None
            ),
            "fusion_label": _classification_fusion(report, evidence_set).get("label"),
            "stability": {
                "status": (report.get("stability") or {}).get("status"),
                "score_range": (report.get("stability") or {}).get("score_range"),
            },
            "transcode": {
                "status": (report.get("transcode_stability") or {}).get("status"),
                "score_delta": (report.get("transcode_stability") or {}).get(
                    "score_delta"
                ),
            },
            "forced_threshold": (
                _forced_label(_classification_fusion(report, evidence_set).get("score"),
                              forced_threshold)
                if forced_threshold is not None else None
            ),
            "forced_threshold_sets": (
                {
                    set_name: _forced_label(result.get("score"), forced_threshold)
                    for set_name, result in
                    (report.get("fusion_sets") or {}).items()
                }
                if forced_threshold is not None else {}
            ),
        }
    score_rows = _numeric_scores(reports, truth, evidence_set=evidence_set)
    positive_scores = [score for _, expected, score in score_rows
                       if expected == "positive"]
    negative_scores = [score for _, expected, score in score_rows
                       if expected == "negative"]
    ranking = {
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "positive_median": (
            round(float(median(positive_scores)), 6)
            if positive_scores
            else None
        ),
        "negative_median": (
            round(float(median(negative_scores)), 6)
            if negative_scores
            else None
        ),
    }
    if positive_scores and negative_scores:
        ranking["median_gap_positive_minus_negative"] = round(
            ranking["positive_median"] - ranking["negative_median"], 6
        )
    else:
        ranking["median_gap_positive_minus_negative"] = None
    ranking.update(_rank_metrics(score_rows))
    classified = tp + fp + tn + fn
    return {
        "schema_version": "evaluation-4",
        "analysis_mode": "exploratory" if exploratory else "standard",
        "evidence_set": evidence_set,
        "calibration": "none",
        "mode_note": (
            "exploratory mode is analysis-only, provisional, and not calibrated"
            if exploratory else
            "standard analysis-only evaluation; not calibrated"
        ),
        "thresholds": {
            "positive": float(positive_threshold),
            "negative": float(negative_threshold),
            "use": "analysis-only; not production calibration",
        },
        "forced_threshold": (
            {
                "value": float(forced_threshold),
                "below": "original",
                "at_or_above": "synthetic",
                "use": "analysis-only score bucket; deliberately ignores abstention",
            }
            if forced_threshold is not None else None
        ),
        "items": items,
        "ranking": ranking,
        "exploratory_ranking": (
            _exploratory_anomaly_ranking(reports, decisions)
            if exploratory else {
                "items": [],
                "coverage": {
                    "ranked_report_count": 0,
                    "report_count": len(reports),
                    "fraction": 0.0 if reports else None,
                },
                "note": "exploratory ranking disabled",
            }
        ),
        "branch_diagnostics": _branch_diagnostics(reports, truth),
        "operating_points": _operating_points(
            reports, truth, negative_threshold, score_rows,
            exploratory=exploratory,
        ),
        "consensus_operating_points": _consensus_operating_points(
            reports, truth, negative_threshold, score_rows,
            exploratory=exploratory,
        ),
        "metrics": {
            "labeled_count": labeled,
            "unlabeled_count": len(reports) - labeled,
            "confusion_matrix": {
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "abstain": abstentions,
            },
            "coverage": _metric(classified, labeled),
            "abstention_rate": _metric(abstentions, labeled),
            "provisional_positive_count": provisional_positives,
            "conditional_accuracy": _metric(tp + tn, classified),
            "sensitivity": _metric(tp, tp + fn),
            "specificity": _metric(tn, tn + fp),
            "balanced_accuracy": (
                round(
                    (_metric(tp, tp + fn) + _metric(tn, tn + fp)) / 2.0, 6
                )
                if (tp + fn) and (tn + fp)
                else None
            ),
            "forced_threshold": _forced_metrics(
                items, truth, forced_threshold
            ) if forced_threshold is not None else None,
        },
        "interpretation": (
            "Scores rank anomaly evidence; predicted classes are an analysis "
            "view with fixed thresholds. Abstentions are not errors in the "
            "conditional accuracy, and unlabeled items are excluded from "
            "confusion metrics."
        ),
    }


def _parse_item(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("items must use name=report.json")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("items must use name=report.json")
    return name, Path(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Analysis-only classification audit for forensic reports"
    )
    parser.add_argument(
        "--item",
        action="append",
        type=_parse_item,
        required=True,
        help="report item as name=report.json; repeat for each report",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.45,
        help="analysis-only positive threshold (not production calibration)",
    )
    parser.add_argument(
        "--negative-threshold",
        type=float,
        default=DEFAULT_NEGATIVE_THRESHOLD,
        help="analysis-only negative threshold (not production calibration)",
    )
    parser.add_argument(
        "--forced-threshold",
        type=float,
        default=None,
        help=(
            "analysis-only forced bucket: scores below this are original and "
            "scores at or above it are synthetic; ignores abstention"
        ),
    )
    parser.add_argument(
        "--evidence-set",
        choices=("all", "legacy", "new_additions", "directional"),
        default="all",
        help=(
            "evidence view used for classification; legacy excludes later "
            "diagnostic additions; directional uses face/integrity for "
            "synthetic support"
        ),
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help=(
            "allow provisional consensus for one-window/reduced-budget audits; "
            "analysis-only, not calibrated, and never overrides contradictions "
            "or transcode instability"
        ),
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)
    reports = {}
    for name, path in args.item:
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            # PowerShell's ``>`` redirection may produce UTF-16 on Windows.
            text = raw.decode("utf-16")
        reports[name] = json.loads(text)
    result = evaluate_reports(
        reports,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
        exploratory=args.exploratory,
        forced_threshold=args.forced_threshold,
        evidence_set=args.evidence_set,
    )
    print(
        json.dumps(
            result,
            indent=None if args.indent == 0 else args.indent,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
