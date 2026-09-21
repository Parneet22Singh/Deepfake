import json
import wave
import asyncio

import pytest
import numpy as np


cv2 = pytest.importorskip("cv2")

from forensic_video.analyzer import AnalysisConfig, analyze_video, _evaluate_stability
from forensic_video.synthetic import write_synthetic_video
from forensic_video.audio import analyze_audio
from forensic_video.integrity import analyze_integrity, normalize_content
from forensic_video.fusion import fuse
from forensic_video.fusion import aligned_event_consensus
from forensic_video.analyzer import _apply_robustness_abstention, _localize
from forensic_video.periodicity import _longest_run
from forensic_video.stats import outlier_fraction
from forensic_video.evaluation import classify_report, evaluate_reports
from forensic_video.production import (
    apply_reconciliation_to_fusion,
    build_analysis_outputs,
    reconcile_analysis_outputs,
    run_optional_routers,
)
from forensic_video.production import _apply_router_policy
from forensic_video.api import _router_configuration
from forensic_video import api as forensic_api


def test_synthetic_analysis_is_json_serializable(tmp_path):
    path = write_synthetic_video(str(tmp_path / "fixture.avi"), frames=18, scene_cut_at=9)
    report = analyze_video(path, AnalysisConfig(samples=8, max_frames_per_branch=8))
    data = report.to_dict()
    assert data["schema_version"] == "1.1"
    assert len(data["sampling"]["indices"]) > 0
    assert set(("scene", "temporal", "recompression", "frequency", "face", "audio",
                "spatial", "integrity", "periodicity", "codec", "avsync")) <= set(data["branches"])
    assert set(data["fusion_sets"]) == {"legacy", "new_additions"}
    assert set(data["analysis_outputs"]) == {
        "deterministic_engine",
        "specialist_three_class_router",
        "binary_authenticity_router",
        "directional_analysis",
        "production_five_layer",
    }
    assert data["analysis_outputs"]["deterministic_engine"]["authoritative"] is True
    assert len(data["analysis_outputs"]["production_five_layer"]["layers"]) == 5
    assert data["analysis_outputs"]["specialist_three_class_router"]["status"] == "not_configured"
    assert data["analysis_outputs"]["binary_authenticity_router"]["status"] == "not_configured"
    assert data["analysis_reconciliation"]["status"] == "inconclusive"
    json.dumps(data, allow_nan=False)


def test_api_discovers_standard_protected_snapshot(monkeypatch, tmp_path):
    worktree = tmp_path / "worktree" / "neuroforge-deterministic-forensics"
    snapshot = worktree.parent / "The-Neuroforge-production-final-year-snapshot-2026-09-10"
    specialist = snapshot / "training" / "runs" / "expanded-efficientnet-controlled-saved"
    binary = snapshot / "training" / "runs" / "binary-authenticity-efficientnet"
    specialist.mkdir(parents=True)
    binary.mkdir(parents=True)
    (specialist / "router_best.pt").write_bytes(b"checkpoint")
    (binary / "router_best.pt").write_bytes(b"checkpoint")
    monkeypatch.delenv("NEUROFORGE_PRODUCTION_SNAPSHOT_ROOT", raising=False)
    monkeypatch.delenv("NEUROFORGE_ROUTER_CHECKPOINT", raising=False)
    monkeypatch.delenv("NEUROFORGE_GENERAL_CHECKPOINT", raising=False)
    monkeypatch.setenv("NEUROFORGE_ENABLE_ROUTERS", "1")
    monkeypatch.setattr(
        forensic_api,
        "__file__",
        str(worktree / "forensic_video" / "api.py"),
    )
    root, specialist_path, binary_path = _router_configuration()
    assert root == str(snapshot)
    assert specialist_path == str(specialist / "router_best.pt")
    assert binary_path == str(binary / "router_best.pt")


def test_api_accepts_youtube_url_and_analyzes_downloaded_file(monkeypatch, tmp_path):
    downloaded = tmp_path / "youtube.mp4"
    downloaded.write_bytes(b"fixture")

    monkeypatch.setattr(
        forensic_api,
        "_download_youtube",
        lambda url, directory: downloaded,
    )
    monkeypatch.setattr(
        forensic_api,
        "analyze_video",
        lambda path, config: type("Report", (), {"to_dict": lambda self: {
            "metadata": {"path": path},
            "fusion": {"label": "low-anomaly-signal", "score": 0.1},
        }})(),
    )

    result = asyncio.run(forensic_api.analyze(
        video_path=None,
        file=None,
        youtube_url="https://www.youtube.com/watch?v=fixture",
        samples=48,
        max_frames=32,
    ))
    assert result["fusion"]["label"] == "low-anomaly-signal"
    assert result["metadata"]["path"] == str(downloaded)


def test_cross_system_reconciliation_surfaces_router_conflict():
    outputs = {
        "deterministic_engine": {
            "fusion": {"score": 0.18, "label": "low-anomaly-signal"},
        },
        "specialist_three_class_router": {
            "status": "classified", "label": "ai_generated",
        },
        "binary_authenticity_router": {
            "status": "classified", "label": "synthetic",
        },
    }
    result = reconcile_analysis_outputs(outputs)
    assert result["status"] == "conflict"
    assert result["consensus"] == "review"
    assert "deterministic_engine" in result["conflicting_sources"]
    assert "specialist_three_class_router" in result["conflicting_sources"]


def test_cross_system_reconciliation_agrees_only_with_deterministic_support():
    outputs = {
        "deterministic_engine": {
            "fusion": {"score": 0.80, "label": "high-anomaly-signal"},
        },
        "specialist_three_class_router": {
            "status": "classified", "label": "ai_generated",
        },
        "binary_authenticity_router": {
            "status": "abstain", "label": "unknown",
        },
    }
    result = reconcile_analysis_outputs(outputs)
    assert result["status"] == "agreed"
    assert result["consensus"] == "synthetic"


def test_reconciliation_preserves_explicit_fusion_abstention():
    result = reconcile_analysis_outputs({
        "deterministic_engine": {
            "fusion": {
                "score": 0.63,
                "label": "abstain-insufficient-evidence",
            },
        },
    })
    assert result["status"] == "inconclusive"
    assert result["decision"] == "review"


def test_content_class_router_does_not_claim_authenticity_agreement():
    outputs = {
        "deterministic_engine": {
            "fusion": {"score": 0.18, "label": "low-anomaly-signal"},
        },
        "specialist_three_class_router": {
            "status": "classified", "label": "biological_face",
        },
        "binary_authenticity_router": {
            "status": "abstain", "label": "unknown",
        },
    }
    result = reconcile_analysis_outputs(outputs)
    assert result["status"] == "agreed"
    assert result["consensus"] == "original"
    assert result["sources"]["specialist_three_class_router"] == "uncertain"


def test_conflict_becomes_review_decision_without_changing_score():
    fusion = {
        "score": 0.18,
        "label": "low-anomaly-signal",
        "reason_codes": [],
    }
    reconciliation = {
        "status": "conflict",
        "decision": "review",
        "decision_reason": "Conflicting outputs.",
    }
    apply_reconciliation_to_fusion(fusion, reconciliation)
    assert fusion["score"] == 0.18
    assert fusion["label"] == "low-anomaly-signal"
    assert fusion["decision"] == "review"
    assert "cross-system-conflict" in fusion["reason_codes"]


def test_router_policy_accepts_point_seven_confidence_with_margin():
    result = _apply_router_policy({
        "status": "abstain",
        "label": "unknown",
        "confidence": 0.74,
        "margin": 0.0,
        "probabilities": {
            "original": 0.74,
            "synthetic": 0.26,
        },
    })
    assert result["status"] == "classified"
    assert result["label"] == "original"
    assert result["router_policy"]["minimum_confidence"] == 0.70


def test_router_policy_retains_abstention_below_confidence():
    result = _apply_router_policy({
        "status": "abstain",
        "label": "unknown",
        "probabilities": {
            "original": 0.69,
            "synthetic": 0.31,
        },
    })
    assert result["status"] == "abstain"
    assert result["label"] == "unknown"


def test_optional_router_timeout_does_not_block_deterministic_path(monkeypatch, tmp_path):
    helper_dir = tmp_path / "backend" / "forensic-service"
    helper_dir.mkdir(parents=True)
    (helper_dir / "routing_helpers.py").write_text(
        "import time\n"
        "def classify_with_router(frames, checkpoint):\n"
        "    time.sleep(10)\n"
        "def classify_with_general_model(frames, checkpoint):\n"
        "    time.sleep(10)\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "router.pt"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setenv("NEUROFORGE_ROUTER_TIMEOUT_SECONDS", "0.1")
    outputs = run_optional_routers(
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        snapshot_root=str(tmp_path),
        specialist_checkpoint=str(checkpoint),
    )
    assert outputs["specialist_three_class_router"]["status"] == "error"
    assert "timed out" in outputs["specialist_three_class_router"]["error"]


def test_routers_abstain_when_video_has_no_face_evidence():
    outputs = build_analysis_outputs(
        {"branches": {"face": {"metrics": {"detection_frames": 0}}}},
        frames=[np.zeros((8, 8, 3), dtype=np.uint8)],
        snapshot_root="C:\\protected-snapshot",
    )
    for name in ("specialist_three_class_router", "binary_authenticity_router"):
        assert outputs[name]["status"] == "abstain"
        assert outputs[name]["label"] == "unknown"
        assert "No reliable face detections" in outputs[name]["abstention_reason"]


def test_no_face_quality_evidence_does_not_establish_synthetic_signal():
    result = fuse(
        {
            "face": {
                "status": "ok",
                "score": None,
                "metrics": {"detection_frames": 0},
            },
            "codec": {"status": "ok", "score": 0.70},
            "wavelet": {"status": "ok", "score": 0.62},
            "temporal": {"status": "ok", "score": 0.24},
        },
        metadata={"quality_metrics": {"global_compression_confounder": True}},
    )
    assert result["label"] == "abstain-insufficient-evidence"
    assert "non-face-quality-evidence-only" in result["reason_codes"]


def test_forced_threshold_view_separates_from_conservative_abstention():
    reports = {
        "low": {
            "fusion": {"score": 0.399, "label": "abstain-unstable-evidence"},
            "fusion_sets": {
                "legacy": {"score": 0.25},
                "new_additions": {"score": 0.55},
            },
        },
        "high": {
            "fusion": {"score": 0.4, "label": "abstain-insufficient-evidence"},
            "fusion_sets": {
                "legacy": {"score": 0.35},
                "new_additions": {"score": 0.45},
            },
        },
    }
    result = evaluate_reports(
        reports, truth={"low": "negative", "high": "positive"},
        forced_threshold=0.4,
    )
    assert result["items"]["low"]["predicted"] == "abstain"
    assert result["items"]["low"]["forced_threshold"] == "original"
    assert result["items"]["high"]["forced_threshold"] == "synthetic"
    assert result["items"]["high"]["forced_threshold_sets"] == {
        "legacy": "original", "new_additions": "synthetic",
    }
    assert result["metrics"]["forced_threshold"]["sensitivity"] == 1.0


def test_legacy_evidence_set_can_drive_threshold_classification():
    report = {
        "fusion": {"score": 0.70, "label": "high-anomaly-signal"},
        "fusion_sets": {
            "legacy": {"score": 0.20, "label": "low-anomaly-signal"},
            "new_additions": {"score": 0.70, "label": "high-anomaly-signal"},
        },
    }
    result = evaluate_reports(
        {"clip": report}, truth={"clip": "negative"},
        positive_threshold=0.45, evidence_set="legacy",
    )
    assert result["evidence_set"] == "legacy"
    assert result["items"]["clip"]["predicted"] == "negative"
    assert result["items"]["clip"]["score"] == 0.20


def test_directional_profile_uses_synthetic_specific_branches():
    reports = {
        "fake": {
            "branches": {
                "face": {"score": 0.9, "metrics": {
                    "geometry_cv": 0.5, "candidate_ambiguity": 1.0,
                    "detection_rate": 0.4,
                }},
                "integrity": {"score": 0.2},
            },
            "fusion": {"score": 0.3},
        },
        "real": {
            "branches": {
                "face": {"score": None, "metrics": {}},
                "integrity": {"score": 0.4},
            },
            "fusion": {"score": 0.8},
        },
    }
    result = evaluate_reports(
        reports, truth={"fake": "positive", "real": "negative"},
        positive_threshold=0.45, evidence_set="directional",
    )
    assert result["items"]["fake"]["predicted"] == "positive"
    assert result["items"]["fake"]["directional_label"] == "synthetic"
    assert result["items"]["real"]["predicted"] == "negative"
    assert result["items"]["real"]["directional_label"] == "original"


def test_directional_profile_suppresses_edit_motion_confounds():
    report = {
        "branches": {
            "scene": {"metrics": {"cut_rate": 0.02}},
            "temporal": {"score": 0.2},
            "face": {"score": 0.95, "metrics": {
                "geometry_cv": 0.5, "candidate_ambiguity": 1.0,
                "detection_rate": 0.4,
            }},
            "integrity": {"score": 0.8},
        },
        "fusion": {"score": 0.8},
    }
    result = evaluate_reports(
        {"edited": report}, truth={"edited": "negative"},
        positive_threshold=0.45, evidence_set="directional",
    )
    assert result["items"]["edited"]["predicted"] == "negative"


def test_directional_profile_suppresses_cut_heavy_integrity_without_temporal_score():
    report = {
        "branches": {
            "scene": {"metrics": {"cut_rate": 0.0208}},
            "temporal": {"score": None},
            "face": {"score": 0.65, "metrics": {
                "geometry_cv": 0.27, "candidate_ambiguity": 0.0,
                "detection_rate": 0.38,
            }},
            "integrity": {"score": 0.48},
            "frequency": {"score": 0.28},
        },
        "fusion": {"score": 0.57},
    }
    result = evaluate_reports(
        {"edited": report}, truth={"edited": "negative"},
        positive_threshold=0.45, evidence_set="directional",
    )
    assert result["items"]["edited"]["predicted"] == "negative"


def test_directional_profile_keeps_strong_face_support_in_cut_heavy_clip():
    report = {
        "branches": {
            "scene": {"metrics": {"cut_rate": 0.020}},
            "temporal": {"score": None},
            "face": {"score": 0.80, "metrics": {
                "geometry_cv": 0.34, "candidate_ambiguity": 3.0,
                "detection_rate": 0.90,
            }},
            "integrity": {"score": 0.50},
            "frequency": {"score": 0.70},
        },
        "fusion": {"score": 0.60},
    }
    result = evaluate_reports(
        {"synthetic": report}, truth={"synthetic": "positive"},
        positive_threshold=0.45, evidence_set="directional",
    )
    assert result["items"]["synthetic"]["predicted"] == "positive"


def test_directional_profile_suppresses_face_support_on_severe_cut_heavy_clip():
    report = {
        "branches": {
            "scene": {"metrics": {"cut_rate": 0.029}},
            "temporal": {"score": None},
            "face": {"score": 0.87, "metrics": {
                "geometry_cv": 0.41, "candidate_ambiguity": 2.0,
                "detection_rate": 0.75,
            }},
            "integrity": {"score": 0.53},
            "frequency": {"score": 0.73},
        },
        "fusion": {"score": 0.64},
    }
    result = evaluate_reports(
        {"edited": report}, truth={"edited": "negative"},
        positive_threshold=0.45, evidence_set="directional",
    )
    assert result["items"]["edited"]["predicted"] == "negative"


def test_directional_profile_accepts_high_coverage_moderate_face_support():
    report = {
        "branches": {
            "scene": {"metrics": {"cut_rate": 0.018}},
            "temporal": {"score": None},
            "face": {"score": 0.64, "metrics": {
                "geometry_cv": 0.31, "candidate_ambiguity": 3.0,
                "detection_rate": 0.92,
            }},
            "integrity": {"score": 0.53},
            "frequency": {"score": 0.48},
        },
        "fusion": {"score": 0.57},
    }
    result = evaluate_reports(
        {"synthetic": report}, truth={"synthetic": "positive"},
        positive_threshold=0.45, evidence_set="directional",
    )
    assert result["items"]["synthetic"]["predicted"] == "positive"


def test_sampling_and_scene_are_deterministic(tmp_path):
    path = write_synthetic_video(str(tmp_path / "fixture.avi"), frames=20, scene_cut_at=10)
    first = analyze_video(path, AnalysisConfig(samples=7, max_frames_per_branch=5)).to_dict()
    second = analyze_video(path, AnalysisConfig(samples=7, max_frames_per_branch=5)).to_dict()
    assert first["sampling"] == second["sampling"]
    assert first["branches"]["scene"]["metrics"]["cut_indices"] == second["branches"]["scene"]["metrics"]["cut_indices"]


def test_duplicate_burst_fixture_is_deterministic_and_exercises_periodicity(tmp_path):
    from forensic_video.periodicity import analyze_periodicity

    path = write_synthetic_video(
        str(tmp_path / "duplicate-burst.avi"),
        frames=24,
        scene_cut_at=12,
        manipulation="duplicate-burst",
    )
    indices = list(range(24))
    first = analyze_periodicity(path, max_frames=24, indices=indices)
    second = analyze_periodicity(path, max_frames=24, indices=indices)
    assert first == second
    periodicity = first
    assert periodicity["metrics"]["longest_duplicate_run"] >= 2
    assert periodicity["score"] is not None


def test_multiscale_wavelet_and_resampling_diagnostics_are_deterministic(tmp_path):
    from forensic_video.resampling import analyze_resampling
    from forensic_video.wavelet import analyze_wavelet

    path = write_synthetic_video(
        str(tmp_path / "interpolation.avi"),
        frames=24, scene_cut_at=12, manipulation="interpolation",
    )
    indices = list(range(24))
    wavelet = analyze_wavelet(path, max_frames=24, indices=indices)
    wavelet_again = analyze_wavelet(path, max_frames=24, indices=indices)
    residuals = analyze_resampling(path, max_frames=24, indices=indices)
    assert wavelet == wavelet_again
    assert wavelet["metrics"]["multiscale_levels"] == 2
    assert residuals["metrics"]["fusion_eligible"] is False
    assert residuals["metrics"]["interpolation_event_count"] >= 1


def test_time_aligned_evidence_requires_coincident_independent_events():
    aligned = aligned_event_consensus({
        "temporal": {"status": "ok", "metrics": {"abrupt_transition_indices": [10]}},
        "wavelet": {"status": "ok", "metrics": {"instability_indices": [11]}},
        "resampling": {"status": "ok",
                       "metrics": {"interpolation_event_indices": [40]}},
    })
    assert aligned["status"] == "cross-branch-supported"
    assert aligned["supporting_branch_count"] == 2
    assert aligned["coincident_event_count"] == 1


def test_positive_consensus_synthetic_codec_controls_are_not_manipulation_support(tmp_path):
    """Exercise clean, temporal duplication, re-encoding, and scale changes."""
    import cv2

    clean_path = write_synthetic_video(
        str(tmp_path / "clean.avi"), frames=24, scene_cut_at=12
    )
    duplicate_path = write_synthetic_video(
        str(tmp_path / "duplicate-burst.avi"),
        frames=24,
        scene_cut_at=12,
        manipulation="duplicate-burst",
    )
    capture = cv2.VideoCapture(clean_path)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 35]
        )
        assert ok
        frames.append(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
    capture.release()
    assert len(frames) == 24

    recompressed_path = tmp_path / "recompressed.avi"
    writer = cv2.VideoWriter(
        str(recompressed_path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (96, 64)
    )
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()

    resized_path = tmp_path / "resized-transcoded.avi"
    writer = cv2.VideoWriter(
        str(resized_path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (48, 32)
    )
    assert writer.isOpened()
    for frame in frames:
        writer.write(cv2.resize(frame, (48, 32), interpolation=cv2.INTER_AREA))
    writer.release()

    config = AnalysisConfig(samples=16, max_frames_per_branch=16)
    reports = {
        name: analyze_video(str(path), config).to_dict()
        for name, path in {
            "clean": clean_path,
            "duplicate-burst": duplicate_path,
            "recompressed": recompressed_path,
            "resized-transcoded": resized_path,
        }.items()
    }
    scores = {name: report["fusion"]["score"] for name, report in reports.items()}
    assert scores == {
        "clean": 0.276891,
        "duplicate-burst": 0.444054,
        "recompressed": 0.276049,
        "resized-transcoded": 0.350462,
    }

    assert reports["clean"]["fusion"]["quality_sensitive_only"] is True
    assert reports["recompressed"]["fusion"]["quality_sensitive_only"] is True
    assert reports["resized-transcoded"]["fusion"]["quality_sensitive_only"] is True
    for name in ("clean", "recompressed", "resized-transcoded"):
        fusion = reports[name]["fusion"]
        assert fusion["positive_evidence_score"] is None
        assert "quality-sensitive-evidence-only" in fusion["reason_codes"]
        assert fusion["label"] != "high-anomaly-signal"

    duplicate_fusion = reports["duplicate-burst"]["fusion"]
    assert duplicate_fusion["quality_sensitive_only"] is False
    assert duplicate_fusion["positive_evidence_groups"] == ["codec", "periodicity"]
    assert duplicate_fusion["positive_evidence_score"] == 0.444054


def test_supported_periodicity_consensus_is_not_diluted_by_low_signal(tmp_path):
    config = AnalysisConfig(samples=16, max_frames_per_branch=16)
    clean_path = write_synthetic_video(
        str(tmp_path / "clean.avi"), frames=24, scene_cut_at=12
    )
    manipulated_path = write_synthetic_video(
        str(tmp_path / "duplicate-burst.avi"),
        frames=24,
        scene_cut_at=12,
        manipulation="duplicate-burst",
    )
    clean = analyze_video(clean_path, config).fusion
    manipulated = analyze_video(manipulated_path, config).fusion

    assert clean["score"] == 0.276891
    assert manipulated["positive_evidence_groups"] == ["codec", "periodicity"]
    assert manipulated["positive_evidence_score"] == 0.444054
    assert manipulated["score"] == 0.444054
    assert manipulated["score"] > clean["score"]
    # The positive consensus improves ranking, but the low motion group still
    # prevents a provisional classification because all independent groups do
    # not agree within the production consensus tolerance.
    assert manipulated["provisional_positive"] is False


def test_fusion_abstains_when_only_one_weak_branch_is_available():
    result = fuse({
        "temporal": {"status": "ok", "score": 0.8, "confidence": 0.2,
                     "metrics": {"pairs": 2}},
        "scene": {"status": "ok", "score": None, "metrics": {}},
    })
    assert result["label"] == "abstain-insufficient-evidence"
    assert result["score"] == 0.8
    assert result["used_branches"] == ["temporal"]


def test_fusion_reports_quality_confounder_reason():
    metadata = {"quality_metrics": {"global_compression_confounder": True}}
    result = fuse({
        "frequency": {"status": "ok", "score": 0.8, "confidence": 1.0,
                      "metrics": {"frames": 16}},
        "recompression": {"status": "ok", "score": 0.7, "confidence": 1.0,
                          "metrics": {"frames": 16}},
    }, metadata=metadata)
    assert result["label"] == "abstain-insufficient-evidence"
    assert "global-compression-confounder" in result["reason_codes"]
    assert "only-quality-sensitive-evidence" in result["reason_codes"]


def test_fusion_rejects_nonfinite_branch_reliability_without_nan_output():
    result = fuse({
        "temporal": {"status": "ok", "score": 0.8, "confidence": 1.0,
                     "metrics": {"pairs": 16}},
        "spatial": {"status": "ok", "score": 0.9, "confidence": float("nan"),
                    "metrics": {"frames": 16}},
        "frequency": {"status": "ok", "score": 0.7, "confidence": 1.0,
                      "evidence_coverage": "not-a-number",
                      "metrics": {"frames": 16}},
        "face": {"status": "ok", "score": 0.6, "confidence": 1.0,
                 "metrics": {"frames": "not-a-count"}},
    })
    assert result["used_branches"] == ["temporal"]
    assert result["rejected_branches"] == {
        "face": "invalid-observation-count",
        "frequency": "invalid-coverage",
        "spatial": "invalid-confidence",
    }
    assert "invalid-branch-reliability" in result["reason_codes"]
    json.dumps(result, allow_nan=False)


def test_scene_cuts_are_excluded_from_temporal_metrics(tmp_path):
    from forensic_video.temporal import analyze_temporal
    path = write_synthetic_video(str(tmp_path / "cut.avi"), frames=20, scene_cut_at=10)
    result = analyze_temporal(path, max_pairs=20, indices=list(range(20)),
                              cut_indices=[10])
    assert result["metrics"]["excluded_scene_cut_pairs"] >= 1


def test_outlier_fraction_is_stable_for_constant_series():
    assert outlier_fraction([1.0, 1.0, 1.0, 1.0]) == 0.0
    assert outlier_fraction([1.0, 1.0, 1.0, 10.0]) == 0.25


def test_audio_wav_diagnostics_are_reproducible(tmp_path):
    path = tmp_path / "tone.wav"
    rate = 8000
    samples = (0.1 * np.sin(np.arange(rate * 2) * 2 * np.pi * 440 / rate) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(samples.tobytes())
    first = analyze_audio(str(path))
    second = analyze_audio(str(path))
    assert first["metrics"]["decoder"] in {"wave", "soundfile", "ffmpeg"}
    assert first == second


def test_transform_normalization_and_leave_one_region_out_are_deterministic(tmp_path):
    import cv2
    frame = np.full((80, 160, 3), 18, dtype=np.uint8)
    frame[12:68, 35:125] = (80, 120, 180)
    normalized = normalize_content(frame)
    assert normalized.shape == (144, 256, 3)
    path = write_synthetic_video(str(tmp_path / "integrity.avi"), frames=18, scene_cut_at=9)
    first = analyze_integrity(path, max_frames=8, indices=list(range(18)))
    second = analyze_integrity(path, max_frames=8, indices=list(range(18)))
    assert first == second
    assert 0.0 <= first["metrics"]["leave_one_region_out_support"] <= 1.0
    assert first["metrics"]["normalization"].startswith("border-crop")
    assert cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY).dtype == np.uint8


def test_duplicate_run_helper_is_stable():
    assert _longest_run([False, True, True, True, False, True]) == 3


def test_fusion_exposes_robust_disagreement_without_label_calibration():
    result = fuse({
        "temporal": {"status": "ok", "score": 0.95, "confidence": 1.0,
                     "metrics": {"pairs": 16}},
        "spatial": {"status": "ok", "score": 0.05, "confidence": 1.0,
                    "metrics": {"frames": 16}},
    })
    assert result["robust_group_median"] == 0.5
    assert result["group_score_range"] == 0.9
    assert "independent-evidence-disagrees" in result["reason_codes"]
    assert result["label"] == "abstain-insufficient-evidence"


def test_supported_positive_consensus_keeps_contradiction_as_abstention():
    result = fuse({
        "temporal": {"status": "ok", "score": 0.95, "confidence": 1.0,
                     "metrics": {"pairs": 16}},
        "periodicity": {"status": "ok", "score": 0.90, "confidence": 1.0,
                        "metrics": {"frames": 16}},
        "spatial": {"status": "ok", "score": 0.05, "confidence": 1.0,
                    "metrics": {"frames": 16}},
    })
    assert result["positive_evidence_score"] is not None
    assert result["label"] == "abstain-insufficient-evidence"
    assert "independent-evidence-disagrees" in result["reason_codes"]


def test_fusion_reports_per_group_consensus_without_classifying():
    result = fuse({
        "temporal": {"status": "ok", "score": 0.72, "confidence": 1.0,
                     "metrics": {"pairs": 16}},
        "spatial": {"status": "ok", "score": 0.80, "confidence": 1.0,
                    "metrics": {"frames": 16}},
    })
    assert result["group_consensus"]["status"] == "consistent"
    assert result["group_consensus"]["group_count"] == 2
    assert result["group_consensus"]["consensus_group_count"] == 2
    assert result["group_consensus"]["consensus_fraction"] == 1.0


def test_fusion_exposes_mixed_directional_votes_and_support():
    result = fuse({
        "temporal": {"status": "ok", "score": 0.90, "confidence": 1.0,
                     "metrics": {"pairs": 16}},
        "spatial": {"status": "ok", "score": 0.10, "confidence": 1.0,
                    "metrics": {"frames": 16}},
    })

    assert result["directional_vote_summary"]["status"] == "mixed-directional-votes"
    assert result["directional_vote_summary"]["positive_groups"] == ["motion"]
    assert result["directional_vote_summary"]["negative_groups"] == ["spatial"]
    assert result["group_directional_votes"]["motion"]["direction"] == "positive"
    assert result["group_directional_votes"]["spatial"]["direction"] == "negative"
    assert result["group_directional_votes"]["motion"]["supported"] is True
    assert result["group_support"]["motion"] > 0.0
    assert "mixed-directional-votes" in result["reason_codes"]
    assert result["label"] == "abstain-insufficient-evidence"


def test_fusion_exposes_stable_consensus_hint_below_production_boundary():
    result = fuse({
        "temporal": {"status": "ok", "score": 0.54, "confidence": 1.0,
                     "metrics": {"pairs": 16}},
        "spatial": {"status": "ok", "score": 0.56, "confidence": 1.0,
                    "metrics": {"frames": 16}},
    })
    assert result["label"] == "mixed-anomaly-signal"
    assert result["stable_anomaly_score"] == 0.55
    assert result["instability_score"] == 0.028
    assert result["provisional_positive"] is True
    assert result["provisional_positive_reason"] == (
        "independent-groups-stable-consensus"
    )


def test_analysis_can_report_provisional_consensus_without_lowering_threshold():
    report = {
        "fusion": {
            "score": 0.55, "label": "mixed-anomaly-signal",
            "provisional_positive": True,
        },
        "stability": {"status": "ok", "score_range": 0.1},
        "transcode_stability": {"status": "stable"},
    }
    decision = classify_report(report)
    assert decision["predicted"] == "positive"
    assert decision["decision_kind"] == "provisional-positive"
    assert decision["abstain_reasons"] == []
    assert decision["evidence"] == ["independent-group-consensus"]


def test_exploratory_mode_labels_limited_window_consensus_as_provisional():
    report = {
        "fusion": {
            "score": 0.55,
            "label": "mixed-anomaly-signal",
            "group_score_range": 0.10,
            "provisional_positive": False,
        },
        "stability": {
            "enabled": True,
            "status": "ok",
            "score_range": 0.0,
            "score_mad": 0.0,
            "stability_index": 1.0,
            "provisional_positive_support": False,
            "exploratory_positive_support": True,
            "stable_positive_groups": ["motion", "spatial"],
            "stable_positive_score": 0.72,
            "consensus_policy": {
                "feasible_window_count": 1,
                "requested_budget_count": 1,
                "required_evaluation_count": 1,
                "required_evaluation_fraction": 1.0,
            },
        },
        "transcode_stability": {"enabled": True, "status": "stable"},
    }

    standard = classify_report(report)
    exploratory = classify_report(report, exploratory=True)

    assert standard["predicted"] == "abstain"
    assert "window-consensus-not-stable" in standard["abstain_reasons"]
    assert exploratory["predicted"] == "positive"
    assert exploratory["decision_kind"] == "exploratory-provisional-positive"
    assert exploratory["analysis_mode"] == "exploratory"
    assert exploratory["exploratory"]["eligible"] is True
    assert exploratory["exploratory"]["scope"] == [
        "one-feasible-window", "reduced-budget"
    ]
    assert exploratory["exploratory"]["instability"]["score_range"] == 0.0
    assert exploratory["exploratory"]["instability"]["transcode_status"] == "stable"
    assert "exploratory-limited-window-consensus" in exploratory["evidence"]


def test_exploratory_mode_preserves_contradiction_and_transcode_abstention():
    report = {
        "fusion": {
            "score": 0.80,
            "label": "high-anomaly-signal",
            "group_score_range": 0.40,
            "provisional_positive": False,
        },
        "stability": {
            "enabled": True,
            "status": "ok",
            "score_range": 0.0,
            "exploratory_positive_support": True,
            "stable_positive_groups": ["motion", "spatial"],
            "stable_positive_score": 0.80,
            "consensus_policy": {
                "feasible_window_count": 1,
                "requested_budget_count": 1,
            },
        },
        "transcode_stability": {"enabled": True, "status": "unstable"},
    }
    decision = classify_report(report, exploratory=True)

    assert decision["predicted"] == "abstain"
    assert "unstable-after-transcode" in decision["abstain_reasons"]
    assert "independent-evidence-disagrees" in decision["abstain_reasons"]
    assert decision["exploratory"]["eligible"] is False
    assert "window-score-instability" not in decision["exploratory"]["reasons"]


def test_evaluator_exploratory_mode_is_explicit_and_provisional():
    reports = {
        "short-positive": {
            "fusion": {
                "score": 0.55,
                "label": "mixed-anomaly-signal",
                "group_score_range": 0.1,
                "provisional_positive": False,
            },
            "stability": {
                "enabled": True,
                "status": "ok",
                "score_range": 0.0,
                "exploratory_positive_support": True,
                "stable_positive_groups": ["motion", "spatial"],
                "stable_positive_score": 0.7,
                "consensus_policy": {
                    "feasible_window_count": 1,
                    "requested_budget_count": 1,
                    "required_evaluation_count": 1,
                    "required_evaluation_fraction": 1.0,
                },
            },
            "transcode_stability": {"enabled": True, "status": "stable"},
        },
    }
    result = evaluate_reports(
        reports, truth={"short-positive": "positive"}, exploratory=True
    )

    assert result["analysis_mode"] == "exploratory"
    assert result["calibration"] == "none"
    assert "not calibrated" in result["mode_note"]
    assert result["items"]["short-positive"]["predicted"] == "positive"
    assert result["items"]["short-positive"]["decision_kind"] == (
        "exploratory-provisional-positive"
    )


def test_exploratory_ranking_retains_low_confidence_disagreement():
    reports = {
        "mixed": {
            "fusion": {
                "score": 0.80,
                "label": "high-anomaly-signal",
                "group_score_range": 0.40,
                "evidence_coverage": 0.75,
                "directional_vote_summary": {
                    "status": "mixed-directional-votes",
                    "positive_groups": ["motion"],
                    "negative_groups": ["spatial"],
                },
                "group_directional_votes": {
                    "motion": {"direction": "positive", "support": 0.7},
                    "spatial": {"direction": "negative", "support": 0.6},
                },
                "reason_codes": ["mixed-directional-votes"],
            },
            "stability": {"status": "ok", "score_range": 0.1},
            "transcode_stability": {"status": "stable"},
        },
        "clear": {
            "fusion": {
                "score": 0.60,
                "label": "mixed-anomaly-signal",
                "group_score_range": 0.05,
                "evidence_coverage": 1.0,
            },
            "stability": {"status": "ok", "score_range": 0.1},
            "transcode_stability": {"status": "stable"},
        },
    }
    result = evaluate_reports(
        reports,
        truth={"mixed": "positive", "clear": "positive"},
        exploratory=True,
    )

    ranking = result["exploratory_ranking"]
    assert ranking["coverage"] == {
        "ranked_report_count": 2, "report_count": 2, "fraction": 1.0
    }
    assert [item["name"] for item in ranking["items"]] == ["mixed", "clear"]
    mixed = ranking["items"][0]
    assert mixed["confidence_label"] == "low"
    assert mixed["evidence_coverage"] == 0.75
    assert "independent-evidence-disagrees" in mixed["reasons"]
    assert "mixed-directional-votes" in mixed["reasons"]
    assert result["items"]["mixed"]["predicted"] == "abstain"
    assert "independent-evidence-disagrees" in result["items"]["mixed"][
        "abstain_reasons"
    ]


def test_localization_is_sorted_and_explicitly_non_proof():
    metadata = type("Metadata", (), {"fps": 10.0, "duration_seconds": 4.0})()
    localized = _localize({
        "scene": {"score": None, "metrics": {"cut_indices": [20, 5]}},
        "temporal": {"score": 0.4, "confidence": 0.5,
                     "metrics": {"abrupt_transition_indices": [12]}},
    }, metadata, [0, 10, 20])
    assert [event["frame_index"] for event in localized["events"]] == [5, 12, 20]
    assert localized["segments"][0]["branch"] == "temporal"
    assert "not proof" in localized["notes"][0]


def test_localization_reports_cross_branch_event_consensus():
    metadata = type("Metadata", (), {"fps": 10.0, "duration_seconds": 4.0})()
    localized = _localize({
        "temporal": {"score": 0.4, "confidence": 0.5,
                     "metrics": {"abrupt_transition_indices": [12]}},
        "integrity": {"score": 0.5, "confidence": 0.5,
                      "metrics": {"keyframe_indices": [13]}},
    }, metadata, [0, 10, 20])
    assert localized["consistency"]["status"] == "cross-branch-supported"
    assert localized["consistency"]["supporting_branch_count"] == 2
    assert localized["consistency"]["support_fraction"] == 1.0


def test_analysis_classification_abstains_on_robustness_failures():
    report = {
        "fusion": {"score": 0.95, "label": "high-anomaly-signal"},
        "stability": {"status": "ok", "score_range": 0.4},
        "transcode_stability": {"status": "stable"},
    }
    decision = classify_report(report)
    assert decision["predicted"] == "abstain"
    assert "unstable-window-score" in decision["abstain_reasons"]


def test_stable_negative_consensus_survives_an_outlier_window():
    report = {
        "fusion": {
            "score": 0.42,
            "label": "mixed-anomaly-signal",
            "group_score_range": 0.18,
            "reason_codes": [],
        },
        "stability": {
            "enabled": True,
            "status": "ok",
            "score_range": 0.42,
            "stable_negative_support": True,
            "stable_negative_score": 0.18,
            "stable_negative_groups": ["motion", "spatial"],
            "stable_positive_groups": [],
            "provisional_positive_support": False,
            "decision_reason": "stable-low-independent-group-consensus",
        },
        "transcode_stability": {"enabled": True, "status": "stable"},
    }
    decision = classify_report(report)
    assert decision["predicted"] == "negative"
    assert decision["decision_kind"] == "stable-negative-consensus"
    assert "unstable-window-score" not in decision["abstain_reasons"]
    assert "stable-negative-independent-group-consensus" in decision["evidence"]


def test_unstable_positive_consensus_remains_abstention():
    report = {
        "fusion": {
            "score": 0.86,
            "label": "high-anomaly-signal",
            "group_score_range": 0.12,
            "reason_codes": [],
        },
        "stability": {
            "enabled": True,
            "status": "ok",
            "score_range": 0.46,
            "stable_negative_support": False,
            "stable_negative_score": None,
            "stable_negative_groups": [],
            "stable_positive_groups": ["motion", "periodicity"],
            "provisional_positive_support": False,
            "decision_reason": "window-scores-unstable",
        },
        "transcode_stability": {"enabled": True, "status": "stable"},
    }
    decision = classify_report(report)
    assert decision["predicted"] == "abstain"
    assert "unstable-window-score" in decision["abstain_reasons"]
    assert "window-consensus-not-stable" in decision["abstain_reasons"]


def test_stability_adapts_to_one_feasible_window_without_lowering_score_floors(
    monkeypatch,
):
    def fake_run_branches(*args, **kwargs):
        return {
            "temporal": {
                "status": "ok", "score": 0.18, "confidence": 1.0,
                "metrics": {"pairs": 12},
            },
            "spatial": {
                "status": "ok", "score": 0.20, "confidence": 1.0,
                "metrics": {"frames": 12},
            },
        }

    monkeypatch.setattr("forensic_video.analyzer._run_branches", fake_run_branches)
    metadata = type(
        "Metadata", (), {"fps": 10.0, "frame_count": 100, "duration_seconds": 10.0}
    )()
    result = _evaluate_stability(
        "short.avi", metadata,
        AnalysisConfig(
            stability=True, stability_windows=(10, 20), stability_budgets=(12,)
        ),
        {"cuts": []},
    )

    assert result["windows"] == [10]
    assert result["consensus_policy"]["feasible_window_count"] == 1
    assert result["consensus_policy"]["evidence_group_count"] == 2
    assert result["consensus_policy"]["required_evaluation_count"] == 1
    assert result["stable_negative_support"] is True
    assert result["decision_reason"] == "stable-low-independent-group-consensus"


def test_stability_does_not_invent_a_window_for_a_short_clip():
    metadata = type(
        "Metadata", (), {"fps": 10.0, "frame_count": 50, "duration_seconds": 5.0}
    )()
    result = _evaluate_stability(
        "too-short.avi", metadata,
        AnalysisConfig(stability=True, stability_windows=(10,), stability_budgets=(12,)),
        {"cuts": []},
    )

    assert result["status"] == "unavailable"
    assert result["windows"] == []
    assert result["consensus_policy"]["feasible_window_count"] == 0
    assert result["consensus_policy"]["required_evaluation_count"] == 0


def test_stability_quorum_scales_for_multi_window_reduced_resource_audit(
    monkeypatch,
):
    def fake_run_branches(*args, **kwargs):
        return {
            "temporal": {
                "status": "ok", "score": 0.72, "confidence": 1.0,
                "metrics": {"pairs": 12},
            },
            "spatial": {
                "status": "ok", "score": 0.76, "confidence": 1.0,
                "metrics": {"frames": 12},
            },
        }

    monkeypatch.setattr("forensic_video.analyzer._run_branches", fake_run_branches)
    metadata = type(
        "Metadata", (), {"fps": 10.0, "frame_count": 250, "duration_seconds": 25.0}
    )()
    result = _evaluate_stability(
        "multi-window.avi", metadata,
        AnalysisConfig(
            stability=True, stability_windows=(10, 20), stability_budgets=(12, 24)
        ),
        {"cuts": []},
    )

    assert len(result["evaluations"]) == 4
    assert result["consensus_policy"]["feasible_window_count"] == 2
    assert result["consensus_policy"]["scored_evaluation_count"] == 4
    assert result["consensus_policy"]["evidence_group_count"] == 2
    assert result["consensus_policy"]["reduced_resource_scope"] is True
    assert result["exploratory_positive_support"] is True
    assert result["consensus_policy"]["required_evaluation_count"] == 3
    assert result["provisional_positive_support"] is True
    assert result["decision_reason"] == "stable-positive-independent-group-consensus"


def test_one_window_contradictory_groups_remain_abstention(monkeypatch):
    def fake_run_branches(*args, **kwargs):
        return {
            "temporal": {
                "status": "ok", "score": 0.95, "confidence": 1.0,
                "metrics": {"pairs": 12},
            },
            "spatial": {
                "status": "ok", "score": 0.05, "confidence": 1.0,
                "metrics": {"frames": 12},
            },
        }

    monkeypatch.setattr("forensic_video.analyzer._run_branches", fake_run_branches)
    metadata = type(
        "Metadata", (), {"fps": 10.0, "frame_count": 100, "duration_seconds": 10.0}
    )()
    result = _evaluate_stability(
        "contradictory.avi", metadata,
        AnalysisConfig(stability=True, stability_windows=(10,), stability_budgets=(12,)),
        {"cuts": []},
    )

    assert result["consensus_policy"]["required_evaluation_count"] == 1
    assert result["stable_negative_support"] is False
    assert result["provisional_positive_support"] is False
    assert result["decision_reason"] == "insufficient-window-consensus"


def test_analysis_classification_does_not_hide_group_disagreement():
    report = {
        "fusion": {
            "score": 0.72,
            "label": "high-anomaly-signal",
            "group_score_range": 0.40,
        },
        "stability": {"status": "ok", "score_range": 0.1},
        "transcode_stability": {"status": "stable"},
    }
    decision = classify_report(report)
    assert decision["predicted"] == "abstain"
    assert decision["decision_kind"] == "abstain"
    assert decision["abstain_reasons"] == ["independent-evidence-disagrees"]


def test_production_fusion_abstains_after_transcode_instability():
    fusion = {"label": "high-anomaly-signal", "reason_codes": []}
    _apply_robustness_abstention(
        fusion,
        {"enabled": False},
        {"enabled": True, "status": "unstable"},
    )
    assert fusion["label"] == "abstain-unstable-evidence"
    assert fusion["reason_codes"] == ["transcode-instability"]


def test_evaluation_separates_ranking_from_confusion_metrics():
    reports = {
        "positive": {
            "fusion": {"score": 0.9, "label": "high-anomaly-signal"},
            "stability": {"status": "ok", "score_range": 0.1},
            "transcode_stability": {"status": "stable"},
        },
        "negative": {
            "fusion": {"score": 0.1, "label": "low-anomaly-signal"},
            "stability": {"status": "ok", "score_range": 0.1},
            "transcode_stability": {"status": "stable"},
        },
        "unknown": {
            "fusion": {"score": 0.5, "label": "mixed-anomaly-signal"},
            "stability": {"status": "ok", "score_range": 0.1},
            "transcode_stability": {"status": "stable"},
        },
    }
    result = evaluate_reports(
        reports,
        truth={"positive": "positive", "negative": "negative"},
    )
    assert result["metrics"]["confusion_matrix"] == {
        "true_positive": 1,
        "false_positive": 0,
        "true_negative": 1,
        "false_negative": 0,
        "abstain": 0,
    }
    assert result["items"]["unknown"]["predicted"] == "abstain"
    assert result["metrics"]["conditional_accuracy"] == 1.0
    assert result["ranking"]["median_gap_positive_minus_negative"] == 0.8
    assert result["ranking"]["auroc_like"] == 1.0
    assert result["ranking"]["average_precision_like"] == 1.0
    assert result["operating_points"]
    assert result["consensus_operating_points"]
    point = next(
        item for item in result["operating_points"]
        if item["positive_threshold"] == 0.9
    )
    assert point["coverage"] == 1.0
    assert point["sensitivity"] == 1.0
    assert point["specificity"] == 1.0


def test_evaluation_reports_branch_level_rank_diagnostics():
    reports = {
        "positive": {
            "fusion": {"score": 0.8},
            "branches": {
                "temporal": {"score": 0.9},
                "frequency": {"score": 0.4},
            },
        },
        "negative": {
            "fusion": {"score": 0.2},
            "branches": {
                "temporal": {"score": 0.1},
                "frequency": {"score": 0.5},
            },
        },
    }
    result = evaluate_reports(
        reports,
        truth={"positive": "positive", "negative": "negative"},
    )
    assert result["schema_version"] == "evaluation-4"
    assert result["branch_diagnostics"]["temporal"]["auroc_like"] == 1.0
    assert result["branch_diagnostics"]["temporal"][
        "median_gap_positive_minus_negative"
    ] == 0.8
    assert result["branch_diagnostics"]["temporal"]["missing_or_invalid_count"] == 0
    assert result["branch_diagnostics"]["frequency"]["auroc_like"] == 0.0
