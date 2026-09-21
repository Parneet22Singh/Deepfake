# forensic-video

`forensic-video` is a deterministic, classical-computer-vision video-forensics
package and JSON CLI. Its authoritative engine does not train models and
reports reproducible evidence signals and limitations for human review. The
repository can also invoke the separately trained specialist and binary
authenticity routers from the protected production snapshot as advisory
outputs. Those routers are trained ML models built from custom forensic data
and FaceForensics-derived and related datasets; their checkpoints are not
copied into this repository. A score is an anomaly signal, not a calibrated
probability that a video is manipulated.

## Install and run

```bash
python -m pip install -e ".[video]"
forensic-video input.mp4 --samples 48 --max-frames 32 > report.json
# Optional robustness pass (more decoding; no files are retained):
forensic-video input.mp4 --stability --transcode-check --indent 0 > report.json
```

Each report now exposes four separate analysis outputs under
`analysis_outputs`: the authoritative deterministic engine, an optional
three-class specialist router, an optional binary authenticity router, and a
five-layer production-structure projection. The router outputs are disabled
unless explicitly configured and never replace deterministic fusion.

The protected production snapshot can be used without copying or modifying
its trained checkpoints. All configured paths must be absolute and inside the
snapshot root:

```powershell
python -m forensic_video input.mp4 `
  --production-snapshot-root "C:\path\to\The-Neuroforge-production-final-year-snapshot-2026-09-10" `
  --specialist-checkpoint "C:\path\to\...\training\runs\expanded-efficientnet-controlled-saved\router_best.pt" `
  --binary-checkpoint "C:\path\to\...\training\runs\binary-authenticity-efficientnet\router_best.pt"
```

If PyTorch or the protected snapshot's training dependencies are unavailable,
the report keeps the router outputs explicit with `status: "error"` and
preserves the deterministic result. Router execution is isolated in a bounded
child process so an incompatible Torch/TorchVision native binary cannot block
or crash the deterministic API process. When the protected snapshot and both
checkpoints are present at the standard sibling path, the API discovers and
initializes the routers automatically. Set `NEUROFORGE_ENABLE_ROUTERS=0` to
disable optional router execution explicitly.
An optional local-file API is available
with `pip install -e ".[video,api]"` and `uvicorn forensic_video.api:app`.

### Integrated frontend

The `frontend/` application uses the deterministic API for both local video
uploads and YouTube URLs. Install the API extra (which includes `yt-dlp`) and
start the API:

```powershell
python -m pip install -e ".[video,api]"
python -m uvicorn forensic_video.api:app --host 127.0.0.1 --port 8000
```

Run the frontend with `VITE_FORENSICS_API_URL=http://localhost:8000` in its
environment. The local upload control sends a file to `/analyze`; a YouTube
URL sends `youtube_url` to the same endpoint, where the API downloads one
video with `yt-dlp`, analyzes the temporary file, and removes it afterward.
The frontend renders deterministic branch evidence, authoritative fusion, both
optional router outputs, and the explainable five-layer production projection.
When the protected snapshot is present at the standard sibling path
`The-Neuroforge-production-final-year-snapshot-2026-09-10`, the API discovers
its trained router checkpoints automatically. Explicit environment variables
can override that discovery for another installation. Set
`NEUROFORGE_ENABLE_ROUTERS=0` when running without the protected snapshot or
without a compatible Torch/TorchVision installation. If the routers cannot
produce a result, the report says whether they abstained because no face
evidence was available or failed during isolated execution; `not_configured`
is reserved for missing checkpoint configuration.
Router execution waits for the protected models to finish by default, so a
slow model startup or inference run is not silently excluded. Set
`NEUROFORGE_ROUTER_TIMEOUT_SECONDS` to a positive number when an operator
explicitly wants a hard cap; `0`, `none`, `unlimited`, and `off` mean no cap.
An incompatible native model process may therefore remain running until it is
stopped by the operator when no cap is configured.
Configure allowed frontend origins with `NEUROFORGE_CORS_ORIGINS` when
the default localhost origins are not sufficient.

The minimum install includes NumPy. OpenCV is optional; without it provenance
and a graceful `unavailable` report are still emitted. The output has stable
`schema_version`, `metadata`, `sampling`, independent `branches`, `fusion`,
`localization`, `stability`, `transcode_stability`, and `warnings` fields.
Branches include provenance, scene cuts, temporal
consistency, controlled JPEG recompression, multiscale frequency/noise, a
model-free skin/shape face-like count, spatial residuals, transform-invariant
flow/residual/DCT diagnostics, duplicate-frame periodicity, ffprobe GOP
cadence, and best-effort audio/audio-video timing. The `wavelet` branch adds
one-level Haar LL/LH/HL/HH energies, robust temporal instability, and
localized detail events. The model-free `face` branch now requires repeated
regions and optical-flow-supported geometry before scoring. `resampling`
reports interpolation, duplicate-frame, and frame-rate-conversion residuals
as diagnostics only (`fusion_eligible: false`), never as manipulation claims.

## Interpretation and reproducibility

Sampling always spans the full known duration and deterministically adds
high-change frames. All bounded branches consume endpoint-preserving samples
across that duration (not just the first N frames). Scene detection uses HSV
histograms and is reported as an edit diagnostic, not fused as manipulation
evidence. Temporal diagnostics use Farneback optical flow, feature
round-trips, motion-compensated residuals, and robust median/MAD outliers.
Frequency and recompression branches measure within-video inconsistency rather
than treating a globally noisy or compressed camera as fake. The face branch
only scores repeated, geometrically unstable skin/shape regions; isolated
detections produce `score: null`. The integrity branch removes obvious
letterbox borders, normalizes aspect ratio, and measures flow divergence,
forward/backward occlusion, block/DCT inconsistency, and
motion-compensated residuals. It also reports leave-one-region-out support:
a hotspot that disappears when one tile is removed is not robust evidence.
Periodicity is withheld unless a duplicate burst is surrounded by changing
content. GOP and audio/video timing are triage metadata only; timing is scored
only when both stream start-times and durations are present and disagree beyond
muxing tolerances. Audio uses `soundfile` when available, otherwise ffmpeg to
an in-memory PCM pipe.

Fusion weights branch confidence and observed coverage, collapses correlated
members of the motion, codec, spatial, and audio groups, and reports
per-group representatives, directional votes/support, and a `group_consensus`
summary. Each `group_directional_votes` entry is a diagnostic positive,
negative, or neutral vote with representative score, reliability, and
support; `directional_vote_summary` makes mixed votes explicit. These votes
are not classes and do not override abstention rules. It also reports a
robust `stable_anomaly_score` and an `instability_score`; these are evidence
descriptors, not probabilities. When at least two independent groups agree
within a narrow range, `provisional_positive` is emitted as an analysis-only
hint. It does not change the production `high-anomaly-signal` boundary and is
withheld for quality-only evidence or insufficient coverage. Reason codes
include `correlated-branches-collapsed` and `global-compression-confounder`.
When at least two well-supported groups also agree above the fixed positive
evidence floor, their `positive_evidence_score` is retained separately from
low-signal groups rather than diluted by them. This improves ranking without
overriding disagreement abstention; the supporting groups and policy are
reported in `positive_evidence_groups` and `positive_evidence_policy`.
Codec-sensitive evidence is treated as corroboration rather than manipulation
support: if the codec group is the only group above the positive floor, the
report sets `quality_sensitive_only` and emits
`quality-sensitive-evidence-only`, while retaining the ordinary score for
auditability. A non-codec group such as periodicity may corroborate codec
evidence, but contradictory independent groups still force abstention.
Non-finite or malformed confidence, coverage, and observation-count inputs
are excluded rather than allowed to produce non-reproducible `NaN` output;
excluded branches are listed in `rejected_branches` with an audit reason.
It requires at least two supported independent evidence groups; otherwise it
emits `abstain-insufficient-evidence` (retained branch scores remain visible
for auditability). Record the input SHA-256, CLI options, package version, and
OpenCV/ffmpeg versions with a report.

`--stability` evaluates centered 10, 20, 30, and 40-second windows that fit the
clip at 24, 48, and 96 samples. It reports median/MAD/range and a bounded
stability index in `stability`; it also reports robust per-group medians, MAD,
support fractions, and ranges across the windows. At least two independent
groups with low scores can produce `stable_negative_support`, which keeps a
negative analysis decision available even when one window is an outlier.
Positive provisional support instead requires at least two independent groups
to remain high and agree across the tested windows; a high but unstable score
remains an abstention. The evidence-count quorum is reported in
`stability.consensus_policy`: it is `ceil(0.60 * scored evaluations)`, rounded
up and bounded at one. Thus a short clip with only one feasible window can
still report limited-scope consensus when two independent groups agree, while
multi-window and reduced-resource audits retain a proportional quorum. This
adapts evidence counts, not anomaly score floors or classification thresholds;
contradictory groups still abstain. `decision_reason`,
`stable_negative_groups`, `stable_positive_groups`, and the fusion
`reason_codes` expose which path was available. A one-window result is not
equivalent to temporal robustness across multiple windows. This is an
unsupervised sensitivity check, not calibration from labels. `localization`
contains diagnostic transition/scene/region events and clip-wide branch
segments; events are not proof of manipulation. Its
`time_aligned_evidence` field clusters independent events in a small frame
tolerance so coincident anomalies receive review-priority support, while
unrelated whole-video scores remain separate. This descriptor does not alter
strict production thresholds or the evaluator's explicit exploratory mode.
Each report also contains `fusion_sets.legacy` and
`fusion_sets.new_additions`. The former includes the original temporal,
recompression, frequency, face, spatial, integrity, and audio branches. The
latter contains the later periodicity, codec, AV-sync, wavelet, and
resampling diagnostics. The all-branch `fusion` result remains unchanged, so
the split is an audit comparison rather than a hidden recalibration.
For classification audits, `--evidence-set legacy` uses only the original
branches; `new_additions` remains diagnostic and is not mixed into that
decision. The current legacy audit threshold is `0.45`:

```powershell
python -m forensic_video.evaluation --evidence-set legacy `
  --positive-threshold 0.45 --item clip=eval\clip.json
```

This separates evidence selection from thresholding. It does not manufacture
evidence for a clip whose original branches all report low anomaly scores; such
a clip remains a false negative until a genuinely discriminative deterministic
signal is found.

The evaluator also provides an explicit `--evidence-set directional` audit
profile. This is the two-sided design: face-geometry inconsistency and
motion-compensated integrity residuals are treated as synthetic-support
algorithms, rather than being averaged with codec and frequency diagnostics.
The output retains normal `positive`/`negative` fields and adds
`directional_label` (`synthetic` or `original`). This profile is analysis-only
and must be validated on additional labeled material before production use.
For a constrained-resource audit, `--stability-windows` and
`--stability-budgets` override those lists; omitted durations that do not fit
are not counted as feasible windows. `--transcode-check` performs an ephemeral ffmpeg
half-resolution H.264 transcode, compares the fused score, and deletes the
generated file in a `finally` block. A material change is reported as
`transcode_stability.status: unstable` and now forces the production fusion
label to `abstain-unstable-evidence` rather than silently trusting the score.

The default evaluator remains conservative and can abstain when a requested
stability pass has insufficient window consensus. For exploratory work only,
`python -m forensic_video.evaluation --exploratory` enables a separate
provisional path for reports marked as one-feasible-window or reduced-budget
audits. It requires two non-contradictory positive evidence groups, preserves
the score-range and transcode-instability safeguards, and never overrides
production fusion abstention or independent-group disagreement. Results are
marked `analysis_mode: exploratory`,
`decision_kind: exploratory-provisional-positive`, and include
`exploratory.reasons` plus score-range/MAD/index and transcode status. This is
not a calibrated classifier, a production label, or a justification for
lowering anomaly thresholds; one-window results remain limited-scope evidence.

For a deliberately forced score-bucket comparison that does not abstain, pass
`--forced-threshold 0.4` to the evaluator. It emits `original` for scores
below `0.4` and `synthetic` for scores at or above `0.4`, separately for the
all-branch score and each `fusion_sets` view. This is explicitly an
analysis-only bucket and must not be interpreted as a calibrated authenticity
classification:

```powershell
python -m forensic_video.evaluation --forced-threshold 0.4 `
  --item original_deepfake=eval\original_deepfake.json `
  --item original_real=eval\original_real.json
```

Deterministic fixtures are available via:

```python
from forensic_video.synthetic import write_synthetic_video
write_synthetic_video("fixture.avi")
# A deliberately simple temporal manipulation fixture for regression tests:
write_synthetic_video("duplicate-burst.avi", manipulation="duplicate-burst")
```

The duplicate-burst fixture repeats eight decoded frames while retaining
changing content around the burst. It is useful for checking that periodicity
diagnostics remain deterministic, but it is not a representative deepfake
corpus and must not be treated as detector validation. Run tests (after
installing the `dev` extra) with `python -m pytest -q`, and compile-check with
`python -m compileall forensic_video`.

## Reproducible 30-second labeled evaluation

The analyzer itself never downloads URLs. If `yt-dlp` and ffmpeg are installed,
the following PowerShell commands create a disposable evaluation directory,
download only the first 30 seconds, and write one JSON report per label:

```powershell
$items = @{
  original_deepfake = "https://youtu.be/cQ54GDm1eL0"
  original_real = "https://youtu.be/1KzHEPNmjU8"
  new_real = "https://youtu.be/UqHh6TvGQIQ"
  new_deepfake = "https://youtu.be/iyiOVUbsPcM"
  new_ai_generated = "https://youtu.be/SA6fUs3dsRU"
  normal_video_7Dry = "https://youtu.be/7DryFpCKLzo"
  additional_sP90 = "https://youtu.be/sP90VDiIOXs"
}
New-Item -ItemType Directory -Force eval | Out-Null
foreach ($label in $items.Keys) {
  yt-dlp --quiet --no-warnings --download-sections "*0-30" `
    --merge-output-format mp4 -o "eval\$label.%(ext)s" $items[$label]
  python -m forensic_video "eval\$label.mp4" --samples 48 --max-frames 32 `
    --indent 0 > "eval\$label.json"
}
```

Compare two runs without treating labels as training data:
`Get-FileHash eval\*.json`, or compare the `fusion`, `evidence_coverage`,
branch `metrics`, and `warnings` fields with a JSON diff. Remove `eval\` after
evaluation so the repository remains source-only. Network availability,
region restrictions, transcoding, and YouTube revisions can change the input;
record each downloaded file's SHA-256 and do not interpret a label as ground
truth.

The protocol above is intentionally fixed at 30 seconds, 48 adaptive samples,
and 32 frames per bounded branch. For a compact local report:

```powershell
python -m forensic_video eval\original_real.mp4 --samples 48 --max-frames 32 --indent 0 > eval\original_real.json
Remove-Item -Recurse -Force eval
```

`ffprobe` is optional: without it, GOP and stream-timing branches abstain.
The deterministic branches are not trained detectors, and no deterministic
score is a probability or a claim that the source is fake. The optional
routers are trained models but remain advisory and are gated for confidence
and margin. A useful evaluation compares rank separation and
false-positive behavior against a prior implementation on the same downloaded
bytes; improvement on these seven labels alone can still be overfitting.

### Classification audit (separate from production analysis)

The analyzer intentionally has no calibrated fake/real classifier. To audit
classification on the fixed set without turning these labels into production
calibration, use the analysis-only evaluator after writing reports:

```powershell
python -m forensic_video.evaluation `
  --item original_deepfake=eval\original_deepfake.json `
  --item original_real=eval\original_real.json `
  --item new_real=eval\new_real.json `
  --item new_deepfake=eval\new_deepfake.json `
  --item new_ai_generated=eval\new_ai_generated.json `
  --item normal_video_7Dry=eval\normal_video_7Dry.json `
  --item additional_sP90=eval\additional_sP90.json `
  --indent 2
```

The evaluator reports each anomaly score, an analysis-only
`positive`/`negative`/`abstain` view, stability range, transcode delta,
abstention reasons, confusion metrics, and a threshold sweep. Its fixed
defaults are `positive >= 0.62` and `negative <= 0.30`; the middle is
abstention. `ranking` additionally reports deterministic AUROC-like pair
concordance and average-precision-like rank summaries. If independent groups
agree robustly below 0.62, the audit may report
`decision_kind: provisional-positive`; this is an explicit analysis-only
consensus heuristic, not a calibrated classifier or a production label.
`branch_diagnostics` provides the same rank summaries and positive/negative
median gaps for every branch with labeled scores. Use it to identify which
signals separate a small audit set before changing fusion; its
`missing_or_invalid_count` makes branch coverage explicit. It does not
calibrate thresholds and does not turn an individual branch into a classifier.
Threshold decisions also abstain when the report exposes independent-group
disagreement greater than 0.20, so a weighted mean cannot hide contradictory
evidence. This is a conservative coverage trade-off, not a learned
calibration. With `--exploratory`, `exploratory_ranking` retains a
low-confidence ordering for every report with a numeric score, including
mixed directional votes and group disagreement. Its per-item
`evidence_coverage`, confidence descriptor, vote summary, and reasons keep
ranking coverage explicit; the ranking never converts disagreement into a
positive or negative decision.
Inspect its confusion metrics, coverage, and false-positive rate on new
labeled data before relying on it. `operating_points`
sweeps the positive threshold while retaining the negative threshold and
explicit abstain band, so coverage, sensitivity, specificity, false-positive
rate, and conditional accuracy can be inspected together rather than
selecting a convenient threshold. `consensus_operating_points` repeats that
sweep with the independent-group provisional hint enabled, allowing the
standard threshold-only and consensus-aware operating points to be compared
without conflating them. Rank metrics include every numeric labeled score
(and disclose that robustness failures are not removed); operating points
remain robustness-aware and abstain on unavailable or unstable checks.
`additional_sP90` is included as an unlabeled monitoring item and is excluded
from confusion metrics. Thresholds are not learned, persisted, or used by the
production CLI; the score remains a ranking signal and abstentions are
expected when evidence is weak or unstable.
