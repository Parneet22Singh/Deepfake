import { useState, useCallback, useEffect, useRef } from "react";
import { useAuth } from "@/contexts/AuthContext";

export type AnalysisMode = "quick" | "deep";

export interface ForensicOutput {
  status: string;
  authoritative?: boolean;
  label?: string | null;
  score?: number | null;
  findings?: string[];
  limitations?: string[];
  warnings?: string[];
  [key: string]: unknown;
}

export interface AnalysisOutputs {
  deterministic_engine: ForensicOutput;
  specialist_three_class_router: ForensicOutput;
  binary_authenticity_router: ForensicOutput;
  directional_analysis: ForensicOutput & {
    predicted?: string;
    directional_label?: string;
  };
  production_five_layer: ForensicOutput & {
    layers?: Array<{
      id: string;
      name: string;
      score: number | null;
      evidence_branches: string[];
    }>;
    limitations?: string[];
  };
}

export interface AnalysisReconciliation {
  status: "agreed" | "conflict" | "inconclusive" | string;
  consensus: "synthetic" | "original" | "review" | string;
  sources: Record<string, string>;
  conflicting_sources: string[];
  policy: string;
}

export interface LayerData {
  name: string;
  score: number | null;
  stdDev: number;
  explanation: string;
  color: string;
  icon: string;
  metrics: Record<string, number>;
  rawMetrics?: Record<string, number>;
}

export interface SocialShares {
  x: string;
  redditTitle: string;
  redditBody: string;
  instagram: string;
  linkedin: string;
}

export interface ProcessingMeta {
  framesAnalyzed: number;
  lipFramesAnalyzed: number;
  segmentsDetected: number;
  totalProcessingMs: number;
  whisperActive: boolean;
  librosaActive: boolean;
  realSignalProcessing: boolean;
  engineVersion?: string;
  analysisMode?: AnalysisMode;
  temporalCoverage?: string;
  frequencyFramesAnalyzed?: number;
  biometricFramesAnalyzed?: number;
  faceEvidenceGate?: boolean;
  contentClass?: string;
}

export interface ContentContext {
  content_class: "biological_face" | "cgi_animation" | "game_screen_overlay" | "non_biological" | "unknown" | string;
  confidence: number;
  face_presence_ratio: number;
  roi_quality: number;
  mean_face_count: number;
  screen_score: number;
  cartoon_score: number;
  evidence_gate: boolean;
  reason: string;
}

export interface FusionEngine {
  posterior_probability: number;
  verdict_percent: number;
  margin_of_error: number;
  information_entropy_bits: number;
  layer_contributions: Array<{
    log_likelihood_ratio: number;
    evidence_strength: number;
  }>;
}

export interface TimelineEntry {
  frame_idx: number;
  timestamp_s: number;
  manipulation_probability: number;
  dominant_signal: string;
  evidence_scope?: string;
}

export interface GanFingerprint {
  spectral_signature_class: string;
  confidence: number | null;
  hf_energy_ratio: number | null;
}

export interface ProvenanceData {
  estimated_reencoding_generations: number | null;
  compression_artifact_layers: number | null;
  original_quality_estimate: string;
  quantization_table_anomaly: boolean | null;
}

export interface AnalysisResult {
  verdictPercent: number;
  marginOfError: number;
  confidenceLevel: "Low" | "Moderate" | "High";
  reliabilityScore: number;
  crossLayerConsistency: number;
  modelNonDeepfakeProbability: number;
  interpretationStatus?: "complete" | "unavailable";
  interpretationError?: string | null;
  selfConsistencyValidation: "Passed" | "Review Recommended";
  adversarialShift: number;
  adversarialTests: { topAnomalyRemoved: number | null; compressionRemoved: number | null };
  compressionBiasDetected: boolean;
  biasLog: string;
  modelVersion: string;
  calibrationSource: string;
  traceId: string;
  timestamp: string;
  layers: LayerData[];
  fusionEngine: FusionEngine;
  timeline: TimelineEntry[];
  provenance: ProvenanceData;
  enhancedAudio: {
    metrics: Record<string, number>;
    processing_time_ms: number;
  };
  videoUrl: string;
  videoId: string;
  thumbnailUrl: string;
  reportFrames?: string[];
  reportNarrative?: string;
  humanVerdict: string;
  social: SocialShares;
  processingMeta?: ProcessingMeta;
  decisionStatus?: "verdict" | "abstain";
  abstentionReason?: string | null;
  contentContext?: ContentContext;
  analysis_outputs?: AnalysisOutputs;
  analysis_reconciliation?: AnalysisReconciliation;
  deterministicReport?: {
    metadata?: Record<string, unknown>;
    sampling?: Record<string, unknown>;
    branches?: Record<string, ForensicOutput>;
    fusion?: Record<string, unknown>;
    warnings?: string[];
  };
}

type AppState = "hero" | "analyzing" | "results";

export function useNeuroforge() {
  const { accessToken } = useAuth();
  const [state, setState] = useState<AppState>("hero");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [currentLayer, setCurrentLayer] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>("quick");
  const [localFile, setLocalFile] = useState<File | null>(null);
  const activeAnalysisController = useRef<AbortController | null>(null);

  useEffect(() => {
    const cancelOnPageExit = () => activeAnalysisController.current?.abort();
    window.addEventListener("pagehide", cancelOnPageExit);
    return () => {
      window.removeEventListener("pagehide", cancelOnPageExit);
      cancelOnPageExit();
    };
  }, []);

  const validateYoutubeUrl = (url: string): boolean => {
    const patterns = [
      /^(https?:\/\/)?(www\.)?youtube\.com\/watch\?v=[\w-]+/,
      /^(https?:\/\/)?(www\.)?youtube\.com\/shorts\/[\w-]+/,
      /^(https?:\/\/)?(www\.)?youtu\.be\/[\w-]+/,
      /^(https?:\/\/)?(www\.)?youtube\.com\/clip\/[\w-]+/,
    ];
    return patterns.some((p) => p.test(url));
  };

  const extractVideoId = (url: string): string => {
    const patterns = [
      /[?&]v=([\w-]+)/,
      /youtu\.be\/([\w-]+)/,
      /shorts\/([\w-]+)/,
      /clip\/([\w-]+)/,
    ];
    for (const p of patterns) {
      const m = url.match(p);
      if (m) return m[1];
    }
    return "";
  };

  const parseAnalysisError = (rawMessage: string): string => {
    const message = rawMessage || "Analysis failed. Please try again.";

    if (message === "Failed to fetch" || message.includes("NetworkError")) {
      return "The local forensic API is unreachable. Start the backend at http://127.0.0.1:8000 and retry.";
    }

    if (message.includes("Sign in to confirm youâ€™re not a bot") || message.includes("Sign in to confirm you're not a bot")) {
      return "YouTube blocked automated download on the forensic backend. Refresh the service cookies and redeploy the forensic service, then retry.";
    }

    if (message.includes("non-2xx status code")) {
      return "Analysis request reached the backend but failed upstream. Please retry once; if it persists, check forensic service logs.";
    }

    if (message.includes("capacity reached") || message.includes("rate limit exceeded")) {
      return "The analysis service is busy. Please wait a moment and retry.";
    }

    return message;
  };

  const normalizeLocalReport = (report: Record<string, unknown>, sourceName: string): AnalysisResult => {
    const rawOutputs = (report.analysis_outputs && typeof report.analysis_outputs === "object"
      ? report.analysis_outputs
      : {}) as AnalysisOutputs;
    const fusion = (report.fusion && typeof report.fusion === "object"
      ? report.fusion
      : {}) as Record<string, unknown>;
    const score = typeof fusion.score === "number" ? fusion.score : 0;
    const outputs: AnalysisOutputs = {
      ...rawOutputs,
      deterministic_engine: {
        ...(rawOutputs.deterministic_engine || {}),
        status: rawOutputs.deterministic_engine?.status || "available",
        authoritative: true,
        label: typeof fusion.label === "string" ? fusion.label : null,
        score,
        fusion,
      },
      directional_analysis: {
        ...(rawOutputs.directional_analysis || {}),
        status: rawOutputs.directional_analysis?.status || "available",
        label: rawOutputs.directional_analysis?.label
          || rawOutputs.directional_analysis?.directional_label
          || null,
        score: typeof rawOutputs.directional_analysis?.score === "number"
          ? rawOutputs.directional_analysis.score
          : null,
      },
      production_five_layer: {
        ...(rawOutputs.production_five_layer || {}),
        status: rawOutputs.production_five_layer?.status || "available",
      },
    };
    const branches = report.branches && typeof report.branches === "object"
      ? report.branches as Record<string, { score?: unknown }>
      : {};
    const metadata = report.metadata && typeof report.metadata === "object"
      ? report.metadata as Record<string, unknown>
      : {};
    const localization = report.localization && typeof report.localization === "object"
      ? report.localization as Record<string, unknown>
      : {};
    const qualityMetrics = metadata.quality_metrics && typeof metadata.quality_metrics === "object"
      ? metadata.quality_metrics as Record<string, unknown>
      : {};
    const frequencyBranch = branches.frequency as { score?: unknown; metrics?: Record<string, unknown> } | undefined;
    const frequencyMetrics = frequencyBranch?.metrics || {};
    const numericMetric = (value: unknown): number | null =>
      typeof value === "number" && Number.isFinite(value) ? value : null;
    const layerNames = [
      ["Provenance and media integrity", ["provenance", "codec", "recompression"]],
      ["Temporal and motion consistency", ["temporal", "scene", "periodicity"]],
      ["Face geometry and spatial integrity", ["face", "integrity", "spatial"]],
      ["Frequency, noise, and stream diagnostics", ["frequency", "wavelet", "audio", "avsync"]],
      ["Robust evidence fusion", ["fusion"]],
    ] as const;
    const layers: LayerData[] = layerNames.map(([name, names], index) => {
      const values = names
        .map((branchName) => {
          const value = branches[branchName]?.score;
          return typeof value === "number" ? value : null;
        })
        .filter((value): value is number => value !== null);
      const layerScore = index === 4
        ? score
        : values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
      return {
        name,
        score: layerScore,
        stdDev: 0,
        explanation: `Deterministic evidence from ${names.join(", ")}.`,
        color: ["theta", "p300", "magenta", "shred", "neural-amber"][index],
        icon: "activity",
        metrics: Object.fromEntries(values.map((value, valueIndex) => [`signal_${valueIndex + 1}`, value])),
      };
    });
    const label = typeof fusion.label === "string" ? fusion.label : "deterministic-analysis";
    const branchScores = new Map(
      Object.entries(branches).map(([branchName, branch]) => [
        branchName,
        numericMetric(branch.score),
      ]),
    );
    const localizedEvents = Array.isArray(localization.events)
      ? localization.events
        .filter((event): event is Record<string, unknown> => Boolean(event) && typeof event === "object")
        .map((event) => {
          const timestamp = numericMetric(event.timestamp_seconds);
          const branchName = typeof event.branch === "string" ? event.branch : "diagnostic";
          return timestamp === null
            ? null
            : {
                frame_idx: Math.max(0, Math.round(numericMetric(event.frame_index) || 0)),
                timestamp_s: timestamp,
                manipulation_probability: branchScores.get(branchName) ?? score,
                dominant_signal: typeof event.kind === "string" ? event.kind : "diagnostic",
                evidence_scope: branchName,
              };
        })
        .filter((event): event is TimelineEntry => event !== null)
      : [];
    const localizedDuration = numericMetric(localization.duration_seconds);
    const timeline = localizedEvents.length
      ? localizedEvents
      : localizedDuration !== null
        ? [{
            frame_idx: 0,
            timestamp_s: localizedDuration,
            manipulation_probability: score,
            dominant_signal: "clip-wide fusion",
            evidence_scope: "full analyzed duration",
          }]
        : [];
    return {
      verdictPercent: score * 100,
      marginOfError: 0,
      confidenceLevel: "Moderate",
      reliabilityScore: Math.round((Number(fusion.evidence_coverage || 0) * 10) * 10) / 10,
      crossLayerConsistency: Number(
        (fusion.group_consensus &&
        typeof fusion.group_consensus === "object" &&
        "consensus_fraction" in fusion.group_consensus
          ? fusion.group_consensus.consensus_fraction
          : 0) || 0
      ),
      modelNonDeepfakeProbability: 1 - score,
      selfConsistencyValidation: "Passed",
      adversarialShift: 0,
      adversarialTests: { topAnomalyRemoved: null, compressionRemoved: null },
      compressionBiasDetected: Boolean(fusion.reason_codes?.includes?.("global-compression-confounder")),
      biasLog: "Compression and codec signals remain diagnostic, not decisive.",
      modelVersion: report.schema_version || "deterministic-forensic-engine",
      calibrationSource: "No learned calibration; deterministic evidence only",
      traceId: (
        report.metadata &&
        typeof report.metadata === "object" &&
        "sha256" in report.metadata &&
        typeof report.metadata.sha256 === "string"
          ? report.metadata.sha256
          : sourceName
      ),
      timestamp: new Date().toISOString(),
      layers: layers.map((layer) => ({
        ...layer,
        score: layer.score,
      })),
      fusionEngine: {
        posterior_probability: score,
        verdict_percent: score * 100,
        margin_of_error: 0,
        information_entropy_bits: 0,
        layer_contributions: layers.map((layer) => ({
          log_likelihood_ratio: layer.score - 0.5,
          evidence_strength: layer.score,
        })),
      },
      timeline,
      provenance: {
        estimated_reencoding_generations: numericMetric(qualityMetrics.estimated_reencoding_generations),
        compression_artifact_layers: numericMetric(qualityMetrics.compression_artifact_layers),
        original_quality_estimate: String(qualityMetrics.quality_tier || "unknown"),
        quantization_table_anomaly: typeof qualityMetrics.quantization_table_anomaly === "boolean"
          ? qualityMetrics.quantization_table_anomaly
          : null,
      },
      enhancedAudio: { metrics: {}, processing_time_ms: 0 },
      videoUrl: sourceName,
      videoId: sourceName,
      thumbnailUrl: "",
      humanVerdict: `Deterministic engine: ${label}. Score ${(score * 100).toFixed(1)}%.`,
      social: { x: "", redditTitle: "", redditBody: "", instagram: "", linkedin: "" },
      decisionStatus: "verdict",
      analysis_outputs: outputs,
      analysis_reconciliation: report.analysis_reconciliation as AnalysisResult["analysis_reconciliation"],
      deterministicReport: {
        metadata: report.metadata as AnalysisResult["deterministicReport"]["metadata"],
        sampling: report.sampling as AnalysisResult["deterministicReport"]["sampling"],
        branches,
        fusion,
        warnings: Array.isArray(report.warnings)
          ? report.warnings.filter((warning): warning is string => typeof warning === "string")
          : [],
      },
    };
  };

  const invokeAnalysis = async (url: string, mode: AnalysisMode, signal: AbortSignal, file?: File) => {
    const localApiUrl = import.meta.env.VITE_FORENSICS_API_URL || "http://127.0.0.1:8000";
    if (file || !validateYoutubeUrl(url)) {
      const response = await fetch(`${localApiUrl}/analyze`, {
        method: "POST",
        body: (() => {
          const form = new FormData();
          if (file) form.append("file", file);
          else form.append("video_path", url);
          form.append("samples", mode === "deep" ? "64" : "8");
          form.append("max_frames", mode === "deep" ? "48" : "4");
          return form;
        })(),
        signal,
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(payload?.detail || `Forensic API failed (${response.status})`);
      return normalizeLocalReport(payload, file?.name || url);
    }
    const response = await fetch(`${localApiUrl}/analyze`,
      {
        method: "POST",
        body: (() => {
          const form = new FormData();
          form.append("youtube_url", url);
          form.append("samples", mode === "deep" ? "64" : "8");
          form.append("max_frames", mode === "deep" ? "48" : "4");
          return form;
        })(),
        signal,
      }
    );

    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      const backendError = payload?.detail || payload?.error || `Analysis failed (${response.status})`;
      throw new Error(parseAnalysisError(backendError));
    }

    return normalizeLocalReport(payload, url);
  };

  const runAnalysis = useCallback(async (url: string, mode: AnalysisMode, file?: File) => {
    activeAnalysisController.current?.abort();
    const controller = new AbortController();
    activeAnalysisController.current = controller;
    setState("analyzing");
    setAnalysisProgress(0);
    setCurrentLayer(0);
    setError(null);

    const startTime = Date.now();
    const MIN_LOADER_MS = 12000;
    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const naturalProgress = Math.min(90, (elapsed / MIN_LOADER_MS) * 85 + Math.random() * 3);
      setAnalysisProgress(naturalProgress);
      setCurrentLayer(elapsed < MIN_LOADER_MS * 0.5 ? 0 : 1);
    }, 150);

    try {
      const data = await invokeAnalysis(url, mode, controller.signal, file);

      clearInterval(interval);

      const elapsed = Date.now() - startTime;
      const remaining = Math.max(0, MIN_LOADER_MS - elapsed);

      setAnalysisProgress(95);
      await new Promise((resolve) => setTimeout(resolve, remaining));

      setAnalysisProgress(100);
      setCurrentLayer(1);

      setTimeout(() => {
        setResult(data);
        setState("results");
      }, 500);
    } catch (e) {
      clearInterval(interval);
      if (e instanceof DOMException && e.name === "AbortError") return;
      console.error("Analysis error:", e);
      const message = e instanceof Error ? e.message : "Analysis failed. Please try again.";
      setError(parseAnalysisError(message));
      setState("hero");
    } finally {
      if (activeAnalysisController.current === controller) activeAnalysisController.current = null;
    }
  }, [accessToken]);

  const handleShred = () => {
    if (!localFile && !validateYoutubeUrl(youtubeUrl)) {
      setError("Please enter a valid YouTube URL");
      return;
    }
    runAnalysis(localFile ? localFile.name : youtubeUrl, analysisMode, localFile || undefined);
  };

  const handleReset = () => {
    setState("hero");
    setYoutubeUrl("");
    setLocalFile(null);
    setResult(null);
    setAnalysisProgress(0);
    setCurrentLayer(0);
    setError(null);
  };

  return {
    state,
    youtubeUrl,
    setYoutubeUrl,
    analysisProgress,
    currentLayer,
    result,
    error,
    analysisMode,
    setAnalysisMode,
    handleShred,
    handleReset,
    validateYoutubeUrl,
    extractVideoId,
    localFile,
    setLocalFile,
  };
}
