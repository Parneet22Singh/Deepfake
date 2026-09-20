import { useState } from "react";
import { motion } from "framer-motion";
import {
  ArrowLeft, Share2, RotateCcw, ShieldCheck, ShieldAlert,
  AlertTriangle, CheckCircle, Copy, Check,
  Activity, Cpu, AudioLines, Eye, LineChart, Fingerprint, History
} from "lucide-react";
import type { AnalysisResult, ProcessingMeta, TimelineEntry, FusionEngine, ProvenanceData, GanFingerprint, AnalysisOutputs } from "@/hooks/useNeuroforge";
import { LayerCard } from "./LayerCard";

interface ResultsDashboardProps {
  result: AnalysisResult;
  youtubeUrl: string;
  onReset: () => void;
}

type SocialTab = "x" | "reddit" | "instagram" | "linkedin";

const SOCIAL_ICONS: Record<SocialTab, string> = {
  x: "X", reddit: "R", instagram: "IG", linkedin: "IN",
};

const VerdictCard = ({ result }: { result: AnalysisResult }) => {
  const abstained = result.decisionStatus === "abstain";
  const isElevated = result.verdictPercent >= 62;
  const verdictLabel =
    abstained ? "ABSTAINED — INSUFFICIENT EVIDENCE" :
    isElevated ? "ELEVATED ANOMALY SIGNAL" : "MIXED ANOMALY SIGNAL";

  return (
    <motion.div initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.6, type: "spring" }} className="text-center mb-8">
      <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.2, type: "spring", stiffness: 200 }} className="inline-block mb-4">
        <div className="glass-card px-6 py-5" style={{
          boxShadow: abstained ? "none" : isElevated ? "var(--shadow-neon-red)" : "var(--shadow-neon-green)",
          borderColor: abstained ? "hsl(var(--neural-amber) / 0.4)" : isElevated ? "hsl(var(--shred-red) / 0.4)" : "hsl(var(--theta-green) / 0.4)",
        }}>
          <p className="font-display text-xs tracking-[0.2em] text-muted-foreground mb-1">DETERMINISTIC ANOMALY SIGNAL</p>
          <p className={`font-display text-4xl sm:text-5xl font-black ${abstained ? "neon-text-amber" : isElevated ? "neon-text-red" : "neon-text-green"}`}>
            {abstained ? "—" : `${result.verdictPercent.toFixed(1)}%`}
          </p>
          <p className={`font-display text-sm font-bold mt-1 tracking-wider ${abstained ? "neon-text-amber" : isElevated ? "neon-text-red" : "neon-text-green"}`}>{verdictLabel}</p>
          <p className="mt-2 max-w-xs text-[10px] font-mono text-muted-foreground">Not a calibrated probability or authenticity verdict.</p>
        </div>
      </motion.div>
    </motion.div>
  );
};

const ContentGateNotice = ({ result }: { result: AnalysisResult }) => {
  const context = result.contentContext;
  if (!context) return null;
  const closed = !context.evidence_gate;
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.35 }}
      className={`mb-4 rounded-lg border p-3 text-xs font-mono ${closed ? "border-[hsl(var(--neural-amber)/0.35)] text-[hsl(var(--neural-amber))] bg-[hsl(var(--neural-amber)/0.05)]" : "border-theta/30 text-theta bg-theta/5"}`}>
      <div className="flex items-center gap-2 font-bold uppercase tracking-wider">
        {closed ? <AlertTriangle className="w-3.5 h-3.5" /> : <CheckCircle className="w-3.5 h-3.5" />}
        CONTENT GATE: {context.content_class.replace(/_/g, " ")}
      </div>
      <p className="mt-1 opacity-80">
        Face presence {(context.face_presence_ratio * 100).toFixed(0)}% · ROI quality {(context.roi_quality * 100).toFixed(0)}% · {closed ? (result.abstentionReason || context.reason) : "Face-local evidence eligible; global rendering artifacts remain diagnostic only."}
      </p>
    </motion.div>
  );
};

const ReportFrames = ({ result }: { result: AnalysisResult }) => {
  const frames = result.reportFrames ?? [];
  if (!frames.length) return null;
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.38 }} className="mb-6">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-display text-xs tracking-[0.3em] text-muted-foreground uppercase">REPRESENTATIVE FRAMES</h3>
      </div>
      <div className="grid grid-cols-2 gap-3">
        {frames.slice(0, 2).map((frameUrl, index) => (
          <div key={`${frameUrl}-${index}`} className="overflow-hidden rounded-xl border border-border bg-muted/30">
            <img src={frameUrl} alt={`Representative frame ${index + 1}`} className="h-40 w-full object-cover" />
          </div>
        ))}
      </div>
    </motion.div>
  );
};

const MetaBadges = ({ result }: { result: AnalysisResult }) => (
  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }} className="flex flex-wrap justify-center gap-2 mb-4">
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono border ${
      result.confidenceLevel === "High" ? "border-theta/30 text-theta bg-theta/10" :
      result.confidenceLevel === "Moderate" ? "border-neural-amber/30 text-[hsl(var(--neural-amber))] bg-[hsl(var(--neural-amber)/0.1)]" :
      "border-shred/30 text-shred bg-shred/10"
    }`}>
      {result.confidenceLevel === "High" ? <CheckCircle className="w-3 h-3" /> : <AlertTriangle className="w-3 h-3" />}
      {result.confidenceLevel} Confidence
    </span>
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono border border-p300/30 text-p300 bg-p300/10">
      {result.reliabilityScore}/10 Reliability
    </span>
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono border ${
      result.selfConsistencyValidation === "Passed" ? "border-theta/30 text-theta bg-theta/10" :
      "border-[hsl(var(--neural-amber)/0.3)] text-[hsl(var(--neural-amber))] bg-[hsl(var(--neural-amber)/0.1)]"
    }`}>
      {result.selfConsistencyValidation === "Passed" ? <ShieldCheck className="w-3 h-3" /> : <ShieldAlert className="w-3 h-3" />}
      Self-Check: {result.selfConsistencyValidation}
    </span>
  </motion.div>
);

const BiasNotice = ({ result }: { result: AnalysisResult }) => {
  if (!result.compressionBiasDetected) return null;
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="flex justify-center mb-3">
      <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-mono border border-[hsl(var(--neural-amber)/0.3)] text-[hsl(var(--neural-amber))] bg-[hsl(var(--neural-amber)/0.05)]">
        <AlertTriangle className="w-3 h-3" />
        {result.biasLog}
      </span>
    </motion.div>
  );
};

const InterpretationNotice = ({ result }: { result: AnalysisResult }) => {
  if (result.interpretationStatus !== "unavailable") return null;
  return (
    <div className="flex justify-center mb-4">
      <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-mono border border-[hsl(var(--neural-amber)/0.3)] text-[hsl(var(--neural-amber))] bg-[hsl(var(--neural-amber)/0.05)]">
        <AlertTriangle className="w-3 h-3" />
        Narrative interpretation unavailable; numeric forensic results are shown.
      </span>
    </div>
  );
};

const RealProcessingBanner = ({ meta }: { meta?: ProcessingMeta }) => {
  if (!meta?.realSignalProcessing) return null;
  return (
    <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }}
      className="flex justify-center mb-4">
      <div className="inline-flex items-center gap-3 px-4 py-2 rounded-lg border border-theta/30 bg-theta/5">
        <Activity className="w-4 h-4 text-theta" />
        <span className="text-xs font-display font-bold tracking-wider text-theta">REAL SIGNAL PROCESSING ACTIVE</span>
      </div>
    </motion.div>
  );
};

const ProcessingStats = ({ meta }: { meta?: ProcessingMeta }) => {
  if (!meta) return null;
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.45 }}
      className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
      <div className="glass-card p-3 text-center">
        <Eye className="w-3.5 h-3.5 text-theta mx-auto mb-1" />
        <p className="text-[10px] font-mono text-muted-foreground">FRAMES</p>
        <p className="font-display text-lg font-bold neon-text-green">{meta.framesAnalyzed}</p>
      </div>
      <div className="glass-card p-3 text-center">
        <AudioLines className="w-3.5 h-3.5 text-p300 mx-auto mb-1" />
        <p className="text-[10px] font-mono text-muted-foreground">SPEECH SEGMENTS</p>
        <p className="font-display text-lg font-bold neon-text-blue">{meta.segmentsDetected}</p>
      </div>
      <div className="glass-card p-3 text-center">
        <Cpu className="w-3.5 h-3.5 text-magenta mx-auto mb-1" />
        <p className="text-[10px] font-mono text-muted-foreground">PROCESSING</p>
        <p className="font-display text-lg font-bold text-magenta">{(meta.totalProcessingMs / 1000).toFixed(1)}s</p>
      </div>
      <div className="glass-card p-3 text-center">
        <Activity className="w-3.5 h-3.5 text-theta mx-auto mb-1" />
        <p className="text-[10px] font-mono text-muted-foreground">LIP FRAMES</p>
        <p className="font-display text-lg font-bold neon-text-green">{meta.lipFramesAnalyzed}</p>
      </div>
    </motion.div>
  );
};

const ForensicStats = ({ result }: { result: AnalysisResult }) => (
  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }} className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
    {[
      { label: "CROSS-LAYER SYNC", value: `${(result.crossLayerConsistency * 100).toFixed(0)}%`, color: "neon-text-blue" },
      { label: "ANOMALY SCORE (NOT PROBABILITY)", value: `${result.verdictPercent.toFixed(1)}%`, color: "neon-text-blue" },
      { label: "MARGIN OF ERROR", value: `±${result.marginOfError.toFixed(1)}%`, color: result.marginOfError > 15 ? "neon-text-red" : "neon-text-blue" },
      { label: "ADV. SHIFT", value: `${result.adversarialShift.toFixed(1)}%`, color: result.adversarialShift > 10 ? "neon-text-red" : "neon-text-green" },
    ].map((stat) => (
      <div key={stat.label} className="glass-card p-4 text-center">
        <p className="text-xs font-mono text-muted-foreground mb-1">{stat.label}</p>
        <p className={`font-display text-xl font-bold ${stat.color}`}>{stat.value}</p>
      </div>
    ))}
  </motion.div>
);

const AdversarialSection = ({ result }: { result: AnalysisResult }) => (
  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.55 }} className="glass-card p-4 sm:p-5 mb-6">
    <h3 className="font-display text-xs tracking-[0.3em] text-muted-foreground mb-3 flex items-center gap-2">
      ADVERSARIAL SELF-VALIDATION
      <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${
        result.selfConsistencyValidation === "Passed" ? "bg-theta/10 text-theta" : "bg-[hsl(var(--neural-amber)/0.1)] text-[hsl(var(--neural-amber))]"
      }`}>{result.selfConsistencyValidation}</span>
    </h3>
    <div className="grid grid-cols-2 gap-3 text-xs font-mono">
      <div className="bg-muted/30 rounded-lg p-3">
        <p className="text-muted-foreground mb-1">Top anomaly removed</p>
        <p className="text-foreground font-bold">{result.adversarialTests.topAnomalyRemoved.toFixed(1)}%</p>
        <p className="text-muted-foreground/60 text-[10px] mt-1">deterministic branch perturbation</p>
      </div>
      <div className="bg-muted/30 rounded-lg p-3">
        <p className="text-muted-foreground mb-1">Compression removed</p>
        <p className="text-foreground font-bold">{result.adversarialTests.compressionRemoved.toFixed(1)}%</p>
        <p className="text-muted-foreground/60 text-[10px] mt-1">deterministic branch perturbation</p>
      </div>
    </div>
    <p className="text-[10px] text-muted-foreground/50 font-mono mt-2">
      Max shift: {result.adversarialShift.toFixed(1)}% — {result.adversarialShift < 10 ? "Score is stable under adversarial perturbation." : "Score shows sensitivity — interpret with caution."}
    </p>
  </motion.div>
);

const SocialSharing = ({ result, youtubeUrl }: { result: AnalysisResult; youtubeUrl: string }) => {
  const [socialTab, setSocialTab] = useState<SocialTab>("x");
  const [copied, setCopied] = useState(false);

  const getSocialContent = (): string => {
    switch (socialTab) {
      case "x": return result.social.x;
      case "reddit": return `${result.social.redditTitle}\n\n${result.social.redditBody}`;
      case "instagram": return result.social.instagram;
      case "linkedin": return result.social.linkedin;
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(getSocialContent() + "\n" + youtubeUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getSocialAction = () => {
    switch (socialTab) {
      case "x":
        window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(result.social.x)}&url=${encodeURIComponent(youtubeUrl)}`, "_blank");
        break;
      case "reddit":
        window.open(`https://www.reddit.com/submit?title=${encodeURIComponent(result.social.redditTitle)}&text=${encodeURIComponent(result.social.redditBody + "\n\n" + youtubeUrl)}`, "_blank");
        break;
      case "instagram":
        handleCopy();
        window.open("https://www.instagram.com/create/story", "_blank");
        break;
      case "linkedin":
        window.open(`https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(youtubeUrl)}&summary=${encodeURIComponent(result.social.linkedin)}`, "_blank");
        break;
    }
  };

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1 }} className="glass-card p-5 sm:p-6 mb-6">
      <div className="flex items-center gap-2 mb-4">
        <Share2 className="w-4 h-4 text-theta" />
        <h3 className="font-display text-xs tracking-[0.3em] text-muted-foreground">SHARE YOUR VERDICT</h3>
      </div>
      
      <div className="mb-4 p-3 bg-muted/30 rounded-lg border border-border">
        <p className="text-xs font-display tracking-[0.2em] text-muted-foreground mb-2">DETERMINISTIC ANALYSIS SUMMARY</p>
        <p className="text-sm text-foreground/90 leading-relaxed whitespace-pre-line">{result.humanVerdict}</p>
      </div>
      <div className="flex gap-2 mb-4">
        {(["x", "reddit", "instagram", "linkedin"] as SocialTab[]).map((tab) => (
          <button key={tab} onClick={() => setSocialTab(tab)}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-display font-bold tracking-wider transition-colors border ${
              socialTab === tab ? "border-theta/40 bg-theta/10 text-theta" : "border-border bg-muted/30 text-muted-foreground hover:text-foreground"
            }`}>
            <span>{SOCIAL_ICONS[tab]}</span>
            {tab === "x" ? "X" : tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>
      <div className="bg-muted/50 rounded-lg p-3 mb-4 font-mono text-xs text-muted-foreground whitespace-pre-line break-all">{getSocialContent()}</div>
      <div className="flex gap-3">
        <button onClick={getSocialAction} className="flex items-center gap-2 px-4 py-2.5 bg-theta/10 text-theta border border-theta/20 rounded-lg font-display text-xs font-bold tracking-wider hover:bg-theta/20 transition-colors">
          {socialTab === "x" ? "Post on X" : socialTab === "reddit" ? "Post on Reddit" : socialTab === "instagram" ? "Post on Instagram" : "Post on LinkedIn"}
        </button>
        <button onClick={handleCopy} className="flex items-center gap-2 px-4 py-2.5 bg-muted text-muted-foreground border border-border rounded-lg font-display text-xs font-bold tracking-wider hover:bg-muted/80 transition-colors">
          {copied ? <Check className="w-4 h-4 text-theta" /> : <Copy className="w-4 h-4" />}
          {copied ? "COPIED" : "COPY"}
        </button>
      </div>
    </motion.div>
  );
};

const ManipulationTimeline = ({ timeline }: { timeline: TimelineEntry[] }) => (
  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.65 }} className="glass-card p-5 mb-6">
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-2">
        <LineChart className="w-4 h-4 text-magenta" />
        <h3 className="font-display text-xs tracking-[0.3em] text-muted-foreground uppercase">Rendering Diagnostic Timeline</h3>
      </div>
      <span className="text-[10px] font-mono text-muted-foreground/40">NOT VERDICT EVIDENCE</span>
    </div>

    <div className="h-24 flex items-end gap-1 mb-2">
      {timeline.map((entry, i) => (
        <motion.div
          key={i}
          initial={{ height: 0 }}
          animate={{ height: `${Math.max(5, entry.manipulation_probability * 100)}%` }}
          transition={{ delay: 0.8 + i * 0.05, duration: 0.5 }}
          className={`group relative flex-1 rounded-t-sm transition-colors ${
            entry.manipulation_probability > 0.7 ? "bg-shred" :
            entry.manipulation_probability > 0.4 ? "bg-[hsl(var(--neural-amber))]" : "bg-theta/40"
          }`}
        >
          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity z-20 pointer-events-none">
            <div className="glass-card px-2 py-1 whitespace-nowrap text-[8px] font-mono border-p300/30">
              <p className="text-foreground font-bold">{(entry.manipulation_probability * 100).toFixed(1)}%</p>
              <p className="text-muted-foreground/60">{entry.timestamp_s}s | {entry.dominant_signal}</p>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
    <div className="flex justify-between text-[8px] font-mono text-muted-foreground/40 px-1">
      <span>0s</span>
      <span>VIDEO DURATION (SAMPLED)</span>
      <span>{timeline[timeline.length - 1]?.timestamp_s}s</span>
    </div>
  </motion.div>
);

const BayesianBreakdown = ({ fusion }: { fusion: FusionEngine }) => (
  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.7 }} className="glass-card p-5 mb-6 border-p300/20">
    <div className="flex items-center gap-2 mb-4">
      <Fingerprint className="w-4 h-4 text-p300" />
      <h3 className="font-display text-xs tracking-[0.3em] text-muted-foreground uppercase">Deterministic Evidence Fusion</h3>
    </div>
    <div className="space-y-3">
      {fusion.layer_contributions.map((layer, i) => (
        <div key={i} className="flex items-center gap-3">
          <div className="w-16 text-[10px] font-mono text-muted-foreground/60">LAYER {i + 1}</div>
          <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
            <div className={`h-full rounded-full transition-all duration-1000 ${
              layer.log_likelihood_ratio > 0 ? "bg-p300" : "bg-theta/30"
            }`} style={{ width: `${Math.min(100, Math.abs(layer.log_likelihood_ratio) * 20)}%` }} />
          </div>
          <div className="w-12 text-right text-[10px] font-mono font-bold">
            {layer.log_likelihood_ratio > 0 ? "+" : ""}{layer.log_likelihood_ratio.toFixed(2)}
          </div>
        </div>
      ))}
    </div>
    <div className="mt-4 pt-4 border-t border-border/50 flex justify-between items-center">
      <div>
        <p className="text-[10px] font-mono text-muted-foreground">INFORMATION ENTROPY</p>
        <p className="text-xs font-bold text-p300">{fusion.information_entropy_bits.toFixed(3)} bits</p>
      </div>
      <div className="text-right">
        <p className="text-[10px] font-mono text-muted-foreground">ANOMALY SCORE</p>
        <p className="text-xs font-bold text-p300">{fusion.verdict_percent.toFixed(2)}%</p>
      </div>
    </div>
  </motion.div>
);

const VideoProvenance = ({ provenance, gan }: { provenance: ProvenanceData; gan: GanFingerprint }) => (
  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.75 }} className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
    <div className="glass-card p-5">
      <div className="flex items-center gap-2 mb-4">
        <History className="w-4 h-4 text-theta" />
        <h3 className="font-display text-xs tracking-[0.3em] text-muted-foreground uppercase">Provenance Chain</h3>
      </div>
      <div className="space-y-3 font-mono text-[10px]">
        <div className="flex justify-between">
          <span className="text-muted-foreground">Original Quality</span>
          <span className="text-theta font-bold">{provenance.original_quality_estimate}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">Re-encoding Gens</span>
          <span className="text-foreground">{provenance.estimated_reencoding_generations}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">Quantization Anomaly</span>
          <span className={provenance.quantization_table_anomaly ? "text-shred" : "text-theta"}>
            {provenance.quantization_table_anomaly ? "DETECTED" : "NONE"}
          </span>
        </div>
      </div>
    </div>
    <div className="glass-card p-5 border-magenta/20">
      <div className="flex items-center gap-2 mb-4">
        <Cpu className="w-4 h-4 text-magenta" />
        <h3 className="font-display text-xs tracking-[0.3em] text-muted-foreground uppercase">Frequency and noise diagnostics</h3>
      </div>
      <div className="space-y-3 font-mono text-[10px]">
        <div className="flex justify-between">
          <span className="text-muted-foreground">Diagnostic class</span>
          <span className="text-magenta font-bold">{gan.spectral_signature_class}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">Spectral Score</span>
          <span className="text-foreground">{(gan.confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">HF Energy Ratio</span>
          <span className="text-foreground">{gan.hf_energy_ratio.toFixed(3)}</span>
        </div>
      </div>
    </div>
  </motion.div>
);

const AnalysisOutputsPanel = ({ outputs }: { outputs?: AnalysisOutputs }) => {
  if (!outputs) return null;
  const entries: Array<[string, AnalysisOutputs[keyof AnalysisOutputs]]> = [
    ["Deterministic engine", outputs.deterministic_engine],
    ["Specialist three-class router", outputs.specialist_three_class_router],
    ["Binary authenticity router", outputs.binary_authenticity_router],
    ["Production five-layer structure", outputs.production_five_layer],
  ];
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1 }} className="mb-8">
      <h2 className="font-display text-xs tracking-[0.3em] text-muted-foreground mb-4">
        SEPARATE ANALYSIS OUTPUTS
      </h2>
      <div className="grid gap-3 sm:grid-cols-2">
        {entries.map(([name, output]) => (
          <div key={name} className="glass-card p-4 border-border/70">
            <div className="flex items-center justify-between gap-3">
              <span className="font-display text-[10px] tracking-wider text-muted-foreground uppercase">{name}</span>
              <span className="font-mono text-[10px] text-theta">{output.status}</span>
            </div>
            <div className="mt-3 flex items-baseline justify-between">
              <span className="font-mono text-sm font-bold text-foreground">
                {output.label || "No classified label"}
              </span>
              {typeof output.score === "number" ? (
                <span className="font-mono text-xs text-p300">score {output.score.toFixed(3)}</span>
              ) : typeof output.confidence === "number" ? (
                <span className="font-mono text-xs text-p300">confidence {output.confidence.toFixed(3)}</span>
              ) : null}
            </div>
            {typeof output.margin === "number" && (
              <p className="mt-1 text-[10px] font-mono text-muted-foreground">
                decision margin {output.margin.toFixed(3)}
              </p>
            )}
            {output.status === "not_configured" && (
              <p className="mt-2 text-[10px] font-mono text-muted-foreground">
                Optional checkpoint output is not configured.
              </p>
            )}
            {output.status === "error" && (
              <p className="mt-2 text-[10px] font-mono text-shred">
                Router unavailable: {typeof output.error === "string" ? output.error : "dependency or checkpoint error"}
              </p>
            )}
            {name === "Production five-layer structure" && output.layers && (
              <div className="mt-3 space-y-2">
                {output.layers.map((layer) => (
                  <div key={layer.id}>
                    <div className="flex justify-between text-[10px] font-mono text-muted-foreground">
                      <span>{layer.name}</span>
                      <span>{typeof layer.score === "number" ? `${(layer.score * 100).toFixed(1)}%` : "n/a"}</span>
                    </div>
                    <div className="mt-1 h-1 rounded-full bg-muted overflow-hidden">
                      <div className="h-full bg-p300" style={{ width: `${Math.max(0, Math.min(100, (layer.score || 0) * 100))}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </motion.div>
  );
};

const DeterministicEvidencePanel = ({ result }: { result: AnalysisResult }) => {
  const branches = result.deterministicReport?.branches;
  if (!branches) return null;
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1 }} className="glass-card p-5 mb-8">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="w-4 h-4 text-theta" />
        <h2 className="font-display text-xs tracking-[0.3em] text-muted-foreground uppercase">Deterministic Evidence Branches</h2>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {Object.entries(branches).map(([name, branch]) => (
          <div key={name} className="rounded-lg border border-border/70 bg-muted/20 p-3">
            <div className="flex justify-between gap-2 text-[10px] font-mono">
              <span className="uppercase text-muted-foreground">{name}</span>
              <span className={branch.status === "ok" ? "text-theta" : "text-neural-amber"}>{branch.status}</span>
            </div>
            <p className="mt-2 font-display text-lg font-bold text-foreground">
              {typeof branch.score === "number" ? `${(branch.score * 100).toFixed(1)}%` : "n/a"}
            </p>
            {branch.findings?.[0] && <p className="mt-1 text-[10px] leading-4 text-muted-foreground">{branch.findings[0]}</p>}
          </div>
        ))}
      </div>
      {result.deterministicReport.warnings?.length ? (
        <p className="mt-4 text-[10px] font-mono text-neural-amber">
          Warnings: {result.deterministicReport.warnings.join(" · ")}
        </p>
      ) : null}
    </motion.div>
  );
};

export const ResultsDashboard = ({ result, youtubeUrl, onReset }: ResultsDashboardProps) => {
  return (
    <div className="relative min-h-screen pb-20">
      <div className="absolute inset-0" style={{ background: "var(--gradient-void)" }} />

      <div className="relative z-10 max-w-3xl mx-auto px-4 pt-6 sm:pt-10">
        
        <motion.button initial={{ opacity: 0 }} animate={{ opacity: 1 }} onClick={onReset}
          className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors mb-8 group">
          <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
          <span className="text-sm font-mono">NEW ANALYSIS</span>
        </motion.button>

        <VerdictCard result={result} />
        <ContentGateNotice result={result} />
        <RealProcessingBanner meta={result.processingMeta} />
        <ReportFrames result={result} />
        {result.reportNarrative && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }} className="mb-6 rounded-xl border border-border bg-muted/30 p-4 text-sm text-foreground/90 leading-relaxed">
            <div className="mb-2 font-display text-[10px] tracking-[0.3em] text-muted-foreground uppercase">FORENSIC SUMMARY</div>
            <p className="font-mono text-xs leading-6 whitespace-pre-line">{result.reportNarrative}</p>
          </motion.div>
        )}
        <MetaBadges result={result} />
        <BiasNotice result={result} />
        <InterpretationNotice result={result} />
        
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.45 }} className="text-center mb-6">
          <p className="text-[10px] font-mono text-theta/60 uppercase tracking-widest flex items-center justify-center gap-2">
            <Fingerprint className="w-3 h-3" />
            {result.processingMeta?.engineVersion || "NEUROFORGE CORE ENGINE v4.0"}
          </p>
        </motion.div>

        <ProcessingStats meta={result.processingMeta} />
        <AnalysisOutputsPanel outputs={result.analysis_outputs} />
        <DeterministicEvidencePanel result={result} />

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="text-center mb-6">
          <p className="text-sm text-muted-foreground font-mono max-w-md mx-auto break-all">{youtubeUrl}</p>
          <p className="text-[10px] text-muted-foreground/30 font-mono mt-1">Trace: {result.traceId} • {new Date(result.timestamp).toLocaleString()}</p>
        </motion.div>

        <ForensicStats result={result} />

        {result.timeline && <ManipulationTimeline timeline={result.timeline} />}
        {result.fusionEngine && <BayesianBreakdown fusion={result.fusionEngine} />}
        {result.provenance && (
           <VideoProvenance 
             provenance={result.provenance} 
             gan={(result.layers[2]?.metrics as unknown as { gan_fingerprint: GanFingerprint })?.gan_fingerprint || { spectral_signature_class: "No Anomalies", confidence: 0, hf_energy_ratio: 0 }} 
           />
        )}

        <AdversarialSection result={result} />

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }} className="mb-6">
          <h2 className="font-display text-xs tracking-[0.3em] text-muted-foreground mb-4">PRODUCTION FIVE-LAYER EVIDENCE</h2>
          <div className="space-y-3">
            {result.layers.map((layer, i) => (
              <LayerCard key={layer.name} layer={layer} index={i} />
            ))}
          </div>
        </motion.div>

        <SocialSharing result={result} youtubeUrl={youtubeUrl} />

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.1 }} className="text-center">
          <button onClick={onReset} className="inline-flex items-center gap-2 px-6 py-3 bg-muted text-foreground border border-border rounded-lg font-display text-xs font-bold tracking-wider hover:bg-muted/80 transition-colors">
            <RotateCcw className="w-4 h-4" /> ANALYZE ANOTHER VIDEO
          </button>
        </motion.div>

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.2 }} className="text-center mt-10 pb-8">
          <p className="text-xs text-muted-foreground/40 font-mono">{result.modelVersion} • {result.calibrationSource}</p>
          <p className="text-xs text-muted-foreground/30 font-mono mt-1">Results are probabilistic — no system can guarantee 100% certainty.</p>
        </motion.div>
      </div>
    </div>
  );
};
