import { motion } from "framer-motion";
import { Zap, Shield, ExternalLink } from "lucide-react";
import { AnalysisMode } from "@/hooks/useNeuroforge";

interface HeroSectionProps {
  youtubeUrl: string;
  setYoutubeUrl: (url: string) => void;
  onShred: () => void;
  error: string | null;
  analysisMode: AnalysisMode;
  setAnalysisMode: (mode: AnalysisMode) => void;
  localFile: File | null;
  setLocalFile: (file: File | null) => void;
}

export const HeroSection = ({ youtubeUrl, setYoutubeUrl, onShred, error, analysisMode, setAnalysisMode, localFile, setLocalFile }: HeroSectionProps) => {
  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center px-4 overflow-hidden">
      
      <div className="absolute inset-0 opacity-20" style={{ background: "radial-gradient(circle at center, hsl(var(--p300) / 0.18), transparent 65%)" }} />

      <div className="absolute inset-0" style={{ background: "var(--gradient-void)" }} />

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="relative z-10 w-full max-w-3xl text-center"
      >
        
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="inline-flex items-center gap-2 glass-card px-4 py-2 mb-8"
        >
          <Shield className="w-4 h-4 text-theta" />
          <span className="text-sm font-medium text-muted-foreground">
            Deterministic computer vision • Optional protected routers
          </span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="font-display text-4xl sm:text-5xl md:text-7xl font-black tracking-tight mb-2"
        >
          <span className="gradient-text-neural">NEUROFORGE</span>
        </motion.h1>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="font-display text-lg sm:text-xl md:text-2xl font-bold tracking-[0.3em] text-muted-foreground mb-6"
        >
          ALPHA-1
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="text-base sm:text-lg text-muted-foreground mb-10 max-w-xl mx-auto leading-relaxed"
        >
          Analyze local video with transparent deterministic forensic evidence and
          optional advisory routers.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="w-full max-w-2xl mx-auto"
        >
          <div className="glass-card p-2 animate-pulse-neon">
            <div className="flex flex-col sm:flex-row gap-2">
              <input
                type="url"
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && onShred()}
                disabled={Boolean(localFile)}
                placeholder={localFile ? localFile.name : "https://youtube.com/watch?v=... or Shorts link"}
                className="flex-1 bg-transparent border-none outline-none px-4 py-4 text-foreground text-base sm:text-lg placeholder:text-muted-foreground/50 font-mono"
              />
              <button
                onClick={onShred}
                className="group relative px-6 py-4 bg-primary text-primary-foreground font-display font-bold text-sm sm:text-base tracking-wider rounded-lg overflow-hidden transition-all duration-300 hover:shadow-[var(--shadow-neon-green)] active:scale-95 whitespace-nowrap"
              >
                <span className="relative z-10 flex items-center justify-center gap-2">
                  <Zap className="w-5 h-5" />
                  SHRED REALITY
                </span>
                <div className="absolute inset-0 bg-gradient-to-r from-primary via-secondary to-primary opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              </button>
            </div>
            <div className="flex items-center justify-center gap-3 px-4 pt-3 text-xs font-mono text-muted-foreground">
              <label className="cursor-pointer rounded-lg border border-border px-3 py-2 hover:border-theta/40 hover:text-theta">
                {localFile ? "CHANGE LOCAL VIDEO" : "ANALYZE LOCAL VIDEO"}
                <input
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={(event) => {
                    const file = event.target.files?.[0] || null;
                    setLocalFile(file);
                    if (file) setYoutubeUrl("");
                  }}
                />
              </label>
              {localFile && (
                <button type="button" onClick={() => setLocalFile(null)} className="text-muted-foreground hover:text-shred">
                  CLEAR
                </button>
              )}
            </div>
            <div className="flex items-center justify-center gap-3 px-4 py-3 text-sm text-muted-foreground">
              <span>Quick Scan</span>
              <button
                type="button"
                role="switch"
                aria-checked={analysisMode === "deep"}
                onClick={() => setAnalysisMode(analysisMode === "deep" ? "quick" : "deep")}
                className={`relative flex h-6 w-11 shrink-0 items-center rounded-full p-0 transition-colors ${analysisMode === "deep" ? "bg-primary" : "bg-muted"}`}
              >
                <span className={`block h-4 w-4 rounded-full bg-background transition-transform ${analysisMode === "deep" ? "translate-x-6" : "translate-x-1"}`} />
              </button>
              <span className={analysisMode === "deep" ? "text-primary" : ""}>Deep Forensics</span>
            </div>
          </div>

          {error && (
            <motion.p
              initial={{ opacity: 0, y: -5 }}
              animate={{ opacity: 1, y: 0 }}
              className="neon-text-red text-sm mt-3"
            >
              {error}
            </motion.p>
          )}
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="text-xs text-muted-foreground/60 mt-6 flex items-center justify-center gap-1"
        >
          {analysisMode === "deep" ? "Full video • Native frequency frames • 5 layers" : "First 30s • 720p • 5 layers"}
        </motion.p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.4 }}
        transition={{ delay: 1.2 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2"
      >
        <div className="w-5 h-8 rounded-full border border-muted-foreground/30 flex items-start justify-center p-1">
          <motion.div
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="w-1.5 h-1.5 rounded-full bg-theta"
          />
        </div>
      </motion.div>
    </div>
  );
};
