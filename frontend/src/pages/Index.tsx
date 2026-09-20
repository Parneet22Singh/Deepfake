import { AnimatePresence, motion } from "framer-motion";
import { NeuralParticles } from "@/components/NeuralParticles";
import { HeroSection } from "@/components/HeroSection";
import { AnalysisLoader } from "@/components/AnalysisLoader";
import { ResultsDashboard } from "@/components/ResultsDashboard";
import { useNeuroforge } from "@/hooks/useNeuroforge";
import { AuthPanel } from "@/components/AuthPanel";

const Index = () => {
  const {
    state,
    youtubeUrl,
    setYoutubeUrl,
    analysisProgress,
    currentLayer,
    result,
    error,
    handleShred,
    handleReset,
    extractVideoId,
    analysisMode,
    setAnalysisMode,
    localFile,
    setLocalFile,
  } = useNeuroforge();

  const videoId = extractVideoId(youtubeUrl);
  const thumbnailUrl = videoId ? `https://img.youtube.com/vi/${videoId}/hqdefault.jpg` : undefined;

  return (
    <div className="min-h-screen bg-background neural-grid relative">
      <AuthPanel />
      
      <AnimatePresence mode="wait">
        {state === "hero" && (
          <motion.div
            key="hero"
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3 }}
          >
            <HeroSection
              youtubeUrl={youtubeUrl}
              setYoutubeUrl={setYoutubeUrl}
              onShred={handleShred}
              error={error}
              analysisMode={analysisMode}
              setAnalysisMode={setAnalysisMode}
              localFile={localFile}
              setLocalFile={setLocalFile}
            />
          </motion.div>
        )}

        {state === "analyzing" && (
          <motion.div
            key="analyzing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <AnalysisLoader
              progress={analysisProgress}
              currentLayer={currentLayer}
              thumbnailUrl={thumbnailUrl}
              videoId={videoId}
            />
          </motion.div>
        )}

        {state === "results" && result && (
          <motion.div
            key="results"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <ResultsDashboard
              result={result}
              youtubeUrl={youtubeUrl}
              onReset={handleReset}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Index;
