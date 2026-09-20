import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

const LAYER_NAMES = [
  "Pixel Fracture Analysis",
  "Audio-Visual Sync Drift",
];

const LAYER_COLORS = ["bg-theta", "bg-p300"];
const LAYER_DESCRIPTIONS = [
  "Scanning frame-by-frame for GAN artifacts, morphing signatures, iris geometry anomalies...",
  "Analyzing lip-sync drift, phoneme alignment, spectral consistency, formant patterns...",
];
const STATUS_TEXTS = [
  "Extracting temporal keyframes...",
  "Computing pixel fracture gradient map...",
  "Analyzing GAN fingerprint residuals...",
  "Isolating facial boundary morphing vectors...",
  "Measuring iris ellipse geometry deviation...",
  "Decoding audio spectrogram...",
  "Correlating lip-sync phoneme drift...",
  "Computing bilabial consonant alignment...",
  "Evaluating spectral formant naturalness...",
  "Cross-referencing neural signature database...",
];

interface AnalysisLoaderProps {
  progress: number;
  currentLayer: number;
  thumbnailUrl?: string;
  videoId?: string;
}

const ScanLineOverlay = () => (
  <motion.div
    className="absolute inset-0 pointer-events-none z-20"
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
  >
    <motion.div
      className="absolute left-0 right-0 h-[2px] z-30"
      style={{
        background: "linear-gradient(90deg, transparent, hsl(var(--theta-green)), transparent)",
        boxShadow: "0 0 20px hsl(var(--theta-green) / 0.8), 0 0 60px hsl(var(--theta-green) / 0.3)",
      }}
      animate={{ top: ["0%", "100%", "0%"] }}
      transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
    />
  </motion.div>
);

const DetectionBoxes = ({ seed }: { seed: number }) => {
  const boxes = Array.from({ length: 3 }).map((_, i) => ({
    x: 15 + ((seed * (i + 1) * 37) % 50),
    y: 10 + ((seed * (i + 1) * 23) % 60),
    w: 20 + ((seed * (i + 1) * 13) % 30),
    h: 15 + ((seed * (i + 1) * 17) % 25),
  }));

  return (
    <>
      {boxes.map((box, i) => (
        <motion.div
          key={i}
          className="absolute border z-20 pointer-events-none"
          style={{
            left: `${box.x}%`,
            top: `${box.y}%`,
            width: `${box.w}%`,
            height: `${box.h}%`,
            borderColor: i === 0 ? "hsl(var(--shred-red) / 0.8)" : "hsl(var(--theta-green) / 0.6)",
          }}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{
            opacity: [0, 1, 1, 0],
            scale: [0.8, 1, 1, 0.9],
          }}
          transition={{
            duration: 2,
            delay: i * 0.7,
            repeat: Infinity,
            repeatDelay: 1,
          }}
        >
          <span
            className="absolute -top-5 left-0 font-mono text-[10px] whitespace-nowrap"
            style={{
              color: i === 0 ? "hsl(var(--shred-red))" : "hsl(var(--theta-green))",
            }}
          >
            {i === 0 ? "ANOMALY" : "SCAN"} {(0.3 + i * 0.25).toFixed(2)}
          </span>
        </motion.div>
      ))}
    </>
  );
};

const AudioWaveform = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const offsetRef = useRef(0);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    offsetRef.current += 2;

    ctx.beginPath();
    ctx.strokeStyle = "hsl(192, 100%, 50%)";
    ctx.lineWidth = 1.5;
    ctx.shadowColor = "hsl(192, 100%, 50%)";
    ctx.shadowBlur = 8;

    for (let x = 0; x < W; x++) {
      const t = (x + offsetRef.current) * 0.03;
      const y =
        H / 2 +
        Math.sin(t) * H * 0.2 +
        Math.sin(t * 2.7) * H * 0.1 +
        Math.sin(t * 5.3) * H * 0.05 +
        (Math.random() - 0.5) * 4;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.beginPath();
    ctx.strokeStyle = "hsl(330, 100%, 50%)";
    ctx.lineWidth = 1;
    ctx.shadowColor = "hsl(330, 100%, 50%)";
    ctx.shadowBlur = 6;

    for (let x = 0; x < W; x++) {
      const t = (x + offsetRef.current * 0.7) * 0.05;
      const y =
        H / 2 +
        Math.sin(t + 1) * H * 0.15 +
        Math.sin(t * 3.1 + 2) * H * 0.08;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    animRef.current = requestAnimationFrame(draw);
  }, []);

  useEffect(() => {
    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      width={400}
      height={80}
      className="w-full h-20 rounded-lg"
      style={{ background: "hsl(0 0% 4% / 0.6)" }}
    />
  );
};

const HexDataStream = () => {
  const [lines, setLines] = useState<string[]>([]);

  useEffect(() => {
    const genLine = () => {
      const hex = Array.from({ length: 16 })
        .map(() => Math.floor(Math.random() * 256).toString(16).padStart(2, "0"))
        .join(" ");
      const addr = Math.floor(Math.random() * 0xffff).toString(16).padStart(4, "0");
      return `0x${addr}: ${hex}`;
    };

    const interval = setInterval(() => {
      setLines((prev) => {
        const next = [...prev, genLine()];
        return next.slice(-8);
      });
    }, 120);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="font-mono text-[9px] text-muted-foreground/40 leading-tight overflow-hidden h-[72px]">
      {lines.map((line, i) => (
        <motion.div
          key={`${i}-${line}`}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 0.6, x: 0 }}
          className="whitespace-nowrap"
        >
          {line}
        </motion.div>
      ))}
    </div>
  );
};

const FrameCounter = ({ progress }: { progress: number }) => {
  const frame = Math.floor(progress * 9);
  const totalFrames = 900;
  return (
    <div className="flex items-center gap-4 font-mono text-xs">
      <div>
        <span className="text-muted-foreground/50">FRAME </span>
        <span className="neon-text-green">{frame}</span>
        <span className="text-muted-foreground/50">/{totalFrames}</span>
      </div>
      <div>
        <span className="text-muted-foreground/50">FPS </span>
        <span className="neon-text-blue">29.97</span>
      </div>
      <div>
        <span className="text-muted-foreground/50">RES </span>
        <span className="text-muted-foreground/70">1920×1080</span>
      </div>
    </div>
  );
};

export const AnalysisLoader = ({
  progress,
  currentLayer,
  thumbnailUrl,
  videoId,
}: AnalysisLoaderProps) => {
  const [statusText, setStatusText] = useState("");
  useEffect(() => {
    let idx = 0;
    const interval = setInterval(() => {
      setStatusText(STATUS_TEXTS[idx % STATUS_TEXTS.length]);
      idx++;
    }, 1800);
    setStatusText(STATUS_TEXTS[0]);
    return () => clearInterval(interval);
  }, []);

  const seed = videoId
    ? videoId.split("").reduce((a, c) => a + c.charCodeAt(0), 0)
    : 42;

  return (
    <div className="relative min-h-screen flex items-center justify-center px-4">
      <div className="absolute inset-0" style={{ background: "var(--gradient-void)" }} />

      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="relative z-10 w-full max-w-4xl"
      >
        
        <div className="text-center mb-6">
          <motion.h2
            className="font-display text-xl sm:text-2xl font-bold gradient-text-neural mb-1"
            animate={{ opacity: [0.7, 1, 0.7] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            SHREDDING REALITY
          </motion.h2>
          <FrameCounter progress={progress} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          
          <div className="glass-card p-3 sm:p-4">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-xs font-mono text-muted-foreground/60">LAYER 1</span>
              <span className="text-xs font-display font-bold neon-text-green">PIXEL FRACTURE</span>
              <motion.span
                className="w-2 h-2 rounded-full bg-theta ml-auto"
                animate={{ opacity: [1, 0.3, 1] }}
                transition={{ duration: 0.8, repeat: Infinity }}
              />
            </div>

            <div className="relative aspect-video rounded-lg overflow-hidden bg-muted/20 mb-3">
              {thumbnailUrl ? (
                <img
                  src={thumbnailUrl}
                  alt="Video frame"
                  className="w-full h-full object-cover"
                  style={{ filter: "brightness(0.7) contrast(1.2) saturate(0.8)" }}
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <span className="text-sm font-mono text-muted-foreground/30">NO FRAME</span>
                </div>
              )}
              <ScanLineOverlay />
              <DetectionBoxes seed={seed} />

              <div className="absolute bottom-2 left-2 right-2 z-20 flex justify-between">
                <span className="font-mono text-[10px] text-theta/80 bg-background/60 px-1.5 py-0.5 rounded">
                  SCAN ACTIVE
                </span>
                <span className="font-mono text-[10px] text-shred/80 bg-background/60 px-1.5 py-0.5 rounded">
                  {(progress * 0.01 * 100).toFixed(1)}% SCANNED
                </span>
              </div>
            </div>

            <HexDataStream />
          </div>

          <div className="space-y-4">
            <div className="glass-card p-3 sm:p-4">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xs font-mono text-muted-foreground/60">LAYER 2</span>
                <span className="text-xs font-display font-bold neon-text-blue">AUDIO-VISUAL SYNC</span>
                <motion.span
                  className="w-2 h-2 rounded-full bg-p300 ml-auto"
                  animate={{ opacity: [1, 0.3, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                />
              </div>

              <AudioWaveform />

              <div className="mt-3 grid grid-cols-3 gap-2 font-mono text-[10px]">
                <div className="bg-muted/30 rounded p-2 text-center">
                  <div className="text-muted-foreground/50">DRIFT</div>
                  <motion.div
                    className="neon-text-blue text-sm font-bold"
                    animate={{ opacity: [0.6, 1, 0.6] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  >
                    {Math.floor(progress * 2.3 + 30)}ms
                  </motion.div>
                </div>
                <div className="bg-muted/30 rounded p-2 text-center">
                  <div className="text-muted-foreground/50">SYNC</div>
                  <div className="neon-text-green text-sm font-bold">
                    {(100 - progress * 0.4).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-muted/30 rounded p-2 text-center">
                  <div className="text-muted-foreground/50">PHONEME</div>
                  <div className="text-magenta text-sm font-bold">
                    {(progress * 0.8).toFixed(0)}/128
                  </div>
                </div>
              </div>
            </div>

            <div className="glass-card p-3 sm:p-4">
              <div className="space-y-3">
                {LAYER_NAMES.map((name, i) => (
                  <div key={name}>
                    <div className="flex items-center gap-2 mb-1">
                      <div className="relative w-4 h-4 flex items-center justify-center">
                        {i < currentLayer ? (
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            className={`w-3 h-3 rounded-full ${LAYER_COLORS[i]}`}
                          />
                        ) : i === currentLayer ? (
                          <motion.div
                            animate={{ scale: [0.8, 1.2, 0.8] }}
                            transition={{ duration: 0.8, repeat: Infinity }}
                            className={`w-3 h-3 rounded-full ${LAYER_COLORS[i]} opacity-80`}
                          />
                        ) : (
                          <div className="w-3 h-3 rounded-full border border-muted-foreground/20" />
                        )}
                      </div>
                      <span
                        className={`text-xs font-mono ${
                          i <= currentLayer ? "text-foreground" : "text-muted-foreground/30"
                        }`}
                      >
                        L{i + 1}: {name}
                      </span>
                      {i === currentLayer && (
                        <motion.span
                          animate={{ opacity: [0, 1, 0] }}
                          transition={{ duration: 0.6, repeat: Infinity }}
                          className="text-theta text-xs ml-auto"
                        >
                          ●
                        </motion.span>
                      )}
                    </div>
                    <AnimatePresence>
                      {i === currentLayer && (
                        <motion.p
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          className="text-[10px] text-muted-foreground/50 ml-6 leading-relaxed"
                        >
                          {LAYER_DESCRIPTIONS[i]}
                        </motion.p>
                      )}
                    </AnimatePresence>
                  </div>
                ))}
              </div>
            </div>

            <div className="glass-card p-3">
              <motion.p
                key={statusText}
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-xs font-mono text-muted-foreground/60 text-center"
              >
                {statusText}
              </motion.p>
            </div>
          </div>
        </div>

        <div className="mt-4">
          <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{ background: "var(--gradient-neural)" }}
              initial={{ width: "0%" }}
              animate={{ width: `${progress}%` }}
              transition={{ ease: "easeOut" }}
            />
          </div>
          <div className="flex justify-between mt-1">
            <span className="text-[10px] font-mono text-muted-foreground/40">
              NEUROFORGE ALPHA-1
            </span>
            <motion.span
              className="text-[10px] font-mono text-theta/60"
              animate={{ opacity: [0.4, 0.8, 0.4] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            >
              {Math.round(progress)}% COMPLETE
            </motion.span>
          </div>
        </div>
      </motion.div>
    </div>
  );
};
