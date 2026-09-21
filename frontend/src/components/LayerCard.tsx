import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { LayerData } from "@/hooks/useNeuroforge";

const COLOR_MAP: Record<string, string> = {
  theta: "neon-text-green",
  p300: "neon-text-blue",
  magenta: "neon-text-magenta",
  shred: "neon-text-red",
  "shred-red": "neon-text-red",
  "neural-amber": "neon-text-amber",
};

const BAR_COLOR_MAP: Record<string, string> = {
  theta: "bg-theta",
  p300: "bg-p300",
  magenta: "bg-magenta",
  shred: "bg-shred",
  "shred-red": "bg-shred",
  "neural-amber": "bg-neural-amber",
};

interface LayerCardProps {
  layer: LayerData;
  index: number;
}

export const LayerCard = ({ layer, index }: LayerCardProps) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 + index * 0.15 }}
      className="glass-card cursor-pointer"
      onClick={() => setExpanded(!expanded)}
    >
      <div className="p-5 sm:p-6">
        
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <span className="text-sm font-mono text-muted-foreground/60">L{index + 1}</span>
            <div>
              <p className="text-xs font-mono text-muted-foreground mb-0.5">
                LAYER {index + 1}
              </p>

              <h3 className="font-display text-sm sm:text-base font-semibold">
                {layer.name}
              </h3>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className={`font-display text-2xl font-bold ${COLOR_MAP[layer.color]}`}>
              {typeof layer.score === "number" ? `${(layer.score * 100).toFixed(0)}%` : "n/a"}
            </span>
            {expanded ? (
              <ChevronUp className="w-5 h-5 text-muted-foreground" />
            ) : (
              <ChevronDown className="w-5 h-5 text-muted-foreground" />
            )}
          </div>
        </div>

        <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${typeof layer.score === "number" ? layer.score * 100 : 0}%` }}
            transition={{ delay: 0.5 + index * 0.15, duration: 0.8 }}
            className={`h-full rounded-full ${BAR_COLOR_MAP[layer.color]}`}
          />
        </div>

        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="overflow-hidden"
            >
              <p className="text-sm text-muted-foreground mt-4 leading-relaxed">
                {layer.explanation}
              </p>
              {layer.score == null && (
                <p className="mt-3 text-xs font-mono text-muted-foreground/70">
                  No usable score was produced for this layer because its contributing evidence branches did not return an authenticity score for this video.
                </p>
              )}

              <div className="grid grid-cols-2 gap-3 mt-4">
                {Object.entries(layer.metrics).map(([key, value]) => (
                  <div key={key} className="bg-muted/50 rounded-lg p-3">
                    <p className="text-xs font-mono text-muted-foreground mb-1">
                      {key.replace(/_/g, " ").toUpperCase()}
                    </p>
                    <p className={`font-display text-lg font-bold ${COLOR_MAP[layer.color]}`}>
                      {typeof value === "number" && value < 1
                        ? (value * 100).toFixed(1) + "%"
                        : value}
                    </p>
                  </div>
                ))}
              </div>

              <div className="mt-4 h-16 flex items-end gap-0.5 overflow-hidden rounded-lg bg-muted/30 p-2">
                {Array.from({ length: 40 }).map((_, i) => {
                  const metricValues = Object.values(layer.metrics).filter((value): value is number => typeof value === "number");
                  const source = metricValues[i % Math.max(1, metricValues.length)] ?? layer.score ?? 0;
                  const height = 20 + Math.min(60, Math.max(0, Math.abs(source) * 60));
                  return (
                    <motion.div
                      key={i}
                      initial={{ height: 0 }}
                      animate={{ height: `${height}%` }}
                      transition={{ delay: 0.8 + i * 0.02, duration: 0.5 }}
                      className={`flex-1 rounded-t-sm ${BAR_COLOR_MAP[layer.color]} opacity-60`}
                    />
                  );
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};
