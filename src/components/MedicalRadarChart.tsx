import { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { Activity, Zap, Layers } from 'lucide-react';

interface RadiomicsFeatures {
  meanIntensity: number;
  contrast: number;
  skewness: number;
  entropy: number;
  edgeDensity: number;
  homogeneity: number;
}

interface MedicalRadarChartProps {
  features: RadiomicsFeatures | undefined;
}

export default function MedicalRadarChart({ features }: MedicalRadarChartProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [rotation, setRotation] = useState(0);

  // Animate the radar sweep line
  useEffect(() => {
    let animationFrameId: number;
    const animate = () => {
      setRotation((prev) => (prev + 1) % 360);
      animationFrameId = requestAnimationFrame(animate);
    };
    animationFrameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrameId);
  }, []);

  if (!features) {
    return (
      <div className="h-64 bg-surface/30 border border-border rounded-3xl border-dashed flex flex-col items-center justify-center p-6 text-center">
        <Activity className="w-8 h-8 text-text-muted/40 animate-pulse mb-3" />
        <span className="text-xs text-text-muted font-bold uppercase tracking-wider">Awaiting Feature Extraction</span>
        <span className="text-[10px] text-text-muted/60 mt-1">Extract radiomics to run real-time structural geometry scanning.</span>
      </div>
    );
  }

  // Parse raw features or fall back to mock standard bounds
  const data = [
    { label: "Mean Density", raw: features.meanIntensity, val: Math.min(1, features.meanIntensity / 255), desc: "Mean gray-level voxel intensity" },
    { label: "Contrast", raw: features.contrast, val: Math.min(1, features.contrast / 100), desc: "Local variation standard deviation" },
    { label: "Skewness", raw: features.skewness, val: Math.min(1, Math.abs(features.skewness) / 5), desc: "Voxel value asymmetry index" },
    { label: "Entropy", raw: features.entropy, val: Math.min(1, features.entropy / 8), desc: "Tissue randomness level" },
    { label: "Edge Detail", raw: features.edgeDensity, val: Math.min(1, features.edgeDensity / 0.5), desc: "High-frequency edge sharpness" },
    { label: "Homogeneity", raw: features.homogeneity, val: Math.min(1, features.homogeneity / 1.2), desc: "Structural texture uniformity" },
  ];

  const size = 300;
  const center = size / 2;
  const radius = center - 40;

  // Polar coordinate helper
  const getCoordinates = (index: number, value: number) => {
    const angle = (Math.PI * 2 / data.length) * index - Math.PI / 2;
    const x = center + radius * value * Math.cos(angle);
    const y = center + radius * value * Math.sin(angle);
    return { x, y };
  };

  // Generate radar polygon points
  const points = data.map((d, i) => {
    const { x, y } = getCoordinates(i, d.val);
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="bg-surface/60 border border-border p-6 rounded-[32px] shadow-2xl relative overflow-hidden flex flex-col lg:flex-row gap-6 items-center">
      <div className="absolute top-0 right-0 w-32 h-32 bg-accent-blue/5 rounded-full blur-3xl pointer-events-none" />
      
      {/* Visual Chart Canvas */}
      <div className="relative w-[300px] h-[300px] shrink-0">
        <svg width={size} height={size} className="overflow-visible">
          {/* Definitions for Glow/Gradients */}
          <defs>
            <radialGradient id="radarGlow" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#00E5FF" stopOpacity="0.15" />
              <stop offset="100%" stopColor="#000000" stopOpacity="0" />
            </radialGradient>
            <linearGradient id="polyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#00E5FF" stopOpacity="0.6" />
              <stop offset="100%" stopColor="#10B981" stopOpacity="0.3" />
            </linearGradient>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="6" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Glowing Center Core */}
          <circle cx={center} cy={center} r={radius} fill="url(#radarGlow)" />

          {/* Grid Circles */}
          {[0.2, 0.4, 0.6, 0.8, 1.0].map((level, idx) => (
            <circle
              key={idx}
              cx={center}
              cy={center}
              r={radius * level}
              fill="none"
              stroke="#1E293B"
              strokeWidth="1"
              strokeDasharray={idx === 4 ? "0" : "3, 3"}
            />
          ))}

          {/* Grid Axis Lines */}
          {data.map((_, i) => {
            const { x, y } = getCoordinates(i, 1);
            return (
              <line
                key={i}
                x1={center}
                y1={center}
                x2={x}
                y2={y}
                stroke="#1E293B"
                strokeWidth="1"
              />
            );
          })}

          {/* Live Scanning Sweep Line */}
          <line
            x1={center}
            y1={center}
            x2={center + radius * Math.cos((rotation * Math.PI) / 180 - Math.PI / 2)}
            y2={center + radius * Math.sin((rotation * Math.PI) / 180 - Math.PI / 2)}
            stroke="#00E5FF"
            strokeWidth="1.5"
            strokeOpacity="0.4"
          />

          {/* Radar Polygon Shape */}
          <polygon
            points={points}
            fill="url(#polyGrad)"
            stroke="#00E5FF"
            strokeWidth="2"
            filter="url(#glow)"
            className="transition-all duration-700"
          />

          {/* Data Vertex Dots */}
          {data.map((d, i) => {
            const { x, y } = getCoordinates(i, d.val);
            return (
              <g
                key={i}
                onMouseEnter={() => setHoveredIndex(i)}
                onMouseLeave={() => setHoveredIndex(null)}
                className="cursor-pointer group"
              >
                <circle
                  cx={x}
                  cy={y}
                  r={hoveredIndex === i ? 7 : 4}
                  fill={hoveredIndex === i ? "#FFFFFF" : "#00E5FF"}
                  stroke="#0E131F"
                  strokeWidth="2"
                  className="transition-all duration-200"
                  filter={hoveredIndex === i ? "url(#glow)" : ""}
                />
              </g>
            );
          })}

          {/* Labels */}
          {data.map((d, i) => {
            const { x, y } = getCoordinates(i, 1.22);
            return (
              <text
                key={i}
                x={x}
                y={y}
                textAnchor="middle"
                alignmentBaseline="middle"
                className={`text-[8px] font-black uppercase tracking-wider font-mono ${
                  hoveredIndex === i ? "fill-accent-blue scale-105" : "fill-text-muted"
                } transition-all duration-200`}
              >
                {d.label}
              </text>
            );
          })}
        </svg>
      </div>

      {/* Info Stats Panel */}
      <div className="flex-1 w-full space-y-4">
        <div className="flex items-center gap-2 border-b border-border pb-3">
          <Layers className="w-4 h-4 text-accent-blue" />
          <h4 className="text-xs font-black uppercase tracking-wider text-accent">Quantitative Voxel Scans</h4>
        </div>

        <div className="space-y-3">
          {data.map((d, i) => (
            <div 
              key={i} 
              onMouseEnter={() => setHoveredIndex(i)}
              onMouseLeave={() => setHoveredIndex(null)}
              className={`p-3 rounded-2xl border transition-all flex items-center justify-between ${
                hoveredIndex === i 
                  ? "bg-accent-blue/10 border-accent-blue/30 shadow-md translate-x-1" 
                  : "bg-bg/40 border-border"
              }`}
            >
              <div className="space-y-0.5">
                <span className="text-[10px] font-bold text-accent block leading-none">{d.label}</span>
                <span className="text-[8px] text-text-muted block leading-none">{d.desc}</span>
              </div>
              <div className="text-right">
                <span className="text-xs font-black font-mono text-accent-blue">{d.raw}</span>
                <div className="w-16 h-1 bg-border rounded-full overflow-hidden mt-1.5">
                  <div className="h-full bg-accent-blue" style={{ width: `${d.val * 100}%` }} />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
