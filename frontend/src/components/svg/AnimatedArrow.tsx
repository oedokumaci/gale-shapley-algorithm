import { useMemo } from 'react';

interface AnimatedArrowProps {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: 'neutral' | 'green' | 'red';
  visible: boolean;
  index: number;
  dashed?: boolean;
}

const STROKE_COLORS = {
  neutral: '#94a3b8',
  green: '#22c55e',
  red: '#ef4444',
} as const;

export function AnimatedArrow({ x1, y1, x2, y2, color, visible, index, dashed }: AnimatedArrowProps) {
  const stroke = STROKE_COLORS[color];
  const gradientId = `arrow-grad-${index}`;

  // Bezier control points for a smooth S-curve
  const dx = x2 - x1;
  const cp1x = x1 + dx * 0.4;
  const cp1y = y1;
  const cp2x = x1 + dx * 0.6;
  const cp2y = y2;

  const pathD = `M ${x1} ${y1} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${x2} ${y2}`;

  // Arrowhead at the end — compute tangent direction from last control point
  const arrowhead = useMemo(() => {
    const tdx = x2 - cp2x;
    const tdy = y2 - cp2y;
    const len = Math.sqrt(tdx * tdx + tdy * tdy);
    if (len === 0) return '';
    const ux = tdx / len;
    const uy = tdy / len;
    const size = 8;
    const px = -uy * size * 0.5;
    const py = ux * size * 0.5;
    const baseX = x2 - ux * size;
    const baseY = y2 - uy * size;
    return `M ${x2} ${y2} L ${baseX + px} ${baseY + py} L ${baseX - px} ${baseY - py} Z`;
  }, [x2, y2, cp2x, cp2y]);

  return (
    <g
      style={{
        opacity: visible ? 1 : 0,
        transition: 'opacity 0.35s ease',
      }}
    >
      <defs>
        <linearGradient id={gradientId} x1={x1} y1={y1} x2={x2} y2={y2} gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor={stroke} stopOpacity={0.3} />
          <stop offset="100%" stopColor={stroke} stopOpacity={0.9} />
        </linearGradient>
      </defs>

      {/* Main curve */}
      <path
        d={pathD}
        fill="none"
        stroke={dashed || color === 'red' ? stroke : `url(#${gradientId})`}
        strokeWidth={2}
        strokeLinecap="round"
        strokeDasharray={dashed ? '6,4' : undefined}
        strokeOpacity={dashed && color !== 'red' ? 0.6 : undefined}
        style={{ transition: 'stroke 0.35s ease' }}
      />

      {/* Arrowhead — hidden for dashed (bidirectional match) lines */}
      {!dashed && (
        <path
          d={arrowhead}
          fill={stroke}
          opacity={0.85}
          style={{ transition: 'fill 0.35s ease' }}
        />
      )}
    </g>
  );
}
