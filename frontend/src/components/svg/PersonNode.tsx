import { useState } from 'react';

interface PersonNodeProps {
  name: string;
  cx: number;
  cy: number;
  radius: number;
  imageUrl?: string;
  status: 'default' | 'matched' | 'rejected' | 'self-matched' | 'unmatched';
  side: 'left' | 'right';
  index: number;
}

interface StatusStyle {
  fill: string;
  stroke: string;
  strokeWidth: number;
  textClass: string;
}

const STATUS_STYLES: Record<PersonNodeProps['status'], StatusStyle> = {
  default: {
    fill: 'var(--color-muted)',
    stroke: 'var(--color-border)',
    strokeWidth: 1.5,
    textClass: 'fill-foreground',
  },
  matched: {
    fill: '#dcfce7',
    stroke: '#22c55e',
    strokeWidth: 2.5,
    textClass: 'fill-foreground',
  },
  rejected: {
    fill: '#fee2e2',
    stroke: '#ef4444',
    strokeWidth: 2.5,
    textClass: 'fill-foreground',
  },
  'self-matched': {
    fill: '#dbeafe',
    stroke: '#3b82f6',
    strokeWidth: 2,
    textClass: 'fill-foreground',
  },
  unmatched: {
    fill: 'var(--color-muted)',
    stroke: 'var(--color-border)',
    strokeWidth: 1.5,
    textClass: 'fill-muted-foreground',
  },
};

export function PersonNode({ name, cx, cy, radius, imageUrl, status, side, index }: PersonNodeProps) {
  // Remember which URL failed (not just that one did), so a different photo still gets a try
  const [failedUrl, setFailedUrl] = useState<string>();
  const showImage = !!imageUrl && imageUrl !== failedUrl;
  const style = STATUS_STYLES[status];
  const clipId = `clip-${side}-${index}`;
  const textX = side === 'left' ? cx - radius - 4 : cx + radius + 4;
  const textAnchor = side === 'left' ? 'end' : 'start';

  return (
    <g style={{ transition: 'opacity 0.3s ease' }}>
      <defs>
        <clipPath id={clipId}>
          <circle cx={cx} cy={cy} r={radius - 2} />
        </clipPath>
      </defs>

      {/* Shadow */}
      <circle
        cx={cx}
        cy={cy + 2}
        r={radius}
        fill="black"
        opacity={0.06}
      />

      {/* Main circle */}
      <circle
        cx={cx}
        cy={cy}
        r={radius}
        fill={style.fill}
        stroke={style.stroke}
        strokeWidth={style.strokeWidth}
        style={{ transition: 'fill 0.3s ease, stroke 0.3s ease, stroke-width 0.3s ease' }}
      />

      {/* Image if present */}
      {showImage && (
        <image
          href={imageUrl}
          x={cx - radius + 2}
          y={cy - radius + 2}
          width={(radius - 2) * 2}
          height={(radius - 2) * 2}
          clipPath={`url(#${clipId})`}
          preserveAspectRatio="xMidYMid slice"
          onError={() => setFailedUrl(imageUrl)}
        />
      )}

      {/* Initials if no image, or it failed to load */}
      {!showImage && (
        <text
          x={cx}
          y={cy + 1}
          textAnchor="middle"
          dominantBaseline="central"
          fontSize={radius * 0.7}
          fontWeight={600}
          className="fill-muted-foreground"
          style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}
        >
          {name.charAt(0).toUpperCase()}
        </text>
      )}

      {/* Name label */}
      <text
        x={textX}
        y={cy + 1}
        textAnchor={textAnchor}
        dominantBaseline="central"
        fontSize={9}
        fontWeight={500}
        className={style.textClass}
        style={{
          fontFamily: "'DM Sans', system-ui, sans-serif",
          transition: 'fill 0.3s ease',
        }}
      >
        {name}
      </text>
    </g>
  );
}
