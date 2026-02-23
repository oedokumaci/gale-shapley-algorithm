import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { PersonNode } from './svg/PersonNode';
import { AnimatedArrow } from './svg/AnimatedArrow';
import type { RoundStep, AnimationPhase, PersonImages } from '@/types';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface SVGMatchingVisualizationProps {
  step: RoundStep | null; // null = initial "ready" frame
  phase: AnimationPhase | null; // null = ready
  proposerNames: string[];
  responderNames: string[];
  personImages: PersonImages;
  onStepBack: () => void;
  onStepForward: () => void;
  canStepBack: boolean;
  canStepForward: boolean;
  isFinalRound: boolean;
  previousMatches?: { proposer: string; responder: string }[];
}

const VIEW_WIDTH = 900;
const PROPOSER_X = 300;
const RESPONDER_X = 600;
const PADDING_Y = 55;
const LABEL_Y = 22;
const MIN_RADIUS = 16;
const MAX_RADIUS = 24;
const NARRATIVE_FONT = 7;
const NARRATIVE_LINE_HEIGHT = 10;

function computeLayout(names: string[], viewHeight: number) {
  const count = names.length;
  if (count === 0) return { positions: [] as { name: string; y: number }[], radius: MAX_RADIUS };
  const usableHeight = viewHeight - PADDING_Y * 2;
  const spacing = count > 1 ? usableHeight / (count - 1) : 0;
  const radius = Math.max(MIN_RADIUS, Math.min(MAX_RADIUS, spacing * 0.42));
  const positions = names.map((name, i) => ({
    name,
    y: count === 1 ? viewHeight / 2 : PADDING_Y + i * spacing,
  }));
  return { positions, radius };
}

function computeViewHeight(maxCount: number) {
  return Math.max(280, PADDING_Y * 2 + (maxCount - 1) * 56);
}

function arrowEndpoints(
  x1: number, y1: number, x2: number, y2: number,
  r1: number, r2: number,
) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist === 0) return { ax1: x1, ay1: y1, ax2: x2, ay2: y2 };
  const ux = dx / dist;
  const uy = dy / dist;
  return {
    ax1: x1 + ux * (r1 + 3),
    ay1: y1 + uy * (r1 + 3),
    ax2: x2 - ux * (r2 + 3),
    ay2: y2 - uy * (r2 + 3),
  };
}

function firstName(fullName: string): string {
  if (fullName === 'Bad Bunny') return 'Bad Bunny';
  return fullName.split(' ')[0];
}

const SVG_NARRATIVE_FILLS = {
  neutral: 'var(--color-muted-foreground)',
  green: '#22c55e',
  red: '#ef4444',
} as const;

interface PositionedNarrative {
  personName: string;
  y: number;
  lines: { text: string; color: 'neutral' | 'green' | 'red' }[];
}

function buildProposerNarratives(
  step: RoundStep,
  phase: AnimationPhase,
  proposerYMap: Map<string, number>,
): PositionedNarrative[] {
  if (phase === 'proposals' || phase === 'responses') {
    // Each proposer: "→ {responder}"
    return step.proposals.map((p) => ({
      personName: p.proposer,
      y: proposerYMap.get(p.proposer) ?? 0,
      lines: [{ text: `→ ${firstName(p.responder)}`, color: 'neutral' as const }],
    }));
  }
  // No narrative text for matches phase
  if (phase === 'matches') {
    return [];
  }
  return [];
}

function buildResponderNarratives(
  step: RoundStep,
  phase: AnimationPhase,
  responderYMap: Map<string, number>,
): PositionedNarrative[] {
  if (phase !== 'responses') return [];

  const rejectedProposers = new Set(step.rejections.map((r) => r.proposer));
  const matchByResponder = new Map(step.tentative_matches.map((m) => [m.responder, m.proposer]));

  // Group proposals by responder
  const byResponder = new Map<string, typeof step.proposals>();
  for (const p of step.proposals) {
    if (!byResponder.has(p.responder)) byResponder.set(p.responder, []);
    byResponder.get(p.responder)!.push(p);
  }

  const narratives: PositionedNarrative[] = [];

  for (const [responder, proposals] of byResponder) {
    const y = responderYMap.get(responder) ?? 0;
    const lines: { text: string; color: 'neutral' | 'green' | 'red' }[] = [];

    for (const p of proposals) {
      if (rejectedProposers.has(p.proposer)) {
        const preferred = matchByResponder.get(responder);
        lines.push({
          text: preferred
            ? `✗ ${firstName(p.proposer)} (prefers ${firstName(preferred)})`
            : `✗ ${firstName(p.proposer)}`,
          color: 'red',
        });
      } else {
        lines.push({
          text: `✓ ${firstName(p.proposer)}`,
          color: 'green',
        });
      }
    }

    narratives.push({ personName: responder, y, lines });
  }

  return narratives;
}

export function SVGMatchingVisualization({
  step,
  phase,
  proposerNames,
  responderNames,
  personImages,
  onStepBack,
  onStepForward,
  canStepBack,
  canStepForward,
  isFinalRound,
  previousMatches,
}: SVGMatchingVisualizationProps) {
  const maxCount = Math.max(proposerNames.length, responderNames.length, 1);
  const viewHeight = computeViewHeight(maxCount);
  const proposerLayout = computeLayout(proposerNames, viewHeight);
  const responderLayout = computeLayout(responderNames, viewHeight);

  const isReady = step === null || phase === null;

  const proposerYMap = new Map(proposerLayout.positions.map((p) => [p.name, p.y]));
  const responderYMap = new Map(responderLayout.positions.map((p) => [p.name, p.y]));

  const matchMap = step ? new Map(step.tentative_matches.map((m) => [m.proposer, m.responder])) : new Map();
  const reverseMatchMap = step ? new Map(step.tentative_matches.map((m) => [m.responder, m.proposer])) : new Map();
  const rejectedSet = step ? new Set(step.rejections.map((r) => r.proposer)) : new Set<string>();
  const selfMatchSet = step ? new Set(step.self_matches) : new Set<string>();

  function getProposerStatus(name: string): 'default' | 'matched' | 'rejected' | 'self-matched' | 'unmatched' {
    if (isReady || phase === 'proposals') return 'default';
    if (selfMatchSet.has(name)) return 'self-matched';
    if (matchMap.has(name)) return phase === 'responses' ? 'default' : 'matched';
    if (rejectedSet.has(name)) return phase === 'responses' ? 'rejected' : 'unmatched';
    return 'default';
  }

  function getResponderStatus(name: string): 'default' | 'matched' | 'unmatched' {
    if (isReady || phase === 'proposals') return 'default';
    if (reverseMatchMap.has(name)) return phase === 'responses' ? 'default' : 'matched';
    return phase === 'matches' ? 'unmatched' : 'default';
  }

  type ArrowData = {
    key: string;
    proposer: string;
    responder: string;
    color: 'neutral' | 'green' | 'red';
    dashed?: boolean;
  };

  // Background arrows: previous round's tentative matches shown during proposals/responses
  const bgArrows: ArrowData[] = [];
  if (!isReady && previousMatches && previousMatches.length > 0 && (phase === 'proposals' || phase === 'responses')) {
    const currentMatchByResponder = step
      ? new Map(step.tentative_matches.map((m) => [m.responder, m.proposer]))
      : new Map<string, string>();

    for (const pm of previousMatches) {
      let color: 'green' | 'red' = 'green';
      if (phase === 'responses') {
        const currentProposer = currentMatchByResponder.get(pm.responder);
        if (currentProposer !== undefined && currentProposer !== pm.proposer) {
          color = 'red';
        }
      }
      bgArrows.push({
        key: `bg-${pm.proposer}-${pm.responder}`,
        proposer: pm.proposer,
        responder: pm.responder,
        color,
        dashed: true,
      });
    }
  }

  const arrows: ArrowData[] = [];

  if (!isReady && step) {
    if (phase === 'proposals') {
      for (const p of step.proposals) {
        arrows.push({ key: `${p.proposer}-${p.responder}`, proposer: p.proposer, responder: p.responder, color: 'neutral' });
      }
    } else if (phase === 'responses') {
      for (const p of step.proposals) {
        const wasRejected = rejectedSet.has(p.proposer);
        arrows.push({
          key: `${p.proposer}-${p.responder}`,
          proposer: p.proposer,
          responder: p.responder,
          color: wasRejected ? 'red' : 'green',
        });
      }
    } else {
      for (const m of step.tentative_matches) {
        arrows.push({ key: `${m.proposer}-${m.responder}`, proposer: m.proposer, responder: m.responder, color: 'green' });
      }
    }
  }

  const phaseLabel = isReady
    ? null
    : phase === 'proposals' ? 'Proposals' : phase === 'responses' ? 'Responses' : 'Current Matches';

  // Compute stats
  const matchedProposers = step ? step.tentative_matches.length : 0;
  const matchedResponders = step ? new Set(step.tentative_matches.map((m) => m.responder)).size : 0;
  const unmatchedProposers = proposerNames.length - matchedProposers - (step?.self_matches.length ?? 0);
  const unmatchedResponders = responderNames.length - matchedResponders;

  // Build positioned narratives
  const leftNarratives = !isReady && step && phase
    ? buildProposerNarratives(step, phase, proposerYMap)
    : [];
  const rightNarratives = !isReady && step && phase
    ? buildResponderNarratives(step, phase, responderYMap)
    : [];

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center">
          <div className="w-24 shrink-0">
            <CardTitle className="text-sm font-semibold" style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}>
              {isReady ? 'Ready' : `Round ${step!.round}`}
            </CardTitle>
          </div>

          <div className="flex-1 flex items-center justify-center gap-2">
            {isReady ? (
              <span className="text-xs text-muted-foreground">Press play to begin</span>
            ) : (
              <>
                <Button size="icon" variant="ghost" className="h-7 w-7" onClick={onStepBack} disabled={!canStepBack} aria-label="Previous phase (h)">
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <Badge variant="outline" className="text-sm font-semibold px-4 py-1 min-w-[140px] justify-center">
                  {phaseLabel}
                </Badge>
                <Button size="icon" variant="ghost" className="h-7 w-7" onClick={onStepForward} disabled={!canStepForward} aria-label="Next phase (l)">
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </>
            )}
          </div>

          <div className="w-24 shrink-0 flex justify-end">
            {!isReady && phase === 'matches' && (
              <div className="flex gap-3 text-[11px] text-muted-foreground">
                <span className="inline-flex items-center gap-1">
                  <span className="text-[10px] font-medium opacity-50">P</span>
                  <span className="inline-block w-2 h-2 rounded-full bg-green-500" />
                  {matchedProposers}
                  <span className="inline-block w-2 h-2 rounded-full bg-blue-500 ml-1" />
                  {unmatchedProposers}
                </span>
                <span className="inline-flex items-center gap-1">
                  <span className="text-[10px] font-medium opacity-50">R</span>
                  <span className="inline-block w-2 h-2 rounded-full bg-green-500" />
                  {matchedResponders}
                  <span className="inline-block w-2 h-2 rounded-full bg-blue-500 ml-1" />
                  {unmatchedResponders}
                </span>
              </div>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <svg
          viewBox={`0 0 ${VIEW_WIDTH} ${viewHeight}`}
          preserveAspectRatio="xMidYMid meet"
          className="w-full h-auto"
        >
          {/* Column labels */}
          <text x={PROPOSER_X} y={LABEL_Y} textAnchor="middle"
            fontSize={10} fontWeight={600} letterSpacing="0.08em"
            className="fill-muted-foreground"
            style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}
          >
            PROPOSERS
          </text>
          <text x={RESPONDER_X} y={LABEL_Y} textAnchor="middle"
            fontSize={10} fontWeight={600} letterSpacing="0.08em"
            className="fill-muted-foreground"
            style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}
          >
            RESPONDERS
          </text>

          {/* Vertical guide lines */}
          <line x1={PROPOSER_X} y1={PADDING_Y - 15} x2={PROPOSER_X} y2={viewHeight - PADDING_Y + 15}
            className="stroke-border" strokeWidth="1" strokeDasharray="3,6" opacity={0.5} />
          <line x1={RESPONDER_X} y1={PADDING_Y - 15} x2={RESPONDER_X} y2={viewHeight - PADDING_Y + 15}
            className="stroke-border" strokeWidth="1" strokeDasharray="3,6" opacity={0.5} />

          {/* Background arrows: previous round's tentative matches */}
          {bgArrows.map((a, i) => {
            const py = proposerYMap.get(a.proposer);
            const ry = responderYMap.get(a.responder);
            if (py === undefined || ry === undefined) return null;
            const { ax1, ay1, ax2, ay2 } = arrowEndpoints(
              PROPOSER_X, py, RESPONDER_X, ry,
              proposerLayout.radius, responderLayout.radius,
            );
            return (
              <AnimatedArrow
                key={a.key}
                x1={ax1} y1={ay1} x2={ax2} y2={ay2}
                color={a.color} visible={true} index={1000 + i}
                dashed={true}
              />
            );
          })}

          {/* Arrows */}
          {arrows.map((a, i) => {
            const py = proposerYMap.get(a.proposer);
            const ry = responderYMap.get(a.responder);
            if (py === undefined || ry === undefined) return null;
            const { ax1, ay1, ax2, ay2 } = arrowEndpoints(
              PROPOSER_X, py, RESPONDER_X, ry,
              proposerLayout.radius, responderLayout.radius,
            );
            return (
              <AnimatedArrow
                key={a.key}
                x1={ax1} y1={ay1} x2={ax2} y2={ay2}
                color={a.color} visible={true} index={i}
                dashed={phase === 'matches' && !isFinalRound}
              />
            );
          })}

          {/* Proposer nodes */}
          {proposerLayout.positions.map((p, i) => (
            <PersonNode
              key={p.name} name={p.name} cx={PROPOSER_X} cy={p.y}
              radius={proposerLayout.radius} imageUrl={personImages[p.name]}
              status={getProposerStatus(p.name)} side="left" index={i}
            />
          ))}

          {/* Responder nodes */}
          {responderLayout.positions.map((p, i) => (
            <PersonNode
              key={p.name} name={p.name} cx={RESPONDER_X} cy={p.y}
              radius={responderLayout.radius} imageUrl={personImages[p.name]}
              status={getResponderStatus(p.name)} side="right" index={i}
            />
          ))}

          {/* Left narrative text — positioned at each proposer's Y */}
          {leftNarratives.map((n) => (
            <g key={`left-${n.personName}`}>
              {n.lines.map((line, li) => (
                <text
                  key={li}
                  x={8}
                  y={n.y + li * NARRATIVE_LINE_HEIGHT}
                  textAnchor="start"
                  dominantBaseline="central"
                  fontSize={NARRATIVE_FONT}
                  fill={SVG_NARRATIVE_FILLS[line.color]}
                  opacity={0.85}
                  style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}
                >
                  {line.text}
                </text>
              ))}
            </g>
          ))}

          {/* Right narrative text — positioned at each responder's Y */}
          {rightNarratives.map((n) => (
            <g key={`right-${n.personName}`}>
              {n.lines.map((line, li) => (
                <text
                  key={li}
                  x={VIEW_WIDTH - 8}
                  y={n.y + li * NARRATIVE_LINE_HEIGHT}
                  textAnchor="end"
                  dominantBaseline="central"
                  fontSize={NARRATIVE_FONT}
                  fill={SVG_NARRATIVE_FILLS[line.color]}
                  opacity={0.85}
                  style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}
                >
                  {line.text}
                </text>
              ))}
            </g>
          ))}
        </svg>
      </CardContent>
    </Card>
  );
}
