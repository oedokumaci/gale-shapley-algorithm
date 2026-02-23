import { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { SVGMatchingVisualization } from './SVGMatchingVisualization';
import { ResultsPanel } from './ResultsPanel';
import type { StepsResponse, AnimationPhase, PersonImages, RoundStep } from '@/types';
import { Play, Pause, SkipBack, SkipForward, RotateCcw } from 'lucide-react';

interface AnimationPlayerProps {
  data: StepsResponse;
  proposerNames: string[];
  responderNames: string[];
  personImages: PersonImages;
}

const PHASES: AnimationPhase[] = ['proposals', 'responses', 'matches'];

function buildFinalStep(data: StepsResponse): RoundStep {
  const result = data.final_result;
  const matches = Object.entries(result.matches).map(([proposer, responder]) => ({
    proposer,
    responder,
  }));
  return {
    round: result.rounds,
    proposals: [],
    rejections: [],
    tentative_matches: matches,
    self_matches: result.self_matches,
  };
}

export function AnimationPlayer({
  data,
  proposerNames,
  responderNames,
  personImages,
}: AnimationPlayerProps) {
  const [currentPhase, setCurrentPhase] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1500);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const totalPhases = data.steps.length * 3;
  const isAtEnd = currentPhase > totalPhases;

  const roundIndex = currentPhase > 0 ? Math.floor((currentPhase - 1) / 3) : -1;
  const phaseInRound = currentPhase > 0 ? (currentPhase - 1) % 3 : -1;
  const currentStep = roundIndex >= 0 && roundIndex < data.steps.length ? data.steps[roundIndex] : null;
  const animationPhase: AnimationPhase = phaseInRound >= 0 ? PHASES[phaseInRound] : 'proposals';
  const previousStep = roundIndex > 0 ? data.steps[roundIndex - 1] : null;

  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setPlaying(false);
  }, []);

  const advance = useCallback(() => {
    setCurrentPhase((prev) => {
      if (prev > totalPhases) {
        stop();
        return prev;
      }
      return prev + 1;
    });
  }, [totalPhases, stop]);

  const stepBack = useCallback(() => {
    stop();
    setCurrentPhase((p) => Math.max(0, p - 1));
  }, [stop]);

  const stepForward = useCallback(() => {
    stop();
    setCurrentPhase((p) => Math.min(totalPhases + 1, p + 1));
  }, [stop, totalPhases]);

  const togglePlay = useCallback(() => {
    if (isAtEnd) {
      setCurrentPhase(1);
      setPlaying(true);
    } else if (currentPhase === 0) {
      setCurrentPhase(1);
      setPlaying(true);
    } else {
      setPlaying((p) => !p);
    }
  }, [isAtEnd, currentPhase]);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(advance, speed);
      return () => {
        if (intervalRef.current) clearInterval(intervalRef.current);
      };
    }
  }, [playing, speed, advance]);

  useEffect(() => {
    if (currentPhase > totalPhases && playing) {
      stop();
    }
  }, [currentPhase, totalPhases, playing, stop]);

  // Keyboard shortcuts
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // Don't capture if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch (e.key) {
        case 'h':
        case 'ArrowLeft':
          e.preventDefault();
          stepBack();
          break;
        case 'l':
        case 'ArrowRight':
          e.preventDefault();
          stepForward();
          break;
        case ' ':
          e.preventDefault();
          togglePlay();
          break;
        case 'r':
          e.preventDefault();
          stop();
          setCurrentPhase(0);
          break;
      }
    }

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [stepBack, stepForward, togglePlay, stop]);

  function reset() {
    stop();
    setCurrentPhase(0);
  }

  const progress = totalPhases > 0 ? Math.min(currentPhase / totalPhases, 1) : 0;

  const phaseLabels: Record<AnimationPhase, string> = {
    proposals: 'Proposals',
    responses: 'Responses',
    matches: 'Current Matches',
  };

  let statusText = 'Ready';
  if (currentPhase > 0 && currentPhase <= totalPhases) {
    statusText = `Round ${roundIndex + 1} — ${phaseLabels[animationPhase]}`;
  } else if (isAtEnd) {
    statusText = 'Done';
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg" style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}>
              Step-by-Step Animation
            </CardTitle>
            <span className="text-sm text-muted-foreground">{statusText}</span>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Progress bar */}
          <div className="h-1 rounded-full bg-muted overflow-hidden">
            <div
              className="h-full rounded-full bg-primary transition-all duration-300 ease-out"
              style={{ width: `${progress * 100}%` }}
            />
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            <Button size="icon" variant="outline" onClick={reset} aria-label="Reset (r)" className="h-8 w-8">
              <RotateCcw className="h-3.5 w-3.5" />
            </Button>
            <Button size="icon" variant="outline" onClick={stepBack} disabled={currentPhase === 0} aria-label="Step back (h)" className="h-8 w-8">
              <SkipBack className="h-3.5 w-3.5" />
            </Button>
            <Button size="icon" onClick={togglePlay} aria-label={playing ? 'Pause (space)' : 'Play (space)'} className="h-10 w-10 rounded-full">
              {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 ml-0.5" />}
            </Button>
            <Button size="icon" variant="outline" onClick={stepForward} disabled={isAtEnd} aria-label="Step forward (l)" className="h-8 w-8">
              <SkipForward className="h-3.5 w-3.5" />
            </Button>

            <div className="flex-1" />

            {/* Time between steps */}
            <span className="text-xs text-muted-foreground whitespace-nowrap">Time between steps</span>
            <div className="w-24">
              <Slider
                value={[speed]}
                onValueChange={([v]) => setSpeed(v)}
                min={100}
                max={5000}
                step={100}
              />
            </div>
            <span className="text-xs text-muted-foreground tabular-nums w-12 text-right">{speed >= 1000 ? `${(speed / 1000).toFixed(1)}s` : `${speed}ms`}</span>
          </div>

          {/* Keyboard shortcuts */}
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-muted-foreground/60">
            <span><kbd className="px-1 py-0.5 rounded border border-border bg-muted text-[10px] font-mono">Space</kbd> play/pause</span>
            <span><kbd className="px-1 py-0.5 rounded border border-border bg-muted text-[10px] font-mono">h</kbd> / <kbd className="px-1 py-0.5 rounded border border-border bg-muted text-[10px] font-mono">&larr;</kbd> prev</span>
            <span><kbd className="px-1 py-0.5 rounded border border-border bg-muted text-[10px] font-mono">l</kbd> / <kbd className="px-1 py-0.5 rounded border border-border bg-muted text-[10px] font-mono">&rarr;</kbd> next</span>
            <span><kbd className="px-1 py-0.5 rounded border border-border bg-muted text-[10px] font-mono">r</kbd> reset</span>
          </div>
        </CardContent>
      </Card>

      {/* Ready state — show all participants */}
      {currentPhase === 0 && (
        <SVGMatchingVisualization
          step={null}
          phase={null}
          proposerNames={proposerNames}
          responderNames={responderNames}
          personImages={personImages}
          onStepBack={stepBack}
          onStepForward={stepForward}
          canStepBack={false}
          canStepForward={true}
          isFinalRound={false}
        />
      )}

      {/* Current phase visualization */}
      {currentStep && currentPhase > 0 && currentPhase <= totalPhases && (
        <SVGMatchingVisualization
          step={currentStep}
          phase={animationPhase}
          proposerNames={proposerNames}
          responderNames={responderNames}
          personImages={personImages}
          onStepBack={stepBack}
          onStepForward={stepForward}
          canStepBack={currentPhase > 0}
          canStepForward={!isAtEnd}
          isFinalRound={roundIndex === data.steps.length - 1}
          previousMatches={previousStep?.tentative_matches}
        />
      )}

      {/* Final results — SVG visualization + text details */}
      {isAtEnd && (
        <>
          <SVGMatchingVisualization
            step={buildFinalStep(data)}
            phase="matches"
            proposerNames={proposerNames}
            responderNames={responderNames}
            personImages={personImages}
            onStepBack={stepBack}
            onStepForward={() => {}}
            canStepBack={true}
            canStepForward={false}
            isFinalRound={true}
          />
          <ResultsPanel result={data.final_result} />
        </>
      )}
    </div>
  );
}
