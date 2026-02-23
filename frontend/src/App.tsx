import { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { PreferenceEditor } from '@/components/PreferenceEditor';
import { ResultsPanel } from '@/components/ResultsPanel';
import { AnimationPlayer } from '@/components/AnimationPlayer';
import { SVGMatchingVisualization } from '@/components/SVGMatchingVisualization';
import { HowItWorksDialog } from '@/components/HowItWorksDialog';
import { usePersonImages } from '@/hooks/usePersonImages';
import { runMatching, runMatchingSteps } from '@/api/client';
import { Sun, Moon, BookOpen, Github } from 'lucide-react';
import type { MatchingRequest, MatchingResponse, StepsResponse, RoundStep } from '@/types';
import {
  DEFAULT_PROPOSER_NAMES,
  DEFAULT_RESPONDER_NAMES,
  buildRandomPrefs,
  DEFAULT_IMAGES,
} from '@/defaults';

type ViewMode = 'edit' | 'results' | 'animation';

function useTheme() {
  const [dark, setDark] = useState(() => {
    if (typeof window === 'undefined') return false;
    const stored = localStorage.getItem('theme');
    if (stored) return stored === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark);
    localStorage.setItem('theme', dark ? 'dark' : 'light');
  }, [dark]);

  return { dark, toggle: () => setDark((d) => !d) };
}

function buildFinalStep(result: MatchingResponse): RoundStep {
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

function App() {
  const { dark, toggle: toggleTheme } = useTheme();
  const [proposerNames, setProposerNames] = useState<string[]>(DEFAULT_PROPOSER_NAMES);
  const [responderNames, setResponderNames] = useState<string[]>(DEFAULT_RESPONDER_NAMES);
  const [proposerPrefs, setProposerPrefs] = useState<Record<string, string[]>>(() =>
    buildRandomPrefs(DEFAULT_PROPOSER_NAMES, DEFAULT_RESPONDER_NAMES),
  );
  const [responderPrefs, setResponderPrefs] = useState<Record<string, string[]>>(() =>
    buildRandomPrefs(DEFAULT_RESPONDER_NAMES, DEFAULT_PROPOSER_NAMES),
  );

  const [result, setResult] = useState<MatchingResponse | null>(null);
  const [stepsData, setStepsData] = useState<StepsResponse | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('edit');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { images: personImages, uploadImage, removeImage } = usePersonImages(DEFAULT_IMAGES);

  const addProposer = useCallback((name: string) => {
    setProposerNames((prev) => [...prev, name]);
    setProposerPrefs((prev) => ({ ...prev, [name]: [...responderNames] }));
    setResponderPrefs((prev) => {
      const updated = { ...prev };
      for (const r of Object.keys(updated)) {
        updated[r] = [...updated[r], name];
      }
      return updated;
    });
  }, [responderNames]);

  const removeProposer = useCallback((name: string) => {
    setProposerNames((prev) => prev.filter((n) => n !== name));
    setProposerPrefs((prev) => {
      const updated = { ...prev };
      delete updated[name];
      return updated;
    });
    setResponderPrefs((prev) => {
      const updated = { ...prev };
      for (const r of Object.keys(updated)) {
        updated[r] = updated[r].filter((p) => p !== name);
      }
      return updated;
    });
    removeImage(name);
  }, [removeImage]);

  const addResponder = useCallback((name: string) => {
    setResponderNames((prev) => [...prev, name]);
    setResponderPrefs((prev) => ({ ...prev, [name]: [...proposerNames] }));
    setProposerPrefs((prev) => {
      const updated = { ...prev };
      for (const p of Object.keys(updated)) {
        updated[p] = [...updated[p], name];
      }
      return updated;
    });
  }, [proposerNames]);

  const removeResponder = useCallback((name: string) => {
    setResponderNames((prev) => prev.filter((n) => n !== name));
    setResponderPrefs((prev) => {
      const updated = { ...prev };
      delete updated[name];
      return updated;
    });
    setProposerPrefs((prev) => {
      const updated = { ...prev };
      for (const p of Object.keys(updated)) {
        updated[p] = updated[p].filter((r) => r !== name);
      }
      return updated;
    });
    removeImage(name);
  }, [removeImage]);

  const reorderProposerPref = useCallback((person: string, newOrder: string[]) => {
    setProposerPrefs((prev) => ({ ...prev, [person]: newOrder }));
  }, []);

  const reorderResponderPref = useCallback((person: string, newOrder: string[]) => {
    setResponderPrefs((prev) => ({ ...prev, [person]: newOrder }));
  }, []);

  const randomizePrefs = useCallback(() => {
    setProposerPrefs(buildRandomPrefs(proposerNames, responderNames));
    setResponderPrefs(buildRandomPrefs(responderNames, proposerNames));
  }, [proposerNames, responderNames]);

  const clearAll = useCallback(() => {
    setProposerNames([]);
    setResponderNames([]);
    setProposerPrefs({});
    setResponderPrefs({});
  }, []);

  function buildRequest(): MatchingRequest {
    return {
      proposer_preferences: { ...proposerPrefs },
      responder_preferences: { ...responderPrefs },
    };
  }

  const canRun = proposerNames.length > 0 && responderNames.length > 0;

  async function handleRunMatching() {
    setLoading(true);
    setError(null);
    try {
      const res = await runMatching(buildRequest());
      setResult(res);
      setStepsData(null);
      setViewMode('results');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  async function handleRunAnimation() {
    setLoading(true);
    setError(null);
    try {
      const res = await runMatchingSteps(buildRequest());
      setStepsData(res);
      setResult(null);
      setViewMode('animation');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  function handleBackToEdit() {
    setViewMode('edit');
    setResult(null);
    setStepsData(null);
    setError(null);
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold tracking-tight" style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}>
              Gale-Shapley Algorithm
            </h1>
            <p className="text-xs text-muted-foreground">
              Interactive matching with step-by-step visualization
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button size="icon" variant="ghost" className="h-8 w-8" asChild>
              <a href="https://oedokumaci.github.io/gale-shapley-algorithm" target="_blank" rel="noopener noreferrer" aria-label="Documentation">
                <BookOpen className="h-4 w-4" />
              </a>
            </Button>
            <Button size="icon" variant="ghost" className="h-8 w-8" asChild>
              <a href="https://github.com/oedokumaci/gale-shapley-algorithm" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
                <Github className="h-4 w-4" />
              </a>
            </Button>
            <Button size="icon" variant="ghost" onClick={toggleTheme} aria-label="Toggle theme" className="h-8 w-8">
              {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
            {viewMode !== 'edit' && (
              <Button variant="outline" size="sm" onClick={handleBackToEdit}>
                Back to Editor
              </Button>
            )}
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-4 space-y-4">
        {viewMode === 'edit' && (
          <>
            {/* Action buttons — sticky at top */}
            <div className="flex gap-3 justify-center sticky top-0 z-10 py-2 bg-background/95 backdrop-blur-sm border-b border-border/50 -mx-4 px-4 -mt-4 mb-1">
              <Button onClick={handleRunAnimation} disabled={!canRun || loading} size="sm">
                {loading ? 'Running...' : 'Animate Step-by-Step'}
              </Button>
              <Button variant="outline" onClick={handleRunMatching} disabled={!canRun || loading} size="sm">
                {loading ? 'Running...' : 'Run Matching'}
              </Button>
              <HowItWorksDialog />
            </div>

            <PreferenceEditor
              proposerNames={proposerNames}
              responderNames={responderNames}
              proposerPrefs={proposerPrefs}
              responderPrefs={responderPrefs}
              onAddProposer={addProposer}
              onRemoveProposer={removeProposer}
              onAddResponder={addResponder}
              onRemoveResponder={removeResponder}
              onReorderProposerPref={reorderProposerPref}
              onReorderResponderPref={reorderResponderPref}
              onRandomizePrefs={randomizePrefs}
              onClearAll={clearAll}
              personImages={personImages}
              onUploadImage={uploadImage}
            />
          </>
        )}

        {error && (
          <div className="rounded-md bg-destructive/10 border border-destructive/20 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {viewMode === 'results' && result && (
          <div className="space-y-4">
            <SVGMatchingVisualization
              step={buildFinalStep(result)}
              phase="matches"
              proposerNames={proposerNames}
              responderNames={responderNames}
              personImages={personImages}
              onStepBack={() => {}}
              onStepForward={() => {}}
              canStepBack={false}
              canStepForward={false}
              isFinalRound={true}
            />
            <ResultsPanel result={result} />
          </div>
        )}

        {viewMode === 'animation' && stepsData && (
          <AnimationPlayer
            data={stepsData}
            proposerNames={proposerNames}
            responderNames={responderNames}
            personImages={personImages}
          />
        )}
      </main>
    </div>
  );
}

export default App;
