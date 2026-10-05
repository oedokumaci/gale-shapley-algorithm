import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { PersonList } from './PersonList';
import { CompactPreferenceGrid } from './CompactPreferenceGrid';
import { Shuffle, Trash2 } from 'lucide-react';
import type { PersonImages, Side } from '@/types';

// The cap has to sit on the viewport: with only a max-height on the root, the viewport's
// height:100% can't resolve, so it grows with the grid and spills out of the card.
// 420px fits the default 6x6 grid (404px) on wide screens; larger grids scroll.
const GRID_SCROLL_AREA = '[&>[data-slot=scroll-area-viewport]]:max-h-[420px]';

interface PreferenceEditorProps {
  proposerNames: string[];
  responderNames: string[];
  proposerPrefs: Record<string, string[]>;
  responderPrefs: Record<string, string[]>;
  onAddProposer: (name: string) => void;
  onRemoveProposer: (name: string) => void;
  onAddResponder: (name: string) => void;
  onRemoveResponder: (name: string) => void;
  onReorderProposerPref: (person: string, newOrder: string[]) => void;
  onReorderResponderPref: (person: string, newOrder: string[]) => void;
  onRandomizePrefs: () => void;
  onClearAll: () => void;
  personImages: PersonImages;
  onUploadImage: (side: Side, name: string, file: File) => void;
}

export function PreferenceEditor({
  proposerNames,
  responderNames,
  proposerPrefs,
  responderPrefs,
  onAddProposer,
  onRemoveProposer,
  onAddResponder,
  onRemoveResponder,
  onReorderProposerPref,
  onReorderResponderPref,
  onRandomizePrefs,
  onClearAll,
  personImages,
  onUploadImage,
}: PreferenceEditorProps) {
  const canRandomize = proposerNames.length > 0 && responderNames.length > 0;
  const canClear = proposerNames.length > 0 || responderNames.length > 0;

  return (
    <div className="space-y-3">
      {(canRandomize || canClear) && (
        <div className="flex justify-center gap-2">
          {canRandomize && (
            <Button variant="outline" size="sm" onClick={onRandomizePrefs}>
              <Shuffle className="h-3.5 w-3.5 mr-1.5" />
              Randomize Preferences
            </Button>
          )}
          {canClear && (
            <Button variant="outline" size="sm" onClick={onClearAll}>
              <Trash2 className="h-3.5 w-3.5 mr-1.5" />
              Clear All
            </Button>
          )}
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Proposers column */}
      <Card>
        <CardHeader className="pb-2 px-4 pt-4">
          <CardTitle className="text-base text-center">Proposers</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 px-4 pb-4">
          <PersonList
            label="Proposers"
            persons={proposerNames}
            onAdd={onAddProposer}
            onRemove={onRemoveProposer}
            images={personImages.proposers}
            onUploadImage={(name, file) => onUploadImage('proposers', name, file)}
          />
          {proposerNames.length > 0 && responderNames.length > 0 && (
            <>
              <Separator />
              <h4 className="text-xs font-semibold">
                Preference Rankings
                <span className="text-[10px] font-normal text-muted-foreground ml-1">
                  (drag to reorder)
                </span>
              </h4>
              <ScrollArea className={GRID_SCROLL_AREA}>
                <CompactPreferenceGrid
                  names={proposerNames}
                  prefs={proposerPrefs}
                  onReorder={onReorderProposerPref}
                />
              </ScrollArea>
            </>
          )}
        </CardContent>
      </Card>

      {/* Responders column */}
      <Card>
        <CardHeader className="pb-2 px-4 pt-4">
          <CardTitle className="text-base text-center">Responders</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 px-4 pb-4">
          <PersonList
            label="Responders"
            persons={responderNames}
            onAdd={onAddResponder}
            onRemove={onRemoveResponder}
            images={personImages.responders}
            onUploadImage={(name, file) => onUploadImage('responders', name, file)}
          />
          {proposerNames.length > 0 && responderNames.length > 0 && (
            <>
              <Separator />
              <h4 className="text-xs font-semibold">
                Preference Rankings
                <span className="text-[10px] font-normal text-muted-foreground ml-1">
                  (drag to reorder)
                </span>
              </h4>
              <ScrollArea className={GRID_SCROLL_AREA}>
                <CompactPreferenceGrid
                  names={responderNames}
                  prefs={responderPrefs}
                  onReorder={onReorderResponderPref}
                />
              </ScrollArea>
            </>
          )}
        </CardContent>
      </Card>
      </div>
    </div>
  );
}
