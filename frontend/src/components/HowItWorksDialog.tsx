import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { HelpCircle } from 'lucide-react';

export function HowItWorksDialog() {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" className="text-muted-foreground">
          <HelpCircle className="h-3.5 w-3.5 mr-1" />
          How does it work?
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-2xl p-0 gap-0">
        <DialogHeader className="px-6 pt-6 pb-0">
          <DialogTitle
            className="text-xl font-bold tracking-tight"
            style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}
          >
            The Gale–Shapley Algorithm
          </DialogTitle>
        </DialogHeader>
        <ScrollArea className="max-h-[70vh] px-6 pb-6 pt-4">
          <div
            className="space-y-4 text-sm leading-relaxed text-foreground/90"
            style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}
          >
            <p>
              David Gale and Lloyd Shapley showed in 1962 that a short
              procedure always finds a stable matching in a two-sided market.
              Each person on one side ranks everyone on the other side. The
              algorithm then runs in rounds: every unmatched proposer asks the
              highest-ranked person they haven't yet approached. Each responder
              holds onto the best offer received so far and turns down the rest.
              Rejected proposers move to their next choice. Once no one has
              anyone left to ask, the held offers become final.
            </p>

            <p>
              The outcome is stable, meaning there is no pair of people who
              would both rather be with each other than with their assigned
              partners. When the two sides differ in size, some people stay
              unmatched, but stability still holds: no mutually preferred
              switch exists.
            </p>

            <p>
              The procedure favors whichever side proposes. Among all stable
              matchings, the proposing side gets its best possible result and the
              responding side gets its worst. Swapping who proposes yields a
              different stable outcome. The algorithm itself is neutral; the
              choice of who proposes is a design decision.
            </p>

            <p>
              Lloyd Shapley and Alvin Roth won the 2012 Nobel Prize in Economic
              Sciences for this work. Shapley built the theory; Roth applied it
              to real institutions.
            </p>

            <p>
              The U.S. medical residency match, run since the early 1950s as
              the National Resident Matching Program, is the oldest large-scale
              application. Roth and collaborators redesigned it in the late 1990s
              to handle couples applying jointly, bringing it closer to the
              applicant-proposing version of deferred acceptance.
            </p>

            <p>
              Kidney exchange posed a different problem. Incompatible
              donor-patient pairs form a directed graph, and clearinghouses
              search for disjoint swap cycles and chains that satisfy medical
              constraints while maximizing the number of transplants. The
              structure differs from one-to-one matching, but the principle is
              the same: state preferences explicitly, run a central algorithm,
              and prevent side deals. These programs have made thousands of
              additional transplants possible.
            </p>

            <p>
              Public school assignment in New York and Boston followed a similar
              path. Families rank schools; schools apply priority rules based on
              siblings, distance, and other criteria. The redesigned systems cut
              strategic gaming and sharply reduced the number of students placed
              at schools they never listed.
            </p>

            <p>
              The core logic has not changed since 1962: collect ranked
              preferences, run the procedure, and produce a matching that no
              pair can improve on by going around the system.
            </p>
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
