import type { MatchingResponse, RoundStep } from '@/types';

/** The final matching as a visualization frame (results view and end of the animation). */
export function buildFinalStep(result: MatchingResponse): RoundStep {
  const matches = Object.entries(result.matches).map(([proposer, responder]) => ({
    proposer,
    responder,
  }));
  return {
    round: result.rounds,
    proposals: [],
    rejections: [],
    tentative_matches: matches,
    // Unlike a round's list, this also names responders left on their own
    self_matches: result.self_matches,
  };
}
