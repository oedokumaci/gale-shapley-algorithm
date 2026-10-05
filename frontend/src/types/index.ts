export interface ProposalAction {
  proposer: string;
  responder: string;
}

export interface MatchingRequest {
  proposer_preferences: Record<string, string[]>;
  responder_preferences: Record<string, string[]>;
}

export interface MatchingResponse {
  rounds: number;
  matches: Record<string, string>;
  unmatched: string[];
  self_matches: string[];
  all_matched: boolean;
  is_stable: boolean;
  is_individually_rational: boolean;
  blocking_pairs: [string, string][];
}

export interface RoundStep {
  round: number;
  proposals: ProposalAction[];
  rejections: ProposalAction[];
  tentative_matches: ProposalAction[];
  self_matches: string[];
}

export interface StepsResponse {
  steps: RoundStep[];
  final_result: MatchingResponse;
}

export type AnimationPhase = 'proposals' | 'responses' | 'matches';

export type Side = 'proposers' | 'responders';

// Photo URLs by side, then name: a proposer and a responder may share a name
export type PersonImages = Record<Side, Record<string, string>>;
