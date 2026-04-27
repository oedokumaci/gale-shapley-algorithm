"""Pydantic request/response models for the API."""

from typing import Self

from pydantic import BaseModel, model_validator


class MatchingRequest(BaseModel):
    """Request body for matching endpoints."""

    proposer_preferences: dict[str, list[str]]
    responder_preferences: dict[str, list[str]]

    @model_validator(mode="after")
    def _names_in_preferences_must_exist(self) -> Self:
        # Names referenced in a preference list but absent from the opposite
        # side's keys would otherwise be silently dropped by the algorithm,
        # producing self-matches that misrepresent the user's intent.
        proposer_names = set(self.proposer_preferences)
        responder_names = set(self.responder_preferences)
        unknown_responders = {
            r for prefs in self.proposer_preferences.values() for r in prefs if r not in responder_names
        }
        unknown_proposers = {
            p for prefs in self.responder_preferences.values() for p in prefs if p not in proposer_names
        }
        problems: list[str] = []
        if unknown_responders:
            problems.append(f"unknown responders referenced by proposers: {sorted(unknown_responders)!r}")
        if unknown_proposers:
            problems.append(f"unknown proposers referenced by responders: {sorted(unknown_proposers)!r}")
        if problems:
            raise ValueError("; ".join(problems))
        return self


class ProposalAction(BaseModel):
    """A proposer-responder pair representing an action."""

    proposer: str
    responder: str


class MatchingResponse(BaseModel):
    """Response for the matching endpoint, combines MatchingResult and StabilityResult."""

    rounds: int
    matches: dict[str, str]
    unmatched: list[str]
    self_matches: list[str]
    all_matched: bool
    is_stable: bool
    is_individually_rational: bool
    blocking_pairs: list[tuple[str, str]]


class RoundStep(BaseModel):
    """Snapshot of a single round of the algorithm."""

    round: int
    proposals: list[ProposalAction]
    rejections: list[ProposalAction]
    tentative_matches: list[ProposalAction]
    self_matches: list[str]


class StepsResponse(BaseModel):
    """Response for the step-through endpoint."""

    steps: list[RoundStep]
    final_result: MatchingResponse
