"""Step-through execution of the Gale-Shapley algorithm, capturing per-round state."""

from gale_shapley_algorithm._api.models import (
    MatchingResponse,
    ProposalAction,
    RoundStep,
    StepsResponse,
)
from gale_shapley_algorithm.matching import _build_algorithm
from gale_shapley_algorithm.person import Responder
from gale_shapley_algorithm.result import MatchingResult, StabilityResult
from gale_shapley_algorithm.stability import check_stability


def _build_matching_response(result: MatchingResult, stability: StabilityResult) -> MatchingResponse:
    """Combine MatchingResult and StabilityResult into a MatchingResponse."""
    return MatchingResponse(
        rounds=result.rounds,
        matches=result.matches,
        unmatched=result.unmatched,
        self_matches=result.self_matches,
        all_matched=result.all_matched,
        is_stable=stability.is_stable,
        is_individually_rational=stability.is_individually_rational,
        blocking_pairs=stability.blocking_pairs,
    )


def run_step_through(
    proposer_preferences: dict[str, list[str]],
    responder_preferences: dict[str, list[str]],
) -> StepsResponse:
    """Run the algorithm step by step, capturing a RoundStep per round."""
    algorithm = _build_algorithm(proposer_preferences, responder_preferences)
    steps: list[RoundStep] = []

    while not algorithm.terminate():
        round_num = algorithm.round + 1
        unmatched_before = {p.name for p in algorithm.unmatched_proposers}

        algorithm.proposers_propose()

        proposals: list[ProposalAction] = []
        round_self_matches: list[str] = []
        for proposer in algorithm.proposers:
            if proposer.name not in unmatched_before:
                continue
            if proposer.last_proposal is proposer:
                round_self_matches.append(proposer.name)
            elif proposer.last_proposal is not None and isinstance(proposer.last_proposal, Responder):
                proposals.append(ProposalAction(proposer=proposer.name, responder=proposer.last_proposal.name))

        algorithm.responders_respond()
        algorithm.round += 1

        rejections: list[ProposalAction] = []
        tentative_matches: list[ProposalAction] = []

        for proposer in algorithm.proposers:
            if proposer.name not in unmatched_before:
                if (
                    proposer.match is not None
                    and proposer.match is not proposer
                    and isinstance(proposer.match, Responder)
                ):
                    tentative_matches.append(ProposalAction(proposer=proposer.name, responder=proposer.match.name))
                continue
            if proposer.name in round_self_matches:
                continue
            if proposer.match is not None and isinstance(proposer.match, Responder):
                tentative_matches.append(ProposalAction(proposer=proposer.name, responder=proposer.match.name))
            elif proposer.last_proposal is not None and isinstance(proposer.last_proposal, Responder):
                rejections.append(ProposalAction(proposer=proposer.name, responder=proposer.last_proposal.name))

        steps.append(
            RoundStep(
                round=round_num,
                proposals=proposals,
                rejections=rejections,
                tentative_matches=tentative_matches,
                self_matches=round_self_matches,
            )
        )

    # Algorithm.execute() short-circuits the loop (terminate() is True) and runs
    # the same self-match fixup + result construction we used to inline here.
    result = algorithm.execute()
    stability = check_stability(algorithm)

    return StepsResponse(
        steps=steps,
        final_result=_build_matching_response(result, stability),
    )
