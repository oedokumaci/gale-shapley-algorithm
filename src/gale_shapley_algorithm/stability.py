"""Stability analysis for matchings."""

from gale_shapley_algorithm.algorithm import Algorithm
from gale_shapley_algorithm.person import Proposer, Responder
from gale_shapley_algorithm.result import StabilityResult


def is_individually_rational(proposers: list[Proposer], responders: list[Responder]) -> bool:
    """Check if the matching is individually rational.

    A matching is individually rational if every matched person finds their
    match acceptable (ranked at or above themselves).

    Args:
        proposers: List of proposers in the matching.
        responders: List of responders in the matching.

    Returns:
        True if individually rational, False otherwise.
    """
    persons = proposers + responders
    return all(person.match is None or person.is_acceptable(person.match) for person in persons)


def find_blocking_pairs(proposers: list[Proposer], responders: list[Responder]) -> list[tuple[str, str]]:  # noqa: ARG001
    """Find all blocking pairs in a matching.

    A blocking pair (p, r) exists when proposer p and responder r both prefer
    each other over their current matches. A responder whose match is None
    counts as single, so she blocks only with proposers she finds acceptable.

    Args:
        proposers: List of proposers in the matching.
        responders: List of responders in the matching.

    Returns:
        List of (proposer_name, responder_name) blocking pairs.
    """
    blocking: list[tuple[str, str]] = []
    for proposer in proposers:
        if not (bool(proposer.preferences) and proposer.is_matched):
            continue

        better_than_match = proposer.preferences[: proposer.preferences.index(proposer.match)]
        for responder in better_than_match:
            if not isinstance(responder, Responder):
                continue

            match responder.is_matched:
                case False if all(
                    item in responder.preferences for item in [proposer, responder]
                ) and responder.is_acceptable(proposer):
                    blocking.append((proposer.name, responder.name))
                case True if (
                    bool(responder.preferences)
                    and all(item in responder.preferences for item in [proposer, responder.match])
                    and responder.preferences.index(proposer) < responder.preferences.index(responder.match)
                ):
                    blocking.append((proposer.name, responder.name))

    return blocking


def check_stability(algorithm: Algorithm) -> StabilityResult:
    """Check the stability of an algorithm's matching.

    Args:
        algorithm: An Algorithm instance that has been executed.

    Returns:
        StabilityResult with is_stable, is_individually_rational, and blocking_pairs.
    """
    ir = is_individually_rational(algorithm.proposers, algorithm.responders)
    bp = find_blocking_pairs(algorithm.proposers, algorithm.responders)
    return StabilityResult(
        is_stable=ir and len(bp) == 0,
        is_individually_rational=ir,
        blocking_pairs=bp,
    )
