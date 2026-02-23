"""gale-shapley-algorithm: A Python implementation of the Gale-Shapley algorithm."""

from gale_shapley_algorithm.algorithm import Algorithm
from gale_shapley_algorithm.matching import create_matching
from gale_shapley_algorithm.person import Person, Proposer, Responder
from gale_shapley_algorithm.result import MatchingResult, StabilityResult
from gale_shapley_algorithm.stability import check_stability, find_blocking_pairs, is_individually_rational

__version__ = "1.5.0"
__all__ = [
    "Algorithm",
    "MatchingResult",
    "Person",
    "Proposer",
    "Responder",
    "StabilityResult",
    "check_stability",
    "create_matching",
    "find_blocking_pairs",
    "is_individually_rational",
]
