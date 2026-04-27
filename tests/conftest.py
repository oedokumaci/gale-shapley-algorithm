"""Test fixtures for the Gale-Shapley algorithm."""

import pytest

from gale_shapley_algorithm.algorithm import Algorithm
from gale_shapley_algorithm.person import Proposer, Responder

# --- Person fixtures (function-scoped for isolation) ---


@pytest.fixture
def m_1_fix() -> Proposer:
    return Proposer("m_1", "man")


@pytest.fixture
def m_2_fix() -> Proposer:
    return Proposer("m_2", "man")


@pytest.fixture
def w_1_fix() -> Responder:
    return Responder("w_1", "woman")


@pytest.fixture
def w_2_fix() -> Responder:
    return Responder("w_2", "woman")


@pytest.fixture
def deterministic_proposers_and_responders(
    m_1_fix: Proposer, m_2_fix: Proposer, w_1_fix: Responder, w_2_fix: Responder
) -> tuple[list[Proposer], list[Responder]]:
    """Create proposers and responders with deterministic preferences.

       m_1 m_2 w_1 w_2
       --- --- --- ---
    1. w_1 w_1 m_1 m_2
    2. w_2 m_2 m_2 m_1
    3. m_1     w_1 w_2
    """
    proposers = [m_1_fix, m_2_fix]
    responders = [w_1_fix, w_2_fix]

    proposers[0].preferences = (responders[0], responders[1], proposers[0])
    proposers[1].preferences = (responders[0], proposers[1], responders[1])
    responders[0].preferences = (proposers[0], proposers[1], responders[0])
    responders[1].preferences = (proposers[1], proposers[0], responders[1])

    return proposers, responders


# --- Algorithm fixtures ---


@pytest.fixture
def ran_algorithm_fix(
    deterministic_proposers_and_responders: tuple[list[Proposer], list[Responder]],
) -> Algorithm:
    """Algorithm that has been run to completion."""
    proposers, responders = deterministic_proposers_and_responders
    algo = Algorithm(proposers, responders)
    algo.execute()
    return algo
