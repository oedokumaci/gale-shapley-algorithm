"""Tests for the algorithm module."""

from gale_shapley_algorithm.algorithm import Algorithm
from gale_shapley_algorithm.person import Proposer, Responder
from gale_shapley_algorithm.result import MatchingResult


class TestAlgorithmProperties:
    """Tests for Algorithm properties."""

    def test_persons_property(
        self,
        deterministic_proposers_and_responders: tuple[list[Proposer], list[Responder]],
    ) -> None:
        proposers, responders = deterministic_proposers_and_responders
        algo = Algorithm(proposers, responders)
        assert algo.persons == proposers + responders


class TestProposeRespondTerminate:
    """Tests for the core propose/respond/terminate cycle."""

    def test_proposers_propose(
        self,
        deterministic_proposers_and_responders: tuple[list[Proposer], list[Responder]],
    ) -> None:
        proposers, responders = deterministic_proposers_and_responders
        m_1, m_2 = proposers
        w_1, w_2 = responders
        algo = Algorithm(proposers, responders)

        algo.proposers_propose()

        assert m_1.last_proposal == w_1
        assert m_2.last_proposal == w_1
        assert w_1.current_proposals == [m_1, m_2]
        assert not w_2.current_proposals

    def test_responders_respond(
        self,
        deterministic_proposers_and_responders: tuple[list[Proposer], list[Responder]],
    ) -> None:
        proposers, responders = deterministic_proposers_and_responders
        m_1, m_2 = proposers
        w_1, w_2 = responders
        algo = Algorithm(proposers, responders)

        algo.proposers_propose()
        algo.responders_respond()

        assert m_1.match == w_1
        assert w_1.match == m_1
        assert not m_2.match
        assert not w_2.match
        assert not w_1.current_proposals
        assert not w_2.current_proposals

    def test_terminate_false_initially(
        self,
        deterministic_proposers_and_responders: tuple[list[Proposer], list[Responder]],
    ) -> None:
        proposers, responders = deterministic_proposers_and_responders
        algo = Algorithm(proposers, responders)
        assert not algo.terminate()

    def test_terminate_true_after_full_run(self, ran_algorithm_fix: Algorithm) -> None:
        assert ran_algorithm_fix.terminate()


class TestFormatMatches:
    """Tests for format_matches covering all match branches."""

    def test_proposer_unmatched(self) -> None:
        m = Proposer("m")
        m.match = None
        algo = Algorithm([m], [])
        result = algo.format_matches()
        assert "m is unmatched" in result

    def test_proposer_self_match(self) -> None:
        m = Proposer("m")
        m.match = m
        algo = Algorithm([m], [])
        result = algo.format_matches()
        assert "m is matched to self" in result

    def test_proposer_normal_match(self) -> None:
        m = Proposer("m")
        w = Responder("w")
        m.match = w
        w.match = m
        algo = Algorithm([m], [w])
        result = algo.format_matches()
        assert "m is matched to w" in result

    def test_responder_self_match(self) -> None:
        r = Responder("r")
        r.match = r
        algo = Algorithm([], [r])
        result = algo.format_matches()
        assert "r is matched to self" in result

    def test_responder_unmatched(self) -> None:
        r = Responder("r")
        r.match = None
        algo = Algorithm([], [r])
        result = algo.format_matches()
        assert "r is unmatched" in result


class TestFormatAllPreferences:
    """Tests for format_all_preferences."""

    def test_compact_true(
        self,
        deterministic_proposers_and_responders: tuple[list[Proposer], list[Responder]],
    ) -> None:
        proposers, responders = deterministic_proposers_and_responders
        algo = Algorithm(proposers, responders)
        result = algo.format_all_preferences(compact=True)
        assert "compact format" in result
        assert "m_1" in result
        assert "w_1" in result

    def test_compact_false(
        self,
        deterministic_proposers_and_responders: tuple[list[Proposer], list[Responder]],
    ) -> None:
        proposers, responders = deterministic_proposers_and_responders
        algo = Algorithm(proposers, responders)
        result = algo.format_all_preferences(compact=False)
        assert "separately" in result
        assert "has the following preferences" in result


class TestExecute:
    """Tests for execute() returning MatchingResult."""

    def test_execute_returns_matching_result(
        self,
        deterministic_proposers_and_responders: tuple[list[Proposer], list[Responder]],
    ) -> None:
        proposers, responders = deterministic_proposers_and_responders
        algo = Algorithm(proposers, responders)
        result = algo.execute()
        assert isinstance(result, MatchingResult)
        assert result.rounds > 0
        assert algo.terminate()

    def test_execute_deterministic_matching(
        self,
        deterministic_proposers_and_responders: tuple[list[Proposer], list[Responder]],
    ) -> None:
        """m_1 prefs: w_1, w_2, m_1 | m_2 prefs: w_1, m_2, w_2 (w_2 unacceptable).

        w_1 prefs: m_1, m_2, w_1 | w_2 prefs: m_2, m_1, w_2.
        Round 1: m_1, m_2 -> w_1. w_1 accepts m_1.
        Round 2: m_2 self-matches (next after w_1 is self).
        w_2 unmatched -> self-match.
        """
        proposers, responders = deterministic_proposers_and_responders
        algo = Algorithm(proposers, responders)
        result = algo.execute()
        assert result.matches == {"m_1": "w_1"}
        assert result.unmatched == []
        assert "m_2" in result.self_matches
        assert "w_2" in result.self_matches
        assert not result.all_matched

    def test_execute_with_self_match(self) -> None:
        """When a proposer exhausts all acceptable, they self-match."""
        m = Proposer("m")
        w = Responder("w")
        # m only finds self acceptable (self before w)
        m.preferences = (m, w)
        w.preferences = (m, w)
        algo = Algorithm([m], [w])
        result = algo.execute()
        assert "m" in result.self_matches
        assert "w" in result.self_matches
        assert not result.all_matched


class TestRun:
    """Tests for backward-compatible run() via execute()."""

    def test_run_completes(
        self,
        deterministic_proposers_and_responders: tuple[list[Proposer], list[Responder]],
    ) -> None:
        proposers, responders = deterministic_proposers_and_responders
        algo = Algorithm(proposers, responders)
        result = algo.execute()
        assert algo.terminate()
        assert isinstance(result, MatchingResult)
