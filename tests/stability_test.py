"""Tests for the stability module."""

from gale_shapley_algorithm.algorithm import Algorithm
from gale_shapley_algorithm.person import Proposer, Responder
from gale_shapley_algorithm.stability import check_stability, find_blocking_pairs, is_individually_rational


class TestIsIndividuallyRational:
    """Tests for is_individually_rational."""

    def test_after_gs_run(self, ran_algorithm_fix: Algorithm) -> None:
        assert is_individually_rational(ran_algorithm_fix.proposers, ran_algorithm_fix.responders)


class TestFindBlockingPairs:
    """Tests for find_blocking_pairs."""

    def test_empty_after_gs(self, ran_algorithm_fix: Algorithm) -> None:
        assert find_blocking_pairs(ran_algorithm_fix.proposers, ran_algorithm_fix.responders) == []

    def test_blocking_pair_responder_unmatched(self) -> None:
        """A blocking pair exists when proposer prefers an unmatched responder."""
        m1 = Proposer("m1")
        m2 = Proposer("m2")
        w1 = Responder("w1")
        w2 = Responder("w2")

        m1.preferences = (w1, w2, m1)
        m2.preferences = (w2, m2, w1)
        w1.preferences = (m1, m2, w1)
        w2.preferences = (m2, m1, w2)

        m1.match = w2
        w2.match = m1
        m2.match = m2
        w1.match = None

        blocking = find_blocking_pairs([m1, m2], [w1, w2])
        assert any(bp == ("m1", "w1") for bp in blocking)

    def test_blocking_pair_responder_prefers_proposer(self) -> None:
        """A blocking pair when responder prefers a different proposer."""
        m1 = Proposer("m1")
        m2 = Proposer("m2")
        w1 = Responder("w1")
        w2 = Responder("w2")

        m1.preferences = (w1, w2, m1)
        m2.preferences = (w1, w2, m2)
        w1.preferences = (m1, m2, w1)
        w2.preferences = (m1, m2, w2)

        m1.match = w2
        w2.match = m1
        m2.match = w1
        w1.match = m2

        assert ("m1", "w1") in find_blocking_pairs([m1, m2], [w1, w2])

    def test_blocking_pairs_skip_self(self) -> None:
        """Self (Proposer) in better_than_match is skipped."""
        m = Proposer("m")
        w = Responder("w")

        m.preferences = (m, w)
        w.preferences = (m, w)
        m.match = w
        w.match = m

        assert find_blocking_pairs([m], [w]) == []

    def test_blocking_pairs_proposer_no_preferences(self) -> None:
        m = Proposer("m")
        w = Responder("w")
        m.preferences = ()
        m.match = m

        assert find_blocking_pairs([m], [w]) == []

    def test_blocking_pairs_proposer_not_matched(self) -> None:
        m = Proposer("m")
        w = Responder("w")
        m.preferences = (w, m)
        m.match = None

        assert find_blocking_pairs([m], [w]) == []


class TestCheckStability:
    """Tests for check_stability."""

    def test_stable_after_gs(self, ran_algorithm_fix: Algorithm) -> None:
        result = check_stability(ran_algorithm_fix)
        assert result.is_stable
        assert result.is_individually_rational
        assert result.blocking_pairs == []

    def test_unstable_matching(self) -> None:
        m1 = Proposer("m1")
        m2 = Proposer("m2")
        w1 = Responder("w1")
        w2 = Responder("w2")

        m1.preferences = (w1, w2, m1)
        m2.preferences = (w1, w2, m2)
        w1.preferences = (m1, m2, w1)
        w2.preferences = (m1, m2, w2)

        # Bad matching: m1-w2, m2-w1
        m1.match = w2
        w2.match = m1
        m2.match = w1
        w1.match = m2

        algo = Algorithm([m1, m2], [w1, w2])
        result = check_stability(algo)
        assert not result.is_stable
        assert len(result.blocking_pairs) > 0
