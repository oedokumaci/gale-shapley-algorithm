"""Tests for the create_matching convenience function."""

from gale_shapley_algorithm.matching import _build_algorithm, create_matching
from gale_shapley_algorithm.result import MatchingResult


class TestCreateMatching:
    """Tests for create_matching."""

    def test_basic_matching(self) -> None:
        result = create_matching(
            proposer_preferences={"alice": ["bob", "charlie"], "dave": ["charlie", "bob"]},
            responder_preferences={"bob": ["alice", "dave"], "charlie": ["dave", "alice"]},
        )
        assert isinstance(result, MatchingResult)
        assert result.all_matched
        assert len(result.matches) == 2

    def test_deterministic_result(self) -> None:
        """GS always produces proposer-optimal stable matching."""
        result = create_matching(
            proposer_preferences={"m1": ["w1", "w2"], "m2": ["w1", "w2"]},
            responder_preferences={"w1": ["m1", "m2"], "w2": ["m1", "m2"]},
        )
        # m1 gets w1 (proposer-optimal)
        assert result.matches["m1"] == "w1"
        assert result.matches["m2"] == "w2"

    def test_unequal_sides(self) -> None:
        """More proposers than responders leads to self-matches."""
        result = create_matching(
            proposer_preferences={"m1": ["w1"], "m2": ["w1"], "m3": ["w1"]},
            responder_preferences={"w1": ["m1", "m2", "m3"]},
        )
        assert isinstance(result, MatchingResult)
        # Only one can match w1
        assert len(result.matches) == 1
        assert len(result.self_matches) > 0

    def test_empty_preferences(self) -> None:
        """Empty preference lists result in self-matches."""
        result = create_matching(
            proposer_preferences={"m1": []},
            responder_preferences={"w1": []},
        )
        assert "m1" in result.self_matches
        assert "w1" in result.self_matches
        assert not result.all_matched

    def test_repeated_preference_entry_terminates(self) -> None:
        """A name listed twice used to make its proposer re-propose to it forever."""
        algorithm = _build_algorithm({"P1": ["A", "A"], "P2": ["A"]}, {"A": ["P2", "P1"]})
        # GS finishes within |P| * (|R| + 1) rounds; bounding the loop makes a regression fail instead of hang.
        for _ in range(10):
            if algorithm.terminate():
                break
            algorithm.proposers_propose()
            algorithm.responders_respond()
        assert algorithm.terminate()
        result = algorithm.execute()
        assert result.matches == {"P2": "A"}
        assert result.self_matches == ["P1"]
