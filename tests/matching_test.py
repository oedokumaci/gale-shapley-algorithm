"""Tests for the create_matching convenience function."""

import pytest

from gale_shapley_algorithm.matching import _build_algorithm, create_matching
from gale_shapley_algorithm.person import Person
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


class TestScaling:
    """Guards the O(1) work per proposal without a wall-clock bound, which would be flaky in CI."""

    def test_large_instance_does_constant_work_per_proposal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Everyone shares one ranking: n rounds and n(n+1)/2 proposals, a quadratic workload.

        Every dict lookup and every scan of a preference tuple calls __hash__ or __eq__ on persons, so
        counting those calls measures the work deterministically. Building and running take about
        20 * n**2 calls; a linear scan per proposal would add about n**3 / 2 and trip the budget early.
        """
        n = 200
        budget = 40 * n * n
        calls = 0

        def count_call() -> None:
            nonlocal calls
            calls += 1
            if calls > budget:
                raise AssertionError(f"more than {budget} hash/eq calls on persons for n={n}")

        def counting_hash(person: Person) -> int:
            count_call()
            return id(person)

        def counting_eq(person: Person, other: object) -> bool:
            count_call()
            return person is other

        monkeypatch.setattr(Person, "__hash__", counting_hash)
        monkeypatch.setattr(Person, "__eq__", counting_eq)
        proposers = [f"m{i}" for i in range(n)]
        responders = [f"w{i}" for i in range(n)]
        result = create_matching(
            proposer_preferences=dict.fromkeys(proposers, responders),
            responder_preferences=dict.fromkeys(responders, proposers),
        )
        assert result.rounds == n
        assert result.matches == {f"m{i}": f"w{i}" for i in range(n)}
        assert calls > 0  # the counting hooks were really in place
