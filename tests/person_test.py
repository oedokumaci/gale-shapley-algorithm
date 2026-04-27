"""Tests for the person module."""

import pytest

from gale_shapley_algorithm.person import Person, Proposer, Responder


class TestPerson:
    """Tests for Person base class."""

    def test_repr_unmatched(self) -> None:
        p = Person("alice", "proposer")
        assert "Match: None" in repr(p)

    def test_repr_matched(self) -> None:
        p = Person("alice", "proposer")
        other = Person("bob", "responder")
        p.match = other
        assert "Match: bob" in repr(p)

    def test_is_acceptable_true(
        self,
        deterministic_proposers_and_responders: tuple[list[Proposer], list[Responder]],
    ) -> None:
        proposers, responders = deterministic_proposers_and_responders
        m_1, m_2 = proposers
        w_1, w_2 = responders
        # m_1 prefs: w_1, w_2, m_1 (all acceptable since self is last)
        assert m_1.is_acceptable(w_1)
        assert m_1.is_acceptable(w_2)
        assert m_1.is_acceptable(m_1)
        # m_2 prefs: w_1, m_2, w_2 (w_1 and m_2 acceptable, w_2 not)
        assert m_2.is_acceptable(w_1)
        assert m_2.is_acceptable(m_2)
        assert not m_2.is_acceptable(w_2)

    def test_is_acceptable_value_error(self) -> None:
        """Person not in preferences raises ValueError."""
        m = Proposer("m", "man")
        w1 = Responder("w1", "woman")
        w2 = Responder("w2", "woman")
        m.preferences = (w1, m)  # w2 not in preferences
        with pytest.raises(ValueError, match="not in preferences"):
            m.is_acceptable(w2)

    def test_format_preferences(self) -> None:
        m = Proposer("m", "man")
        w1 = Responder("w1", "woman")
        w2 = Responder("w2", "woman")
        m.preferences = (w1, m, w2)  # w1 and m acceptable, w2 not
        result = m.format_preferences()
        assert "m has the following preferences" in result
        assert "w1" in result
        assert "w2" in result
        assert "*" in result

    def test_is_matched_property(self) -> None:
        p = Person("p", "side")
        assert not p.is_matched
        p.match = p
        assert p.is_matched


class TestProposer:
    """Tests for Proposer class."""

    def test_acceptable_to_propose(self) -> None:
        m = Proposer("m", "man")
        w1 = Responder("w1", "woman")
        w2 = Responder("w2", "woman")
        # m prefs: w1, m, w2 -> w1 and m acceptable (at or before self), w2 not
        m.preferences = (w1, m, w2)
        acceptable = m.acceptable_to_propose
        assert w1 in acceptable
        assert m in acceptable
        assert w2 not in acceptable

    def test_next_proposal_first(self) -> None:
        m = Proposer("m", "man")
        w = Responder("w", "woman")
        m.preferences = (w, m)
        assert m.next_proposal == w

    def test_next_proposal_after_last(self) -> None:
        m = Proposer("m", "man")
        w1 = Responder("w1", "woman")
        w2 = Responder("w2", "woman")
        m.preferences = (w1, w2, m)
        m.last_proposal = w1
        assert m.next_proposal == w2

    def test_next_proposal_exhausted_returns_self(self) -> None:
        """When all acceptable proposals exhausted, returns self via IndexError."""
        m = Proposer("m", "man")
        w = Responder("w", "woman")
        m.preferences = (w, m)
        # last_proposal = m (self), which is the last acceptable
        m.last_proposal = m
        assert m.next_proposal == m

    def test_propose_normal(self) -> None:
        """Proposing adds self to responder's current_proposals."""
        m = Proposer("m", "man")
        w = Responder("w", "woman")
        m.preferences = (w, m)
        m.propose()
        assert m in w.current_proposals
        assert m.last_proposal == w

    def test_propose_self_match(self) -> None:
        """When next proposal is self (Proposer), sets match to self."""
        m = Proposer("m", "man")
        w = Responder("w", "woman")
        m.preferences = (w, m)
        m.last_proposal = w  # already proposed to w
        m.propose()
        assert m.match == m


class TestResponder:
    """Tests for Responder class."""

    def test_awaiting_to_respond_empty(self) -> None:
        r = Responder("r", "woman")
        assert not r.awaiting_to_respond

    def test_awaiting_to_respond_nonempty(self) -> None:
        r = Responder("r", "woman")
        r.current_proposals.append(Proposer("m", "man"))
        assert r.awaiting_to_respond

    def test_acceptable_proposals(self) -> None:
        r = Responder("r", "woman")
        m1 = Proposer("m1", "man")
        m2 = Proposer("m2", "man")
        # r prefs: m1, r, m2 -> m1 acceptable, m2 not
        r.preferences = (m1, r, m2)
        r.current_proposals = [m1, m2]
        assert r.acceptable_proposals == [m1]

    def test_most_preferred_normal(self) -> None:
        r = Responder("r", "woman")
        m1 = Proposer("m1", "man")
        m2 = Proposer("m2", "man")
        r.preferences = (m1, m2, r)
        assert r._most_preferred([m1, m2]) == m1
        assert r._most_preferred([m2, m1]) == m1

    def test_most_preferred_empty_preferences(self) -> None:
        r = Responder("r", "woman")
        m = Proposer("m", "man")
        r.preferences = ()
        with pytest.raises(ValueError):
            r._most_preferred([m])

    def test_most_preferred_empty_proposals(self) -> None:
        r = Responder("r", "woman")
        r.preferences = (Proposer("m", "man"), r)
        with pytest.raises(ValueError):
            r._most_preferred([])

    def test_most_preferred_proposal_not_in_preferences(self) -> None:
        r = Responder("r", "woman")
        m1 = Proposer("m1", "man")
        m2 = Proposer("m2", "man")
        r.preferences = (m1, r)  # m2 not in prefs
        with pytest.raises(ValueError):
            r._most_preferred([m2])

    def test_respond_no_match_accepts_best(self) -> None:
        """Unmatched responder accepts best acceptable proposal."""
        r = Responder("r", "woman")
        m1 = Proposer("m1", "man")
        m2 = Proposer("m2", "man")
        r.preferences = (m1, m2, r)
        r.current_proposals = [m2, m1]
        r.respond()
        assert r.match == m1
        assert m1.match == r
        assert r.current_proposals == []

    def test_respond_swap_better_proposal(self) -> None:
        """Responder drops current match for a better proposal."""
        r = Responder("r", "woman")
        m1 = Proposer("m1", "man")
        m2 = Proposer("m2", "man")
        r.preferences = (m2, m1, r)  # prefers m2 over m1
        r.match = m1
        m1.match = r
        r.current_proposals = [m2]
        r.respond()
        assert r.match == m2
        assert m2.match == r
        assert m1.match is None  # dropped

    def test_respond_keep_current_match(self) -> None:
        """Responder keeps current match when it's better than new proposals."""
        r = Responder("r", "woman")
        m1 = Proposer("m1", "man")
        m2 = Proposer("m2", "man")
        r.preferences = (m1, m2, r)  # prefers m1 over m2
        r.match = m1
        m1.match = r
        r.current_proposals = [m2]
        r.respond()
        assert r.match == m1
        assert m1.match == r
        assert r.current_proposals == []

    def test_respond_no_acceptable_proposals(self) -> None:
        """When no proposals are acceptable, clears proposals without matching."""
        r = Responder("r", "woman")
        m = Proposer("m", "man")
        r.preferences = (r, m)  # self is above m, so m not acceptable
        r.current_proposals = [m]
        r.respond()
        assert r.match is None
        assert r.current_proposals == []
