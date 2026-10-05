"""Tests for the person module."""

import copy
import pickle

import pytest

from gale_shapley_algorithm.algorithm import Algorithm
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

    def test_is_acceptable_self_not_in_preferences(self) -> None:
        m = Proposer("m", "man")
        w = Responder("w", "woman")
        m.preferences = (w,)
        with pytest.raises(ValueError, match="not in preferences"):
            m.is_acceptable(w)

    def test_preferences_default_to_empty_tuple(self) -> None:
        assert Person("p", "side").preferences == ()

    def test_is_acceptable_uses_first_copy_of_repeated_entry(self) -> None:
        """Like preferences.index, a repeated entry ranks at its first position (self included)."""
        m = Proposer("m", "man")
        w1 = Responder("w1", "woman")
        w2 = Responder("w2", "woman")
        m.preferences = (w1, m, w1, w2)
        assert m.is_acceptable(w1)
        assert not m.is_acceptable(w2)
        m.preferences = (m, w1, m)
        assert not m.is_acceptable(w1)

    def test_is_acceptable_follows_reassigned_preferences(self) -> None:
        m = Proposer("m", "man")
        w1 = Responder("w1", "woman")
        w2 = Responder("w2", "woman")
        m.preferences = (w1, m, w2)
        assert m.is_acceptable(w1)
        assert not m.is_acceptable(w2)
        m.preferences = (w2, m, w1)
        assert not m.is_acceptable(w1)
        assert m.is_acceptable(w2)
        m.preferences = (w1, m)
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

    def test_acceptable_to_propose_skips_repeated_entries(self) -> None:
        """A responder listed twice is proposed to once, at its first position."""
        m = Proposer("m", "man")
        w1 = Responder("w1", "woman")
        w2 = Responder("w2", "woman")
        m.preferences = (w1, w1, w2, m)
        assert m.acceptable_to_propose == (w1, w2, m)

    def test_next_proposal_moves_past_repeated_entry(self) -> None:
        """Re-proposing to a repeated responder used to loop forever."""
        m = Proposer("m", "man")
        w1 = Responder("w1", "woman")
        w2 = Responder("w2", "woman")
        m.preferences = (w1, w1, w2, m)
        m.last_proposal = w1
        assert m.next_proposal == w2

    def test_next_proposal_exhausted_returns_self(self) -> None:
        """When all acceptable proposals are exhausted, returns self."""
        m = Proposer("m", "man")
        w = Responder("w", "woman")
        m.preferences = (w, m)
        # last_proposal = m (self), which is the last acceptable
        m.last_proposal = m
        assert m.next_proposal == m

    def test_next_proposal_empty_preferences_returns_self(self) -> None:
        m = Proposer("m", "man")
        assert m.acceptable_to_propose == ()
        assert m.next_proposal == m

    def test_next_proposal_last_proposal_not_acceptable_raises(self) -> None:
        m = Proposer("m", "man")
        w1 = Responder("w1", "woman")
        w2 = Responder("w2", "woman")
        m.preferences = (w1, m, w2)
        m.last_proposal = w2
        with pytest.raises(ValueError, match=r"tuple\.index\(x\): x not in tuple"):
            _ = m.next_proposal

    def test_self_not_in_preferences_raises(self) -> None:
        m = Proposer("m", "man")
        w = Responder("w", "woman")
        m.preferences = (w,)
        with pytest.raises(ValueError, match="not in preferences"):
            _ = m.acceptable_to_propose
        with pytest.raises(ValueError, match="not in preferences"):
            _ = m.next_proposal

    def test_acceptable_to_propose_and_next_proposal_follow_reassigned_preferences(self) -> None:
        """Reassigning preferences (even mid-run) must not leave a stale cached order behind."""
        m = Proposer("m", "man")
        w1 = Responder("w1", "woman")
        w2 = Responder("w2", "woman")
        w3 = Responder("w3", "woman")
        m.preferences = (w1, w2, w3, m)
        m.last_proposal = w2
        assert m.next_proposal == w3
        m.preferences = (w2, w1, w3, m)  # w2 moves to the front
        assert m.acceptable_to_propose == (w2, w1, w3, m)
        assert m.next_proposal == w1
        m.preferences = (w2, m, w1, w3)  # w1 and w3 become unacceptable
        assert m.acceptable_to_propose == (w2, m)
        assert m.next_proposal == m

    def test_next_proposal_follows_overridden_acceptable_to_propose(self) -> None:
        """next_proposal walks whatever acceptable_to_propose returns, as it did with tuple.index."""

        class ReverseProposer(Proposer):
            @property
            def acceptable_to_propose(self) -> tuple[Responder | Proposer, ...]:
                return (*reversed(super().acceptable_to_propose[:-1]), self)

        m = ReverseProposer("m", "man")
        w1 = Responder("w1", "woman")
        w2 = Responder("w2", "woman")
        m.preferences = (w1, w2, m)
        assert m.next_proposal == w2
        m.last_proposal = w2
        assert m.next_proposal == w1
        m.last_proposal = w1
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

    def test_most_preferred_uses_first_copy_of_repeated_entry(self) -> None:
        r = Responder("r", "woman")
        m1 = Proposer("m1", "man")
        m2 = Proposer("m2", "man")
        r.preferences = (m2, m1, m2, r)
        assert r._most_preferred([m1, m2]) == m2

    def test_most_preferred_follows_reassigned_preferences(self) -> None:
        r = Responder("r", "woman")
        m1 = Proposer("m1", "man")
        m2 = Proposer("m2", "man")
        r.preferences = (m1, m2, r)
        assert r._most_preferred([m1, m2]) == m1
        r.preferences = (m2, m1, r)
        assert r._most_preferred([m1, m2]) == m2
        r.preferences = (m2, r)
        with pytest.raises(ValueError):
            r._most_preferred([m1, m2])

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


class TestCacheFallbacks:
    """Inputs the rank caches can't serve must behave exactly as plain preference lookups do."""

    def test_subclass_with_its_own_preferences_property(self) -> None:
        class StoredResponder(Responder):
            @property
            def preferences(self) -> tuple[Proposer | Responder, ...]:
                return self._stored

            @preferences.setter
            def preferences(self, value: tuple[Proposer | Responder, ...]) -> None:
                self._stored = value

        m = Proposer("m", "man")
        w = StoredResponder("w", "woman")
        m.preferences = (w, m)
        w.preferences = (m, w)
        assert w.is_acceptable(m)
        w.current_proposals = [m]
        w.respond()
        assert w.match is m

    def test_unhashable_person_that_is_never_acceptable(self) -> None:
        class UnhashableResponder(Responder):
            __hash__ = None  # type: ignore[assignment]

        m = Proposer("m", "man")
        w = UnhashableResponder("w", "woman")
        m.preferences = (m, w)  # w is unacceptable to m, so it is never put in a set or dict
        w.preferences = (m, w)
        result = Algorithm([m], [w]).execute()
        assert result.rounds == 1
        assert result.self_matches == ["m", "w"]

    def test_overridden_is_acceptable_is_asked_again(self) -> None:
        class PickyProposer(Proposer):
            excluded: frozenset[str] = frozenset()

            def is_acceptable(self, person: Proposer | Responder) -> bool:
                return person.name not in self.excluded and super().is_acceptable(person)

        m = PickyProposer("m", "man")
        w1 = Responder("w1", "woman")
        w2 = Responder("w2", "woman")
        m.preferences = (w1, w2, m)
        m.excluded = frozenset({"w2"})
        assert m.acceptable_to_propose == (w1, m)
        m.last_proposal = w1
        m.excluded = frozenset()  # same preferences tuple, different answers
        assert m.next_proposal == w2

    def test_preferences_list_changed_in_place(self) -> None:
        m = Proposer("m", "man")
        w1 = Responder("w1", "woman")
        w2 = Responder("w2", "woman")
        preferences = [w1, m, w2]
        m.preferences = preferences  # type: ignore[assignment]
        assert not m.is_acceptable(w2)
        preferences.insert(0, preferences.pop())  # w2 moves to the front of the same list
        assert m.is_acceptable(w2)
        assert m.acceptable_to_propose == (w2, w1, m)

    def test_preferences_list_lookups_for_a_responder(self) -> None:
        r = Responder("r", "woman")
        m1 = Proposer("m1", "man")
        m2 = Proposer("m2", "man")
        stranger = Proposer("s", "man")
        r.preferences = [m2, m1, r]  # type: ignore[assignment]
        assert r._most_preferred([m1, m2]) is m2
        with pytest.raises(ValueError, match="not in preferences"):
            r.is_acceptable(stranger)

    def test_acceptable_to_propose_overridden_to_return_a_list(self) -> None:
        class ListProposer(Proposer):
            @property
            def acceptable_to_propose(self) -> list[Responder | Proposer]:  # type: ignore[override]
                return list(super().acceptable_to_propose)

        m = ListProposer("m", "man")
        w1 = Responder("w1", "woman")
        w2 = Responder("w2", "woman")
        m.preferences = (w1, w2, m)
        m.last_proposal = w1
        assert m.next_proposal == w2

    def test_unhashable_person_that_is_not_in_preferences(self) -> None:
        class UnhashableResponder(Responder):
            __hash__ = None  # type: ignore[assignment]

        m = Proposer("m", "man")
        w = Responder("w", "woman")
        stranger = UnhashableResponder("s", "woman")
        m.preferences = (w, m)
        with pytest.raises(ValueError, match="not in preferences"):
            m.is_acceptable(stranger)
        m.last_proposal = stranger
        with pytest.raises(ValueError, match=r"tuple\.index\(x\): x not in tuple"):
            _ = m.next_proposal

    def test_is_acceptable_overridden_on_the_instance(self) -> None:
        m = Proposer("m", "man")
        w1 = Responder("w1", "woman")
        w2 = Responder("w2", "woman")
        m.preferences = (w1, w2, m)
        assert m.acceptable_to_propose == (w1, w2, m)
        m.is_acceptable = lambda person: person is not w2 and Person.is_acceptable(m, person)  # type: ignore[method-assign]
        m.last_proposal = w1
        assert m.next_proposal == m

    def test_persons_that_compare_by_a_name_that_changes(self) -> None:
        class NamedResponder(Responder):
            def __eq__(self, other: object) -> bool:
                return isinstance(other, NamedResponder) and other.name == self.name

            def __hash__(self) -> int:
                return 0

        m = Proposer("m", "man")
        w1 = NamedResponder("w", "woman")
        w2 = NamedResponder("w", "woman")  # equal to w1 until renamed
        m.preferences = (w1, w2, m)
        assert m.acceptable_to_propose == (w1, m)
        w2.name = "w2"
        assert m.acceptable_to_propose == (w1, w2, m)

    def test_tuple_subclass_with_its_own_index(self) -> None:
        class LastIndexTuple(tuple):
            __slots__ = ()

            def index(self, value: object, *args: object) -> int:  # type: ignore[override]
                return len(self) - 1 - tuple(reversed(self)).index(value)

        m = Proposer("m", "man")
        w = Responder("w", "woman")
        m.preferences = LastIndexTuple((w, m, w))
        assert not m.is_acceptable(w)

    def test_acceptable_to_propose_getter_raising_index_error(self) -> None:
        class QueueProposer(Proposer):
            queue: tuple[Responder, ...] = ()

            @property
            def acceptable_to_propose(self) -> tuple[Responder | Proposer, ...]:
                return (self.queue[0],)  # IndexError once the queue is empty

        m = QueueProposer("m", "man")
        assert m.next_proposal == m


class TestCacheState:
    """The lookup caches are rebuilt on demand and never travel with pickles or copies."""

    caches = ("_ranks_memo", "_acceptable_memo", "_positions_memo")

    def test_pickles_and_copies_leave_out_the_caches(self) -> None:
        m = Proposer("m", "man")
        w = Responder("w", "woman")
        m.preferences = (w, m)
        w.preferences = (m, w)
        Algorithm([m], [w]).execute()
        assert "_ranks_memo" in vars(m)
        restored = pickle.loads(pickle.dumps(m))  # noqa: S301 - round-trips bytes made just above
        deep = copy.deepcopy(m)
        for clone in (restored, deep, copy.copy(m)):
            assert not [key for key in vars(clone) if key in self.caches]
        for clone in (restored, deep):
            assert clone.match is clone.preferences[0]
            assert clone.is_acceptable(clone.match)

    def test_copies_of_a_subclass_with_slots_keep_the_slot_values(self) -> None:
        class RatedResponder(Responder):
            __slots__ = ("rating",)

        w = RatedResponder("w", "woman")
        w.rating = 5
        m = Proposer("m", "man")
        w.preferences = (m, w)
        assert w.is_acceptable(m)
        clone = copy.deepcopy(w)
        assert clone.rating == 5
        assert not [key for key in vars(clone) if key in self.caches]
