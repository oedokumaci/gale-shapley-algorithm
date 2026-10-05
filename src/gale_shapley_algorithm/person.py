"""Person module.

Creates the units in the matching environment.
Person class is also the base class for Proposer and Responder classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _compares_like_person(obj: object) -> bool:
    """True if obj compares and hashes the way a plain Person does (by identity).

    The rank lookups below are only used for such objects: then a dict lookup finds exactly what a
    scan of the preferences would, and keeps doing so however the objects change.
    """
    cls = type(obj)
    return cls.__eq__ is Person.__eq__ and cls.__hash__ is Person.__hash__


_CACHE_ATTRIBUTES = ("_ranks_memo", "_acceptable_memo", "_positions_memo")


def _without_caches(state: object) -> object:
    """Returns an instance __dict__ state without the lookup caches (unchanged if it has none)."""
    if not isinstance(state, dict) or not any(key in state for key in _CACHE_ATTRIBUTES):
        return state
    return {key: value for key, value in state.items() if key not in _CACHE_ATTRIBUTES}


def _first_positions(items: Sequence[Person]) -> dict[Person, int] | None:
    """Returns `{item: items.index(item)}`, or None if that lookup could disagree with scanning items.

    Only an exact built-in tuple qualifies (a tuple subclass can redefine its lookups, and a list can
    change in place), and only if every item compares like a plain Person.
    """
    if type(items) is not tuple or not all(_compares_like_person(item) for item in items):
        return None
    positions: dict[Person, int] = {}
    for position, item in enumerate(items):
        positions.setdefault(item, position)  # a repeated item keeps its first position, like .index
    return positions


class Person:
    """Person class, base class for Proposer and Responder."""

    def __init__(self, name: str, side: str) -> None:
        self.name = name
        self.side = side
        self.preferences: tuple[Proposer | Responder, ...] = ()
        self.match: Proposer | Responder | None = None

    def __repr__(self) -> str:
        match self.match:
            case None:
                return f"Name: {self.name}, Side: {self.side}, Match: None"
            case _:
                return f"Name: {self.name}, Side: {self.side}, Match: {self.match.name}"

    def __getstate__(self) -> object:
        """Returns the state to pickle or copy, leaving out the lookup caches (rebuilt on first use)."""
        state = super().__getstate__()
        if isinstance(state, tuple):  # (__dict__, slots) when a subclass adds __slots__
            return (_without_caches(state[0]), state[1])
        return _without_caches(state)

    def _ranks(self) -> dict[Person, int] | None:
        """Returns `{person: self.preferences.index(person)}`, or None if it can't stand in for a scan.

        Built once per preferences object and stored together with it in a single attribute, so a
        reassignment is picked up and a concurrent reader never sees a half-updated cache.
        """
        preferences = self.preferences
        memo: tuple[object, dict[Person, int] | None] | None = getattr(self, "_ranks_memo", None)
        if memo is None or memo[0] is not preferences:
            memo = (preferences, _first_positions(preferences) if _compares_like_person(self) else None)
            self._ranks_memo = memo
        return memo[1]

    def is_acceptable(self, person: Proposer | Responder) -> bool:
        """Check if person is acceptable (ranked at or above self in preferences).

        Args:
            person: The person to check acceptability for.

        Raises:
            ValueError: If person is not in preferences.

        Returns:
            True if person is acceptable, False otherwise.
        """
        ranks = self._ranks()
        if ranks is not None and _compares_like_person(person):
            if person in ranks and self in ranks:
                return ranks[person] <= ranks[self]
        elif person in self.preferences and self in self.preferences:
            return self.preferences.index(person) <= self.preferences.index(self)
        raise ValueError(f"Either {self} or {person} is not in preferences.")

    def format_preferences(self) -> str:
        """Format the preferences of the person as a string, * indicates acceptable."""
        lines = [f"{self.name} has the following preferences, * indicates acceptable:"]
        offset_one: int = len(str(len(self.preferences)))
        offset_two: int = max(len(person.name) for person in self.preferences)
        for i, person in enumerate(self.preferences, start=1):
            acceptable = "*" if self.is_acceptable(person) else ""
            lines.append(f"{i}.{'':{offset_one - len(str(i)) + 1}}{person.name:<{offset_two + 1}}{acceptable}")
        return "\n".join(lines)

    @property
    def is_matched(self) -> bool:
        """Returns True if the person is matched to someone or self, False if match is None."""
        return bool(self.match)


class Proposer(Person):
    """Proposer class, subclass of Person."""

    def __init__(self, name: str, side: str) -> None:
        super().__init__(name, side)
        self.last_proposal: Responder | Proposer | None = None

    @property
    def acceptable_to_propose(self) -> tuple[Responder | Proposer, ...]:
        """Returns a tuple of acceptable responders to propose to, each once, in preference order."""
        preferences = self.preferences
        # Built once per preferences tuple, but only while the base is_acceptable (bound to this
        # person) decides: an override, on the class or the instance, may answer differently later.
        check = self.is_acceptable
        cacheable = (
            getattr(check, "__func__", None) is Person.is_acceptable
            and getattr(check, "__self__", None) is self
            and self._ranks() is not None
        )
        memo: tuple[object, tuple[Responder | Proposer, ...]] | None = (
            getattr(self, "_acceptable_memo", None) if cacheable else None
        )
        if memo is not None and memo[0] is preferences:
            return memo[1]
        # Keep only the first copy of a repeated entry: next_proposal moves past each responder once
        # instead of re-proposing to a repeat forever.
        acceptable = tuple(dict.fromkeys(filter(check, preferences)))
        if cacheable:
            self._acceptable_memo = (preferences, acceptable)
        return acceptable

    def _position_in(self, acceptable: Sequence[Responder | Proposer], person: Responder | Proposer) -> int:
        """Returns `acceptable.index(person)`, using a lookup built once per acceptable tuple."""
        positions = None
        if _compares_like_person(person):
            memo: tuple[object, dict[Person, int] | None] | None = getattr(self, "_positions_memo", None)
            if memo is None or memo[0] is not acceptable:
                memo = (acceptable, _first_positions(acceptable))
                self._positions_memo = memo
            positions = memo[1]
        if positions is None or person not in positions:
            return acceptable.index(person)  # raises the usual ValueError for a missing person
        return positions[person]

    @property
    def next_proposal(self) -> Responder | Proposer:
        """Returns the next acceptable responder to propose to, or self if exhausted."""
        try:
            acceptable = self.acceptable_to_propose
            match self.last_proposal:
                case None:
                    return acceptable[0]
                case last_proposal:
                    return acceptable[self._position_in(acceptable, last_proposal) + 1]
        except IndexError:
            return self

    def propose(self) -> None:
        """Propose to the next acceptable responder. If self is next, set match to self."""
        match self.next_proposal:
            case Proposer():  # meaning self is next
                self.match = self
            case responder:
                responder.current_proposals.append(self)
        self.last_proposal = self.next_proposal


class Responder(Person):
    """Responder class, subclass of Person."""

    def __init__(self, name: str, side: str) -> None:
        super().__init__(name, side)
        self.current_proposals: list[Proposer] = []

    @property
    def awaiting_to_respond(self) -> bool:
        """Returns True if current_proposals is not empty."""
        return bool(self.current_proposals)

    @property
    def acceptable_proposals(self) -> list[Proposer]:
        """Returns a list of acceptable proposals among the current proposals."""
        return [p for p in self.current_proposals if self.is_acceptable(p)]

    def _most_preferred(self, proposals: list[Proposer]) -> Proposer:
        """Returns most preferred of the list.

        Raises:
            ValueError: If preferences or proposals is empty, or proposal not in preferences.
        """
        if bool(self.preferences) and bool(proposals):
            ranks = self._ranks()
            if ranks is not None and all(_compares_like_person(proposal) for proposal in proposals):
                if all(proposal in ranks for proposal in proposals):
                    return min(proposals, key=ranks.__getitem__)
            elif all(proposal in self.preferences for proposal in proposals):
                return min(proposals, key=self.preferences.index)
        raise ValueError("Either preferences or proposals is empty, or one of the proposals is not in preferences.")

    def respond(self) -> None:
        """Respond to proposals and clear the current_proposals."""
        if bool(self.acceptable_proposals):
            match self.match:
                case Proposer() as current_match:
                    new_match = self._most_preferred(self.acceptable_proposals + [current_match])
                    if new_match != current_match:
                        current_match.match = None
                        self.match = new_match
                        new_match.match = self
                case _:
                    new_match = self._most_preferred(self.acceptable_proposals)
                    self.match = new_match
                    new_match.match = self
        self.current_proposals = []
