"""Person types: a structural Person Protocol with concrete Proposer / Responder.

``Person`` is a Protocol — a structural type describing the surface that
matching code reads (``name``, ``preferences``, ``match``, plus the two
helpers ``is_acceptable`` and ``format_preferences``). ``Proposer`` and
``Responder`` are standalone dataclasses that satisfy this protocol; they
do **not** inherit from ``Person``.

This decouples the data model from a single inheritance hierarchy: any type
with the right shape (e.g. a future synthetic role for testing, or an
RL-agent-driven participant) is a Person.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class Person(Protocol):
    """Structural type for a participant in a matching."""

    name: str
    preferences: tuple[Proposer | Responder, ...]
    match: Proposer | Responder | None

    @property
    def is_matched(self) -> bool:
        """True iff matched to someone or self."""
        ...  # pragma: no cover

    def is_acceptable(self, person: Proposer | Responder) -> bool:
        """True iff person is at or before self in preferences."""
        ...  # pragma: no cover

    def format_preferences(self) -> str:
        """Format the preferences as a string; ``*`` marks acceptable choices."""
        ...  # pragma: no cover


def _is_acceptable(self: Person, person: Proposer | Responder) -> bool:
    """True iff person is ranked at or before self in self's preferences.

    self is always acceptable to itself (it always appears in its own
    preference tuple as the cutoff).

    Raises:
        ValueError: if either operand is not in ``self.preferences``.
    """
    if person in self.preferences and self in self.preferences:
        return self.preferences.index(person) <= self.preferences.index(self)
    raise ValueError(f"Either {self!r} or {person!r} is not in preferences.")


def _format_preferences(self: Person) -> str:
    """Format the preferences as a string; ``*`` marks acceptable choices."""
    lines = [f"{self.name} has the following preferences, * indicates acceptable:"]
    offset_one: int = len(str(len(self.preferences)))
    offset_two: int = max(len(p.name) for p in self.preferences)
    for i, p in enumerate(self.preferences, start=1):
        marker = "*" if _is_acceptable(self, p) else ""
        lines.append(f"{i}.{'':{offset_one - len(str(i)) + 1}}{p.name:<{offset_two + 1}}{marker}")
    return "\n".join(lines)


@dataclass(slots=True, eq=False)
class Proposer:
    """Proposing-side participant in a matching."""

    name: str
    preferences: tuple[Proposer | Responder, ...] = ()
    match: Proposer | Responder | None = None
    last_proposal: Responder | Proposer | None = None

    def __repr__(self) -> str:
        match_repr = "None" if self.match is None else self.match.name
        return f"Name: {self.name}, Match: {match_repr}"

    @property
    def is_matched(self) -> bool:
        """True iff matched to someone or self."""
        return self.match is not None

    def is_acceptable(self, person: Proposer | Responder) -> bool:
        """True iff person is at or before self in preferences."""
        return _is_acceptable(self, person)

    def format_preferences(self) -> str:
        """Format the preferences as a string; ``*`` marks acceptable choices."""
        return _format_preferences(self)

    @property
    def acceptable_to_propose(self) -> tuple[Responder | Proposer, ...]:
        """Tuple of acceptable counterparties to propose to."""
        return tuple(p for p in self.preferences if self.is_acceptable(p))

    @property
    def next_proposal(self) -> Responder | Proposer:
        """Next acceptable counterparty to propose to, or self if exhausted."""
        try:
            match self.last_proposal:
                case None:
                    return self.acceptable_to_propose[0]
                case _:
                    return self.acceptable_to_propose[self.acceptable_to_propose.index(self.last_proposal) + 1]
        except IndexError:
            return self

    def propose(self) -> None:
        """Propose to the next acceptable responder, or self-match if exhausted."""
        match self.next_proposal:
            case Proposer():  # next_proposal returned self
                self.match = self
            case responder:
                responder.current_proposals.append(self)
        self.last_proposal = self.next_proposal


@dataclass(slots=True, eq=False)
class Responder:
    """Responding-side participant in a matching."""

    name: str
    preferences: tuple[Proposer | Responder, ...] = ()
    match: Proposer | Responder | None = None
    current_proposals: list[Proposer] = field(default_factory=list)

    def __repr__(self) -> str:
        match_repr = "None" if self.match is None else self.match.name
        return f"Name: {self.name}, Match: {match_repr}"

    @property
    def is_matched(self) -> bool:
        """True iff matched to someone or self."""
        return self.match is not None

    def is_acceptable(self, person: Proposer | Responder) -> bool:
        """True iff person is at or before self in preferences."""
        return _is_acceptable(self, person)

    def format_preferences(self) -> str:
        """Format the preferences as a string; ``*`` marks acceptable choices."""
        return _format_preferences(self)

    @property
    def awaiting_to_respond(self) -> bool:
        """True iff there are pending proposals to respond to."""
        return bool(self.current_proposals)

    @property
    def acceptable_proposals(self) -> list[Proposer]:
        """Pending proposals filtered to acceptable ones."""
        return [p for p in self.current_proposals if self.is_acceptable(p)]

    def _most_preferred(self, proposals: list[Proposer]) -> Proposer:
        """Return the most-preferred member of ``proposals``.

        Raises:
            ValueError: if preferences or proposals is empty, or any proposal
                is not in preferences.
        """
        if self.preferences and proposals and all(p in self.preferences for p in proposals):
            return min(proposals, key=self.preferences.index)
        raise ValueError("Either preferences or proposals is empty, or one of the proposals is not in preferences.")

    def respond(self) -> None:
        """Respond to current proposals, then clear the queue."""
        if self.acceptable_proposals:
            match self.match:
                case Proposer() as current_match:
                    new_match = self._most_preferred(self.acceptable_proposals + [current_match])
                    if new_match is not current_match:
                        current_match.match = None
                        self.match = new_match
                        new_match.match = self
                case _:
                    new_match = self._most_preferred(self.acceptable_proposals)
                    self.match = new_match
                    new_match.match = self
        self.current_proposals = []
