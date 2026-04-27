"""Person module.

Creates the units in the matching environment.
Person class is also the base class for Proposer and Responder classes.
"""

from __future__ import annotations


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

    def is_acceptable(self, person: Proposer | Responder) -> bool:
        """Check if person is acceptable (ranked at or above self in preferences).

        Args:
            person: The person to check acceptability for.

        Raises:
            ValueError: If person is not in preferences.

        Returns:
            True if person is acceptable, False otherwise.
        """
        if person in self.preferences and self in self.preferences:
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
        """Returns a tuple of acceptable responders to propose to."""
        return tuple(filter(self.is_acceptable, self.preferences))

    @property
    def next_proposal(self) -> Responder | Proposer:
        """Returns the next acceptable responder to propose to, or self if exhausted."""
        try:
            match self.last_proposal:
                case None:
                    return self.acceptable_to_propose[0]
                case _:
                    return self.acceptable_to_propose[self.acceptable_to_propose.index(self.last_proposal) + 1]
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
        if bool(self.preferences) and bool(proposals) and all(proposal in self.preferences for proposal in proposals):
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
