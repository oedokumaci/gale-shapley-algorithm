"""Algorithm module."""

from dataclasses import dataclass
from typing import Final

from gale_shapley_algorithm.person import Proposer, Responder
from gale_shapley_algorithm.result import MatchingResult


@dataclass(slots=True)
class Algorithm:
    """Gale-Shapley Algorithm class.

    Uses slots for memory efficiency.
    """

    proposers: list[Proposer]
    responders: list[Responder]
    round: int = 0

    @property
    def persons(self) -> list[Proposer | Responder]:
        """Returns all proposers and responders."""
        return self.proposers + self.responders

    @property
    def unmatched_proposers(self) -> list[Proposer]:
        """Returns unmatched proposers, excludes self matches."""
        return [proposer for proposer in self.proposers if not proposer.is_matched]

    @property
    def awaiting_to_respond_responders(self) -> list[Responder]:
        """Returns responders that are awaiting to respond."""
        return [responder for responder in self.responders if responder.awaiting_to_respond]

    def proposers_propose(self) -> None:
        """Makes all unmatched proposers propose to their next choice."""
        for proposer in self.unmatched_proposers:
            proposer.propose()

    def responders_respond(self) -> None:
        """Makes all responders that are awaiting to respond respond."""
        for responder in self.awaiting_to_respond_responders:
            responder.respond()

    def terminate(self) -> bool:
        """Returns True if all proposers are matched, False otherwise."""
        return all(proposer.is_matched for proposer in self.proposers)

    def format_matches(self) -> str:
        """Format all matches as a string."""
        lines: list[str] = ["Matching:"]
        for proposer in self.proposers:
            match proposer.match:
                case None:
                    lines.append(f"{proposer.name} is unmatched.")
                case m if m is proposer:
                    lines.append(f"{proposer.name} is matched to self.")
                case m:
                    lines.append(f"{proposer.name} is matched to {m.name}.")

        for responder in self.responders:
            match responder.match:
                case m if m is responder:
                    lines.append(f"{responder.name} is matched to self.")
                case None:
                    lines.append(f"{responder.name} is unmatched.")

        return "\n".join(lines)

    def format_all_preferences(self, compact: bool = True) -> str:
        """Format preferences of all proposers and responders as a string.

        Args:
            compact: If True formats all in one table. Defaults to True.
        """
        lines: list[str] = []
        if compact:
            lines.append("Preferences in compact format, only showing acceptables:")
            header: Final[list[str]] = [p.name for p in self.persons]
            first_column: Final[list[str]] = [
                f"{i}." for i in range(1, max(len(self.proposers), len(self.responders)) + 2)
            ]
            data: list[list[str]] = []
            for i in range(len(first_column)):
                data.append(
                    [
                        (
                            person.preferences[i].name
                            if bool(person.preferences)
                            and i < len(person.preferences)
                            and person.is_acceptable(person.preferences[i])
                            else ""
                        )
                        for person in self.persons
                    ]
                )
            format_row: Final[str] = "{:8}" * (len(header) + 1)
            lines.append(format_row.format("", *header))
            lines.append(format_row.format("", *["-" * len(h) for h in header]))
            for pref, row in zip(first_column, data, strict=False):
                lines.append(format_row.format(pref, *row))
        else:
            lines.append("Preferences for each person separately:")
            for person in self.persons:
                lines.append(person.format_preferences())
                if person != self.persons[-1]:
                    lines.append("")

        return "\n".join(lines)

    def execute(self) -> MatchingResult:
        """Run the algorithm and return structured results.

        Returns:
            MatchingResult with rounds, matches, unmatched, self_matches, all_matched.
        """
        while not self.terminate():
            self.proposers_propose()
            self.responders_respond()
            self.round += 1

        # Change None to self matches for unmatched responders
        for responder in self.responders:
            if not responder.is_matched:
                responder.match = responder

        matches: dict[str, str] = {}
        unmatched: list[str] = []
        self_matches: list[str] = []

        for proposer in self.proposers:
            match proposer.match:
                case None:
                    unmatched.append(proposer.name)
                case m if m is proposer:
                    self_matches.append(proposer.name)
                case m:
                    matches[proposer.name] = m.name

        for responder in self.responders:
            match responder.match:
                case m if m is responder:
                    self_matches.append(responder.name)
                case None:
                    unmatched.append(responder.name)

        return MatchingResult(
            rounds=self.round,
            matches=matches,
            unmatched=unmatched,
            self_matches=self_matches,
            all_matched=len(unmatched) == 0 and len(self_matches) == 0,
        )
