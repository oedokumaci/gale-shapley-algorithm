"""Numpy Gale-Shapley deferred acceptance on rank matrices."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The `numeric` subpackage requires numpy. Install with "
        "`pip install gale-shapley-algorithm[numeric]` or `uv add 'gale-shapley-algorithm[numeric]'`.",
    ) from e

if TYPE_CHECKING:
    from numpy.typing import NDArray


type Selector = Callable[[Sequence[int]], int]
"""Picks which free proposer goes next.

Receives the current free list of proposer ids and returns an index into
that list (``0..len(free)-1``; negative indices supported per ``list.pop``
semantics). The default is :func:`lifo_selector`, which matches the
historical behavior of :func:`gale_shapley` (``free.pop()``).
"""


def lifo_selector(free: Sequence[int]) -> int:
    """Return the last index — equivalent to ``list.pop()`` (LIFO)."""
    del free
    return -1


def fifo_selector(free: Sequence[int]) -> int:
    """Return the first index — pops in arrival order (FIFO)."""
    del free
    return 0


def random_selector(rng: np.random.Generator) -> Selector:
    """Build a uniform-random selector backed by a numpy ``Generator``.

    Roth-Vande Vate-style: pick a free proposer uniformly at random each step.
    Final matching is invariant to selector choice (Knuth's order independence)
    but the proposal count is not — random order is a useful baseline against
    LIFO/FIFO for amortized analysis.

    Args:
        rng: a numpy ``Generator`` (use ``np.random.default_rng(seed)`` for
            determinism).

    Returns:
        A :data:`Selector` closure that draws ``rng.integers(len(free))`` per call.
    """

    def _selector(free: Sequence[int]) -> int:
        return int(rng.integers(len(free)))

    return _selector


@dataclass(slots=True, frozen=True, eq=False)
class GSStats:
    """Result of a traced Gale-Shapley run.

    ``eq=False`` because :attr:`match` and :attr:`proposals_per_proposer` are
    numpy arrays whose ``__eq__`` returns an array, not a bool — so the
    dataclass-generated ``__eq__`` would raise. Compare fields directly.

    Attributes:
        match: shape ``(n,)``, dtype ``int16``. Proposer-indexed:
            ``match[p]`` is the responder paired with proposer ``p``. The
            wrapper :func:`women_optimal_traced` re-indexes this to be
            men-indexed; see its docstring.
        proposals: total number of proposal events. Equals
            ``proposals_per_proposer.sum()``.
        proposals_per_proposer: shape ``(n,)``, dtype ``int16``.
            ``proposals_per_proposer[p]`` is the number of proposals made by
            proposer ``p``, which also equals the rank (1-indexed) of ``p``'s
            final match in ``p``'s preference list.
    """

    match: NDArray[np.int16]
    proposals: int
    proposals_per_proposer: NDArray[np.int16]


def _validate_rank_matrices(
    proposer_rank: NDArray[np.integer],
    responder_rank: NDArray[np.integer],
) -> None:
    """Validate that two rank matrices are square, same-shape, and have permutation rows.

    Each row must be a strict permutation of ``1..n`` per the data convention
    documented in :mod:`gale_shapley_algorithm.numeric`. Without this check,
    non-permutation input silently produces a wrong matching downstream.

    Raises:
        ValueError: if shapes mismatch, are non-square, or any row is not a
            permutation of ``1..n``.
    """
    if proposer_rank.shape != responder_rank.shape:
        raise ValueError(
            f"Rank matrices must have the same shape; got {proposer_rank.shape} vs {responder_rank.shape}.",
        )
    if proposer_rank.ndim != 2 or proposer_rank.shape[0] != proposer_rank.shape[1]:
        raise ValueError(f"Rank matrices must be square; got {proposer_rank.shape}.")
    n = proposer_rank.shape[0]
    expected = np.arange(1, n + 1, dtype=proposer_rank.dtype)
    for name, mat in (("proposer_rank", proposer_rank), ("responder_rank", responder_rank)):
        ok_per_row = (np.sort(mat, axis=1) == expected).all(axis=1)
        if not ok_per_row.all():
            bad_row = int(np.argmin(ok_per_row))
            raise ValueError(
                f"{name} row {bad_row} must be a permutation of 1..{n}; got {mat[bad_row].tolist()!r}.",
            )


def gale_shapley_traced(
    proposer_rank: NDArray[np.integer],
    responder_rank: NDArray[np.integer],
    *,
    selector: Selector = lifo_selector,
) -> GSStats:
    """Run proposer-optimal deferred acceptance and return per-proposer statistics.

    Mechanically identical to :func:`gale_shapley` (sequential McVitie-Wilson
    on the free-proposer pool), but exposes the proposal count and a
    pluggable proposer-selection rule. The final matching is invariant under
    selector choice (Knuth); the proposal count is not.

    Args:
        proposer_rank: ``(n, n)`` array where ``proposer_rank[i, j]`` is
            the 1-indexed position of responder ``j`` in proposer ``i``'s
            preference list. Each row must be a permutation of ``1..n``.
        responder_rank: ``(n, n)`` array with the symmetric convention.
        selector: picks which free proposer acts next; defaults to
            :func:`lifo_selector`. See :data:`Selector`.

    Returns:
        :class:`GSStats` with the matching, total proposal count, and
        per-proposer counts.

    Raises:
        ValueError: if the two arrays don't have the same square shape, or
            any row is not a permutation of ``1..n``.
        IndexError: if ``selector`` returns an index outside the free list.
    """
    _validate_rank_matrices(proposer_rank, responder_rank)
    n = proposer_rank.shape[0]
    next_proposal = np.zeros(n, dtype=np.int16)
    responder_match = np.full(n, -1, dtype=np.int16)
    proposer_pref = np.argsort(proposer_rank, axis=1).astype(np.int16)
    free: list[int] = list(range(n))
    while free:
        p = free.pop(selector(free))
        r = int(proposer_pref[p, next_proposal[p]])
        next_proposal[p] += 1
        current = int(responder_match[r])
        if current == -1:
            responder_match[r] = p
        elif responder_rank[r, p] < responder_rank[r, current]:
            responder_match[r] = p
            free.append(current)
        else:
            free.append(p)
    return GSStats(
        match=np.argsort(responder_match).astype(np.int16),
        proposals=int(next_proposal.sum()),
        proposals_per_proposer=next_proposal.copy(),
    )


def gale_shapley(proposer_rank: NDArray[np.integer], responder_rank: NDArray[np.integer]) -> NDArray[np.int16]:
    """Run proposer-optimal deferred acceptance.

    Args:
        proposer_rank: ``(n, n)`` array where ``proposer_rank[i, j]`` is
            the 1-indexed position of responder ``j`` in proposer ``i``'s
            preference list. Each row must be a permutation of ``1..n``.
        responder_rank: ``(n, n)`` array with the symmetric convention.

    Returns:
        ``match`` of shape ``(n,)``, dtype ``int16``, where ``match[i]`` is
        the responder paired with proposer ``i``.

    Raises:
        ValueError: if the two arrays don't have the same square shape, or
            any row is not a permutation of ``1..n``.
    """
    return gale_shapley_traced(proposer_rank, responder_rank).match


def men_optimal_gs(men_rank: NDArray[np.integer], women_rank: NDArray[np.integer]) -> NDArray[np.int16]:
    """Return the men-optimal stable matching (conventionally men propose, women dispose)."""
    return gale_shapley(men_rank, women_rank)


def women_optimal_gs(men_rank: NDArray[np.integer], women_rank: NDArray[np.integer]) -> NDArray[np.int16]:
    """Return the women-optimal stable matching, still in men-indexed form ``match[m] = w``."""
    return np.argsort(gale_shapley(women_rank, men_rank)).astype(np.int16)


def men_optimal_traced(
    men_rank: NDArray[np.integer],
    women_rank: NDArray[np.integer],
    *,
    selector: Selector = lifo_selector,
) -> GSStats:
    """Men-optimal stable matching with per-man proposal stats.

    Equivalent to ``gale_shapley_traced(men_rank, women_rank, selector=...)``;
    provided for symmetry with :func:`men_optimal_gs`.

    Args:
        men_rank: ``(n, n)`` rank matrix for men (proposers).
        women_rank: ``(n, n)`` rank matrix for women (responders).
        selector: see :data:`Selector`.

    Returns:
        :class:`GSStats` where ``match`` and ``proposals_per_proposer`` are
        both men-indexed.
    """
    return gale_shapley_traced(men_rank, women_rank, selector=selector)


def women_optimal_traced(
    men_rank: NDArray[np.integer],
    women_rank: NDArray[np.integer],
    *,
    selector: Selector = lifo_selector,
) -> GSStats:
    """Women-optimal stable matching with per-woman proposal stats.

    Runs :func:`gale_shapley_traced` with women as the proposing side. The
    returned ``match`` is re-indexed to men-indexed form for consistency with
    :func:`women_optimal_gs`. ``proposals_per_proposer`` remains
    **women-indexed**, since women are the proposers in this run.

    Args:
        men_rank: ``(n, n)`` rank matrix for men.
        women_rank: ``(n, n)`` rank matrix for women.
        selector: see :data:`Selector`.

    Returns:
        :class:`GSStats` with ``match[m] = w`` (men-indexed) and
        ``proposals_per_proposer[w] = number of proposals woman w made``
        (women-indexed). ``proposals`` is the unambiguous total.
    """
    stats = gale_shapley_traced(women_rank, men_rank, selector=selector)
    return GSStats(
        match=np.argsort(stats.match).astype(np.int16),
        proposals=stats.proposals,
        proposals_per_proposer=stats.proposals_per_proposer,
    )
