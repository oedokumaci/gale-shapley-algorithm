"""Numpy Gale-Shapley deferred acceptance on rank matrices."""

from __future__ import annotations

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


def gale_shapley(proposer_rank: NDArray[np.integer], responder_rank: NDArray[np.integer]) -> NDArray[np.int16]:
    """Run proposer-optimal deferred acceptance.

    Args:
        proposer_rank: ``(n, n)`` array where ``proposer_rank[i, j]`` is
            the 1-indexed position of responder ``j`` in proposer ``i``'s
            preference list.
        responder_rank: ``(n, n)`` array with the symmetric convention.

    Returns:
        ``match`` of shape ``(n,)``, dtype ``int16``, where ``match[i]`` is
        the responder paired with proposer ``i``.

    Raises:
        ValueError: if the two arrays don't have the same square shape.
    """
    if proposer_rank.shape != responder_rank.shape:
        raise ValueError(
            f"Rank matrices must have the same shape; got {proposer_rank.shape} vs {responder_rank.shape}.",
        )
    if proposer_rank.ndim != 2 or proposer_rank.shape[0] != proposer_rank.shape[1]:
        raise ValueError(f"Rank matrices must be square; got {proposer_rank.shape}.")

    n = proposer_rank.shape[0]
    next_proposal = np.zeros(n, dtype=np.int16)
    responder_match = np.full(n, -1, dtype=np.int16)
    proposer_pref = np.argsort(proposer_rank, axis=1).astype(np.int16)
    free = list(range(n))
    while free:
        p = free.pop()
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
    match = np.empty(n, dtype=np.int16)
    for r in range(n):
        match[int(responder_match[r])] = r
    return match


def men_optimal_gs(men_rank: NDArray[np.integer], women_rank: NDArray[np.integer]) -> NDArray[np.int16]:
    """Return the men-optimal stable matching (conventionally men propose, women dispose)."""
    return gale_shapley(men_rank, women_rank)


def women_optimal_gs(men_rank: NDArray[np.integer], women_rank: NDArray[np.integer]) -> NDArray[np.int16]:
    """Return the women-optimal stable matching, still in men-indexed form ``match[m] = w``."""
    women_side = gale_shapley(women_rank, men_rank)  # women_side[w] = m
    n = women_rank.shape[0]
    match = np.empty(n, dtype=np.int16)
    for w in range(n):
        match[int(women_side[w])] = w
    return match
