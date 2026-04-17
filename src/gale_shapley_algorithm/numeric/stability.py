"""Vectorized stability checks for rank-matrix matchings."""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The `numeric` subpackage requires numpy. Install with `pip install gale-shapley-algorithm[numeric]`.",
    ) from e

if TYPE_CHECKING:
    from numpy.typing import NDArray


def find_blocking_pairs(
    proposer_rank: NDArray[np.integer],
    responder_rank: NDArray[np.integer],
    match: NDArray[np.integer],
) -> list[tuple[int, int]]:
    """Return every ``(proposer, responder)`` pair that blocks ``match``.

    A blocking pair ``(p, r)`` is one where ``p`` strictly prefers ``r`` to
    his current partner and ``r`` strictly prefers ``p`` to her current
    partner.

    Args:
        proposer_rank: ``(n, n)`` rank matrix for the proposing side.
        responder_rank: ``(n, n)`` rank matrix for the responding side.
        match: ``(n,)`` men-indexed matching.

    Returns:
        List of 0-indexed ``(proposer, responder)`` blocking pairs.
    """
    n = proposer_rank.shape[0]
    partner_of_responder = np.empty(n, dtype=np.int16)
    for p in range(n):
        partner_of_responder[int(match[p])] = p
    blocking: list[tuple[int, int]] = []
    for p in range(n):
        current_r = int(match[p])
        for r in range(n):
            if r == current_r:
                continue
            if (
                proposer_rank[p, r] < proposer_rank[p, current_r]
                and responder_rank[r, p] < responder_rank[r, partner_of_responder[r]]
            ):
                blocking.append((p, r))
    return blocking


def is_stable(
    proposer_rank: NDArray[np.integer],
    responder_rank: NDArray[np.integer],
    match: NDArray[np.integer],
) -> bool:
    """Return ``True`` iff ``match`` is a perfect matching with no blocking pair."""
    n = proposer_rank.shape[0]
    if match.shape[0] != n:
        return False
    if np.unique(match).shape[0] != n:
        return False
    return len(find_blocking_pairs(proposer_rank, responder_rank, match)) == 0


def is_stable_batch(
    proposer_rank: NDArray[np.integer],
    responder_rank: NDArray[np.integer],
    matchings: NDArray[np.integer],
) -> NDArray[np.bool_]:
    """Vectorized stability check across a batch of matchings.

    Memory use is O(K * n**2) where K is the batch size. For K = n! (full
    permutation enumeration) the tensor grows as O(n**2 * n!), ~12 MB at
    n=8 and ~360 MB at n=10 — caller's responsibility to keep K bounded.

    Args:
        proposer_rank: ``(n, n)`` rank matrix.
        responder_rank: ``(n, n)`` rank matrix.
        matchings: ``(K, n)`` array of candidate men-indexed matchings.

    Returns:
        ``(K,)`` bool array.
    """
    arange_n = np.arange(proposer_rank.shape[0], dtype=np.int16)
    partner_rank_proposer = proposer_rank[arange_n[np.newaxis, :], matchings]
    partner_of_responder = np.argsort(matchings, axis=1)
    partner_rank_responder = responder_rank[arange_n[np.newaxis, :], partner_of_responder]

    block_p = proposer_rank[np.newaxis, :, :] < partner_rank_proposer[:, :, np.newaxis]
    block_r = responder_rank.T[np.newaxis, :, :] < partner_rank_responder[:, np.newaxis, :]
    has_block = np.any(block_p & block_r, axis=(1, 2))
    return ~has_block
