"""Enumerate the full stable-matching lattice of an SMP instance.

Current implementation: vectorized brute-force over ``n!`` permutations. This
is memory-bounded — the block tensor is ``O(K * n**2)`` bools where K is the
number of permutations processed in one batch. At n=9 (K=362,880) it uses
~30 MB; at n=10 (K=3.6M) ~360 MB; we batch beyond that. For n > 12 the
permutation count itself is prohibitive.

Future work: the Gusfield-Irving (1989) rotation algorithm enumerates the
lattice in ``O(n**2 + |L| * n**2)`` time and scales past n=50. That
algorithm requires a fully-reduced list (phase-2 reduction beyond the
one-pass phase-1 reduction) and rotation BFS; a correct implementation is
non-trivial and will be added in a follow-up.

References:
    Gusfield, D. (1987). "Three Fast Algorithms for Four Problems in Stable
        Marriage." SIAM J. Comput. 16(1).
    Gusfield, D. and Irving, R. W. (1989). *The Stable Marriage Problem:
        Structure and Algorithms.* MIT Press.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The `numeric` subpackage requires numpy. Install with `pip install gale-shapley-algorithm[numeric]`.",
    ) from e

from gale_shapley_algorithm.numeric.stability import is_stable_batch

if TYPE_CHECKING:
    from numpy.typing import NDArray


_DEFAULT_MAX_N = 10
_DEFAULT_BATCH_SIZE = 200_000


def _permutation_batches(n: int, batch_size: int) -> tuple[NDArray[np.int16], ...]:
    """Yield batched permutation arrays of shape (B, n) until all n! permutations are emitted."""
    iterator = itertools.permutations(range(n))
    batches: list[NDArray[np.int16]] = []
    while True:
        flat = np.fromiter(
            itertools.islice(itertools.chain.from_iterable(iterator), batch_size * n),
            dtype=np.int16,
        )
        if flat.size == 0:
            break
        batches.append(flat.reshape(-1, n))
    return tuple(batches)


def enumerate_stable_matchings(
    men_rank: NDArray[np.integer],
    women_rank: NDArray[np.integer],
    max_n: int = _DEFAULT_MAX_N,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> NDArray[np.int16]:
    """Return every stable matching of the instance.

    Args:
        men_rank: ``(n, n)`` men's rank matrix.
        women_rank: ``(n, n)`` women's rank matrix.
        max_n: refuse to run above this size (default 10). At n=12 brute
            force enumerates 479M permutations and runs into both memory
            and wall-time trouble.
        batch_size: process permutations in batches of this many at a time
            to bound memory to ``O(batch_size * n**2)``.

    Returns:
        ``(|L|, n)`` int16 array of men-indexed stable matchings, in the
        order they were encountered during enumeration.
    """
    n = men_rank.shape[0]
    if n > max_n:
        raise ValueError(
            f"Brute-force enumeration requested for n={n}, but max_n={max_n}. "
            "A rotation-based enumerator is planned; for now, raise max_n "
            "explicitly if you accept the O(n!) runtime and O(batch_size * n**2) memory.",
        )
    # Seed with an empty (0, n) array so np.concatenate always has something to
    # concatenate even if every batch happened to contain no stable matchings.
    # For valid strict complete SMP this seed is never the only contributor —
    # at least the men-optimal matching is stable.
    stable_rows: list[NDArray[np.int16]] = [np.empty((0, n), dtype=np.int16)]
    for batch in _permutation_batches(n, batch_size):
        mask = is_stable_batch(men_rank, women_rank, batch)
        if bool(mask.any()):
            stable_rows.append(batch[mask])
    return np.concatenate(stable_rows, axis=0)
