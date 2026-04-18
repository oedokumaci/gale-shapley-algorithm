"""Numpy-backed primitives for large-scale stable-matching work.

This subpackage provides fast array-based implementations of Gale-Shapley,
stability checks, and stable-matching lattice enumeration (via the Gusfield-
Irving rotation algorithm).

It is optional: install with ``pip install gale-shapley-algorithm[numeric]``
(or ``uv add 'gale-shapley-algorithm[numeric]'``) to get ``numpy`` as a
dependency. The symbolic ``Person``-based API in ``gale_shapley_algorithm``
remains zero-dependency.

Data convention (all functions):
  * ``proposer_rank[i, j]``: 1-indexed position of responder ``j`` in proposer
    ``i``'s preference list (1 = top choice, n = last). Must be a complete
    strict permutation of ``1..n`` per row.
  * ``responder_rank[j, i]``: symmetrically, responder ``j``'s rank of proposer
    ``i``.
  * ``match[i] = j``: proposer ``i`` is paired with responder ``j``
    (proposer-indexed throughout).

Example:
    >>> import numpy as np
    >>> from gale_shapley_algorithm.numeric import (
    ...     gale_shapley,
    ...     enumerate_stable_matchings,
    ... )
    >>> proposer_rank = np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]], dtype=np.int16)
    >>> responder_rank = np.array([[3, 1, 2], [1, 3, 2], [2, 1, 3]], dtype=np.int16)
    >>> match = gale_shapley(proposer_rank, responder_rank)
    >>> lattice = enumerate_stable_matchings(proposer_rank, responder_rank)
"""

from gale_shapley_algorithm.numeric.gs import gale_shapley, men_optimal_gs, women_optimal_gs
from gale_shapley_algorithm.numeric.lattice import (
    apply_rotation,
    enumerate_stable_matchings,
    exposed_rotations,
)
from gale_shapley_algorithm.numeric.stability import find_blocking_pairs, is_stable

__all__ = [
    "apply_rotation",
    "enumerate_stable_matchings",
    "exposed_rotations",
    "find_blocking_pairs",
    "gale_shapley",
    "is_stable",
    "men_optimal_gs",
    "women_optimal_gs",
]
