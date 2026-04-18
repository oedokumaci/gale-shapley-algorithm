"""Enumerate the full stable-matching lattice of an SMP instance.

Two enumeration strategies are provided:

``rotation`` (default, Gusfield-Irving 1989): navigates the lattice by exposing
and eliminating rotations. Runs in O(n**2 + |L| * n**2) where |L| is the number
of stable matchings. Scales past n=50 comfortably; the only asymptotic cost is
|L| itself, which is empirically small for uniform-random preferences but can
be exponential in adversarial instances.

``brute``: vectorized batched brute force over all n! permutations. Simple,
memory-bounded via batching, but capped at ``max_n`` (default 10) because n!
grows fast. Useful as a correctness oracle for testing the rotation
implementation, and for instances where you want the deterministic n!-enumeration
order.

Also exports two lower-level primitives that expose the rotation structure
itself: :func:`exposed_rotations` returns the rotations currently applicable
at a given stable matching, and :func:`apply_rotation` takes one step down
the lattice. These are the building blocks for any algorithm that walks the
stable-matching lattice (lattice-navigation search, RL agents that learn a
rotation policy, incremental maintenance under preference changes, etc.).

References:
    Gusfield, D. (1987). "Three Fast Algorithms for Four Problems in Stable
        Marriage." SIAM J. Comput. 16(1).
    Irving, R. W. and Leather, P. (1986). "The Complexity of Counting Stable
        Marriages." SIAM J. Comput. 15(3).
    Gusfield, D. and Irving, R. W. (1989). *The Stable Marriage Problem:
        Structure and Algorithms.* MIT Press.
"""

from __future__ import annotations

import itertools
from collections import deque
from typing import TYPE_CHECKING, Literal

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The `numeric` subpackage requires numpy. Install with `pip install gale-shapley-algorithm[numeric]`.",
    ) from e

from gale_shapley_algorithm.numeric.gs import men_optimal_gs
from gale_shapley_algorithm.numeric.stability import is_stable_batch

if TYPE_CHECKING:
    from numpy.typing import NDArray


_DEFAULT_MAX_N_BRUTE = 10
_DEFAULT_BATCH_SIZE = 200_000


# ---------------------------------------------------------------------------
# Rotation primitives
# ---------------------------------------------------------------------------


def _next_rotation_target(
    m: int,
    matching: NDArray[np.integer],
    men_rank: NDArray[np.integer],
    women_rank: NDArray[np.integer],
    partner_of_woman: NDArray[np.integer],
) -> int:
    """Return the next woman m would pair with if displaced from his current partner.

    Specifically: walk m's preference list strictly below his current partner,
    and return the first woman who prefers m to her current ``matching``-partner.
    Returns -1 if no such woman exists.

    This is the classical rotation-target definition: the woman m would
    "promote to" in one step of a breakmarriage cycle. It depends on the
    current matching, not on any precomputed reduced list — a stale reduced
    list was the trap in my first implementation attempt.
    """
    n = int(matching.shape[0])
    current_partner_rank = int(men_rank[m, int(matching[m])])
    for target_rank in range(current_partner_rank + 1, n + 1):
        # Find the woman at preference rank ``target_rank`` on m's list.
        matches = np.where(men_rank[m] == target_rank)[0]
        if matches.size == 0:
            continue
        w = int(matches[0])
        if int(women_rank[w, m]) < int(women_rank[w, int(partner_of_woman[w])]):
            return w
    return -1


def _compute_rotation_pointers(
    matching: NDArray[np.integer],
    men_rank: NDArray[np.integer],
    women_rank: NDArray[np.integer],
) -> tuple[dict[int, int], dict[int, int]]:
    """Build next_target and next_man dictionaries for every man with a valid rotation target."""
    n = int(matching.shape[0])
    partner_of_woman = np.argsort(matching).astype(np.int16)
    next_target: dict[int, int] = {}
    next_man: dict[int, int] = {}
    for m in range(n):
        w = _next_rotation_target(m, matching, men_rank, women_rank, partner_of_woman)
        if w != -1:
            next_target[m] = w
            next_man[m] = int(partner_of_woman[w])
    return next_target, next_man


def _find_cycles(next_man: dict[int, int]) -> list[list[int]]:
    """Return every cycle in a functional graph where each node has at most one outgoing edge."""
    cycles: list[list[int]] = []
    globally_visited: set[int] = set()
    for start in next_man:
        if start in globally_visited:
            continue
        path: list[int] = []
        index_in_path: dict[int, int] = {}
        cur = start
        while cur in next_man and cur not in globally_visited:
            if cur in index_in_path:
                cycles.append(path[index_in_path[cur] :])
                break
            index_in_path[cur] = len(path)
            path.append(cur)
            cur = next_man[cur]
        globally_visited.update(path)
    return cycles


def exposed_rotations(
    men_rank: NDArray[np.integer],
    women_rank: NDArray[np.integer],
    matching: NDArray[np.integer],
) -> list[NDArray[np.int16]]:
    """Return the list of rotations currently exposed at ``matching``.

    Each rotation is an ``(r, 2)`` int16 array; row ``i`` is ``[m_i, w_{i+1}]``,
    where ``w_{i+1}`` is the partner ``m_i`` acquires if the rotation is eliminated.
    Applying the rotation amounts to assigning ``new_matching[m_i] = w_{i+1}``
    cyclically.

    At the men-optimal matching every rotation exposed here corresponds to a
    distinct one-step move down the lattice; at the women-optimal matching no
    rotations are exposed (you have reached the infimum).

    Args:
        men_rank: ``(n, n)`` men's rank matrix.
        women_rank: ``(n, n)`` women's rank matrix.
        matching: ``(n,)`` men-indexed stable matching.

    Returns:
        List of rotations; may be empty if no rotation is exposed.
    """
    next_target, next_man = _compute_rotation_pointers(matching, men_rank, women_rank)
    cycles = _find_cycles(next_man)
    rotations: list[NDArray[np.int16]] = []
    for cycle in cycles:
        rot = np.empty((len(cycle), 2), dtype=np.int16)
        for i, m in enumerate(cycle):
            rot[i, 0] = m
            rot[i, 1] = next_target[m]
        rotations.append(rot)
    return rotations


def apply_rotation(
    matching: NDArray[np.integer],
    rotation: NDArray[np.integer],
) -> NDArray[np.int16]:
    """Apply a rotation to a stable matching, producing the next matching.

    Args:
        matching: ``(n,)`` men-indexed stable matching.
        rotation: ``(r, 2)`` array from :func:`exposed_rotations`.

    Returns:
        A fresh ``(n,)`` int16 matching. Input arrays are not mutated.
    """
    new_matching = matching.astype(np.int16, copy=True)
    for i in range(int(rotation.shape[0])):
        new_matching[int(rotation[i, 0])] = int(rotation[i, 1])
    return new_matching


# ---------------------------------------------------------------------------
# Whole-lattice enumeration
# ---------------------------------------------------------------------------


def _enumerate_via_rotations(
    men_rank: NDArray[np.integer],
    women_rank: NDArray[np.integer],
) -> NDArray[np.int16]:
    """BFS the stable-matching lattice starting from the men-optimal matching."""
    n = int(men_rank.shape[0])
    mo = men_optimal_gs(men_rank, women_rank).astype(np.int16)
    visited: dict[tuple[int, ...], NDArray[np.int16]] = {tuple(int(x) for x in mo): mo}
    queue: deque[NDArray[np.int16]] = deque([mo])
    while queue:
        current = queue.popleft()
        for rotation in exposed_rotations(men_rank, women_rank, current):
            new_matching = apply_rotation(current, rotation)
            key = tuple(int(x) for x in new_matching)
            if key not in visited:
                visited[key] = new_matching
                queue.append(new_matching)
    if not visited:
        return np.empty((0, n), dtype=np.int16)  # pragma: no cover
    return np.stack(list(visited.values()))


def _permutation_batches(n: int, batch_size: int) -> tuple[NDArray[np.int16], ...]:
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


def _enumerate_via_brute_force(
    men_rank: NDArray[np.integer],
    women_rank: NDArray[np.integer],
    max_n: int,
    batch_size: int,
) -> NDArray[np.int16]:
    n = int(men_rank.shape[0])
    if n > max_n:
        raise ValueError(
            f"Brute-force enumeration requested for n={n}, but max_n={max_n}. "
            "Pass method='rotation' (default) to use the scalable Gusfield-Irving algorithm.",
        )
    stable_rows: list[NDArray[np.int16]] = [np.empty((0, n), dtype=np.int16)]
    for batch in _permutation_batches(n, batch_size):
        mask = is_stable_batch(men_rank, women_rank, batch)
        if bool(mask.any()):
            stable_rows.append(batch[mask])
    return np.concatenate(stable_rows, axis=0)


def enumerate_stable_matchings(
    men_rank: NDArray[np.integer],
    women_rank: NDArray[np.integer],
    method: Literal["rotation", "brute"] = "rotation",
    max_n: int = _DEFAULT_MAX_N_BRUTE,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> NDArray[np.int16]:
    """Return every stable matching of the instance.

    Args:
        men_rank: ``(n, n)`` men's rank matrix.
        women_rank: ``(n, n)`` women's rank matrix.
        method: ``"rotation"`` (default) uses the Gusfield-Irving rotation
            algorithm and scales to large n; ``"brute"`` uses vectorized
            permutation enumeration and is capped at ``max_n``.
        max_n: only consulted when ``method="brute"``.
        batch_size: only consulted when ``method="brute"``.

    Returns:
        ``(|L|, n)`` int16 array of men-indexed stable matchings.
    """
    if method == "rotation":
        return _enumerate_via_rotations(men_rank, women_rank)
    if method == "brute":
        return _enumerate_via_brute_force(men_rank, women_rank, max_n=max_n, batch_size=batch_size)
    raise ValueError(f"Unknown method {method!r}. Expected 'rotation' or 'brute'.")
