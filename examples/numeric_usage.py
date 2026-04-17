"""Example: the `numeric` subpackage for large-scale / numerical work.

Install with `pip install 'gale-shapley-algorithm[numeric]'` to pull in numpy.

This demonstrates:
  - Running Gale-Shapley on int16 rank matrices (both men-optimal and women-optimal).
  - Checking stability of an arbitrary matching.
  - Enumerating the full stable-matching lattice of an instance.
"""

import numpy as np

from gale_shapley_algorithm.numeric import (
    enumerate_stable_matchings,
    find_blocking_pairs,
    is_stable,
    men_optimal_gs,
    women_optimal_gs,
)


def random_instance(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Draw a uniformly random SMP instance (strict complete preferences, both sides of size n).

    Encoding: ``men_rank[i, j]`` is the 1-indexed position of woman j in man i's preference
    list (1 = top choice, n = last). Same for ``women_rank`` with roles swapped.
    """
    rng = np.random.default_rng(seed)
    men_rank = np.empty((n, n), dtype=np.int16)
    women_rank = np.empty((n, n), dtype=np.int16)
    for i in range(n):
        men_rank[i, rng.permutation(n)] = np.arange(1, n + 1, dtype=np.int16)
    for j in range(n):
        women_rank[j, rng.permutation(n)] = np.arange(1, n + 1, dtype=np.int16)
    return men_rank, women_rank


def main() -> None:
    men_rank, women_rank = random_instance(n=5, seed=7)

    mo = men_optimal_gs(men_rank, women_rank)
    wo = women_optimal_gs(men_rank, women_rank)
    print(f"Men-optimal matching (match[m] = w):   {mo.tolist()}")
    print(f"Women-optimal matching (match[m] = w): {wo.tolist()}")

    assert is_stable(men_rank, women_rank, mo)
    assert is_stable(men_rank, women_rank, wo)

    lattice = enumerate_stable_matchings(men_rank, women_rank)
    print(f"\nThis instance has {lattice.shape[0]} stable matchings total:")
    for k, m in enumerate(lattice):
        men_sum = int(sum(men_rank[i, m[i]] for i in range(len(m))))
        women_sum = int(sum(women_rank[m[i], i] for i in range(len(m))))
        print(
            f"  {k}: {m.tolist()}  sum(men ranks)={men_sum}, sum(women ranks)={women_sum}, "
            f"sex-equality |men-women|={abs(men_sum - women_sum)}",
        )

    # Example: demonstrate a clearly-unstable matching's blocking pairs.
    bad = np.arange(len(mo), dtype=np.int16)[::-1].copy()  # reverse of identity
    if not is_stable(men_rank, women_rank, bad):
        pairs = find_blocking_pairs(men_rank, women_rank, bad)
        print(f"\nThe reversed-identity matching {bad.tolist()} is blocked by {len(pairs)} pair(s).")


if __name__ == "__main__":
    main()
