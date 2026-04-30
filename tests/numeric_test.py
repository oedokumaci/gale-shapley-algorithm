"""Tests for the numpy-backed `numeric` submodule.

Cross-validates:
  - numeric GS against the symbolic Algorithm/Proposer/Responder API
  - rotation-based lattice enumeration against brute-force permutation
    enumeration on small n
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from gale_shapley_algorithm import Algorithm, Proposer, Responder
from gale_shapley_algorithm.numeric import (
    GSStats,
    apply_rotation,
    enumerate_stable_matchings,
    exposed_rotations,
    fifo_selector,
    find_blocking_pairs,
    gale_shapley,
    gale_shapley_traced,
    is_stable,
    lifo_selector,
    men_optimal_gs,
    men_optimal_traced,
    random_selector,
    women_optimal_gs,
    women_optimal_traced,
)
from gale_shapley_algorithm.numeric.stability import is_stable_batch


def _random_instance(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    men_rank = np.empty((n, n), dtype=np.int16)
    women_rank = np.empty((n, n), dtype=np.int16)
    for i in range(n):
        men_rank[i, rng.permutation(n)] = np.arange(1, n + 1, dtype=np.int16)
    for j in range(n):
        women_rank[j, rng.permutation(n)] = np.arange(1, n + 1, dtype=np.int16)
    return men_rank, women_rank


def _symbolic_men_optimal(men_rank: np.ndarray, women_rank: np.ndarray) -> np.ndarray:
    n = men_rank.shape[0]
    men = [Proposer(f"m{i}", "man") for i in range(n)]
    women = [Responder(f"w{j}", "woman") for j in range(n)]
    for i, p in enumerate(men):
        order = np.argsort(men_rank[i])
        p.preferences = (*[women[int(j)] for j in order], p)
    for j, r in enumerate(women):
        order = np.argsort(women_rank[j])
        r.preferences = (*[men[int(i)] for i in order], r)
    algo = Algorithm(list(men), list(women))
    result = algo.execute()
    out = np.empty(n, dtype=np.int16)
    for i in range(n):
        out[i] = int(result.matches[f"m{i}"][1:])
    return out


def _brute_force_lattice(men_rank: np.ndarray, women_rank: np.ndarray) -> np.ndarray:
    n = men_rank.shape[0]
    perms = np.fromiter(
        itertools.chain.from_iterable(itertools.permutations(range(n))),
        dtype=np.int16,
    ).reshape(-1, n)
    stable = is_stable_batch(men_rank, women_rank, perms)
    return perms[stable]


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_numeric_gs_matches_symbolic_on_random_instances(rng: np.random.Generator) -> None:
    for _ in range(20):
        men, women = _random_instance(6, rng)
        ours = men_optimal_gs(men, women)
        theirs = _symbolic_men_optimal(men, women)
        assert np.array_equal(ours, theirs)
        assert is_stable(men, women, ours)


def test_women_optimal_is_stable(rng: np.random.Generator) -> None:
    for _ in range(10):
        men, women = _random_instance(6, rng)
        match = women_optimal_gs(men, women)
        assert is_stable(men, women, match)


def test_lattice_matches_brute_force_on_small_n(rng: np.random.Generator) -> None:
    for n in (3, 4, 5, 6, 7):
        for _ in range(5):
            men, women = _random_instance(n, rng)
            rot_result = enumerate_stable_matchings(men, women)
            brute_result = _brute_force_lattice(men, women)
            rot_set = {tuple(int(x) for x in row) for row in rot_result}
            brute_set = {tuple(int(x) for x in row) for row in brute_result}
            assert rot_set == brute_set, (men, women, rot_set, brute_set)
            # Every row in the rotation output must actually be stable.
            for k in range(rot_result.shape[0]):
                assert is_stable(men, women, rot_result[k])


def test_lattice_contains_both_extremes(rng: np.random.Generator) -> None:
    for _ in range(10):
        men, women = _random_instance(6, rng)
        lattice = enumerate_stable_matchings(men, women)
        mo = men_optimal_gs(men, women)
        wo = women_optimal_gs(men, women)
        found_mo = any(np.array_equal(lattice[k], mo) for k in range(lattice.shape[0]))
        found_wo = any(np.array_equal(lattice[k], wo) for k in range(lattice.shape[0]))
        assert found_mo
        assert found_wo


def test_find_blocking_pairs_empty_on_stable(rng: np.random.Generator) -> None:
    men, women = _random_instance(5, rng)
    match = men_optimal_gs(men, women)
    assert find_blocking_pairs(men, women, match) == []


def test_find_blocking_pairs_nonempty_on_unstable() -> None:
    # n=2 construction where m0 <-> w1, m1 <-> w0 is unstable if both prefer the other.
    men_rank = np.array([[1, 2], [2, 1]], dtype=np.int16)
    women_rank = np.array([[1, 2], [2, 1]], dtype=np.int16)
    bad = np.array([1, 0], dtype=np.int16)
    bp = find_blocking_pairs(men_rank, women_rank, bad)
    assert len(bp) > 0


def test_gs_rejects_non_square() -> None:
    with pytest.raises(ValueError, match="square"):
        gale_shapley(np.zeros((2, 3), dtype=np.int16), np.zeros((2, 3), dtype=np.int16))


def test_gs_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        gale_shapley(np.zeros((3, 3), dtype=np.int16), np.zeros((4, 4), dtype=np.int16))


def test_gs_rejects_non_permutation_rows() -> None:
    """Silent acceptance of non-permutation rows produces wrong matchings — must raise."""
    bad_proposer = np.array([[1, 1, 2], [1, 2, 3], [3, 2, 1]], dtype=np.int16)
    good_responder = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=np.int16)
    with pytest.raises(ValueError, match="permutation"):
        gale_shapley(bad_proposer, good_responder)


def test_gs_rejects_zero_indexed_input() -> None:
    """1-indexed convention is documented; 0-indexed input must not be silently accepted."""
    proposer_rank = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]], dtype=np.int16)
    responder_rank = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int16)
    with pytest.raises(ValueError, match="permutation"):
        gale_shapley(proposer_rank, responder_rank)


def test_enumerate_stable_matchings_rejects_non_permutation_rows() -> None:
    bad = np.array([[1, 5, 9], [2, 3, 1], [3, 1, 2]], dtype=np.int16)
    good = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=np.int16)
    with pytest.raises(ValueError, match="permutation"):
        enumerate_stable_matchings(bad, good)


def test_exposed_rotations_rejects_non_permutation_rows() -> None:
    bad = np.array([[1, 1, 2], [1, 2, 3], [3, 2, 1]], dtype=np.int16)
    good = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=np.int16)
    matching = np.array([0, 1, 2], dtype=np.int16)
    with pytest.raises(ValueError, match="permutation"):
        exposed_rotations(bad, good, matching)


def test_is_stable_false_on_wrong_length_match() -> None:
    men_rank = np.array([[1, 2], [2, 1]], dtype=np.int16)
    women_rank = np.array([[1, 2], [2, 1]], dtype=np.int16)
    short_match = np.array([0], dtype=np.int16)
    assert is_stable(men_rank, women_rank, short_match) is False


def test_is_stable_false_on_duplicate_partners() -> None:
    men_rank = np.array([[1, 2], [2, 1]], dtype=np.int16)
    women_rank = np.array([[1, 2], [2, 1]], dtype=np.int16)
    not_a_permutation = np.array([0, 0], dtype=np.int16)
    assert is_stable(men_rank, women_rank, not_a_permutation) is False


def test_is_stable_batch_shape() -> None:
    men_rank = np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]], dtype=np.int16)
    women_rank = np.array([[3, 1, 2], [1, 3, 2], [2, 1, 3]], dtype=np.int16)
    matchings = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int16)
    mask = is_stable_batch(men_rank, women_rank, matchings)
    assert mask.shape == (3,)
    assert mask.dtype == np.bool_


def test_lattice_matches_brute_force_on_random_instances(rng: np.random.Generator) -> None:
    """For every random n<=7 instance, rotation-based and brute-force enumeration return the same set."""
    for n in (3, 4, 5, 6, 7):
        for _ in range(8):
            men, women = _random_instance(n, rng)
            rot_set = {tuple(int(x) for x in row) for row in enumerate_stable_matchings(men, women, method="rotation")}
            brute_set = {tuple(int(x) for x in row) for row in enumerate_stable_matchings(men, women, method="brute")}
            assert rot_set == brute_set, (men, women, rot_set, brute_set)


def test_brute_force_refuses_above_max_n() -> None:
    """Brute force is explicitly capped at max_n to avoid n!-runaway."""
    rng = np.random.default_rng(7)
    men, women = _random_instance(11, rng)
    with pytest.raises(ValueError, match="max_n"):
        enumerate_stable_matchings(men, women, method="brute")


def test_rotation_enumeration_scales_past_brute_force() -> None:
    """n=20 is far beyond brute force's n!=2.4e18 but rotation BFS handles it."""
    rng = np.random.default_rng(13)
    men, women = _random_instance(20, rng)
    lattice = enumerate_stable_matchings(men, women, method="rotation")
    assert lattice.shape[0] >= 1
    assert lattice.shape[1] == 20
    # Check a few of the returned matchings are actually stable.
    for k in range(min(lattice.shape[0], 5)):
        assert is_stable(men, women, lattice[k])


def test_exposed_rotations_empty_at_women_optimal(rng: np.random.Generator) -> None:
    """No rotation is exposed at the women-optimal matching (infimum of the lattice)."""
    men, women = _random_instance(6, rng)
    wo = women_optimal_gs(men, women)
    assert exposed_rotations(men, women, wo) == []


def test_exposed_rotations_nonempty_at_men_optimal_when_lattice_has_multiple_matchings(
    rng: np.random.Generator,
) -> None:
    """If the lattice has more than one stable matching, the men-optimal matching must expose at least one rotation."""
    for _ in range(20):
        men, women = _random_instance(6, rng)
        lattice = enumerate_stable_matchings(men, women, method="brute")
        mo = men_optimal_gs(men, women)
        rotations = exposed_rotations(men, women, mo)
        if lattice.shape[0] > 1:
            assert len(rotations) >= 1


def test_apply_rotation_produces_stable_matching(rng: np.random.Generator) -> None:
    """Applying any exposed rotation to a stable matching yields another stable matching."""
    for _ in range(30):
        men, women = _random_instance(6, rng)
        mo = men_optimal_gs(men, women)
        for rot in exposed_rotations(men, women, mo):
            new_match = apply_rotation(mo, rot)
            assert is_stable(men, women, new_match)
            # And it should be strictly different from mo.
            assert not np.array_equal(new_match, mo)


def test_apply_rotation_does_not_mutate_input(rng: np.random.Generator) -> None:
    men, women = _random_instance(5, rng)
    mo = men_optimal_gs(men, women)
    mo_copy = mo.copy()
    rotations = exposed_rotations(men, women, mo)
    if rotations:
        rotation_copy = rotations[0].copy()
        _ = apply_rotation(mo, rotations[0])
        assert np.array_equal(mo, mo_copy)
        assert np.array_equal(rotations[0], rotation_copy)


# --- traced GS API -----------------------------------------------------------


def _same_prefs_instance(n: int) -> tuple[np.ndarray, np.ndarray]:
    """All proposers share one ranking of responders, and vice versa: the
    pathological clustering case where men-MW makes ``n*(n+1)/2`` proposals."""
    rank = np.tile(np.arange(1, n + 1, dtype=np.int16), (n, 1))
    return rank, rank.copy()


def test_traced_match_equals_gale_shapley(rng: np.random.Generator) -> None:
    """gale_shapley_traced.match must agree with the historical gale_shapley."""
    for _ in range(20):
        men, women = _random_instance(6, rng)
        stats = gale_shapley_traced(men, women)
        assert np.array_equal(stats.match, gale_shapley(men, women))


def test_traced_returns_gsstats(rng: np.random.Generator) -> None:
    men, women = _random_instance(5, rng)
    stats = gale_shapley_traced(men, women)
    assert isinstance(stats, GSStats)


def test_traced_proposals_equals_per_proposer_sum(rng: np.random.Generator) -> None:
    for _ in range(20):
        men, women = _random_instance(7, rng)
        stats = gale_shapley_traced(men, women)
        assert stats.proposals == int(stats.proposals_per_proposer.sum())


def test_traced_proposals_per_proposer_is_match_rank(rng: np.random.Generator) -> None:
    """Each proposer's count equals the 1-indexed rank of their final match."""
    for _ in range(20):
        men, women = _random_instance(6, rng)
        stats = gale_shapley_traced(men, women)
        for p in range(stats.match.shape[0]):
            r = int(stats.match[p])
            assert int(stats.proposals_per_proposer[p]) == int(men[p, r])


def test_traced_match_invariant_under_selector(rng: np.random.Generator) -> None:
    """Knuth: proposer-optimal matching is invariant to proposer order."""
    selector_rng = np.random.default_rng(123)
    for _ in range(15):
        men, women = _random_instance(7, rng)
        lifo = gale_shapley_traced(men, women, selector=lifo_selector).match
        fifo = gale_shapley_traced(men, women, selector=fifo_selector).match
        rand = gale_shapley_traced(men, women, selector=random_selector(selector_rng)).match
        assert np.array_equal(lifo, fifo)
        assert np.array_equal(lifo, rand)


def test_traced_proposal_count_invariant_under_selector(rng: np.random.Generator) -> None:
    """One-sided M-W: total proposals depend only on the final match (which is
    Knuth-invariant), so the count is also invariant under selector choice."""
    selector_rng = np.random.default_rng(7)
    for _ in range(15):
        men, women = _random_instance(7, rng)
        lifo = gale_shapley_traced(men, women, selector=lifo_selector).proposals
        fifo = gale_shapley_traced(men, women, selector=fifo_selector).proposals
        rand = gale_shapley_traced(men, women, selector=random_selector(selector_rng)).proposals
        assert lifo == fifo == rand


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 8])
def test_traced_same_prefs_is_quadratic(n: int) -> None:
    """When every proposer shares one ranking and every responder shares one
    ranking, men-MW makes exactly ``1+2+...+n = n*(n+1)/2`` proposals — the
    canonical worst case that motivates this work."""
    p_rank, r_rank = _same_prefs_instance(n)
    stats = gale_shapley_traced(p_rank, r_rank)
    assert stats.proposals == n * (n + 1) // 2
    assert np.array_equal(stats.match, np.arange(n, dtype=np.int16))


def test_random_selector_terminates_and_is_stable(rng: np.random.Generator) -> None:
    selector_rng = np.random.default_rng(42)
    selector = random_selector(selector_rng)
    for _ in range(10):
        men, women = _random_instance(8, rng)
        stats = gale_shapley_traced(men, women, selector=selector)
        assert is_stable(men, women, stats.match)


def test_men_optimal_traced_delegates(rng: np.random.Generator) -> None:
    men, women = _random_instance(6, rng)
    via_wrapper = men_optimal_traced(men, women)
    via_primitive = gale_shapley_traced(men, women)
    assert np.array_equal(via_wrapper.match, via_primitive.match)
    assert via_wrapper.proposals == via_primitive.proposals


def test_women_optimal_traced_match_matches_women_optimal_gs(rng: np.random.Generator) -> None:
    """women_optimal_traced.match must equal women_optimal_gs (men-indexed)."""
    for _ in range(10):
        men, women = _random_instance(6, rng)
        traced = women_optimal_traced(men, women)
        legacy = women_optimal_gs(men, women)
        assert np.array_equal(traced.match, legacy)
        assert is_stable(men, women, traced.match)


def test_women_optimal_traced_per_proposer_indexed_by_woman(rng: np.random.Generator) -> None:
    """proposals_per_proposer in women_optimal_traced is indexed by woman, so
    woman w's count equals the rank of her partner in her own preference list."""
    men, women = _random_instance(6, rng)
    traced = women_optimal_traced(men, women)
    n = traced.match.shape[0]
    for w in range(n):
        partner_m = int(np.argwhere(traced.match == w)[0, 0])
        assert int(traced.proposals_per_proposer[w]) == int(women[w, partner_m])


def test_women_optimal_traced_proposals_total(rng: np.random.Generator) -> None:
    men, women = _random_instance(6, rng)
    traced = women_optimal_traced(men, women)
    assert traced.proposals == int(traced.proposals_per_proposer.sum())


def test_traced_validates_inputs() -> None:
    """Validation should run before the loop, same as gale_shapley."""
    bad = np.array([[1, 1, 2], [1, 2, 3], [3, 2, 1]], dtype=np.int16)
    good = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=np.int16)
    with pytest.raises(ValueError, match="permutation"):
        gale_shapley_traced(bad, good)


def test_lifo_and_fifo_selectors_are_pure() -> None:
    """The default selectors don't read or mutate the free list contents."""
    assert lifo_selector([0, 1, 2]) == -1
    assert fifo_selector([0, 1, 2]) == 0
    free = [4, 5, 6]
    _ = lifo_selector(free)
    _ = fifo_selector(free)
    assert free == [4, 5, 6]
