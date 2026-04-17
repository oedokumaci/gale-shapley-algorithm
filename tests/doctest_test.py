"""Run the executable examples embedded in module docstrings.

Keeps docstring code from bit-rotting relative to the actual API. Only
modules with intentional doctest-style examples are listed here.
"""

from __future__ import annotations

import doctest

import pytest

from gale_shapley_algorithm import numeric as numeric_pkg


@pytest.mark.parametrize(
    "module",
    [
        numeric_pkg,
    ],
    ids=lambda m: m.__name__,
)
def test_doctests(module: object) -> None:
    result = doctest.testmod(module, verbose=False, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.failed == 0, (
        f"Doctest failures in {module.__name__}: {result.failed} failed out of {result.attempted}"
    )
