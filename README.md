# gale-shapley-algorithm

A Python implementation of the celebrated Gale-Shapley (a.k.a. the Deferred Acceptance) Algorithm.

Time complexity is O(n^2), space complexity is O(n).

[![CI](https://github.com/oedokumaci/gale-shapley-algorithm/actions/workflows/ci.yml/badge.svg)](https://github.com/oedokumaci/gale-shapley-algorithm/actions/workflows/ci.yml)
[![Docs](https://github.com/oedokumaci/gale-shapley-algorithm/actions/workflows/docs.yml/badge.svg)](https://oedokumaci.github.io/gale-shapley-algorithm)
[![Docker](https://img.shields.io/docker/v/oedokumaci/gale-shapley-algorithm?sort=semver&label=Docker)](https://hub.docker.com/r/oedokumaci/gale-shapley-algorithm)
[![PyPI](https://img.shields.io/pypi/v/gale-shapley-algorithm)](https://pypi.org/project/gale-shapley-algorithm/)
[![Python](https://img.shields.io/pypi/pyversions/gale-shapley-algorithm)](https://pypi.org/project/gale-shapley-algorithm/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## GUI with Docker

The easiest way to try the algorithm is with the interactive web GUI:

```bash
docker pull oedokumaci/gale-shapley-algorithm
docker run --rm -p 8000:8000 oedokumaci/gale-shapley-algorithm
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

Or build locally for development:

```bash
docker build -t gale-shapley-algorithm .
docker run --rm -p 8000:8000 gale-shapley-algorithm
```

The GUI lets you:

- **Add and remove people** on each side (proposers and responders)
- **Set preferences** by drag-and-drop reordering
- **Randomize** all preferences with one click
- **Run the matching** and see results in a table with stability info
- **Animate step-by-step** to watch proposals, rejections, and tentative matches unfold round by round in an SVG visualization
- **Upload images** for each person to personalize the visualization
- Toggle **dark/light mode**

## Installation

```bash
pip install gale-shapley-algorithm
```

With CLI support:

```bash
pip install "gale-shapley-algorithm[cli]"
```

With numpy-backed primitives for large-scale / numerical work (adds `numpy >= 2.0`):

```bash
pip install "gale-shapley-algorithm[numeric]"
```

## Quick Start

### As a Library

```python
import gale_shapley_algorithm as gsa

result = gsa.create_matching(
    proposer_preferences={
        "alice": ["bob", "charlie"],
        "dave": ["charlie", "bob"],
    },
    responder_preferences={
        "bob": ["alice", "dave"],
        "charlie": ["dave", "alice"],
    },
)
print(result.matches)  # {'alice': 'bob', 'dave': 'charlie'}
```

### Numerical / large-scale usage

For high-throughput work (many random instances, enumerating the stable-matching lattice, using the output as input to downstream ML/RL pipelines), the `numeric` subpackage provides numpy-array APIs:

```python
import numpy as np
from gale_shapley_algorithm.numeric import (
    gale_shapley, men_optimal_gs, women_optimal_gs,
    is_stable, find_blocking_pairs, enumerate_stable_matchings,
)

# Rank matrices: men_rank[i, j] is woman j's 1-indexed position on man i's list.
men_rank   = np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]], dtype=np.int16)
women_rank = np.array([[3, 1, 2], [1, 3, 2], [2, 1, 3]], dtype=np.int16)

mo = men_optimal_gs(men_rank, women_rank)    # match[m] = w
wo = women_optimal_gs(men_rank, women_rank)
lattice = enumerate_stable_matchings(men_rank, women_rank)  # (|L|, n) int16 array
```

See `examples/numeric_usage.py` for a more complete walk-through. The lattice enumerator
handles up to n=10 via batched brute force; a rotation-based enumerator that scales past
n=50 is documented as future work.

### As a CLI

The CLI uses interactive prompts -- no config files needed:

```bash
# Interactive mode: enter names and rank preferences
uvx --from "gale-shapley-algorithm[cli]" python -m gale_shapley_algorithm

# Random mode: auto-generate names and preferences
uvx --from "gale-shapley-algorithm[cli]" python -m gale_shapley_algorithm --random

# Swap proposers and responders
uvx --from "gale-shapley-algorithm[cli]" python -m gale_shapley_algorithm --swap-sides
```

**Interactive mode example:**

```
$ python -m gale_shapley_algorithm

  Gale-Shapley Algorithm

Enter proposer side name [Proposers]: Men
Enter responder side name [Responders]: Women

Enter names for Men (comma-separated): Will, Hampton
Enter names for Women (comma-separated): April, Summer

Ranking preferences for Men...

  Available for Will:
  1. April
  2. Summer
  Enter ranking for Will (e.g. 1,2): 1,2
  -> Will: April > Summer

  Available for Hampton:
  1. April
  2. Summer
  Enter ranking for Hampton (e.g. 1,2): 2,1
  -> Hampton: Summer > April

Ranking preferences for Women...
  ...

┌──────── Matching Result ────────┐
│ Men     │ Women                 │
├─────────┼───────────────────────┤
│ Will    │ April                 │
│ Hampton │ Summer                │
└─────────┴───────────────────────┘
Completed in 1 round. Stable: Yes
```

**Random mode example:**

```
$ python -m gale_shapley_algorithm --random

  Gale-Shapley Algorithm

Enter proposer side name [Proposers]: Cats
Enter responder side name [Responders]: Dogs
Number of Cats [3]: 3
Number of Dogs [3]: 3

  ... (random preferences generated and displayed) ...

Completed in 2 rounds. Stable: Yes
```

## Development

This project is managed with [uv](https://github.com/astral-sh/uv) and uses [taskipy](https://github.com/taskipy/taskipy) for task running.

```bash
git clone https://github.com/oedokumaci/gale-shapley-algorithm
cd gale-shapley-algorithm
uvx --from taskipy task setup   # Install dependencies
uvx --from taskipy task run     # Run the application
uvx --from taskipy task fix     # Auto-format + lint fix
uvx --from taskipy task ci      # Run all CI checks
uvx --from taskipy task test    # Run tests
uvx --from taskipy task docs    # Serve docs locally
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

## Documentation

Full documentation is available at [oedokumaci.github.io/gale-shapley-algorithm](https://oedokumaci.github.io/gale-shapley-algorithm).

## License

[MIT](LICENSE)
