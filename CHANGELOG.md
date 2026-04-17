# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## [v1.6.0](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.6.0) - 2026-04-17

<small>[Compare with v1.5.7](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.5.7...v1.6.0)</small>

### Features

- **New `numeric` subpackage** for numpy-backed primitives (`gale_shapley_algorithm.numeric`). Install the `numeric` extra (`pip install "gale-shapley-algorithm[numeric]"`) to pull in numpy as a dependency. The core remains zero-dep.
    - `numeric.gs.gale_shapley` / `men_optimal_gs` / `women_optimal_gs`: fast numpy deferred acceptance on int16 rank matrices.
    - `numeric.stability.is_stable` / `find_blocking_pairs` / `is_stable_batch`: single-matching and vectorized stability checks. The batched variant filters all K candidate matchings in one tensor operation.
    - `numeric.lattice.enumerate_stable_matchings`: full stable-matching lattice enumeration via batched brute force. Default `max_n=10`; a rotation-based enumerator that scales past n=50 (Gusfield-Irving 1989) is documented as future work.
    - Cross-validated against the existing `Algorithm`/`Proposer`/`Responder` API on random instances; numeric subpackage at 100% test coverage.

### Tests / CI

- Install the `numeric` extra in the CI test job so numpy-backed tests run against a real numpy install on every supported matrix cell.

### Docs

- README + docs landing page now include a "Numerical / large-scale usage" section with a runnable snippet.
- New `examples/numeric_usage.py` demonstrating rank-matrix GS, stability checks, and lattice enumeration.

## [v1.5.7](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.5.7) - 2026-04-07

<small>[Compare with v1.5.6](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.5.6...v1.5.7)</small>

### Bug Fixes

- remove stale ty: ignore directive to fix CI ([986230e](https://github.com/oedokumaci/gale-shapley-algorithm/commit/986230e578b38a4a09753a169162b236aa3c0c34) by oedokumaci).

### Build

- bump vite ([8a71517](https://github.com/oedokumaci/gale-shapley-algorithm/commit/8a7151771bc9791079d1a3d4b4c2f50d7af32e57) by dependabot[bot]).

## [v1.5.6](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.5.6) - 2026-03-26

<small>[Compare with v1.5.5](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.5.5...v1.5.6)</small>

### Security

- bump picomatch from 4.0.3 to 4.0.4 and from 2.3.1 to 2.3.2 to fix CVE-2026-33671 and CVE-2026-33672 ([68dbb1b](https://github.com/oedokumaci/gale-shapley-algorithm/commit/68dbb1ba7252fa2d568b7e87213b76d47163ee4a) by dependabot[bot]).

## [v1.5.5](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.5.5) - 2026-03-25

<small>[Compare with v1.5.4](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.5.4...v1.5.5)</small>

### Security

- bump express-rate-limit from 8.2.1 to 8.3.1 to fix IPv4-mapped IPv6 address bypass for per-client rate limiting ([ee60527](https://github.com/oedokumaci/gale-shapley-algorithm/commit/ee60527)).
- bump flatted from 3.3.3 to 3.4.2 to fix unbounded recursion DoS and prototype pollution in `parse()` ([ee60527](https://github.com/oedokumaci/gale-shapley-algorithm/commit/ee60527)).

### Bug Fixes

- pass --venv to ty typecheck and suppress false positive ([02afa1e](https://github.com/oedokumaci/gale-shapley-algorithm/commit/02afa1e9cefce54f0720bf837cce7281f6747e8b) by Claude).
- bump express-rate-limit and flatted to fix high-severity vulnerabilities ([a377af3](https://github.com/oedokumaci/gale-shapley-algorithm/commit/a377af307481f01216a73794dc30b4e340f0554d) by Claude).

## [v1.5.4](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.5.4) - 2026-03-19

<small>[Compare with v1.5.3](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.5.3...v1.5.4)</small>

### Build

- bump hono ([020c880](https://github.com/oedokumaci/gale-shapley-algorithm/commit/020c880ec575d7e383cbf3b0c6ef734b8a7057ea) by dependabot[bot]).

## [v1.5.3](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.5.3) - 2026-03-04

<small>[Compare with v1.5.2](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.5.2...v1.5.3)</small>

### Build

- bump the npm_and_yarn group across 1 directory with 2 updates ([4f0df8a](https://github.com/oedokumaci/gale-shapley-algorithm/commit/4f0df8af8b6d71836b60d4659c23436b40a03cda) by dependabot[bot]).

## [v1.5.2](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.5.2) - 2026-03-01

<small>[Compare with v1.5.1](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.5.1...v1.5.2)</small>

### Bug Fixes

- add explicit permissions to CI workflow ([27e1265](https://github.com/oedokumaci/gale-shapley-algorithm/commit/27e12655c868dfabe8437d3bbb5d1bf94f77db66) by oedokumaci).

## [v1.5.1](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.5.1) - 2026-03-01

<small>[Compare with v1.5.0](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.5.0...v1.5.1)</small>

### Bug Fixes

- resolve 5 Dependabot security alerts in frontend dependencies ([062d729](https://github.com/oedokumaci/gale-shapley-algorithm/commit/062d729187aa1a69d430c2e6a1e74cdbd6e5b700) by oedokumaci).

## [v1.5.0](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.5.0) - 2026-02-23

<small>[Compare with v1.4.1](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.4.1...v1.5.0)</small>

### Features

- show tentative matches in all animation stages ([c56a355](https://github.com/oedokumaci/gale-shapley-algorithm/commit/c56a35557153a69996572c0cee4fb4e35a4d93f3) by oedokumaci).
- add docs and GitHub links to header ([4765f78](https://github.com/oedokumaci/gale-shapley-algorithm/commit/4765f78ae9a8c22e1f0b3733ac8bdfdf0c4b4ea2) by oedokumaci).

### Bug Fixes

- restore pointer cursor on interactive elements ([682b948](https://github.com/oedokumaci/gale-shapley-algorithm/commit/682b9481b7c2f599decf2d03051ec5aab0353024) by oedokumaci).
- remove default Vite favicon causing 404 ([2eada46](https://github.com/oedokumaci/gale-shapley-algorithm/commit/2eada46b82c4859923a5221bc2739136198fd0b6) by oedokumaci).

## [v1.4.1](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.4.1) - 2026-02-20

<small>[Compare with v1.4.0](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.4.0...v1.4.1)</small>

## [v1.4.0](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.4.0) - 2026-02-20

<small>[Compare with v1.3.2](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.3.2...v1.4.0)</small>

### Features

- add Clear All button to GUI editor (closes #29) ([cbb5834](https://github.com/oedokumaci/gale-shapley-algorithm/commit/cbb583459025591cbe834d34ad8647f9ef58630a) by oedokumaci).

## [v1.3.2](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.3.2) - 2026-02-20

<small>[Compare with v1.3.1](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.3.1...v1.3.2)</small>

### Bug Fixes

- show warning for duplicate names in GUI editor (closes #28) ([8a16033](https://github.com/oedokumaci/gale-shapley-algorithm/commit/8a16033c1ebee1f32f893a12b50f2cc0b8e1767c) by oedokumaci).

## [v1.3.1](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.3.1) - 2026-02-20

<small>[Compare with v1.3.0](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.3.0...v1.3.1)</small>

### Bug Fixes

- allow empty preference ranking in CLI (closes #27) ([a8aec10](https://github.com/oedokumaci/gale-shapley-algorithm/commit/a8aec10c1484c524f514234fa4d4f13fa41ee4b6) by oedokumaci).

## [v1.3.0](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.3.0) - 2026-02-20

<small>[Compare with v1.2.3](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.2.3...v1.3.0)</small>

### Features

- run algorithm twice with --swap-sides (closes #26) ([1572ea3](https://github.com/oedokumaci/gale-shapley-algorithm/commit/1572ea3eab5210e3f8f11443cbea506f4e715895) by oedokumaci).

## [v1.2.3](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.2.3) - 2026-02-20

<small>[Compare with v1.2.2](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.2.2...v1.2.3)</small>

### Build

- bump hono ([3b9e174](https://github.com/oedokumaci/gale-shapley-algorithm/commit/3b9e174861ac61d114e48ea25ac43c098aca0bc2) by dependabot[bot]).

## [v1.2.2](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.2.2) - 2026-02-20

<small>[Compare with v1.2.1](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.2.1...v1.2.2)</small>

### Bug Fixes

- add explicit permissions to docker-publish workflow ([537580e](https://github.com/oedokumaci/gale-shapley-algorithm/commit/537580ebb32449a028ac549c252cee209a226fb3) by oedokumaci).

### Code Refactoring

- rename gale-shapley to gale-shapley-algorithm everywhere ([bbdb4d7](https://github.com/oedokumaci/gale-shapley-algorithm/commit/bbdb4d77b95b84b68ca2ccc9f8c2eb6e63015e09) by oedokumaci).

## [v1.2.1](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.2.1) - 2026-02-18

<small>[Compare with v1.2.0](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.2.0...v1.2.1)</small>

### Bug Fixes

- fix git-changelog config and release skill ([2ff9f0d](https://github.com/oedokumaci/gale-shapley-algorithm/commit/2ff9f0dc9311fdea864181b49e78f99ff5babaee) by oedokumaci).

## [v1.2.0](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.2.0) - 2026-02-18

<small>[Compare with v1.1.3](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.1.3...v1.2.0)</small>

### Features

- replace YAML config with interactive terminal prompts ([7d3d742](https://github.com/oedokumaci/gale-shapley-algorithm/commit/7d3d742029c9873023026f46be34b9218943385f) by oedokumaci).
- add algorithm explainer, randomize button, and polish SVG layout ([9f38fa6](https://github.com/oedokumaci/gale-shapley-algorithm/commit/9f38fa6d6b58192841aca5862cccc06bf3ee94ab) by oedokumaci).
- replace text-based animation with SVG visualization and celebrity defaults ([2d74fd1](https://github.com/oedokumaci/gale-shapley-algorithm/commit/2d74fd10c96cff2a3ef5be799fb1497b816432e3) by oedokumaci).

### Bug Fixes

- fix Docker build and serve frontend from FastAPI ([6741e1d](https://github.com/oedokumaci/gale-shapley-algorithm/commit/6741e1d70f0bef6bf1ba17136f2e59e9280f2130) by oedokumaci).

### Code Refactoring

- address PR review findings ([39e2858](https://github.com/oedokumaci/gale-shapley-algorithm/commit/39e2858171256914c6245158bb8435017bdd2f4d) by oedokumaci).

## [v1.1.3](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.1.3) - 2026-02-18

<small>[Compare with v1.1.2](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.1.2...v1.1.3)</small>

### Bug Fixes

- add missing shadcn/ui utility needed by frontend components ([f444af3](https://github.com/oedokumaci/gale-shapley-algorithm/commit/f444af30a3fd98f9a51e6c2d01a5b47314007de1) by oedokumaci).

### Code Refactoring

- align import name with package name and modernize project infrastructure ([4166c70](https://github.com/oedokumaci/gale-shapley-algorithm/commit/4166c707ca6d5ab64dfa92e10104b35d9684e2ba) by oedokumaci).

## [v1.1.2](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.1.2) - 2026-02-18

<small>[Compare with v1.1.1](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.1.1...v1.1.2)</small>

### Build

- rename PyPI package to gale-shapley-algorithm ([1213c90](https://github.com/oedokumaci/gale-shapley-algorithm/commit/1213c90b3a88fbf6c2718166141925468f1371c2) by oedokumaci).

## [v1.1.1](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.1.1) - 2026-02-18

<small>[Compare with v1.1.0](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.1.0...v1.1.1)</small>

### Bug Fixes

- update hardcoded version test and resolve EncodingWarnings ([44509fb](https://github.com/oedokumaci/gale-shapley-algorithm/commit/44509fbef7e9e8b88a6571ed62031fb071834b68) by oedokumaci).

### Build

- modernize Dockerfile and add comprehensive .dockerignore ([e0d598e](https://github.com/oedokumaci/gale-shapley-algorithm/commit/e0d598ef6ec911c3ad13a81487de3f567fbc8318) by oedokumaci).

## [v1.1.0](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.1.0) - 2026-02-18

<small>[Compare with v1.0.0](https://github.com/oedokumaci/gale-shapley-algorithm/compare/v1.0.0...v1.1.0)</small>

### Features

- add FastAPI backend and React frontend for GUI ([2e35e3a](https://github.com/oedokumaci/gale-shapley-algorithm/commit/2e35e3aca68f27e2ef719753a490e4b1cec1ff22) by oedokumaci).

### Code Refactoring

- convert to zero-dep library with tests and modern tooling ([9888b86](https://github.com/oedokumaci/gale-shapley-algorithm/commit/9888b86616aebf31774168a7dc8551cc79aa2f2d) by oedokumaci).

### Build

- migrate to dependency-groups and align tasks with template-repo ([f0a9576](https://github.com/oedokumaci/gale-shapley-algorithm/commit/f0a9576ffac25b26fa53b94fbdaed57178ae6c0b) by oedokumaci).

## [v1.0.0](https://github.com/oedokumaci/gale-shapley-algorithm/releases/tag/v1.0.0) - 2024-12-29

<small>[Compare with first commit](https://github.com/oedokumaci/gale-shapley-algorithm/compare/f9f02679d80070752869147acf83323217fd81ab...v1.0.0)</small>

### Features

- pass cli args and config to container ([c5082b4](https://github.com/oedokumaci/gale-shapley-algorithm/commit/c5082b4e8d8dd72385da7f2eb73f1930aa6db0d8) by oedokumaci).
- add rich logging to std_out and refactor ([5168a21](https://github.com/oedokumaci/gale-shapley-algorithm/commit/5168a213198acfac3a7a34042bdaeb65d853c996) by oedokumaci).
- add number_of_simulations arg to make run ([f91375f](https://github.com/oedokumaci/gale-shapley-algorithm/commit/f91375fb697f5607bed10bd3cbfbec7b586fe624) by oedokumaci).
- update dependencies ([a06feeb](https://github.com/oedokumaci/gale-shapley-algorithm/commit/a06feeb9816c5e5393059d14447ce4211ca012c8) by oedokumaci).
- add override log user input ([458f507](https://github.com/oedokumaci/gale-shapley-algorithm/commit/458f50734b2a1c1848e1b2513321bd3c27cc2ccd) by oedokumaci).
- add swap_sides cli option ([d288608](https://github.com/oedokumaci/gale-shapley-algorithm/commit/d288608a40207d069b35922869ad6c929c810a54) by oedokumaci).
- custom deterministic input ([d742469](https://github.com/oedokumaci/gale-shapley-algorithm/commit/d742469a578089b0af41d3ee4b90cdc8870a90f2) by oedokumaci).

### Bug Fixes

- workdir already changed ([44ca259](https://github.com/oedokumaci/gale-shapley-algorithm/commit/44ca25973745218067f04d5062514b282a8b6bae) by oedokumaci).
- requirements ([945b99c](https://github.com/oedokumaci/gale-shapley-algorithm/commit/945b99c670342eaca004799c601fabb0dfd609f5) by oedokumaci).
- avoid recursive calls ([fd68e0f](https://github.com/oedokumaci/gale-shapley-algorithm/commit/fd68e0fcd81ab84df72e4fc2cc83998e06f88a7e) by oedokumaci).
- fixes the wrong logging with custom input if number_of_side does not match with input side a warning is produced in that case logger does not log the true number of persons in that side ([444ed37](https://github.com/oedokumaci/gale-shapley-algorithm/commit/444ed37c3a0de3a2e15067aaf88b3f0b088e94a4) by oedokumaci).

### Code Refactoring

- rename config_parser.py to config.py ([87c1e29](https://github.com/oedokumaci/gale-shapley-algorithm/commit/87c1e29e2094a8e04adbab89a9fa025803f452ac) by oedokumaci).
- no need to str wrap also change timer decorator precision to .4f ([86a6ec0](https://github.com/oedokumaci/gale-shapley-algorithm/commit/86a6ec0b87e5b9f64a15269895b70014a084d4ff) by oral.ersoy.dokumaci).
- style prints ([aa9c3ad](https://github.com/oedokumaci/gale-shapley-algorithm/commit/aa9c3ad3d7fa18bdf9b974f3348596e8eb53b812) by oedokumaci).
- add !r ([b0ed891](https://github.com/oedokumaci/gale-shapley-algorithm/commit/b0ed891d45e34db28e39261215401d45f5808380) by oral.ersoy.dokumaci).

### Build

- bump pydantic from 1.10.8 to 1.10.13 in the pip group across 1 directory (#15) ([d243318](https://github.com/oedokumaci/gale-shapley-algorithm/commit/d243318b6fff6035e82a4b250d7c0226b40c8e3d) by dependabot[bot]).

