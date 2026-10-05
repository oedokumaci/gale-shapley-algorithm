# AGENTS.md

This file provides guidance for AI coding agents working on this codebase. Claude Code, Codex, and Cursor
all read it natively. There is deliberately no CLAUDE.md: Claude Code only reads AGENTS.md when CLAUDE.md is absent.

## Project Overview

**gale-shapley-algorithm** - A Python implementation of the Gale-Shapley (Deferred Acceptance) Algorithm, managed with [uv](https://github.com/astral-sh/uv).

| Attribute | Value |
|-----------|-------|
| Package | `gale-shapley-algorithm` |
| Import | `gale_shapley_algorithm` |
| Python | >=3.12 |
| Runtime deps | None (zero-dep core) |
| Extras | `cli`, `gui` (user-facing) |

## Quick Reference

```bash
uvx --from taskipy task setup        # Install dependencies and link agent skills
uvx --from taskipy task agents_link  # Link .agents/skills into .claude/skills (Claude Code)
uvx --from taskipy task run          # Run the application
uvx --from taskipy task fix          # Auto-format + lint fix
uvx --from taskipy task ci           # Run all CI checks (format, lint, typecheck, test)
uvx --from taskipy task test         # Run tests
uvx --from taskipy task docs         # Serve docs locally
uvx --from taskipy task changelog    # Update changelog (for releases)
```

## Standard Workflow

1. **Explore** - Read relevant code first
2. **Plan** - Ask clarifying questions for architectural changes
3. **Test First** - Write failing test, then minimal code to pass, then refactor
4. **Verify** - `uvx --from taskipy task fix && uvx --from taskipy task ci`

**Ask before:** Adding dependencies, changing public API, modifying structure.

## Task Independence

Each agent session is independent — there is no built-in multi-phase pipeline. Structure your requests as self-contained tasks. For multi-step workflows, complete each step fully before starting the next.

## Development Workflow

```bash
git switch -c feat/my-feature             # 1. Create feature branch
# write code and tests                    # 2. Implement changes
uvx --from taskipy task fix               # 3. Auto-format + lint fix
uvx --from taskipy task ci                # 4. Run all checks
git add . && git commit -m "feat: ..."    # 5. Commit (Angular convention)
git rebase -i main                        # 6. (Optional) Clean up commits
git push -u origin feat/my-feature        # 7. Push branch
gh pr create                              # 8. Create pull request
```

## Release Workflow (maintainers)

```bash
git checkout main && git pull                           # 1. Update main
uvx --from taskipy task changelog                       # 2. Update CHANGELOG.md
# update __version__ in src/gale_shapley_algorithm/__init__.py (hatch reads version from here)
git add . && git commit -m "chore: Release vX.Y.Z"     # 3. Commit release
git push && git tag vX.Y.Z && git push --tags           # 4. Tag and push
gh release create vX.Y.Z --generate-notes               # 5. GitHub release
```

## Code Style

### Required in Every File

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
```

- **Type hints**: Mandatory on all functions, methods, and variables
- **Docstrings**: Google style
- **Ruff**: Line length 120, Python 3.12+, absolute imports only

## Project Structure

```
gale-shapley-algorithm/
├── .agents/skills/       # Agent skills (/commit, /fix, /test, etc.), the only tracked copy
├── src/gale_shapley_algorithm/
│   ├── __init__.py       # Public API
│   ├── __main__.py       # Module entry point (run via `task run`)
│   ├── _cli/             # CLI implementation (typer + rich)
│   │   ├── app.py        # Typer app with main command
│   │   ├── prompts.py    # Interactive prompt functions (rich.prompt)
│   │   └── display.py    # Rich output formatting (tables, results)
│   ├── _api/             # FastAPI backend for GUI
│   │   └── app.py        # API endpoints
│   ├── algorithm.py      # Core Gale-Shapley algorithm
│   ├── matching.py       # Matching creation utilities
│   ├── person.py         # Person/Proposer/Responder models
│   ├── result.py         # Result types (MatchingResult, StabilityResult)
│   └── stability.py      # Stability checking
├── frontend/             # React GUI (served by FastAPI)
├── tests/
├── docs/
├── config/               # Tool configs (ruff, pytest, coverage)
└── pyproject.toml        # Metadata, dependencies, and tasks
```

- **Public API**: Exports in `__init__.py` — always import from here (e.g., `from gale_shapley_algorithm import create_matching`)
- **Internal**: `_cli/`, `_api/` — private implementation, never import from these directly
- **CLI**: `__main__.py` entry point calls `_cli/app.py` which uses interactive prompts (no config files)

## Skills

Skills live in `.agents/skills/` (Agent Skills format), the only tracked copy. Codex, Cursor, Copilot, and
Gemini CLI read that folder directly. Claude Code reads only `.claude/skills/`, so `task setup` (or
`task agents_link`) symlinks each skill there; `.claude/` is git-ignored. Invoke with `/skill-name`:

| Skill | Description |
|-------|-------------|
| `/commit` | Create well-structured git commits |
| `/pr` | Create GitHub pull requests |
| `/test` | Run the test suite |
| `/fix` | Auto-format and lint code |
| `/release` | Perform a project release |
| `/review` | Perform code review |
| `/openspec-*` | OpenSpec change workflow (new, continue, apply, verify, archive, ...) |

## Key Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Metadata, dependencies, tasks |
| `config/ruff.toml` | Linting rules |
| `config/pytest.ini` | Test configuration |
| `.pre-commit-config.yaml` | Pre-commit hooks |
