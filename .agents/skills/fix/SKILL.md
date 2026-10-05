---
name: fix
description: Run code formatting and linting with auto-fix using ruff via taskipy. Use when the user asks to fix code style, format code, lint, or clean up their code.
---
# Fix Skill

Auto-format and lint code using ruff via taskipy.

## Commands

```bash
# Format + lint with auto-fix (recommended)
uvx --from taskipy task fix

# Format only
uvx --from taskipy task format

# Lint with auto-fix only
uvx --from taskipy task lint

# Check formatting without changes (CI mode)
uvx --from taskipy task format_check

# Check linting without changes (CI mode)
uvx --from taskipy task lint_check
```

## What Gets Fixed

**Formatting (ruff format):**
- Consistent indentation
- Line length (120 chars max)
- Quote style normalization
- Trailing whitespace

**Linting (ruff check --fix):**
- Import sorting and organization
- Unused imports removal
- Simple code improvements
- PEP 8 compliance

## Process

1. Run `uvx --from taskipy task fix`
2. Review any remaining issues that couldn't be auto-fixed
3. Manually address unfixable issues
4. Run again to verify clean output

## Configuration

Ruff configuration is in `config/ruff.toml`:
- Line length: 120
- Target Python version matches project
- Google-style docstrings
- Absolute imports only
