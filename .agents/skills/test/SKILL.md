---
name: test
description: Run the project test suite using taskipy. Use when the user asks to run tests, check if tests pass, verify their changes, or test specific functionality.
---
# Test Skill

Run the project's test suite with pytest via taskipy.

## Commands

```bash
# Run full test suite with coverage
uvx --from taskipy task test

# Run tests without coverage (faster)
uvx --from taskipy task test_no_cov

# Run specific test file
uv run pytest tests/test_specific.py

# Run specific test function
uv run pytest tests/test_file.py::test_function

# Run tests matching pattern
uv run pytest -k "pattern"

# Run with verbose output
uv run pytest -v
```

## Process

1. Run the appropriate test command
2. Analyze test output for failures
3. If failures occur:
   - Identify the failing test(s)
   - Read the test code to understand expectations
   - Fix the issue in the source code
   - Re-run to verify the fix

## Coverage

Test coverage reports are generated automatically with `task test`. The project maintains coverage thresholds defined in `config/pytest.ini`.

## TDD (Test-Driven Development)

**Three Laws:** (1) No production code without a failing test. (2) Write only enough test to fail. (3) Write only enough code to pass.

**Cycle:** Red -> Green -> Refactor -> Repeat

**FIRST:** **F**ast, **I**ndependent, **R**epeatable, **S**elf-validating, **T**imely

## Guidelines

- Test behavior, not implementation
- Run tests after making changes
- Write tests for new functionality
- Use fixtures for common setup
