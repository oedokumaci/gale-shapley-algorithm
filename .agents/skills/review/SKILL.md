---
name: review
description: Review code for issues, best practices, and improvements. Use when the user asks for a code review, wants feedback on their code, or asks you to check their implementation.
---
# Review Skill

Perform code review focusing on quality, correctness, and maintainability.

## Review Checklist

### Correctness
- Logic errors or edge cases
- Error handling coverage
- Type safety and null checks

### Code Quality (SOLID + Clean Code)
- **Single Responsibility**: One reason to change per class/function
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Subtypes substitutable for base types
- **Interface Segregation**: Specific interfaces over general ones
- **Dependency Inversion**: Depend on abstractions
- DRY, YAGNI, meaningful names, Boy Scout Rule
- Functions: Small (<20 lines), do one thing, few args (0-2)
- No side effects, appropriate abstraction level

### Style Compliance
- Type hints on all functions/methods
- Google-style docstrings
- Absolute imports only
- Line length under 120 chars

### Testing
- Tests cover new functionality
- Tests verify behavior, not implementation
- Edge cases tested
- No flaky tests

### Security
- No hardcoded secrets
- Input validation at boundaries
- Safe handling of user data

## Process

1. Read the code to understand its purpose
2. Run through the review checklist
3. Identify issues by category:
   - **Critical**: Bugs, security issues
   - **Major**: Design problems, missing tests
   - **Minor**: Style, naming, documentation
4. Provide actionable feedback with specific suggestions
5. Note positive aspects of the code

## Output Format

```markdown
## Code Review

### Summary
Brief overview of the code's purpose and quality.

### Critical Issues
- Issue description and location
- Suggested fix

### Suggestions
- Improvement opportunities
- Alternative approaches

### Positive Notes
- Well-done aspects
```
