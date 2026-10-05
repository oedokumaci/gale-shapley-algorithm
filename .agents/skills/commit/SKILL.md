---
name: commit
description: Create well-structured git commits following conventional commit format. Use when the user asks to commit changes, create a commit, or save their work to git.
---
# Commit Skill

Create commits following the project's conventional commit format.

## Commit Format

```
<type>[(scope)]: Subject
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `deps`

**Changelog types:** `build`, `deps`, `feat`, `fix`, `refactor`

## Process

1. Run `git status` to see changes
2. Run `git diff --staged` and `git diff` to understand changes
3. Analyze all changes and draft commit message:
   - Summarize the nature (feat, fix, refactor, etc.)
   - Focus on "why" rather than "what"
   - Keep subject under 72 characters
4. Stage specific files (avoid `git add -A`)
5. Create commit with conventional format
6. Verify with `git status`

## Guidelines

- Never commit sensitive files (.env, credentials)
- Stage specific files rather than using `git add -A`
- One logical change per commit
- Subject should complete: "If applied, this commit will..."
