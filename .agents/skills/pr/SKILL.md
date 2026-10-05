---
name: pr
description: Create GitHub pull requests with proper formatting and descriptions. Use when the user asks to create a PR, open a pull request, or submit their changes for review.
---
# Pull Request Skill

Create well-structured GitHub pull requests.

## Process

1. Check current branch state:
   - `git status` for uncommitted changes
   - `git log origin/main..HEAD` for commits to include
   - `git diff main...HEAD` for full diff
2. Analyze ALL commits (not just the latest)
3. Draft PR title and description:
   - Title: Under 70 characters, descriptive
   - Summary: 1-3 bullet points
   - Test plan: How to verify changes
4. Push branch and create PR

## PR Format

```markdown
## Summary
- Bullet point describing main change
- Additional changes if applicable

## Test plan
- [ ] How to test the changes
- [ ] Expected outcomes
```

## Commands

```bash
# Push branch
git push -u origin <branch-name>

# Create PR
gh pr create --title "Title" --body "Body"
```

## Guidelines

- Keep PR title short and descriptive
- Reference related issues with #number
- Include test plan for reviewers
- One logical change per PR when possible
