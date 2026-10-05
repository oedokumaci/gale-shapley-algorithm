---
name: release
description: Perform a project release by updating the changelog, bumping the version, committing, tagging, and pushing. Use when the user asks to release, cut a release, bump the version, or publish a new version.
---
# Release Skill

Perform a full release for this project.

## Process

1. Ensure you are on `main` and up to date:
   ```bash
   git checkout main
   git pull
   ```
2. Run the changelog task to auto-determine the next version:
   ```bash
   uvx --from taskipy task changelog
   ```
3. Read `CHANGELOG.md` and extract the new version number from the latest heading.
4. Present the changelog diff to the user for review. **Wait for user approval before continuing.**
5. Update `__version__` in `src/gale_shapley_algorithm/__init__.py` to match the new version (without the `v` prefix).
6. Stage and commit:
   ```bash
   git add CHANGELOG.md src/gale_shapley_algorithm/__init__.py
   git commit -m "chore: Release version X.Y.Z"
   ```
7. Push, tag, push the tag, and create the GitHub release:
   ```bash
   git push
   git tag vX.Y.Z
   git push --tags
   gh release create vX.Y.Z --generate-notes
   ```

## Important

- **Always wait for user approval** after showing the changelog diff before committing.
- The version in `__init__.py` must match the tag without the `v` prefix (e.g., `__version__ = "1.2.0"` and tag `v1.2.0`).
- Only changelog-relevant commit types appear: `build`, `deps`, `feat`, `fix`, `refactor`.
- If there are no releasable commits since the last tag, inform the user and stop.
