# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

## Environment setup

Nothing easier!

Fork and clone the repository, then:

```bash
cd gale-shapley-algorithm
uvx --from taskipy task setup
```

> NOTE: If it fails for some reason, you'll need to install [uv](https://github.com/astral-sh/uv) manually.
>
> You can install it with:
>
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```
>
> On Windows (PowerShell):
>
> ```powershell
> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
> ```
>
> Now you can try running `uvx --from taskipy task setup` again, or simply `uv sync`.

You now have the dependencies installed.

Run `uvx --from taskipy task --list` to see all the available tasks!


## Tasks

Tasks are defined in `pyproject.toml` using [taskipy](https://github.com/taskipy/taskipy).
Run tasks with `uvx --from taskipy task <name>`. List all available tasks with `uvx --from taskipy task --list`.

## Development

1. create a new branch: `git switch -c feature-or-bugfix-name`
1. edit the code and/or the documentation

**Before committing:**

1. run `uvx --from taskipy task fix` to auto-format and lint-fix the code
1. run `uvx --from taskipy task ci` to run all checks (fix any warning)
1. if you updated the documentation or the project dependencies:
    1. run `uvx --from taskipy task docs`
    1. go to http://localhost:8000 and check that everything looks good
1. follow our [commit message convention](#commit-message-convention)

**Creating a pull request:**

1. (optional) clean up your commits: `git rebase -i main`
1. push your branch: `git push -u origin feature-or-bugfix-name`
1. create the PR: `gh pr create`
1. have it reviewed (or request an AI review with the `/review` skill)
1. merge when approved

If you are unsure about how to fix or ignore a warning, just let the continuous integration fail, and we will help you during review.

Don't bother updating the changelog, we will take care of this.

## Release workflow (maintainers)

After merging PRs, release from `main`:

```bash
git checkout main
git pull
uvx --from taskipy task changelog      # auto-update CHANGELOG.md
# review CHANGELOG.md and edit if needed
# update __version__ in src/gale_shapley_algorithm/__init__.py (hatch reads version from here)
# and version + date-released in CITATION.cff (CI fails a release whose versions disagree)
git add .
git commit -m "chore: Release version X.Y.Z"
git push
git tag vX.Y.Z
git push --tags
gh release create vX.Y.Z --generate-notes
```

## Commit message convention

Commit messages must follow our convention based on the [Angular style](https://gist.github.com/stephenparish/9941e89d80e2bc58a153#format-of-the-commit-message) or the [Karma convention](https://karma-runner.github.io/4.0/dev/git-commit-msg.html):

```
<type>[(scope)]: Subject

[Body]
```

**Subject and body must be valid Markdown.** Subject must have proper casing (uppercase for first letter if it makes sense), but no dot at the end, and no punctuation in general.

Scope and body are optional. Type can be:

- `build`: About packaging, building wheels, etc.
- `chore`: About packaging or repo/files management.
- `ci`: About Continuous Integration.
- `deps`: Dependencies update.
- `docs`: About documentation.
- `feat`: New feature.
- `fix`: Bug fix.
- `perf`: About performance.
- `refactor`: Changes that are not features or bug fixes.
- `style`: A change in code style/format.
- `tests`: About tests.

If you write a body, please add trailers at the end (for example issues and PR references, or co-authors), without relying on GitHub's flavored Markdown:

```
Body.

Issue #10: https://github.com/oedokumaci/gale-shapley-algorithm/issues/10
Related to PR oedokumaci/gale-shapley-algorithm#15: https://github.com/oedokumaci/gale-shapley-algorithm/pull/15
```

These "trailers" must appear at the end of the body, without any blank lines between them. The trailer title can contain any character except colons `:`. We expect a full URI for each trailer, not just GitHub autolinks (for example, full GitHub URLs for commits and issues, not the hash or the #issue-number).

We do not enforce a line length on commit messages summary and body, but please avoid very long summaries, and very long lines in the body, unless they are part of code blocks that must not be wrapped.

## Pull requests guidelines

Link to any related issue in the Pull Request message.

During the review, we recommend using fixups:

```bash
# SHA is the SHA of the commit you want to fix
git commit --fixup=SHA
```

Once all the changes are approved, you can squash your commits:

```bash
git rebase -i --autosquash main
```

And force-push:

```bash
git push -f
```

If this seems all too complicated, you can push or force-push each new commit, and we will squash them ourselves if needed, before merging.
