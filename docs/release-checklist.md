# Release checklist

Publishing is a maintainer-only operation with a separate explicit approval. Merging a pull
request, pushing a tag, or running the release workflow with `publish_pypi=false` must not publish
anything.

The next intended version is `0.2.0`. The historical `v0.1.0` tag was never published and must not
be moved or reused.

## Evidence gate

- CI is green on the exact `main` commit for Python 3.11–3.13, MCP, TUI, NVML, and package smoke
  jobs.
- Ruff, pytest, `uv build`, and Twine metadata checks pass from a clean checkout.
- The core wheel runs through isolated `uvx`, standard `pip`, and `pipx` without installing extras.
- At least one compact real vLLM capture records engine version, model, server command, traffic
  scenario, raw hashes, and a golden finding. Synthetic fixtures remain labelled synthetic.
- The README contains the captured report and short staged Textual demo; claims match the observed
  environment and do not present one RTX 5080 run as a general benchmark.
- `CHANGELOG.md` has the release date and no known release-blocking limitation is hidden.

## Identity and trusted publishing gate

- Recheck `https://pypi.org/pypi/infertop/json` immediately before tagging. A 404 means no project
  currently exists; it does not reserve the name.
- Confirm `pyproject.toml`, `src/infertop/__init__.py`, `uv.lock`, changelog heading, and proposed tag
  all contain the same version.
- Configure PyPI trusted publishing for owner `Bojje`, repository `infertop`, workflow
  `.github/workflows/release.yml`, and environment `pypi` exactly. Do not add a long-lived PyPI
  token to GitHub secrets.
- Keep required reviewers or equivalent maintainer protection on the GitHub `pypi` environment.

## Tag and dry-run gate

1. Record the exact release commit SHA and verify the worktree is clean.
2. Create and push a new immutable `v0.2.0` tag at that SHA only after the evidence and identity
   gates pass. Never retarget an existing tag.
3. Manually dispatch `Release` with `tag=v0.2.0` and `publish_pypi=false`.
4. Download the `python-distributions` artifact, verify its wheel/sdist names and hashes, run
   `twine check`, inspect wheel contents, and smoke-test `infertop --version` in a clean environment.
5. Confirm the workflow checked out the exact tag commit and that no publish job ran.

## Publication gate

Only after a fresh, explicit maintainer approval, manually dispatch `Release` again with the same
tag and `publish_pypi=true`. The workflow rebuilds from the immutable tag and publishes through
OIDC in the protected `pypi` environment.

After publication:

- verify the PyPI project metadata, files, hashes, and version;
- run `uvx infertop --version`, `pipx run infertop --version`, and a clean `pip install infertop`
  against PyPI rather than the local wheel;
- create the matching GitHub release and attach or link verified hashes only after PyPI succeeds;
- update the changelog/compatibility support policy in the next development commit, not by moving
  the release tag;
- if the artifact is unsafe or materially broken, stop launch work and yank the affected file or
  release according to PyPI policy; never overwrite a published version.

## Launch gate

Launch posts come after artifact verification. Use recorded evidence, state tested engine/model/
hardware versions, link the limitations and security boundaries, and avoid universal performance
claims. Drafts may be prepared earlier, but no external submission is part of the release workflow.
