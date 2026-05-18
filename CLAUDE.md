# CLAUDE.md

Notes for Claude Code when working in this repo (the Nucleus Python SDK).

## What this repo is

The official Python client for Nucleus. Wraps the `/v1/nucleus` REST endpoints on `scaleapi`. Distributed on PyPI as `scale-nucleus`.

- Sources live under `nucleus/`.
- Backend lives in the `scaleapi` repo at `server/src/routes/v1/select.ts` and `server/src/lib/select/api/`.
- The default API base URL is `NUCLEUS_ENDPOINT = "https://api.scale.com/v1/nucleus"` (`nucleus/constants.py`). Override via the `endpoint=` kwarg or `NUCLEUS_ENDPOINT` env var (e.g. point at fedramp).

## Release workflow

Releases are version-numbered with [Semantic Versioning](https://semver.org/) and tracked in `CHANGELOG.md` using the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

When making a user-facing change, the convention (see PRs #459, #455) is:

1. Bump `version = "..."` in `pyproject.toml` under `[tool.poetry]`. This is the single version source — there is no `__version__` in `nucleus/__init__.py`.
   - Patch bump for additive, backwards-compatible changes (new fields, new methods).
   - Minor bump for new features that change behaviour or remove deprecated paths.
   - Major bump for breaking changes (Python version drops, sentinel removal, etc.).
2. Prepend a `## [X.Y.Z](https://github.com/scaleapi/nucleus-python-client/releases/tag/vX.Y.Z) - YYYY-MM-DD` section to `CHANGELOG.md` with `### Added` / `### Changed` / `### Fixed` / `### Removed` subsections as appropriate.
3. Commit the version bump + CHANGELOG entry alongside the code change in the same PR.

Pure refactors / doc-only PRs (#456) sometimes skip the version bump. When in doubt, bump.

## Branch and PR conventions

- Branch naming: `<author>/<kebab-description>` (e.g. `vinayparakala/expose-phash-on-dataset-item`).
- PR title commonly starts with the Linear ticket: `[DE-XXXX] <description>` — see `git log --oneline -20`.
- PRs land via squash merge.

## Architecture pointers

- `nucleus/__init__.py` — `NucleusClient`, top-level operations.
- `nucleus/dataset.py` — `Dataset` class. Most user-facing methods live here (item upload/fetch, generators, queries, slices, autotags, exports). Generators page through the backend via `nucleus/utils.py:paginate_generator`.
- `nucleus/dataset_item.py` — `DatasetItem` dataclass. **`DatasetItem.from_json` is the single deserialization entry point** for items coming back from the API — every SDK method that returns a `DatasetItem` (generators, queries, `iloc`/`refloc`/`loc`, the `items` property) routes through it. To expose a new server-side field on items, add it to the dataclass + `from_json` and you're done on the SDK side.
- `nucleus/utils.py` — `convert_export_payload` and `format_dataset_item_response` are the shared shapers used by the export and single-item endpoints. They wrap raw JSON into typed objects via the respective `from_json` classmethods.
- `nucleus/constants.py` — All API payload keys are constants here. When adding a new field, add a `*_KEY` constant first and reference it from `from_json` / `to_payload` rather than inlining the string.
- `nucleus/annotation.py`, `nucleus/prediction.py` — Annotation and prediction types. Each has its own `from_json` / `to_payload`.

## Testing

Run the suite from the repo root:

```bash
poetry install
poetry run pytest tests
```

Many tests require a real `NUCLEUS_API_KEY` and hit the live API; use `pytest -k <name>` to scope. Pre-commit hooks (`.pre-commit-config.yaml`) run black, ruff, isort.
