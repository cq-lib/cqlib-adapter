# Release checklist

Run this checklist from a clean checkout before publishing.

1. Confirm `git status --short` contains only intentional source, test,
   documentation and configuration changes.
2. Search the staged diff for credentials, task IDs, cloud results and local
   absolute paths. Keep every committed `TIANYAN_API_KEY` value empty.
3. Run Ruff, formatting, mypy, `pip check` and the complete non-cloud pytest
   suite documented in `docs/testing.md`.
4. On Linux/WSL, run the independent CUDA-Q suite in
   `docs/m5-cudaq-testing.md` with `qpp-cpu`.
5. Build the sdist and wheel and run `python -m twine check dist/*`.
6. Inspect both archives and confirm they contain package sources, typing
   metadata, documentation, tests, examples, license and no generated or
   credential files.
7. Remove `.coverage`, caches, logs, `build/`, `dist/` and `*.egg-info/` before
   staging. These artifacts are reproducible and ignored by Git.

Do not run tests marked `cloud` as a release check. They create external tasks
and require separate operator authorization.
