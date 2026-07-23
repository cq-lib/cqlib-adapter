# Contributing

## Development loop

1. Create or activate the environment from `environment-dev.yml`.
2. Install the local `cqlib` and `cqlib-tianyan` 0.1.0 builds.
3. Install this project with `python -m pip install -e ".[dev]"`.
4. Add a success-path test and at least one boundary test before implementation.
5. Run the focused test, the framework suite, and the shared regression suite.
6. Run Ruff, mypy, and the complete non-cloud test suite before committing.

Do not commit credentials, cloud result downloads, virtual environments, build output, or caches. Do commit tests and deterministic fixtures.
