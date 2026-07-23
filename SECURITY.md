# Security policy

## Supported version

Security fixes are applied to the current `2.0.0.dev0` development line.

## Reporting a vulnerability

Do not open a public issue for an unpatched vulnerability or exposed
credential. Report it privately to `tianyan@chinatelecom.cn` with the affected
version, reproduction steps and impact. Do not include a real API key.

## Credentials and cloud execution

- Keep `TIANYAN_API_KEY` out of source files, command lines, logs and Git.
- Inject credentials only into the current process environment using hidden
  terminal input and clear the variable immediately after testing.
- Live tests require explicit `CQLIB_RUN_CLOUD=1`, a key and a device selector.
- Adapter examples use `save_credentials=False` and committed key placeholders
  must remain empty.
- Treat task IDs and downloaded cloud results as private operational metadata.

Default tests are offline. Commands marked `cloud` can create external tasks,
consume quota and must be run only by an authorized operator.
