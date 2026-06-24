# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report vulnerabilities privately via GitHub's
[private vulnerability reporting](https://github.com/Query-farm/vgi-xgboost/security/advisories/new)
(the **Security → Report a vulnerability** tab on this repository). We aim to
acknowledge reports within a few business days and will coordinate a fix and
disclosure timeline with you.

## Scope

This repository is the `vgi-xgboost` VGI worker. Vulnerabilities in the
underlying runtime — [`vgi-python`](https://pypi.org/project/vgi-python/),
[`vgi-rpc`](https://pypi.org/project/vgi-rpc/), DuckDB, or XGBoost — should be
reported to their respective projects, though we're happy to help route a report
if you're unsure.

## Operational notes

- The worker executes arbitrary model training/inference on data passed to it.
  Treat a deployed worker as a compute endpoint and put it behind appropriate
  authentication (the HTTP transport supports VGI signing keys / OAuth).
- The model registry deserializes stored estimators with `joblib`/pickle. Only
  load models written by this worker into a trusted `XGBOOST_MODELS_DIR`; never
  point it at an untrusted directory.
