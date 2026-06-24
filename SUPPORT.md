# Support

`vgi-xgboost` is developed and maintained by **[Query.Farm](https://query.farm)**.
It is a [VGI](https://query.farm) worker that exposes
[XGBoost](https://xgboost.ai) — datasets, a train/predict model registry, and
model interpretation — to DuckDB/SQL.

This document explains how to get help and what level of support to expect.

## Community support

The GitHub issue tracker is the place for questions, bug reports, and feature
requests:

- **Issues:** https://github.com/query-farm/vgi-xgboost/issues

Please search the existing issues before opening a new one. A good bug report
includes:

- the worker version and the DuckDB + `vgi` extension versions you are running;
- a minimal SQL snippet that reproduces the problem (ideally using a built-in
  dataset such as `xgboost.iris()`);
- what you expected to happen, and what happened instead (including the full
  error message).

Query.Farm reviews community issues and contributions and addresses them **at
its own discretion, on a best-effort basis**. We do not guarantee a response
time or a fix for any particular issue, and priorities may change without
notice. If you depend on this worker in production, please consider commercial
support.

## Commercial support

Commercial support is available, including:

- priority triage and response with an agreed service-level agreement (SLA);
- guidance on deployment, scaling, and operations;
- custom features, integrations, and prioritized bug fixes;
- private advisories and long-term maintenance arrangements.

To discuss commercial support, email **[hello@query.farm](mailto:hello@query.farm)**.

## Security

Please do **not** report security vulnerabilities through public GitHub issues.
Instead, email **[hello@query.farm](mailto:hello@query.farm)** with the details
and we will respond as quickly as we are able.

## Contributing

Bug reports and pull requests are welcome via the issue tracker above.
Contributions are accepted at Query.Farm's discretion and must pass the
project's existing test and lint suites.
