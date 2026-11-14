# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-14

### Added
- Initial `vgi-xgboost` VGI worker exposing XGBoost to DuckDB/SQL.
- Model registry surface: `fit`, `predict` (with `with_proba`),
  `cross_val_predict`, `list_models`, `model_info`, `drop_model`. Estimators:
  `xgb_classifier`, `xgb_regressor`, `xgb_rf_classifier`, `xgb_rf_regressor`.
- XGBoost-specific interpretation: `feature_importance` and SHAP `explain`.
- Self-contained datasets (via scikit-learn): `iris`, `wine`, `breast_cancer`,
  `diabetes`, `california_housing`, `make_classification`, `make_regression`.
- Local-disk model registry behind a swappable `ModelStore` (S3/R2 seam).
- Fly.io deployment (Dockerfile installs from PyPI, no vendoring).
- Quality gate: ruff + mypy, pytest unit tests, and the `test/sql/*.test`
  integration suite over three transports (stdio, HTTP, in-process haybarn).
- GitHub Actions CI and Dependabot (pip / actions / docker).
- Composed examples joining `cross_val_predict` out-of-fold predictions back to
  `iris` (confusion matrix) and `diabetes` (held-out error per BMI quartile), plus
  an out-of-fold agent test task on `breast_cancer`.
- Requires `vgi-python>=0.51.1` (VGI protocol 3.3); `0.50.x` speaks protocol 3.2
  and fails ATTACH against the 3.3 community extension.

[Unreleased]: https://github.com/Query-farm/vgi-xgboost/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Query-farm/vgi-xgboost/releases/tag/v0.1.0
