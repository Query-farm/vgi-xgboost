# vgi-xgboost

[![CI](https://github.com/rustyconover/vgi-xgboost/actions/workflows/ci.yml/badge.svg)](https://github.com/rustyconover/vgi-xgboost/actions/workflows/ci.yml)

A [VGI](https://github.com/query-farm/vgi-python) worker that brings
[XGBoost](https://xgboost.readthedocs.io/) into DuckDB/SQL: train gradient-boosted
models, persist them in a registry, predict over SQL tables, and interpret them
(feature importance + SHAP contributions) — all as SQL functions.

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'xgboost' (TYPE vgi, LOCATION 'uv run xgboost_worker.py');

-- train + persist a model
SELECT * FROM xgboost.fit(
  (SELECT * FROM xgboost.iris()),
  model_name := 'iris_clf', estimator := 'xgb_classifier', target := 'target', id := 'sample_id');

-- predict later
SELECT * FROM xgboost.predict((SELECT * FROM new_flowers), model_name := 'iris_clf', id := 'id');
```

## How it maps XGBoost onto SQL

XGBoost is built around stateful *fit / predict* estimators; SQL is set-oriented.
Each piece is mapped to the VGI primitive that fits its data flow:

| Area | SQL surface | VGI primitive |
| --- | --- | --- |
| **Datasets** | `SELECT * FROM xgboost.iris()` | table function (source) |
| **Fit** | `xgboost.fit((SELECT ...), model_name := 'm', ...)` | table-buffering → registry |
| **Predict** | `xgboost.predict((SELECT ...), model_name := 'm')` | streaming table-in-out |
| **Cross-val** | `xgboost.cross_val_predict((SELECT ...), ...)` | table-buffering (no persistence) |
| **Importance** | `xgboost.feature_importance('m')` | table function (reads the registry) |
| **Explain (SHAP)** | `xgboost.explain((SELECT ...), model_name := 'm')` | streaming table-in-out |

**Conventions** for the fit / predict / explain functions:

- The input relation **is** the feature matrix `X`, passed as a `(SELECT ...)`
  subquery. Named arguments use DuckDB's `name := value` (or `=>`) syntax.
- **`id`** names a passthrough column: it is *excluded from the features* and
  copied unchanged onto each output row, so you can join results back to the
  source. It is optional.
- **`target`** (required for `fit` / `cross_val_predict`) names the label column,
  also excluded from features. Classification targets must be integer class
  labels encoded `0..n_classes-1` (XGBoost's requirement); the bundled datasets
  already are.
- **Every remaining column is treated as a numeric feature.** Non-numeric
  columns raise a clear error — `SELECT` only the columns you want as features.
- Hyperparameters are passed as a JSON string: `params := '{"n_estimators": 300, "max_depth": 6}'`.
  Unknown hyperparameters are rejected with the list of valid ones for that estimator.
- **`fit`/`predict` align features by name**, not position: `predict` selects the
  model's fitted feature columns by name (input order is irrelevant, extra
  columns are ignored) and errors if a required feature column is missing.

## Function catalog

### Datasets (`xgboost.<name>()`)
Bundled (via scikit-learn) so demos and tests are self-contained: `iris`, `wine`,
`breast_cancer` (classification), `diabetes`, `california_housing` (regression),
and generators `make_classification`, `make_regression`.

```sql
SELECT target_name, avg(petal_length_cm) FROM xgboost.iris() GROUP BY target_name;
SELECT * FROM xgboost.make_classification(n_samples := 500, n_features := 8, n_classes := 3);
```

### Models (registry-backed)
`fit`, `predict`, `cross_val_predict`, `list_models`, `model_info`, `drop_model`.

Estimators: `xgb_classifier`, `xgb_regressor`, `xgb_rf_classifier`,
`xgb_rf_regressor` (the random-forest-flavoured boosters).

```sql
-- train + persist
SELECT * FROM xgboost.fit(
  (SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target FROM xgboost.iris()),
  model_name := 'iris_clf', estimator := 'xgb_classifier', target := 'target', id := 'sample_id',
  params := '{"n_estimators": 200, "max_depth": 4}');

-- predict later (optionally with per-class probabilities)
SELECT * FROM xgboost.predict((SELECT * FROM new_flowers), model_name := 'iris_clf', id := 'id', with_proba := true);

-- evaluate without persisting
SELECT count(*) FROM xgboost.cross_val_predict(
  (SELECT * FROM iris_xy), estimator := 'xgb_classifier', target := 'target', id := 'sample_id', cv := 5);

SELECT * FROM xgboost.list_models();
SELECT * FROM xgboost.drop_model('iris_clf');
```

### Interpretation (XGBoost-specific)
`feature_importance` and `explain`.

```sql
-- ranked per-feature importance for a stored model (weight/gain/cover/total_*)
SELECT * FROM xgboost.feature_importance('iris_clf', importance_type := 'gain');

-- SHAP contributions: base_value + one contrib_<feature> column per feature, per row.
-- base_value + sum(contrib_*) == the model's raw-margin prediction.
-- Supported for regression and binary classification.
SELECT * FROM xgboost.explain((SELECT * FROM xgboost.diabetes()), model_name := 'diab_reg', id := 'sample_id');
```

## Model registry storage

Fitted models are pickled (joblib) plus a JSON metadata sidecar. The store is
chosen behind the `ModelStore` interface in `vgi_xgboost/registry.py`:

- **Local disk** (default): `XGBOOST_MODELS_DIR` (default `./models`).
- **S3 / Cloudflare R2**: not yet implemented — `get_store()` is the single seam
  where an `S3Store` drops in.

On Fly.io the local store is backed by a mounted volume (see `fly.toml`) so models
survive machine restarts. `predict` records the XGBoost version used to fit and
logs a warning (visible in `duckdb_logs()`) if the worker's version differs.

## Local development

```sh
make venv          # create .venv with vgi + xgboost + scikit-learn (from PyPI)
make lint          # ruff + mypy
make pytest        # unit tests
make test-sql      # SQL tests in-process via haybarn (no custom DuckDB build needed)
make test-stdio    # SQL tests with the worker as a subprocess (custom unittest runner)
make test-http     # SQL tests against a local HTTP server
```

The `test/sql/*.test` files are the integration suite. `test-stdio`/`test-http`
run them with DuckDB's `unittest` runner built with the VGI extension
(`VGI_BUILD_DIR`) and are the local authority. `test-sql` replays the **same**
files in-process against the `haybarn` DuckDB distribution (which can
`INSTALL vgi FROM community`), so they also run on a stock CI runner.

### Continuous integration
`.github/workflows/ci.yml` runs ruff, mypy, the unit tests, the haybarn SQL
suite, and a Docker build + `/health` smoke test on every push and PR.
Dependabot (`.github/dependabot.yml`) keeps the Python deps, GitHub Actions, and
the Docker base image up to date weekly.

## Deployment (Fly.io)

`vgi-python` / `vgi-rpc` are on PyPI, so the Docker image installs everything
directly — no vendoring step.

```sh
make deploy        # build, smoke-test, push, and deploy
fly volumes create xgboost_models --size 1 --region iad   # one-time, for the registry
```

`serve.py` runs the worker over HTTP; attach the deployed endpoint with
`ATTACH 'xgboost' (TYPE vgi, LOCATION 'https://<app>.fly.dev');`.

## Layout

```
xgboost_worker.py      entry point; assembles the `xgboost` catalog
serve.py               HTTP entry point (Fly.io)
vgi_xgboost/
  datasets.py          dataset table functions (bundled via scikit-learn)
  models.py            fit / predict / cross_val_predict / registry mgmt
  importance.py        feature_importance + SHAP explain
  registry.py          ModelStore (local disk; S3/R2 seam)
  buffering.py         shared sink/combine/matrix helpers
  schema_utils.py      Arrow schema helpers
```
