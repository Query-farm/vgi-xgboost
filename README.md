<p align="center">
  <img src="https://raw.githubusercontent.com/Query-farm/vgi-xgboost/main/assets/vgi-logo.png" alt="Vector Gateway Interface" height="104">
</p>

# vgi-xgboost

[![CI](https://github.com/Query-farm/vgi-xgboost/actions/workflows/ci.yml/badge.svg)](https://github.com/Query-farm/vgi-xgboost/actions/workflows/ci.yml)

A [VGI](https://github.com/query-farm/vgi-python) worker that brings
[XGBoost](https://xgboost.readthedocs.io/) into DuckDB/SQL: train gradient-boosted
models, persist them in a registry, predict over SQL tables, and interpret them
(feature importance + SHAP contributions) — all as SQL functions.

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'xgboost' (TYPE vgi, LOCATION 'uv run xgboost_worker.py');

-- train + persist a model (every non-id/target column becomes a feature)
SELECT * FROM xgboost.fit(
  (SELECT * EXCLUDE (target_name) FROM xgboost.iris()),
  model_name := 'iris_clf', estimator := 'xgb_classifier', target := 'target', id := 'sample_id');

-- predict later (input needs the same feature columns; selected by name)
SELECT * FROM xgboost.predict((SELECT * FROM new_flowers), model_name := 'iris_clf', id := 'id');
```

## How it maps XGBoost onto SQL

XGBoost is built around stateful *fit / predict* estimators; SQL is set-oriented.
Each piece is mapped to the VGI primitive that fits its data flow:

| Area | SQL surface | VGI primitive |
| --- | --- | --- |
| **Datasets** | `SELECT * FROM xgboost.iris()` | table function (source) |
| **Fit** | `xgboost.fit((SELECT ...), model_name := 'm', ...)` | table-buffering → registry + BLOB |
| **Typed fit** | `xgboost.fit_xgb_classifier((SELECT ...), n_estimators := 300, ...)` | table-buffering → registry + BLOB |
| **Predict** | `xgboost.predict((SELECT ...), model_name := 'm')` | streaming table-in-out |
| **Cross-val** | `xgboost.cross_val_predict` / `cross_val_score((SELECT ...), ...)` | table-buffering (no persistence) |
| **Tuning** | `xgboost.grid_search` / `randomized_search((SELECT ...), grid := '...')` | table-buffering (CV leaderboard) |
| **Importance** | `xgboost.feature_importance('m')` / `permutation_importance(...)` | table function / buffering |
| **Explain (SHAP)** | `xgboost.explain` / `shap_values((SELECT ...), model_name := 'm')` | streaming table-in-out |

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
- **Every remaining column is a feature.** Numeric columns are used directly;
  **string columns become native categorical features** (no one-hot needed —
  this is XGBoost's edge), and **NULLs flow through as missing values**. Only
  genuinely unsupported types (e.g. blobs, structs) raise a clear error.
- Hyperparameters can be passed as a JSON string: `params := '{"n_estimators": 300, "max_depth": 6}'`,
  or — better — as **native typed arguments** via the `fit_<estimator>` functions
  (see below). Unknown hyperparameters are rejected with the list of valid ones.
- **`fit`/`predict` align features by name**, not position: `predict` selects the
  model's fitted feature columns by name (input order is irrelevant, extra
  columns are ignored) and errors if a required feature column is missing. Unseen
  categories at predict time are treated as missing rather than raising.
- **`fit` always returns the model as a `model` BLOB** (estimator + metadata,
  serialized with XGBoost's native format). It also persists to the registry
  *only if* `model_name` is given (so `model_name` is optional). `predict`,
  `explain`, `shap_values`, and `permutation_importance` take **either**
  `model_name :=` or `model :=` (a BLOB).

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
`fit`, `predict`, `cross_val_predict`, `cross_val_score`, `list_models`,
`model_info`, `drop_model`.

Estimators: `xgb_classifier`, `xgb_regressor`, `xgb_rf_classifier`,
`xgb_rf_regressor` (the random-forest-flavoured boosters).

```sql
-- train + persist (params as JSON)
SELECT * FROM xgboost.fit(
  (SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target FROM xgboost.iris()),
  model_name := 'iris_clf', estimator := 'xgb_classifier', target := 'target', id := 'sample_id',
  params := '{"n_estimators": 200, "max_depth": 4}');

-- predict later (optionally with per-class probabilities)
SELECT * FROM xgboost.predict((SELECT * FROM new_flowers), model_name := 'iris_clf', id := 'id', with_proba := true);

-- predict output modes: raw margin or per-tree leaf indices
SELECT * FROM xgboost.predict((SELECT * FROM new_flowers), model_name := 'iris_clf', id := 'id', output_margin := true);
SELECT * FROM xgboost.predict((SELECT * FROM new_flowers), model_name := 'iris_clf', id := 'id', pred_leaf := true);

-- evaluate without persisting: out-of-fold predictions, or per-fold held-out scores
SELECT count(*) FROM xgboost.cross_val_predict(
  (SELECT * FROM iris_xy), estimator := 'xgb_classifier', target := 'target', id := 'sample_id', cv := 5);
SELECT fold, score FROM xgboost.cross_val_score(
  (SELECT * FROM iris_xy), estimator := 'xgb_classifier', target := 'target', cv := 5);

SELECT * FROM xgboost.list_models();
SELECT * FROM xgboost.drop_model('iris_clf');
```

### Typed fit functions (`fit_<estimator>`)
Each estimator also has a `fit_<estimator>` form that exposes XGBoost's most-tuned
hyperparameters as **native typed SQL arguments** — discoverable in autocomplete,
type-checked, no JSON: `n_estimators`, `max_depth`, `learning_rate`, `subsample`,
`colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`,
`objective`, `booster` (`gbtree`/`gblinear`/`dart`), `tree_method`, `random_state`.

```sql
SELECT * FROM xgboost.fit_xgb_classifier(
  (SELECT * FROM iris_xy), model_name := 'iris_clf', target := 'target', id := 'sample_id',
  n_estimators := 300, max_depth := 6, learning_rate := 0.1);
```

### Hyperparameter search
`grid_search` and `randomized_search` run cross-validated tuning and return the
leaderboard (one row per combination) with the refit best model BLOB on the best
row. The grid is a JSON object; grab the best model with `WHERE model IS NOT NULL`.

```sql
SELECT params, mean_test_score, rank
FROM xgboost.grid_search((SELECT * FROM iris_xy),
  estimator := 'xgb_classifier', target := 'target', id := 'sample_id',
  grid := '{"n_estimators": [100, 300], "max_depth": [3, 5, 8]}')
ORDER BY rank;

SELECT params, mean_test_score FROM xgboost.randomized_search((SELECT * FROM iris_xy),
  estimator := 'xgb_classifier', target := 'target', n_iter := 10,
  grid := '{"learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5, 8]}');
```

### Interpretation (XGBoost-specific)
`feature_importance`, `explain`, `shap_values`, and `permutation_importance`.

```sql
-- ranked per-feature importance for a stored model (weight/gain/cover/total_*)
SELECT * FROM xgboost.feature_importance('iris_clf', importance_type := 'gain');

-- SHAP contributions, wide: base_value + one contrib_<feature> column per row.
-- base_value + sum(contrib_*) == the model's raw-margin prediction (regression / binary).
SELECT * FROM xgboost.explain((SELECT * FROM xgboost.diabetes()), model_name := 'diab_reg', id := 'sample_id');

-- SHAP contributions, long: one row per (input row, feature) — easy to pivot/aggregate.
SELECT * FROM xgboost.shap_values((SELECT * FROM xgboost.diabetes()), model_name := 'diab_reg', id := 'sample_id');

-- model-agnostic importance: the drop in score when each feature is shuffled.
SELECT * FROM xgboost.permutation_importance((SELECT * FROM xgboost.diabetes()),
  model_name := 'diab_reg', target := 'target') ORDER BY importance_mean DESC;
```

### Native categorical + missing values
String columns are used as categorical features directly (no one-hot), and NULLs
are XGBoost's native missing value — both at fit and predict, and unseen
categories at predict map to missing rather than erroring:

```sql
SELECT * FROM xgboost.fit((SELECT id, color, score, churned FROM customers),
  model_name := 'churn', estimator := 'xgb_classifier', target := 'churned', id := 'id');
```

## Model registry storage

Fitted models are serialized with XGBoost's **native `save_model`** (UBJSON, not
pickle) plus a JSON metadata sidecar — so they are forward-compatible across
library upgrades and load without arbitrary code execution. The same packing
flows through SQL as a self-contained `model` BLOB. The store is chosen behind the
`ModelStore` interface in `vgi_xgboost/registry.py`:

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
  models.py            fit / predict / cross_val_predict / cross_val_score / registry mgmt
  typed_models.py      generated fit_<estimator> functions with typed hyperparams
  search.py            grid_search / randomized_search (JSON grid)
  features.py          native categorical + missing-value feature assembly
  importance.py        feature_importance + SHAP explain/shap_values + permutation_importance
  registry.py          ModelStore (local disk; S3/R2 seam) + native serialization + model BLOB
  buffering.py         shared sink/combine helpers
  schema_utils.py      Arrow schema helpers
```
