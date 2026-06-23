# CLAUDE.md — vgi-xgboost

Contributor/agent notes for this repo. User-facing docs live in `README.md`;
this file is the "how it's built and where the sharp edges are" companion.

## What this is

A [VGI](https://github.com/query-farm/vgi-python) worker exposing XGBoost to
DuckDB/SQL. `xgboost_worker.py` assembles every function into one `xgboost`
catalog (single `main` schema) and runs it over stdio (local) or HTTP (Fly.io).
Depends on the published `vgi-python` / `vgi-rpc` from PyPI; modeled on the
sibling `~/Development/vgi-scikit-learn` worker (which vendored local checkouts —
that complexity is gone here now those packages are on PyPI).

XGBoost's value is gradient-boosted train/predict, so this worker is focused on a
**model registry** (fit/predict/cross_val_predict/cross_val_score), **typed
fit + hyperparameter search**, and XGBoost-specific interpretation (feature
importance, SHAP, permutation importance). It deliberately does *not* mirror
vgi-sklearn's metrics/transforms surface — those are scikit-learn's, and
vgi-sklearn already exposes them. A small datasets module (reusing scikit-learn's
bundled data) is kept only so demos and the SQL tests are self-contained.

## Layout

```
xgboost_worker.py     entry point: builds the `xgboost` Catalog, XGBoostWorker, main()
serve.py              HTTP entry point (injects --http into Worker.main())
vgi_xgboost/
  datasets.py         dataset table functions (toy sets + make_* generators)
  models.py           fit / predict / cross_val_predict / cross_val_score + registry mgmt
  typed_models.py     generated fit_<estimator> functions with typed hyperparams
  search.py           grid_search / randomized_search (single-estimator JSON grid)
  features.py         native categorical + missing-value feature assembly (pandas)
  importance.py       feature_importance + explain (wide SHAP) + shap_values (long) + permutation_importance
  registry.py         ModelStore + LocalDiskStore (S3/R2 seam) + native serialization + model-BLOB pack/unpack
  buffering.py        shared sink/combine/serialize helpers
  schema_utils.py     pa.Field comment helper, name sanitisation, NoArgs
tests/                pytest (in-process harness in tests/harness.py)
test/sql/*.test       DuckDB sqllogictest — the authoritative integration tests
```

To add functions: implement in the relevant `vgi_xgboost/*.py`, export a
`*_FUNCTIONS` list, and splice it into `_FUNCTIONS` in `xgboost_worker.py`.

## Which VGI primitive for which job

| Need | Primitive | Example here |
| --- | --- | --- |
| Emit rows, no input | `TableFunctionGenerator` (`@bind_fixed_schema` / `@init_single_worker`, or custom `on_bind` for schema-from-args) | `datasets.py`, `importance.FeatureImportance` |
| `fit` (needs whole input) | `TableBufferingFunction` via `buffering.SinkBuffer` | `models.FitModel`, `CrossValPredict`, `CrossValScore`, typed `fit_<estimator>`, `search.GridSearch` |
| Score/explain a stream with an already-fit model | `TableInOutGenerator` | `models.PredictModel`, `importance.ExplainModel`, `ShapValues` |

Conventions for fit/predict/explain: input relation is X via a `(SELECT ...)`
subquery (Arg(0)); name `target` (features = the rest) and an optional `id`
passthrough; hyperparameters as a JSON-string arg (generic `fit`/`search`) or
typed named args (`fit_<estimator>`).

## Models: registry + BLOB + typed functions + search

- **fit always returns a `model` BLOB** (estimator + metadata packed by
  `registry.pack_model`) and persists to the registry only if `model_name` is
  given (so `model_name` is optional). `predict` / `explain` / `shap_values` /
  `permutation_importance` take **either** `model_name :=` or `model :=` (a BLOB);
  `unpack_meta` reads metadata at bind, `unpack_model` loads the estimator at
  process. Pass a BLOB into a table function via `SET VARIABLE`+`getvariable()`
  (a table function gets only one subquery slot — the table input).
- **Serialization is XGBoost-native, not pickle** (`registry._native_dumps/_loads`
  via `Booster.save_model`/`load_model` to a temp `.ubj`). UBJSON is XGBoost's own
  forward-compatible format and loads without arbitrary code execution. The
  estimator *class* is recorded in metadata (`_ESTIMATOR_CLASSES`) so the right
  scikit-learn wrapper is rebuilt on load.
- **Typed `fit_<estimator>` functions** are generated in `typed_models.py` from
  the `_HPARAMS` spec via `types.new_class(name, (SinkBuffer[args, DrainState],),
  ...)` — plain `type()` can't resolve the subscripted-generic base. Each shares
  `models._fit_and_emit`. Numeric knobs use a sentinel default (`n_estimators := 0`,
  `gamma := -1.0`, etc.) that is *dropped* so the hyperparameter stays at XGBoost's
  own default — an omitted arg is a true no-op. `test_every_typed_param_is_valid`
  guards that every exposed param is real for its estimator.
- **`search.grid_search` / `randomized_search`** wrap sklearn's
  `GridSearchCV`/`RandomizedSearchCV`. The grid is a **JSON object** arg
  (`grid := '{"n_estimators": [50, 100]}'`), *not* vgi-sklearn's discriminated-
  union `union_value` surface: the released PyPI vgi-python (0.8.2) does **not**
  export `TaggedUnion` / preserve union tags, so the union approach is unavailable
  here. Returns the CV leaderboard (one row per combo) with the refit best model
  BLOB on the best row — grab it with `WHERE model IS NOT NULL`. When a vgi-python
  with union-tag decoding is pinned, the union form from vgi-sklearn's `search.py`
  could replace this.
- **Native categorical + missing values (`features.py`).** Feature assembly builds
  a **pandas DataFrame** (pandas is a hard dep) so XGBoost's `enable_categorical`
  + `tree_method='hist'` (both in `_COMMON_DEFAULTS`) handle string columns
  natively (no one-hot) and NaN as missing. `categorical_mask` flags string/dict
  Arrow columns; `build_x_fit` captures each categorical column's category list
  into `ModelMetadata.categories`; `build_x_predict` reuses those categories so
  **unseen categories at predict map to NaN/missing instead of raising** (XGBoost
  3.x has strict category encoding by default). Decimal columns are cast to float.
- **predict output modes:** default label/value; `with_proba` (per-class probs);
  `output_margin` (raw margin scalar — collapsed to max for multiclass); `pred_leaf`
  (one leaf index per tree as a `list<int>` column — uses the Booster, not the
  sklearn `predict()`, which rejects `pred_leaf`). The modes are mutually
  exclusive (validated at bind).
- **SHAP:** `explain` is wide (`base_value` + one `contrib_<feature>` per row);
  `shap_values` is long (`(id, feature, shap_value, base_value)`). Both use
  `booster.predict(DMatrix(x, enable_categorical=True), pred_contribs=True)` and
  are regression / binary only (multiclass contribs are 3D — rejected at bind).

## Sharp edges (read before debugging)

1. **Don't name the worker module `xgboost.py`.** It would shadow the real
   `xgboost` package import. The entry point is `xgboost_worker.py`; the package
   is `vgi_xgboost`.
2. **XGBoost classification labels must be `0..n_classes-1`.** Unlike some
   sklearn estimators, `XGBClassifier.fit` errors on arbitrary integer labels.
   `_xy` rounds the target to int but does *not* re-encode; the bundled datasets
   are already 0-based. If you add a dataset with non-contiguous labels, encode
   it first.
3. **`explain` / `shap_values` are regression / binary only.** `booster.predict(...,
   pred_contribs=True)` returns a 2D `(n_rows, n_features+1)` array for those;
   multiclass returns a 3D array. `on_bind` rejects multiclass with a clear error.
   The **last** column of the contribs array is the base value.
4. **`_xy` now returns a pandas DataFrame** (was a numpy matrix), so the booster
   sees the real feature names and categorical dtypes. `feature_importance` still
   maps `f{i}` → `feature_names[i]` defensively, but `get_score()` may also key by
   name; features never used in a split are absent and reported as importance `0`.
   When fitting on the DataFrame XGBoost needs `enable_categorical=True` +
   `tree_method='hist'` (both in `_COMMON_DEFAULTS`) or string columns error.
4a. **Unseen categories at predict raise in XGBoost 3.x** unless you reuse the
   training category set — `build_x_predict` does this (maps unseen → NaN). A bare
   `pd.Categorical(values, categories=...)` triggers a pandas-4 deprecation
   warning for out-of-dtype entries; filter unseen values to `None` *before*
   `.astype(CategoricalDtype(...))` (as `features.py` does).
5. **`pa.Float64Array` does not exist** — the class is `pa.DoubleArray`. A bad
   `Param` type hint does NOT error; the framework warns and registers the
   function with **zero input columns**. Watch for `UserWarning: ... type hints
   could not be resolved`.
6. **Table argument syntax is `(SELECT ...)`, not `TABLE(...)`.**
7. **`Arg(0)` = positional, `Arg("name")` = named-only.** The table input is
   always `Arg(0)`.
8. **Buffering / in-out state classes must extend `ArrowSerializableDataclass`**
   (e.g. `buffering.DrainState`).
9. **Output schema is fixed at bind.** Fine here: predict/explain widths come
   from the stored model's metadata (known at bind via `load_meta`).
10. **HTTP entry point:** current vgi-python has **no `main_http`**. Serve HTTP
    via `Worker.main()` with `--http`; `serve.py` injects that flag.
11. **Distribution vs import name:** the distribution is `vgi-python` but the
    import is `vgi` (and `vgi-rpc` imports as `vgi_rpc`). Deps everywhere
    (PEP 723 headers, `pyproject.toml`, Dockerfile, `make venv`) name the
    distributions `vgi-python` / `vgi-rpc`.

## Testing

```sh
make venv          # .venv with vgi + xgboost + scikit-learn (from PyPI) + ruff/mypy
make lint          # ruff + mypy (config in pyproject.toml; both run clean)
make pytest        # in-process unit tests (fast; uses tests/harness.py)
make test-stdio    # SQL tests, worker as subprocess  (authoritative)
make test-http     # SQL tests against a local HTTP server
```

- **SQL tests are authoritative.** Unit tests call classmethods directly and can
  pass while the real RPC path is broken. Always run `test-stdio`.
- The same `test/sql/*.test` files run over **three transports**: stdio and HTTP
  (via DuckDB's `unittest` runner at `$(VGI_BUILD_DIR)/test/unittest`, the local
  authority) and **in-process via haybarn** (`make test-sql`). The haybarn path
  is what CI uses — it `INSTALL vgi FROM community` on Query Farm's DuckDB build,
  so it needs no custom binary. `tests/sqllogic.py` is a small subset
  sqllogictest runner; if you use a directive it doesn't support, extend it.
- `make test-stdio` / `test-http` point `XGBOOST_MODELS_DIR` at an isolated
  `.test-models/` so the registry tests don't pollute `./models`.
- **CI:** `.github/workflows/ci.yml` runs ruff + mypy + unit + haybarn SQL +
  Docker smoke. Dependabot watches pip / actions / docker. Keep all five CI
  steps green; the haybarn SQL step needs network (community extension fetch).

## Deployment (Fly.io)

The Docker image `pip install`s `vgi-python` / `vgi-rpc` straight from PyPI — no
vendoring or local wheel build.

```sh
make deploy        # build (linux/amd64) -> smoke-test -> push -> fly deploy
fly volumes create xgboost_models --size 1 --region iad   # one-time, registry
```

`fly.toml` bumps VM memory to 1gb (xgboost/scipy are heavy) and mounts a volume
at `/data` for the model registry (`XGBOOST_MODELS_DIR=/data/models`). The Docker
smoke test verifies imports + `/health`.

## Model registry

`registry.get_store()` is the single seam selecting the backend. `LocalDiskStore`
(`<name>.ubj` native XGBoost artifact + `<name>.json` metadata sidecar, root from
`XGBOOST_MODELS_DIR`, default `./models`) is the only impl today; an `S3Store` for
S3/R2 drops in here without touching `models.py`. Serialization is XGBoost-native
(`save_model`/`load_model`), not pickle — see the registry note above. `predict`
warns via `duckdb_logs()` if the worker's XGBoost version differs from the one a
model was fitted with. The same pack/unpack also produces the self-contained
`model` BLOB that flows through SQL.

## Dependencies

PyPI deps live in **four** places that must stay in sync when adding one:
`pyproject.toml`, the PEP 723 header in `xgboost_worker.py`, `make venv` in the
`Makefile`, and the Dockerfile `pip install` line. **pandas** is a hard dep
(needed for native categorical DataFrames in `features.py`); joblib was dropped
when serialization moved to XGBoost-native.
