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
fit + discriminated-union hyperparameter search**, and interpretation
(feature_importance, SHAP `explain`, permutation_importance, partial_dependence).
It deliberately does *not* mirror
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
  search.py           grid_search / randomized_search (discriminated-union estimator arg via _GRID_UNION/_HPARAMS)
  features.py         native categorical + missing-value feature assembly (pandas)
  importance.py       feature_importance + explain (long-format SHAP) + permutation_importance + partial_dependence
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
| `fit` / buffer whole input | `TableBufferingFunction` via `buffering.SinkBuffer` | `models.FitModel`, `CrossValPredict`, `CrossValScore`, typed `fit_<estimator>`, `search.GridSearch`/`RandomizedSearch`, `importance.PermutationImportance`/`PartialDependence` |
| Score/explain a stream with an already-fit model | `TableInOutGenerator` | `models.PredictModel`, `importance.ExplainModel` |

Conventions for fit/predict/explain: input relation is X via a `(SELECT ...)`
subquery (Arg(0)); name `target` (features = the rest) and an optional `id`
passthrough; hyperparameters as a JSON-string arg on the generic `fit`, or typed
named args on `fit_<estimator>`. `grid_search`/`randomized_search` instead take a
typed **discriminated-union** estimator arg (see below), not JSON.

## Models: registry + BLOB + typed functions + search

- **fit always returns a `model` BLOB** (estimator + metadata packed by
  `registry.pack_model`) and persists to the registry only if `model_name` is
  given (so `model_name` is optional). `predict` / `explain` / `feature_importance`
  / `permutation_importance` / `partial_dependence` take **either** `model_name :=`
  or `model :=` (a BLOB); `unpack_meta` reads metadata at bind, `unpack_model`
  loads the estimator at process. Pass a BLOB into a table function via
  `SET VARIABLE`+`getvariable()` (a table function gets only one subquery slot —
  the table input).
- **Classification labels can be any dtype** (string, int, bool). `models._target_array`
  builds a stable sorted `classes` list and label-encodes the target to `0..n-1`
  codes for XGBoost (which requires contiguous int labels); `ModelMetadata.classes`
  stores the *original* labels. `predict` decodes `code → classes[code]` and types
  the `prediction` column from the label dtype (VARCHAR for strings, BIGINT for
  ints); `with_proba` emits `proba_<original_label>` columns. `permutation_importance`
  re-encodes string targets through `meta.classes`.
- **Serialization is XGBoost-native, not pickle** (`registry._native_dumps/_loads`
  via `Booster.save_model`/`load_model` to a temp `.ubj`). UBJSON is XGBoost's own
  forward-compatible format and loads without arbitrary code execution. The
  estimator *class* is recorded in metadata (`_ESTIMATOR_CLASSES`) so the right
  scikit-learn wrapper is rebuilt on load.
- **Typed `fit_<estimator>` functions** are generated in `typed_models.py` from
  the `_HPARAMS` spec via `types.new_class(name, (SinkBuffer[args, DrainState],),
  ...)` — plain `type()` can't resolve the subscripted-generic base. Each shares
  `models._fit_and_emit`. Typed knobs carry XGBoost's **real documented defaults**
  (`n_estimators=100, max_depth=6, learning_rate=0.3, …`) — the catalog shows the
  true default and every value is forwarded; the only sentinel left is `none_if=""`
  on `objective`/`booster` (empty string → keep the task default). A param-validity
  test guards that every exposed param is real for its estimator.
- **`search.grid_search` / `randomized_search` are a discriminated union** (same
  design as vgi-sklearn's `search.py`). The `estimator` arg is a sparse Arrow union
  `_GRID_UNION` (one member per estimator from `_HPARAMS`, each field a
  `list<scalar>`); SQL calls it `union_value(<estimator> := {param: [values]})`.
  The worker reads it as a `vgi.TaggedUnion` (`.tag` = estimator, `.value` = grid
  dict); `_param_grid` translates a member into a sklearn grid (omitted/NULL params
  stay at their default). Returns the CV leaderboard (one row per combo) with the
  refit best model BLOB on the best row — grab it with `WHERE model IS NOT NULL`.
  `randomized_search` adds `n_iter` (capped at the grid size) + `random_state`.
  **Requires `vgi-python >=0.8.3`** (ships `vgi.TaggedUnion` / union-tag-preserving
  decode) — already the pin. Dense unions are unsupported by the C++ extension;
  `union_value` produces sparse, which works.
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
- **SHAP:** a single `explain` in **long format** — optional `<id>`, optional
  `class` (only for multiclass), `feature`, `shap_value`, `base_value`, one row per
  (row, [class], feature). Uses `booster.predict(DMatrix(x,
  enable_categorical=True), pred_contribs=True)`; for multiclass the 3-D contribs
  array is unrolled into one row per (row, class, feature). (There is no separate
  wide `shap_values` function — it was removed.)
- **`feature_importance` and `permutation_importance` are ranked** (a `rank`
  int32 column, sorted by importance desc) so server-side output is already
  ordered. `partial_dependence` (`importance.py`, buffering) shows how a feature
  moves the prediction over a grid: numeric-feature-only (categorical → clear
  error), output `(feature_value, class, partial_dependence)` with one curve per
  class for multiclass / NULL `class` for regression+binary.
- **Schema consistency:** `_FIT_SCHEMA` and `_MODEL_INFO_SCHEMA` agree —
  `n_samples`/`n_features`/`n_classes` are `int64` and `task` is a plain `string`,
  so `fit` output joins cleanly to `model_info` (the `rank` columns are
  intentionally `int32`).

## Sharp edges (read before debugging)

1. **Don't name the worker module `xgboost.py`.** It would shadow the real
   `xgboost` package import. The entry point is `xgboost_worker.py`; the package
   is `vgi_xgboost`.
2. **Labels are label-encoded internally; you no longer pre-encode.** `XGBClassifier.fit`
   requires contiguous `0..n-1` int labels, so `models._target_array` builds a
   stable sorted `classes` and maps the target to codes; `predict` decodes back via
   `_decode_labels`. So string/non-contiguous labels just work — don't add a manual
   re-encoding step (it would double-encode). The prediction column dtype follows
   the original label dtype.
3. **`explain` supports multiclass.** `booster.predict(..., pred_contribs=True)`
   returns a 2-D `(n_rows, n_features+1)` array for regression/binary and a 3-D
   array for multiclass; `explain` unrolls both into long rows (a `class` column
   appears for multiclass). The **last** contribs column is the base value. (No
   multiclass rejection, and no wide `shap_values` — both were removed.)
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
