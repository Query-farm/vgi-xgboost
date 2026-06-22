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
**model registry** (fit/predict/cross_val_predict) plus XGBoost-specific
interpretation (feature importance, SHAP). It deliberately does *not* mirror
vgi-sklearn's metrics/transforms surface — those are scikit-learn's, and
vgi-sklearn already exposes them. A small datasets module (reusing scikit-learn's
bundled data) is kept only so demos and the SQL tests are self-contained.

## Layout

```
xgboost_worker.py     entry point: builds the `xgboost` Catalog, XGBoostWorker, main()
serve.py              HTTP entry point (injects --http into Worker.main())
vgi_xgboost/
  datasets.py         dataset table functions (toy sets + make_* generators)
  models.py           fit / predict / cross_val_predict + registry mgmt
  importance.py       feature_importance (from registry) + explain (SHAP pred_contribs)
  registry.py         ModelStore interface + LocalDiskStore (S3/R2 seam)
  buffering.py        shared sink/combine/serialize/matrix helpers
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
| `fit` (needs whole input) | `TableBufferingFunction` via `buffering.SinkBuffer` | `models.FitModel`, `CrossValPredict` |
| Score/explain a stream with an already-fit model | `TableInOutGenerator` | `models.PredictModel`, `importance.ExplainModel` |

Conventions for fit/predict/explain: input relation is X via a `(SELECT ...)`
subquery (Arg(0)); name `target` (features = the rest) and an optional `id`
passthrough; hyperparameters as a JSON-string arg.

## Sharp edges (read before debugging)

1. **Don't name the worker module `xgboost.py`.** It would shadow the real
   `xgboost` package import. The entry point is `xgboost_worker.py`; the package
   is `vgi_xgboost`.
2. **XGBoost classification labels must be `0..n_classes-1`.** Unlike some
   sklearn estimators, `XGBClassifier.fit` errors on arbitrary integer labels.
   `_xy` rounds the target to int but does *not* re-encode; the bundled datasets
   are already 0-based. If you add a dataset with non-contiguous labels, encode
   it first.
3. **`explain` (SHAP) is regression / binary only.** `booster.predict(...,
   pred_contribs=True)` returns a 2D `(n_rows, n_features+1)` array for those;
   multiclass returns a 3D array. `ExplainModel.on_bind` rejects multiclass with
   a clear error. The **last** column of the contribs array is the base value.
4. **Feature names in the booster are `f0..fN`** when fit on a numpy matrix (we
   always do). `feature_importance` maps `f{i}` back to the stored
   `feature_names[i]`; features never used in a split are absent from
   `get_score()` and reported as importance `0`.
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
(joblib pickle + JSON metadata, root from `XGBOOST_MODELS_DIR`, default
`./models`) is the only impl today; an `S3Store` for S3/R2 drops in here without
touching `models.py`. `predict` warns via `duckdb_logs()` if the worker's XGBoost
version differs from the one a model was fitted with.
