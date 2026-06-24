"""Hyperparameter search exposed as a single discriminated-union SQL function.

``xgboost.grid_search`` / ``xgboost.randomized_search`` run scikit-learn's
``GridSearchCV`` / ``RandomizedSearchCV`` over a training table and return the
cross-validation leaderboard (one row per parameter combination tried) plus the
refit best model as a BLOB on the rank-1 row. The estimator and its search grid
are a single **tagged-union** argument: the union *tag* is the estimator name and
the *value* is a struct of hyperparameter value-lists. Each member therefore only
exposes the hyperparameters that estimator actually has -- a discriminated union,
realized with DuckDB's ``UNION`` type:

    SELECT params, mean_test_score, rank
    FROM xgboost.grid_search(
      (SELECT * FROM xgboost.iris()),
      target := 'target', id := 'sample_id',
      estimator := union_value(xgb_classifier := {
        'n_estimators': [50, 100], 'max_depth': [3, 5]}))
    ORDER BY rank;

Only the hyperparameters you list are searched; the rest stay at the estimator's
defaults. Grab the best model with ``WHERE model IS NOT NULL``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import pyarrow as pa
import xgboost
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from vgi import TaggedUnion
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
from vgi.table_function import BindParams

from .buffering import DrainState, SinkBuffer, input_schema_of
from .models import _ESTIMATORS, _features_excluding, _xy, build_estimator
from .registry import ModelMetadata, get_store, now_iso, pack_model, validate_name
from .schema_utils import columns_md
from .schema_utils import field as sfield
from .typed_models import _HPARAMS, _UNSET

_PYTYPE_TO_ARROW: dict[type, pa.DataType] = {
    int: pa.int64(),
    float: pa.float64(),
    str: pa.string(),
    bool: pa.bool_(),
}


def _member_struct(spec: list) -> pa.DataType:
    """Struct type for one estimator's grid: each hyperparameter as a list of its scalar type."""
    return pa.struct([pa.field(hp.name, pa.list_(_PYTYPE_TO_ARROW[hp.type])) for hp in spec])


# One sparse-union member per estimator, tagged by the estimator name. This is
# the discriminated union surfaced to SQL via union_value(<estimator> := {...}).
_GRID_UNION = pa.sparse_union([pa.field(name, _member_struct(spec)) for name, spec in _HPARAMS.items()])


def _param_grid(tag: str, value: dict[str, Any] | None) -> dict[str, list[Any]]:
    """Translate a union member value (``{param: [values]}``) into a scikit-learn param grid.

    Applies the same per-hyperparameter translation as the typed ``fit_<estimator>``
    functions, element-wise. After dropping the magic sentinels (XGBoost's typed
    defaults are truthful), the only remaining translation is ``none_if`` (e.g.
    ``objective``/``booster`` "" -> omit). Hyperparameters left unset (NULL) are
    omitted, so they stay at the estimator default rather than being searched.
    """
    grid: dict[str, list[Any]] = {}
    for hp in _HPARAMS[tag]:
        vals = (value or {}).get(hp.name)
        if vals is None:
            continue
        items = [v for v in vals if not (hp.none_if is not _UNSET and v == hp.none_if)]
        if not items:
            continue
        grid[hp.kwarg or hp.name] = items
    return grid


def _grid_size(space: dict[str, Any]) -> int:
    """Total number of combinations in a (list-valued) parameter grid."""
    total = 1
    for values in space.values():
        total *= max(1, len(values))
    return total


_SEARCH_SCHEMA = pa.schema(
    [
        sfield("estimator", pa.string(), "Estimator that was searched.", nullable=False),
        sfield("params", pa.string(), "This combination's hyperparameters (JSON).", nullable=False),
        sfield("mean_test_score", pa.float64(), "Mean cross-validated score.", nullable=False),
        sfield("std_test_score", pa.float64(), "Std-dev of the cross-validated score.", nullable=False),
        sfield("rank", pa.int64(), "Rank by mean score (1 = best).", nullable=False),
        sfield("model", pa.binary(), "The refit best model as a BLOB (only on the rank-1 row)."),
    ]
)


def _validate_search_bind(name: str, params: BindParams[Any]) -> BindResponse:
    """Shared bind validation for grid_search / randomized_search."""
    a = params.args
    if not a.target:
        raise ValueError(f"{name} requires 'target' (the label column name, e.g. target := 'label')")
    tag = getattr(a.estimator, "tag", None)
    if tag is not None and tag not in _ESTIMATORS:
        raise ValueError(f"unknown estimator {tag!r}; choose one of: {', '.join(sorted(_ESTIMATORS))}")
    if a.model_name:
        validate_name(a.model_name)
    input_schema = params.bind_call.input_schema
    assert input_schema is not None
    if a.target not in input_schema.names:
        raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")
    return BindResponse(output_schema=_SEARCH_SCHEMA)


def _run_search(cls: type, params: Any, state: DrainState, out: OutputCollector, build_search: Any) -> None:
    """Shared finalize; ``build_search(est, space, args)`` makes the CV object."""
    if state.done:
        out.finish()
        return
    state.done = True

    a = params.args
    tag = a.estimator.tag
    if tag not in _ESTIMATORS:
        raise ValueError(f"unknown estimator {tag!r}")
    task, estimator = build_estimator(tag, {})
    grid = _param_grid(tag, a.estimator.value)

    input_schema = input_schema_of(params)
    feats = _features_excluding(input_schema, a.target, a.id)
    table = cls.buffered_table(params, input_schema)  # type: ignore[attr-defined]
    if table is None or table.num_rows == 0:
        raise ValueError(f"{cls.Meta.name} received no training rows")  # type: ignore[attr-defined]

    x, y, cat_mask, categories, classes = _xy(table, feats, a.target, task)
    search = build_search(estimator, grid, a)
    search.fit(x, y)

    results = search.cv_results_
    n = len(results["params"])
    best_idx = int(search.best_index_)

    meta = ModelMetadata(
        name=a.model_name,
        estimator=tag,
        task=task,
        target=a.target,
        feature_names=feats,
        categorical=cat_mask,
        categories=categories,
        params={k: _json_safe(v) for k, v in search.best_params_.items()},
        classes=classes,
        n_samples=int(table.num_rows),
        n_features=len(feats),
        train_score=float(search.best_score_),
        xgboost_version=xgboost.__version__,
        created_at=now_iso(),
    )
    if a.model_name:
        get_store().save(search.best_estimator_, meta)
    best_blob = pack_model(search.best_estimator_, meta)

    out.emit(
        pa.RecordBatch.from_pydict(
            {
                "estimator": [tag] * n,
                "params": [json.dumps({k: _json_safe(v) for k, v in p.items()}) for p in results["params"]],
                "mean_test_score": [float(s) for s in results["mean_test_score"]],
                "std_test_score": [float(s) for s in results["std_test_score"]],
                "rank": [int(r) for r in results["rank_test_score"]],
                "model": [best_blob if i == best_idx else None for i in range(n)],
            },
            schema=params.output_schema,
        )
    )


def _json_safe(v: Any) -> Any:
    if isinstance(v, tuple):
        return list(v)
    return v


@dataclass(slots=True, frozen=True)
class GridSearchArgs:
    data: Annotated[TableInput, Arg(0, doc="Training table (features + target [+ id]).")]
    estimator: Annotated[
        TaggedUnion,
        Arg(
            "estimator",
            arrow_type=_GRID_UNION,
            doc="union_value(<estimator> := {param: [values], ...}); the tag picks the estimator.",
        ),
    ]
    target: Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")]
    cv: Annotated[int, Arg("cv", default=5, doc="Number of cross-validation folds.")]
    scoring: Annotated[str, Arg("scoring", default="", doc="Scorer name (default: the estimator's own scorer).")]
    model_name: Annotated[str, Arg("model_name", default="", doc="Optional registry name for the refit best model.")]


class GridSearch(SinkBuffer[GridSearchArgs, DrainState]):
    FunctionArguments: ClassVar[type] = GridSearchArgs

    class Meta:
        name = "grid_search"
        description = "Cross-validated grid search over an XGBoost estimator's hyperparameters"
        categories = ["models", "supervised", "tuning"]
        tags = {"vgi.columns_md": columns_md(_SEARCH_SCHEMA)}
        examples = [
            FunctionExample(
                sql=(
                    "SELECT params, mean_test_score, rank FROM xgboost.grid_search("
                    "(SELECT * FROM xgboost.iris()), target := 'target', id := 'sample_id', "
                    "estimator := union_value(xgb_classifier := "
                    "{'n_estimators': [50, 100], 'max_depth': [3, 5]})) ORDER BY rank"
                ),
                description="Grid-search an XGBoost classifier on iris",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[GridSearchArgs]) -> BindResponse:
        return _validate_search_bind(cls.Meta.name, params)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[GridSearchArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[GridSearchArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        _run_search(
            cls,
            params,
            state,
            out,
            lambda est, space, a: GridSearchCV(est, space, cv=a.cv, scoring=(a.scoring or None), refit=True),
        )


@dataclass(slots=True, frozen=True)
class RandomizedSearchArgs:
    data: Annotated[TableInput, Arg(0, doc="Training table (features + target [+ id]).")]
    estimator: Annotated[
        TaggedUnion,
        Arg(
            "estimator",
            arrow_type=_GRID_UNION,
            doc="union_value(<estimator> := {param: [values], ...}); the tag picks the estimator.",
        ),
    ]
    target: Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")]
    n_iter: Annotated[int, Arg("n_iter", default=10, doc="Number of random combinations to sample.")]
    cv: Annotated[int, Arg("cv", default=5, doc="Number of cross-validation folds.")]
    scoring: Annotated[str, Arg("scoring", default="", doc="Scorer name (default: the estimator's own scorer).")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed for the sampler.")]
    model_name: Annotated[str, Arg("model_name", default="", doc="Optional registry name for the refit best model.")]


class RandomizedSearch(SinkBuffer[RandomizedSearchArgs, DrainState]):
    FunctionArguments: ClassVar[type] = RandomizedSearchArgs

    class Meta:
        name = "randomized_search"
        description = "Cross-validated randomized search: sample n_iter hyperparameter combinations"
        categories = ["models", "supervised", "tuning"]
        tags = {"vgi.columns_md": columns_md(_SEARCH_SCHEMA)}
        examples = [
            FunctionExample(
                sql=(
                    "SELECT params, mean_test_score, rank FROM xgboost.randomized_search("
                    "(SELECT * FROM xgboost.iris()), target := 'target', id := 'sample_id', n_iter := 4, "
                    "estimator := union_value(xgb_classifier := "
                    "{'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 8]})) ORDER BY rank"
                ),
                description="Randomized-search an XGBoost classifier on iris",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[RandomizedSearchArgs]) -> BindResponse:
        return _validate_search_bind(cls.Meta.name, params)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[RandomizedSearchArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[RandomizedSearchArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        # n_iter can't exceed the number of distinct combinations (the grid is discrete).
        _run_search(
            cls,
            params,
            state,
            out,
            lambda est, space, a: RandomizedSearchCV(
                est,
                space,
                n_iter=min(a.n_iter, _grid_size(space)),
                cv=a.cv,
                scoring=(a.scoring or None),
                random_state=a.random_state,
                refit=True,
            ),
        )


SEARCH_FUNCTIONS: list[type] = [GridSearch, RandomizedSearch]
