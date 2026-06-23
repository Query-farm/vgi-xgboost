"""XGBoost-specific extras that read a stored model.

* ``feature_importance`` -- the booster's per-feature importance (weight / gain /
  cover) for a stored model, one row per feature, ranked.
* ``explain``            -- SHAP-style per-row prediction contributions
  (``pred_contribs``): how much each feature pushed each row's raw margin away
  from the model's base value. Streams a table through the model like
  ``predict``. Supported for regression and binary classification (multiclass
  contributions are 3D and not emitted).

    SELECT * FROM xgboost.feature_importance('iris_clf', importance_type => 'gain');
    SELECT * FROM xgboost.explain((SELECT * FROM new_data), model_name => 'house_reg', id => 'id');
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
import pyarrow as pa
import xgboost
from sklearn.inspection import permutation_importance as sk_permutation_importance
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import OutputCollector as BufferingOutputCollector
from vgi.table_buffering_function import TableBufferingParams
from vgi.table_function import (
    BindParams,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)
from vgi.table_in_out_function import OutputCollector as InOutCollector
from vgi.table_in_out_function import TableInOutGenerator
from vgi_rpc.rpc import OutputCollector

from .buffering import DrainState, SinkBuffer, input_schema_of
from .features import build_x_predict
from .models import CLASSIFICATION
from .registry import ModelMetadata, ModelNotFoundError, get_store, unpack_meta, unpack_model
from .schema_utils import field as sfield

_IMPORTANCE_TYPES = {"weight", "gain", "cover", "total_gain", "total_cover"}


# ===========================================================================
# feature_importance
# ===========================================================================


@dataclass(slots=True, frozen=True)
class FeatureImportanceArgs:
    model_name: Annotated[str, Arg(0, doc="Name of a stored model.")]
    importance_type: Annotated[
        str,
        Arg("importance_type", default="gain", doc="weight, gain, cover, total_gain, or total_cover."),
    ]


_IMPORTANCE_SCHEMA = pa.schema(
    [
        sfield("feature", pa.string(), "Feature column name.", nullable=False),
        sfield("importance", pa.float64(), "Importance score for the chosen importance_type.", nullable=False),
        sfield("rank", pa.int32(), "1-based rank by importance (1 = most important).", nullable=False),
    ]
)


@init_single_worker
@bind_fixed_schema
class FeatureImportance(TableFunctionGenerator[FeatureImportanceArgs]):
    FIXED_SCHEMA: ClassVar[pa.Schema] = _IMPORTANCE_SCHEMA

    class Meta:
        name = "feature_importance"
        description = "Per-feature importance (weight/gain/cover) for a stored model, ranked"
        categories = ["models", "interpretation"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM xgboost.feature_importance('iris_clf', importance_type => 'gain')",
                description="Gain-based feature importance for a stored model",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[FeatureImportanceArgs]) -> BindResponse:
        a = params.args
        if a.importance_type not in _IMPORTANCE_TYPES:
            raise ValueError(
                f"invalid importance_type {a.importance_type!r}; choose one of: {', '.join(sorted(_IMPORTANCE_TYPES))}"
            )
        try:
            get_store().load_meta(a.model_name)
        except ModelNotFoundError as exc:
            raise ValueError(f"model {a.model_name!r} not found in the registry") from exc
        return BindResponse(output_schema=cls.FIXED_SCHEMA)

    @classmethod
    def cardinality(cls, params: BindParams[FeatureImportanceArgs]) -> TableCardinality:
        return TableCardinality(estimate=20, max=100000)

    @classmethod
    def process(cls, params: ProcessParams[FeatureImportanceArgs], state: None, out: OutputCollector) -> None:
        a = params.args
        estimator, meta = get_store().load(a.model_name)
        booster = estimator.get_booster()
        # Models are fit on a numpy matrix, so the booster names features f0..fN
        # in feature order; fall back to the stored name in case names were set.
        scores = booster.get_score(importance_type=a.importance_type)
        rows = [
            (name, float(scores.get(f"f{i}", scores.get(name, 0.0))))
            for i, name in enumerate(meta.feature_names)
        ]
        rows.sort(key=lambda r: r[1], reverse=True)
        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "feature": [name for name, _ in rows],
                    "importance": [imp for _, imp in rows],
                    "rank": [i + 1 for i in range(len(rows))],
                },
                schema=params.output_schema,
            )
        )
        out.finish()


# ===========================================================================
# explain (SHAP prediction contributions)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class ExplainArgs:
    data: Annotated[TableInput, Arg(0, doc="Table to explain (must contain the model's feature columns).")]
    model_name: Annotated[
        str, Arg("model_name", default="", doc="Name of a model in the registry. Provide this OR model.")
    ]
    model: Annotated[
        bytes, Arg("model", default=b"", doc="A model BLOB (as returned by fit). Provide this OR model_name.")
    ]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through.")]


_EXPLAIN_CACHE: dict[bytes, tuple[Any, ModelMetadata]] = {}


def _contrib_name(feature: str) -> str:
    return f"contrib_{feature}"


def _explain_meta(a: Any) -> ModelMetadata:
    if a.model_name:
        try:
            return get_store().load_meta(a.model_name)
        except ModelNotFoundError as exc:
            raise ValueError(f"model {a.model_name!r} not found in the registry") from exc
    return unpack_meta(a.model)


def _explain_model(cache: dict[bytes, tuple[Any, ModelMetadata]], params: Any) -> tuple[Any, ModelMetadata]:
    assert params.init_response is not None
    key = params.init_response.execution_id
    cached = cache.get(key)
    if cached is None:
        a = params.args
        cached = get_store().load(a.model_name) if a.model_name else unpack_model(a.model)
        cache[key] = cached
    return cached


class ExplainModel(TableInOutGenerator[ExplainArgs]):
    FunctionArguments: ClassVar[type] = ExplainArgs

    class Meta:
        name = "explain"
        description = "Per-row SHAP feature contributions toward the model's raw margin (base_value + contrib_*)"
        categories = ["models", "interpretation", "inference"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM xgboost.explain((SELECT * FROM xgboost.diabetes()), "
                    "model_name => 'diab_reg', id => 'sample_id')"
                ),
                description="Explain each row's prediction with per-feature contributions",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[ExplainArgs]) -> BindResponse:
        a = params.args
        if not a.model_name and not a.model:
            raise ValueError("explain requires either 'model_name' (a registry name) or 'model' (a model BLOB)")
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        meta = _explain_meta(a)

        if meta.task == CLASSIFICATION and meta.classes is not None and len(meta.classes) > 2:
            raise ValueError(
                f"explain supports regression and binary classification; model {a.model_name!r} has "
                f"{len(meta.classes)} classes (multiclass contributions are not emitted)"
            )

        missing = [f for f in meta.feature_names if f not in input_schema.names]
        if missing:
            raise ValueError(
                f"model {a.model_name!r} requires feature column(s) {', '.join(missing)} "
                f"not present in the input; model features: {', '.join(meta.feature_names)}; "
                f"input columns: {', '.join(input_schema.names)}"
            )

        fields: list[pa.Field] = []
        if a.id:
            fields.append(input_schema.field(a.id))
        fields.append(sfield("base_value", pa.float64(), "Model base (expected) raw-margin value.", nullable=False))
        for f in meta.feature_names:
            fields.append(
                sfield(_contrib_name(f), pa.float64(), f"Contribution of {f} to the raw margin.", nullable=False)
            )
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def _model(cls, params: ProcessParams[ExplainArgs]) -> tuple[Any, ModelMetadata]:
        return _explain_model(_EXPLAIN_CACHE, params)

    @classmethod
    def process(
        cls,
        params: ProcessParams[ExplainArgs],
        state: None,
        batch: pa.RecordBatch,
        out: InOutCollector,
    ) -> None:
        a = params.args
        estimator, meta = cls._model(params)
        cat_mask = meta.categorical or [False] * len(meta.feature_names)
        x = build_x_predict(pa.Table.from_batches([batch]), meta.feature_names, cat_mask, meta.categories)

        booster = estimator.get_booster()
        dmat = xgboost.DMatrix(x, enable_categorical=True)
        contribs = booster.predict(dmat, pred_contribs=True)  # (n_rows, n_features + 1); last col = base value

        columns: dict[str, list[Any]] = {}
        if a.id:
            columns[a.id] = batch.column(a.id).to_pylist()
        columns["base_value"] = [float(v) for v in contribs[:, -1]]
        for j, f in enumerate(meta.feature_names):
            columns[_contrib_name(f)] = [float(v) for v in contribs[:, j]]

        out.emit(pa.RecordBatch.from_pydict(columns, schema=params.output_schema))


# ===========================================================================
# shap_values (per-row long format: one row per (input row, feature))
# ===========================================================================


@dataclass(slots=True, frozen=True)
class ShapValuesArgs:
    data: Annotated[TableInput, Arg(0, doc="Table to explain (must contain the model's feature columns).")]
    model_name: Annotated[
        str, Arg("model_name", default="", doc="Name of a model in the registry. Provide this OR model.")
    ]
    model: Annotated[bytes, Arg("model", default=b"", doc="A model BLOB. Provide this OR model_name.")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through onto every emitted row.")]


_SHAP_CACHE: dict[bytes, tuple[Any, ModelMetadata]] = {}


class ShapValues(TableInOutGenerator[ShapValuesArgs]):
    FunctionArguments: ClassVar[type] = ShapValuesArgs

    class Meta:
        name = "shap_values"
        description = "Per-row SHAP contributions in long format: one row per (input row, feature)"
        categories = ["models", "interpretation", "inference"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM xgboost.shap_values((SELECT * FROM xgboost.diabetes()), "
                    "model_name => 'diab_reg', id => 'sample_id') ORDER BY sample_id, feature LIMIT 5"
                ),
                description="Long-format SHAP values, one row per feature per sample",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[ShapValuesArgs]) -> BindResponse:
        a = params.args
        if not a.model_name and not a.model:
            raise ValueError("shap_values requires either 'model_name' (a registry name) or 'model' (a model BLOB)")
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        meta = _explain_meta(a)
        if meta.task == CLASSIFICATION and meta.classes is not None and len(meta.classes) > 2:
            raise ValueError(
                f"shap_values supports regression and binary classification; this model has "
                f"{len(meta.classes)} classes (multiclass contributions are not emitted)"
            )
        missing = [f for f in meta.feature_names if f not in input_schema.names]
        if missing:
            raise ValueError(
                f"model requires feature column(s) {', '.join(missing)} not present in the input; "
                f"model features: {', '.join(meta.feature_names)}"
            )
        fields: list[pa.Field] = []
        if a.id:
            fields.append(input_schema.field(a.id))
        fields.append(sfield("feature", pa.string(), "Feature name.", nullable=False))
        fields.append(
            sfield("shap_value", pa.float64(), "This feature's contribution to the row's raw margin.", nullable=False)
        )
        fields.append(sfield("base_value", pa.float64(), "Model base (expected) raw-margin value.", nullable=False))
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def _model(cls, params: ProcessParams[ShapValuesArgs]) -> tuple[Any, ModelMetadata]:
        return _explain_model(_SHAP_CACHE, params)

    @classmethod
    def process(
        cls,
        params: ProcessParams[ShapValuesArgs],
        state: None,
        batch: pa.RecordBatch,
        out: InOutCollector,
    ) -> None:
        a = params.args
        estimator, meta = cls._model(params)
        cat_mask = meta.categorical or [False] * len(meta.feature_names)
        x = build_x_predict(pa.Table.from_batches([batch]), meta.feature_names, cat_mask, meta.categories)

        booster = estimator.get_booster()
        dmat = xgboost.DMatrix(x, enable_categorical=True)
        contribs = booster.predict(dmat, pred_contribs=True)  # (n_rows, n_features + 1); last col = base value

        feats = meta.feature_names
        n_rows = contribs.shape[0]
        ids = batch.column(a.id).to_pylist() if a.id else None

        id_col: list[Any] = []
        feat_col: list[str] = []
        shap_col: list[float] = []
        base_col: list[float] = []
        for r in range(n_rows):
            base = float(contribs[r, -1])
            for j, f in enumerate(feats):
                if ids is not None:
                    id_col.append(ids[r])
                feat_col.append(f)
                shap_col.append(float(contribs[r, j]))
                base_col.append(base)

        columns: dict[str, list[Any]] = {}
        if a.id:
            columns[a.id] = id_col
        columns["feature"] = feat_col
        columns["shap_value"] = shap_col
        columns["base_value"] = base_col
        out.emit(pa.RecordBatch.from_pydict(columns, schema=params.output_schema))


# ===========================================================================
# permutation_importance (model-agnostic feature importance)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class PermImportanceArgs:
    data: Annotated[TableInput, Arg(0, doc="Evaluation table (the model's features + the target column).")]
    model_name: Annotated[
        str, Arg("model_name", default="", doc="Name of a model in the registry. Provide this OR model.")
    ]
    model: Annotated[bytes, Arg("model", default=b"", doc="A model BLOB. Provide this OR model_name.")]
    target: Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")]
    n_repeats: Annotated[int, Arg("n_repeats", default=5, doc="Number of times each feature is shuffled.")]
    scoring: Annotated[str, Arg("scoring", default="", doc="Scorer name (default: the estimator's own scorer).")]
    random_state: Annotated[int, Arg("random_state", default=0, doc="Random seed.")]


_PERM_SCHEMA = pa.schema(
    [
        sfield("feature", pa.string(), "Feature column name.", nullable=False),
        sfield("importance_mean", pa.float64(), "Mean drop in score when the feature is shuffled.", nullable=False),
        sfield("importance_std", pa.float64(), "Std-dev of the importance across repeats.", nullable=False),
    ]
)


class PermutationImportance(SinkBuffer[PermImportanceArgs, DrainState]):
    FunctionArguments: ClassVar[type] = PermImportanceArgs

    class Meta:
        name = "permutation_importance"
        description = "Model-agnostic feature importance: the drop in score when each feature is shuffled"
        categories = ["models", "interpretation", "evaluation"]
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM xgboost.permutation_importance("
                    "(SELECT * EXCLUDE (target_name) FROM xgboost.iris()), "
                    "model_name := 'iris_clf', target := 'target') ORDER BY importance_mean DESC"
                ),
                description="Rank iris features by permutation importance for a stored model",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[PermImportanceArgs]) -> BindResponse:
        a = params.args
        if not a.model_name and not a.model:
            raise ValueError("permutation_importance requires either 'model_name' or 'model' (a model BLOB)")
        if not a.target:
            raise ValueError("permutation_importance requires 'target' (the label column name)")
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        meta = _explain_meta(a)
        missing = [f for f in meta.feature_names if f not in input_schema.names]
        if missing:
            raise ValueError(
                f"model requires feature column(s) {', '.join(missing)} not present in the input; "
                f"model features: {', '.join(meta.feature_names)}"
            )
        if a.target not in input_schema.names:
            raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")
        return BindResponse(output_schema=_PERM_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[PermImportanceArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[PermImportanceArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: BufferingOutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        input_schema = input_schema_of(params)
        estimator, meta = get_store().load(a.model_name) if a.model_name else unpack_model(a.model)

        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            raise ValueError("permutation_importance received no rows")

        cat_mask = meta.categorical or [False] * len(meta.feature_names)
        x = build_x_predict(table, meta.feature_names, cat_mask, meta.categories)
        y = np.asarray(table.column(a.target).to_numpy(zero_copy_only=False))
        y = np.rint(y.astype(float)).astype(int) if meta.task == CLASSIFICATION else y.astype(float)

        result = sk_permutation_importance(
            estimator, x, y, n_repeats=a.n_repeats, random_state=a.random_state, scoring=(a.scoring or None)
        )
        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "feature": list(meta.feature_names),
                    "importance_mean": [float(v) for v in result.importances_mean],
                    "importance_std": [float(v) for v in result.importances_std],
                },
                schema=params.output_schema,
            )
        )


IMPORTANCE_FUNCTIONS: list[type] = [
    FeatureImportance,
    ExplainModel,
    ShapValues,
    PermutationImportance,
]
