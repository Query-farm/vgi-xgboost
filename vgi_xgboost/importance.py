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

import pyarrow as pa
import xgboost
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
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

from .buffering import matrix
from .models import CLASSIFICATION
from .registry import ModelMetadata, ModelNotFoundError, get_store
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
    model_name: Annotated[str, Arg("model_name", default="", doc="Name of a stored model (required).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through.")]


_EXPLAIN_CACHE: dict[bytes, tuple[Any, ModelMetadata]] = {}


def _contrib_name(feature: str) -> str:
    return f"contrib_{feature}"


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
        if not a.model_name:
            raise ValueError("explain requires 'model_name' (e.g. model_name := 'my_model')")
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        try:
            meta = get_store().load_meta(a.model_name)
        except ModelNotFoundError as exc:
            raise ValueError(f"model {a.model_name!r} not found in the registry") from exc

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
        assert params.init_response is not None
        key = params.init_response.execution_id
        cached = _EXPLAIN_CACHE.get(key)
        if cached is None:
            cached = get_store().load(params.args.model_name)
            _EXPLAIN_CACHE[key] = cached
        return cached

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
        x = matrix(pa.Table.from_batches([batch]), meta.feature_names)

        booster = estimator.get_booster()
        dmat = xgboost.DMatrix(x)
        contribs = booster.predict(dmat, pred_contribs=True)  # (n_rows, n_features + 1); last col = base value

        columns: dict[str, list[Any]] = {}
        if a.id:
            columns[a.id] = batch.column(a.id).to_pylist()
        columns["base_value"] = [float(v) for v in contribs[:, -1]]
        for j, f in enumerate(meta.feature_names):
            columns[_contrib_name(f)] = [float(v) for v in contribs[:, j]]

        out.emit(pa.RecordBatch.from_pydict(columns, schema=params.output_schema))


IMPORTANCE_FUNCTIONS: list[type] = [FeatureImportance, ExplainModel]
