"""XGBoost-specific extras that read a stored model.

* ``feature_importance`` -- the booster's per-feature importance (weight / gain /
  cover) for a stored model, one row per feature, ranked.
* ``explain``            -- SHAP-style per-row prediction contributions
  (``pred_contribs``) in **long format** ``(row, [class], feature, shap_value,
  base_value)``: how much each feature pushed each row's raw margin away from the
  model's base value. Streams a table through the model like ``predict`` and
  supports multiclass (one row per (row, class, feature)).
* ``permutation_importance`` -- model-agnostic ranked importance: the drop in
  score when each feature is shuffled.
* ``partial_dependence`` -- how the model's average prediction moves as one
  numeric feature varies over a grid (one curve per class for multiclass).

    SELECT * FROM xgboost.feature_importance('iris_clf', importance_type => 'gain');
    SELECT * FROM xgboost.explain((SELECT * FROM new_data), model_name => 'house_reg', id => 'id');
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
import pyarrow as pa
import xgboost
from sklearn.inspection import partial_dependence as sk_partial_dependence
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
from .schema_utils import columns_md, columns_md_rows
from .schema_utils import field as sfield

_IMPORTANCE_TYPES = {"weight", "gain", "cover", "total_gain", "total_cover"}


def _resolve_meta(model_name: str, model: bytes) -> ModelMetadata:
    """Read a model's metadata from a BLOB if given, else from the registry by name."""
    if model:
        return unpack_meta(model)
    if not model_name:
        raise ValueError("requires either a 'model_name' (a registry name) or 'model' (a model BLOB)")
    try:
        return get_store().load_meta(model_name)
    except ModelNotFoundError as exc:
        raise ValueError(f"model {model_name!r} not found in the registry") from exc


def _resolve_model(model_name: str, model: bytes) -> tuple[Any, ModelMetadata]:
    """Load a model (estimator + metadata) from a BLOB if given, else the registry."""
    if model:
        return unpack_model(model)
    return get_store().load(model_name)


# ===========================================================================
# feature_importance
# ===========================================================================


@dataclass(slots=True, frozen=True)
class FeatureImportanceArgs:
    model_name: Annotated[str, Arg(0, doc="Name of a stored model (pass '' to use model:= instead).")]
    model: Annotated[
        bytes, Arg("model", default=b"", doc="A model BLOB (as returned by fit). Provide this OR model_name.")
    ]
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
        tags = {
            "vgi.result_columns_md": columns_md(_IMPORTANCE_SCHEMA),
            "vgi.doc_llm": (
                "Returns the booster's built-in per-feature importance for a stored model, one ranked row "
                "per feature (`feature`, `importance`, `rank`; rank 1 = most important, already sorted "
                "descending). Identify the model positionally by name (`feature_importance('m')`) or with "
                "`model :=` (a BLOB). Choose the `importance_type :=` — `gain` (default; average loss "
                "reduction), `weight` (split count), `cover`, `total_gain`, or `total_cover`. Features "
                "never used in a split report importance 0. A fast, model-native ranking; for a "
                "model-agnostic alternative use `permutation_importance`."
            ),
            "vgi.doc_md": (
                "**Booster feature importance** — model-native, ranked.\n\n"
                "- Identify the model by name (`feature_importance('m')`) or `model :=` BLOB\n"
                "- `importance_type :=` `gain` (default) | `weight` | `cover` | `total_gain` | "
                "`total_cover`\n"
                "- Returns `(feature, importance, rank)`, sorted by importance descending\n\n"
                "Unused features score 0. See `permutation_importance` for a model-agnostic measure."
            ),
        }
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
        _resolve_meta(a.model_name, a.model)
        return BindResponse(output_schema=cls.FIXED_SCHEMA)

    @classmethod
    def cardinality(cls, params: BindParams[FeatureImportanceArgs]) -> TableCardinality:
        return TableCardinality(estimate=20, max=100000)

    @classmethod
    def process(cls, params: ProcessParams[FeatureImportanceArgs], state: None, out: OutputCollector) -> None:
        a = params.args
        estimator, meta = _resolve_model(a.model_name, a.model)
        booster = estimator.get_booster()
        # Models are fit on a numpy matrix, so the booster names features f0..fN
        # in feature order; fall back to the stored name in case names were set.
        scores = booster.get_score(importance_type=a.importance_type)
        rows = [(name, float(scores.get(f"f{i}", scores.get(name, 0.0)))) for i, name in enumerate(meta.feature_names)]
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
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through onto every emitted row.")]


_EXPLAIN_CACHE: dict[bytes, tuple[Any, ModelMetadata]] = {}


class ExplainModel(TableInOutGenerator[ExplainArgs]):
    FunctionArguments: ClassVar[type] = ExplainArgs

    class Meta:
        name = "explain"
        description = "Per-row SHAP feature contributions, long format (row, [class], feature, shap_value, base_value)"
        categories = ["models", "interpretation", "inference"]
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [
                    ("feature", "VARCHAR", "Feature column name."),
                    ("shap_value", "DOUBLE", "Contribution of the feature to the raw margin."),
                    ("base_value", "DOUBLE", "Model base (expected) raw-margin value."),
                ],
                note=(
                    "Long format: one row per (input row, feature). If an `id` column is named, it is carried "
                    "through as the first column. For multiclass models a `class` BIGINT column is added "
                    "(one row per (input row, class, feature))."
                ),
            ),
            "vgi.doc_llm": (
                "Streams a table through a stored model and emits SHAP-style per-row prediction "
                "contributions (XGBoost's `pred_contribs`) in long format: one row per (input row, "
                "feature) with `feature`, `shap_value` (how much that feature pushed this row's raw margin "
                "away from the model's base), and `base_value` (the expected margin). Identify the model "
                "with `model_name :=` or `model :=` (a BLOB); name an `id :=` to tag every emitted row. "
                "For multiclass models an extra `class` column appears (one row per (row, class, "
                "feature)). Sum `shap_value` over a row's features and add `base_value` to recover the raw "
                "margin. Use it for local, per-prediction explanations."
            ),
            "vgi.doc_md": (
                "**SHAP explanations** — per-row feature contributions, long format.\n\n"
                "- Identify the model with `model_name :=` or `model :=`; optional `id :=` passthrough\n"
                "- One row per (input row, feature): `feature`, `shap_value`, `base_value`\n"
                "- Multiclass adds a `class` BIGINT column (one row per (row, class, feature))\n\n"
                "`sum(shap_value) + base_value` = the row's raw margin. Local, per-prediction explanation."
            ),
        }
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
        meta = _resolve_meta(a.model_name, a.model)

        missing = [f for f in meta.feature_names if f not in input_schema.names]
        if missing:
            raise ValueError(
                f"model requires feature column(s) {', '.join(missing)} "
                f"not present in the input; model features: {', '.join(meta.feature_names)}; "
                f"input columns: {', '.join(input_schema.names)}"
            )

        fields: list[pa.Field] = []
        if a.id:
            fields.append(input_schema.field(a.id))
        multiclass = meta.task == CLASSIFICATION and meta.classes is not None and len(meta.classes) > 2
        if multiclass:
            fields.append(sfield("class", pa.int64(), "Class index the contribution applies to.", nullable=False))
        fields.append(sfield("feature", pa.string(), "Feature column name.", nullable=False))
        fields.append(
            sfield("shap_value", pa.float64(), "Contribution of the feature to the raw margin.", nullable=False)
        )
        fields.append(sfield("base_value", pa.float64(), "Model base (expected) raw-margin value.", nullable=False))
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def _model(cls, params: ProcessParams[ExplainArgs]) -> tuple[Any, ModelMetadata]:
        assert params.init_response is not None
        key = params.init_response.execution_id
        cached = _EXPLAIN_CACHE.get(key)
        if cached is None:
            cached = _resolve_model(params.args.model_name, params.args.model)
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
        cat_mask = meta.categorical or [False] * len(meta.feature_names)
        x = build_x_predict(pa.Table.from_batches([batch]), meta.feature_names, cat_mask, meta.categories)

        booster = estimator.get_booster()
        dmat = xgboost.DMatrix(x, enable_categorical=True)
        contribs = np.asarray(booster.predict(dmat, pred_contribs=True))

        feats = meta.feature_names
        n_feat = len(feats)
        n_rows = contribs.shape[0]
        ids = batch.column(a.id).to_pylist() if a.id else None
        n_classes = len(meta.classes) if (meta.classes is not None) else 0
        multiclass = meta.task == CLASSIFICATION and n_classes > 2

        # Regression / binary: contribs is (n_rows, n_feat+1), last col = base.
        # Multiclass: XGBoost returns (n_rows, n_classes, n_feat+1).
        id_out: list[Any] = []
        class_out: list[int] = []
        feature_out: list[str] = []
        shap_out: list[float] = []
        base_out: list[float] = []

        n_blocks = n_classes if multiclass else 1
        for r in range(n_rows):
            for b in range(n_blocks):
                row = contribs[r, b] if multiclass else contribs[r]
                base = float(row[n_feat])
                for j, fname in enumerate(feats):
                    if ids is not None:
                        id_out.append(ids[r])
                    if multiclass:
                        class_out.append(b)
                    feature_out.append(fname)
                    shap_out.append(float(row[j]))
                    base_out.append(base)

        columns: dict[str, list[Any]] = {}
        if a.id:
            columns[a.id] = id_out
        if multiclass:
            columns["class"] = class_out
        columns["feature"] = feature_out
        columns["shap_value"] = shap_out
        columns["base_value"] = base_out
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
        sfield("rank", pa.int32(), "1-based rank by importance_mean (1 = most important).", nullable=False),
    ]
)


class PermutationImportance(SinkBuffer[PermImportanceArgs, DrainState]):
    FunctionArguments: ClassVar[type] = PermImportanceArgs

    class Meta:
        name = "permutation_importance"
        description = "Model-agnostic feature importance: the drop in score when each feature is shuffled"
        categories = ["models", "interpretation", "evaluation"]
        tags = {
            "vgi.result_columns_md": columns_md(_PERM_SCHEMA),
            "vgi.doc_llm": (
                "Computes model-agnostic permutation importance over an evaluation table: for each "
                "feature, shuffle its values `n_repeats :=` times and measure the resulting drop in the "
                "model's score; a larger drop means a more important feature. Returns one ranked row per "
                "feature (`feature`, `importance_mean`, `importance_std`, `rank`; rank 1 = most "
                "important). Identify the model with `model_name :=` or `model :=`, name the held-out "
                "`target :=` column, and optionally set `scoring :=` and `random_state :=`. Unlike "
                "`feature_importance` this measures impact on real predictions and respects feature "
                "correlations, but it needs labelled data and is slower."
            ),
            "vgi.doc_md": (
                "**Permutation importance** — model-agnostic, ranked.\n\n"
                "- Identify the model with `model_name :=` or `model :=`; name the `target :=` column\n"
                "- `n_repeats :=` shuffles per feature, optional `scoring :=`, `random_state :=`\n"
                "- Returns `(feature, importance_mean, importance_std, rank)`, sorted descending\n\n"
                "Measures impact on real predictions (needs labels; slower than `feature_importance`)."
            ),
        }
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
        meta = _resolve_meta(a.model_name, a.model)
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
        estimator, meta = _resolve_model(a.model_name, a.model)

        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            raise ValueError("permutation_importance received no rows")

        cat_mask = meta.categorical or [False] * len(meta.feature_names)
        x = build_x_predict(table, meta.feature_names, cat_mask, meta.categories)
        if meta.task == CLASSIFICATION:
            # Score against the same label-encoded codes the estimator predicts,
            # mapping the (possibly string) target labels through the stored classes.
            code_of = {c: i for i, c in enumerate(meta.classes or [])}
            raw = table.column(a.target).to_pylist()
            y = np.asarray([code_of.get(v, -1) for v in raw], dtype=int)
        else:
            y = np.asarray(table.column(a.target).to_numpy(zero_copy_only=False)).astype(float)

        result = sk_permutation_importance(
            estimator, x, y, n_repeats=a.n_repeats, random_state=a.random_state, scoring=(a.scoring or None)
        )
        rows = sorted(
            zip(meta.feature_names, result.importances_mean, result.importances_std, strict=True),
            key=lambda r: r[1],
            reverse=True,
        )
        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "feature": [name for name, _, _ in rows],
                    "importance_mean": [float(m) for _, m, _ in rows],
                    "importance_std": [float(s) for _, _, s in rows],
                    "rank": [i + 1 for i in range(len(rows))],
                },
                schema=params.output_schema,
            )
        )


# ===========================================================================
# partial_dependence (how a model's prediction moves with one feature)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class PartialDependenceArgs:
    data: Annotated[TableInput, Arg(0, doc="Background table (the model's feature columns).")]
    model_name: Annotated[
        str, Arg("model_name", default="", doc="Name of a model in the registry. Provide this OR model.")
    ]
    model: Annotated[bytes, Arg("model", default=b"", doc="A model BLOB. Provide this OR model_name.")]
    feature: Annotated[str, Arg("feature", default="", doc="Numeric feature column to vary (required).")]
    grid_resolution: Annotated[int, Arg("grid_resolution", default=100, doc="Number of grid points along the feature.")]


_PD_SCHEMA = pa.schema(
    [
        sfield("feature_value", pa.float64(), "Value the feature was set to.", nullable=False),
        sfield("class", pa.int64(), "Class index (NULL for regression / the single binary curve)."),
        sfield("partial_dependence", pa.float64(), "Average model output at this feature value.", nullable=False),
    ]
)


class PartialDependence(SinkBuffer[PartialDependenceArgs, DrainState]):
    FunctionArguments: ClassVar[type] = PartialDependenceArgs

    class Meta:
        name = "partial_dependence"
        description = "How a stored model's average prediction changes as one feature varies over a grid"
        categories = ["models", "inspection"]
        tags = {
            "vgi.result_columns_md": columns_md(_PD_SCHEMA),
            "vgi.doc_llm": (
                "Computes the partial-dependence curve for one numeric feature: sweeps `feature :=` over a "
                "grid of `grid_resolution :=` points and reports the model's average prediction at each "
                "value, holding the rest of the buffered background table fixed. Returns `(feature_value, "
                "class, partial_dependence)` ordered along the grid; `class` is NULL for regression and "
                "the single binary curve, and is the class index (one curve per class) for multiclass. "
                "Identify the model with `model_name :=` or `model :=`. Numeric features only "
                "(categorical features raise a clear error). Use it to see the marginal effect and shape "
                "of a feature on the prediction."
            ),
            "vgi.doc_md": (
                "**Partial dependence** — marginal effect of one feature.\n\n"
                "- `feature :=` (numeric only) swept over `grid_resolution :=` points\n"
                "- Identify the model with `model_name :=` or `model :=`; input is the background table\n"
                "- Returns `(feature_value, class, partial_dependence)` along the grid\n\n"
                "`class` is NULL for regression/binary, the class index for multiclass (one curve each)."
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM xgboost.partial_dependence((SELECT * FROM xgboost.iris()), "
                    "model_name := 'iris_clf', feature := 'petal_length_cm') ORDER BY feature_value"
                ),
                description="Partial dependence of 'iris_clf' on petal length",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[PartialDependenceArgs]) -> BindResponse:
        a = params.args
        if not a.model_name and not a.model:
            raise ValueError("partial_dependence requires either 'model_name' or 'model' (a model BLOB)")
        if not a.feature:
            raise ValueError("partial_dependence requires 'feature' (the column to vary)")
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        meta = _resolve_meta(a.model_name, a.model)
        if a.feature not in meta.feature_names:
            raise ValueError(
                f"feature {a.feature!r} is not one of the model's features: {', '.join(meta.feature_names)}"
            )
        idx = meta.feature_names.index(a.feature)
        if (meta.categorical or [False] * len(meta.feature_names))[idx]:
            raise ValueError(f"partial_dependence supports numeric features only; {a.feature!r} is categorical")
        missing = [f for f in meta.feature_names if f not in input_schema.names]
        if missing:
            raise ValueError(f"model requires feature column(s) {', '.join(missing)} not present in the input")
        return BindResponse(output_schema=_PD_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[PartialDependenceArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[PartialDependenceArgs],
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
        estimator, meta = _resolve_model(a.model_name, a.model)

        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            raise ValueError("partial_dependence received no rows")

        cat_mask = meta.categorical or [False] * len(meta.feature_names)
        x = build_x_predict(table, meta.feature_names, cat_mask, meta.categories)
        idx = meta.feature_names.index(a.feature)
        result = sk_partial_dependence(estimator, x, [idx], grid_resolution=a.grid_resolution, kind="average")
        grid = result["grid_values"][0]
        averages = np.asarray(result["average"])  # shape (n_outputs, n_grid)

        # Label each output's curve: regression -> NULL; binary -> the single
        # curve (NULL); multiclass -> one curve per class index.
        if meta.task == CLASSIFICATION and averages.shape[0] > 1:
            labels: list[int | None] = list(range(averages.shape[0]))
        else:
            labels = [None] * averages.shape[0]

        feature_value: list[float] = []
        class_col: list[Any] = []
        pd_col: list[float] = []
        for o in range(averages.shape[0]):
            for g in range(len(grid)):
                feature_value.append(float(grid[g]))
                class_col.append(labels[o])
                pd_col.append(float(averages[o, g]))
        out.emit(
            pa.RecordBatch.from_pydict(
                {"feature_value": feature_value, "class": class_col, "partial_dependence": pd_col},
                schema=params.output_schema,
            )
        )


IMPORTANCE_FUNCTIONS: list[type] = [
    FeatureImportance,
    ExplainModel,
    PermutationImportance,
    PartialDependence,
]
