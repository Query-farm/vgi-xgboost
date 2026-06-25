"""Supervised learning: fit XGBoost estimators into the registry and predict from it.

* ``fit``       -- TableBufferingFunction: buffer the training table, fit an
  estimator, persist it to the registry, return a one-row training summary.
* ``predict``   -- TableInOutGenerator: stream a table through a stored model.
* ``cross_val_predict`` -- buffering: out-of-fold predictions, no persistence.
* ``list_models`` / ``model_info`` / ``drop_model`` -- registry management.

Column roles follow the project convention: name the ``target`` column (for
fit / cross_val_predict) and optionally an ``id`` column to carry through; every
other column is a numeric feature. Hyperparameters are passed as a JSON string.
Classification targets may be **any label dtype** (string, int, …): they are
label-encoded to ``0..n_classes-1`` codes for XGBoost, the original ordered
labels are stored in the model, and ``predict`` decodes back to those labels.

    SELECT * FROM xgboost.fit((SELECT * FROM training), model_name => 'iris_clf',
                              estimator => 'xgb_classifier', target => 'species', id => 'id');
    SELECT * FROM xgboost.predict((SELECT * FROM new_data), model_name => 'iris_clf', id => 'id');
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

import numpy as np
import pyarrow as pa
import xgboost
from sklearn.model_selection import cross_val_predict as sk_cross_val_predict
from sklearn.model_selection import cross_val_score as sk_cross_val_score
from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
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
from vgi_rpc.log import Level
from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor

from .buffering import DrainState, SinkBuffer, input_schema_of
from .features import build_x_fit, build_x_predict, categorical_mask
from .registry import (
    ModelMetadata,
    ModelNotFoundError,
    get_store,
    now_iso,
    pack_model,
    unpack_meta,
    unpack_model,
    validate_name,
)
from .schema_utils import columns_md, columns_md_rows
from .schema_utils import field as sfield

CLASSIFICATION = "classification"
REGRESSION = "regression"

# Defaults shared by every estimator: a fixed seed, and native categorical
# support via the histogram tree method (so string features work without
# one-hot encoding and missing values flow through as NaN).
_COMMON_DEFAULTS: dict[str, Any] = {"random_state": 0, "enable_categorical": True, "tree_method": "hist"}

# name -> (task, estimator class, default kwargs)
_ESTIMATORS: dict[str, tuple[str, type, dict[str, Any]]] = {
    "xgb_classifier": (CLASSIFICATION, XGBClassifier, dict(_COMMON_DEFAULTS)),
    "xgb_regressor": (REGRESSION, XGBRegressor, dict(_COMMON_DEFAULTS)),
    "xgb_rf_classifier": (CLASSIFICATION, XGBRFClassifier, dict(_COMMON_DEFAULTS)),
    "xgb_rf_regressor": (REGRESSION, XGBRFRegressor, dict(_COMMON_DEFAULTS)),
}


def _parse_params(params: str) -> dict[str, Any]:
    params = (params or "").strip()
    if not params:
        return {}
    parsed = json.loads(params)
    if not isinstance(parsed, dict):
        raise ValueError("params must be a JSON object, e.g. '{\"n_estimators\": 200}'")
    return parsed


def estimator_param_names(name: str) -> list[str]:
    """Sorted list of hyperparameters accepted by an estimator (for discovery/errors)."""
    _task, cls, _defaults = _ESTIMATORS[name]
    return sorted(cls().get_params().keys())


def build_estimator(name: str, params: dict[str, Any]) -> tuple[str, Any]:
    """Return ``(task, estimator)`` for a registered estimator name + hyperparams."""
    if name not in _ESTIMATORS:
        raise ValueError(f"unknown estimator {name!r}; choose one of: {', '.join(sorted(_ESTIMATORS))}")
    task, cls, defaults = _ESTIMATORS[name]
    # Reject unknown hyperparameters up front with the valid set, rather than
    # surfacing XGBoost's opaque error later.
    valid = set(cls().get_params().keys())
    unknown = [k for k in params if k not in valid]
    if unknown:
        raise ValueError(
            f"unknown hyperparameter(s) for {name!r}: {', '.join(sorted(unknown))}. "
            f"valid params: {', '.join(sorted(valid))}"
        )
    try:
        return task, cls(**{**defaults, **params})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid hyperparameters for {name!r}: {exc}") from exc


def _target_array(column: pa.ChunkedArray | pa.Array, task: str) -> tuple[np.ndarray, list[Any] | None]:
    """Turn the target column into ``(y, classes)``.

    For regression ``y`` is float64 and ``classes`` is ``None``. For
    classification any label dtype is accepted: the labels are sorted into a
    stable order, label-encoded to integer codes ``0..n_classes-1`` (XGBoost
    requires contiguous integer codes), and the *original* ordered labels are
    returned as ``classes`` so predictions can be decoded back. A friendly error
    is raised when the target has no usable (non-null) labels.
    """
    raw = column.to_pylist()
    if task != CLASSIFICATION:
        return np.asarray(raw, dtype=float), None
    labels = [v for v in raw if v is not None]
    if not labels:
        raise ValueError("classification target has no usable (non-null) labels")
    if len({type(v) for v in labels}) > 1:
        raise ValueError("classification target mixes incompatible label types; use a single label dtype")
    try:
        classes = sorted(set(labels))
    except TypeError as exc:
        raise ValueError(f"classification target labels are not orderable: {exc}") from exc
    code_of = {c: i for i, c in enumerate(classes)}
    if any(v is None for v in raw):
        raise ValueError("classification target contains NULL labels; drop or impute them before fitting")
    y = np.asarray([code_of[v] for v in raw], dtype=int)
    return y, classes


def _xy(
    table: pa.Table, feature_names: list[str], target: str, task: str
) -> tuple[Any, np.ndarray, list[bool], list[list[str] | None], list[Any] | None]:
    """Assemble ``(X, y, categorical_mask, categories, classes)`` from a buffered table.

    ``X`` is a pandas DataFrame so XGBoost can use native categorical + missing
    handling. ``categories`` records each categorical column's category list for
    reproducible scoring later. For classification, ``y`` is integer codes and
    ``classes`` the original ordered labels (``None`` for regression).
    """
    cat_mask = categorical_mask([table.schema.field(n).type for n in feature_names])
    x, categories = build_x_fit(table, feature_names, cat_mask)
    y, classes = _target_array(table.column(target), task)
    return x, y, cat_mask, categories, classes


def _features_excluding(input_schema: pa.Schema, *exclude: str) -> list[str]:
    drop = {e for e in exclude if e}
    return [n for n in input_schema.names if n not in drop]


def _label_arrow_type(classes: list[Any] | None) -> pa.DataType:
    """Arrow type for a decoded class label, inferred from the stored class labels."""
    if classes and all(isinstance(c, str) for c in classes):
        return pa.string()
    return pa.int64()


def _prediction_field(task: str, classes: list[Any] | None = None) -> pa.Field:
    if task == CLASSIFICATION:
        return sfield("prediction", _label_arrow_type(classes), "Predicted class label.", nullable=False)
    return sfield("prediction", pa.float64(), "Predicted value.", nullable=False)


def _decode_labels(codes: Any, classes: list[Any] | None) -> list[Any]:
    """Map integer class codes back to the original labels (identity if no classes)."""
    if classes is None:
        return [int(v) for v in codes]
    return [classes[int(v)] for v in codes]


# ===========================================================================
# fit
# ===========================================================================


# Required string args carry a "" default so an omitted value reaches on_bind as
# "" and we can raise a friendly error, instead of the framework's raw KeyError
# during argument parsing.
@dataclass(slots=True, frozen=True)
class FitArgs:
    data: Annotated[TableInput, Arg(0, doc="Training table (features + target [+ id]).")]
    model_name: Annotated[
        str, Arg("model_name", default="", doc="Optional registry name; the model is always returned as a BLOB.")
    ]
    estimator: Annotated[str, Arg("estimator", default="xgb_classifier", doc="Estimator name.")]
    target: Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")]
    params: Annotated[str, Arg("params", default="", doc="JSON object of hyperparameters.")]


_FIT_SCHEMA = pa.schema(
    [
        sfield("model_name", pa.string(), "Name the model was stored under ('' if not persisted).", nullable=False),
        sfield("estimator", pa.string(), "Estimator type used.", nullable=False),
        sfield("task", pa.string(), "classification or regression.", nullable=False),
        sfield("n_samples", pa.int64(), "Number of training rows.", nullable=False),
        sfield("n_features", pa.int64(), "Number of features.", nullable=False),
        sfield("n_classes", pa.int64(), "Number of classes (NULL for regression)."),
        sfield("train_score", pa.float64(), "In-sample score (accuracy or R^2)."),
        sfield("features", pa.list_(pa.string()), "Ordered feature column names.", nullable=False),
        sfield(
            "model", pa.binary(), "The fitted model as a self-contained BLOB (estimator + metadata).", nullable=False
        ),
    ]
)


def _fit_and_emit(
    out: OutputCollector,
    output_schema: pa.Schema,
    *,
    table: pa.Table | None,
    input_schema: pa.Schema,
    estimator_label: str,
    task: str,
    estimator: Any,
    model_name: str,
    target: str,
    id_col: str,
    params_dict: dict[str, Any],
) -> None:
    """Fit ``estimator`` on the buffered table, persist if named, emit summary + BLOB.

    Shared by the generic ``fit`` and the typed ``fit_<estimator>`` functions.
    """
    if table is None or table.num_rows == 0:
        raise ValueError("fit received no training rows")
    feats = _features_excluding(input_schema, target, id_col)
    x, y, cat_mask, categories, classes = _xy(table, feats, target, task)
    estimator.fit(x, y)
    train_score = float(estimator.score(x, y))

    meta = ModelMetadata(
        name=model_name,
        estimator=estimator_label,
        task=task,
        target=target,
        feature_names=feats,
        categorical=cat_mask,
        categories=categories,
        params=params_dict,
        classes=classes,
        n_samples=int(table.num_rows),
        n_features=len(feats),
        train_score=train_score,
        xgboost_version=xgboost.__version__,
        created_at=now_iso(),
    )
    if model_name:
        get_store().save(estimator, meta)

    out.emit(
        pa.RecordBatch.from_pydict(
            {
                "model_name": [model_name],
                "estimator": [estimator_label],
                "task": [task],
                "n_samples": [meta.n_samples],
                "n_features": [meta.n_features],
                "n_classes": [len(classes) if classes is not None else None],
                "train_score": [train_score],
                "features": [feats],
                "model": [pack_model(estimator, meta)],
            },
            schema=output_schema,
        )
    )


class FitModel(SinkBuffer[FitArgs, DrainState]):
    FunctionArguments: ClassVar[type] = FitArgs

    class Meta:
        name = "fit"
        description = "Fit an XGBoost estimator and store it in the model registry"
        categories = ["models", "supervised"]
        tags = {
            "vgi.result_columns_md": columns_md(_FIT_SCHEMA),
            "vgi.doc_llm": (
                "Buffers a training table, fits an XGBoost estimator (`estimator :=` one of "
                "`xgb_classifier`, `xgb_regressor`, `xgb_rf_classifier`, `xgb_rf_regressor`), and returns a "
                "one-row training summary whose `model` column is a self-contained BLOB (estimator + "
                "metadata). Name the label column with `target :=`; every other column except an optional "
                "`id :=` passthrough becomes a feature (strings/missing values handled natively). "
                "Hyperparameters go in a JSON `params :=` string. Pass `model_name :=` to also persist the "
                "model to the registry; otherwise it lives only in the returned BLOB. Feed that BLOB to "
                "`predict`/`explain`/`feature_importance` via `SET VARIABLE` + `getvariable()`."
            ),
            "vgi.doc_md": (
                "**Fit an XGBoost model** — train and return a reusable model BLOB.\n\n"
                "- Input: a training table `(SELECT ...)`; name the label with `target :=`, optionally an "
                "`id :=` passthrough\n"
                "- `estimator :=` `xgb_classifier` | `xgb_regressor` | `xgb_rf_classifier` | "
                "`xgb_rf_regressor`; hyperparameters via JSON `params :=`\n"
                "- Returns one row: `estimator`, `task`, `n_samples`/`n_features`/`n_classes`, "
                "`train_score`, `features`, and the `model` BLOB\n"
                "- `model_name :=` also persists to the registry; classification labels may be any dtype "
                "(string/int/bool) and are decoded back on predict"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM xgboost.fit("
                    "(SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target "
                    "FROM xgboost.iris()), model_name => 'iris_clf', "
                    "estimator => 'xgb_classifier', target => 'target', id => 'sample_id')"
                ),
                description="Train an XGBoost classifier on iris and store it as 'iris_clf'",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[FitArgs]) -> BindResponse:
        a = params.args
        # model_name is optional: the model is always returned as a BLOB; when a
        # name is given it is also persisted to the registry.
        if a.model_name:
            validate_name(a.model_name)
        if not a.target:
            raise ValueError("fit requires 'target' (the label column name, e.g. target := 'label')")
        # Validate estimator + hyperparameters now so errors surface at bind.
        build_estimator(a.estimator, _parse_params(a.params))
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.target not in input_schema.names:
            raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")
        return BindResponse(output_schema=_FIT_SCHEMA)

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[FitArgs]) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[FitArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        task, estimator = build_estimator(a.estimator, _parse_params(a.params))
        table = cls.buffered_table(params, input_schema_of(params))
        _fit_and_emit(
            out,
            params.output_schema,
            table=table,
            input_schema=input_schema_of(params),
            estimator_label=a.estimator,
            task=task,
            estimator=estimator,
            model_name=a.model_name,
            target=a.target,
            id_col=a.id,
            params_dict=_parse_params(a.params),
        )


# ===========================================================================
# predict
# ===========================================================================


@dataclass(slots=True, frozen=True)
class PredictArgs:
    data: Annotated[TableInput, Arg(0, doc="Table to score (must contain the model's feature columns).")]
    model_name: Annotated[
        str, Arg("model_name", default="", doc="Name of a model in the registry. Provide this OR model.")
    ]
    model: Annotated[
        bytes, Arg("model", default=b"", doc="A model BLOB (as returned by fit). Provide this OR model_name.")
    ]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through.")]
    with_proba: Annotated[
        bool, Arg("with_proba", default=False, doc="Also emit per-class probabilities (classifiers).")
    ]
    output_margin: Annotated[
        bool, Arg("output_margin", default=False, doc="Emit the raw (untransformed) margin score instead of the label.")
    ]
    pred_leaf: Annotated[
        bool,
        Arg("pred_leaf", default=False, doc="Emit the leaf index each tree assigns the row (a list per row)."),
    ]


# Loaded estimators cached per query execution to avoid reloading each batch.
_PREDICT_CACHE: dict[bytes, tuple[Any, ModelMetadata]] = {}
# Execution ids for which a version-mismatch warning was already emitted.
_VERSION_WARNED: set[bytes] = set()


class PredictModel(TableInOutGenerator[PredictArgs]):
    FunctionArguments: ClassVar[type] = PredictArgs

    class Meta:
        name = "predict"
        description = "Score a table through a stored model, emitting predictions"
        categories = ["models", "supervised", "inference"]
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [
                    ("prediction", "BIGINT or DOUBLE", "Predicted class label (classification) or value (regression)."),
                ],
                note=(
                    "If an `id` column is named, it is carried through as the first column. The middle column "
                    "varies with the prediction mode: default is `prediction`; `output_margin := true` emits a "
                    "`margin` DOUBLE; `pred_leaf := true` emits a `leaf` INTEGER[] (one leaf index per tree). "
                    "With `with_proba := true` on a classifier, one `proba_<class>` DOUBLE column is added per class."
                ),
            ),
            "vgi.doc_llm": (
                "Streams a table through an already-fit XGBoost model and emits its predictions. Identify "
                "the model with either `model_name :=` (a registry name) or `model :=` (a BLOB from `fit`, "
                "passed via `SET VARIABLE` + `getvariable()` since a table function has only one subquery "
                "slot). Features are matched by name (order-independent; extra columns ignored; missing "
                "ones error at bind). The default output is the decoded `prediction` (the original label "
                "dtype for classifiers, value for regressors); the mutually exclusive modes `with_proba`, "
                "`output_margin`, and `pred_leaf` switch to per-class probabilities, the raw margin, or "
                "per-tree leaf indices respectively. Name an `id :=` column to carry it through."
            ),
            "vgi.doc_md": (
                "**Predict with a stored model** — score a table row by row.\n\n"
                "- Identify the model with `model_name :=` *or* `model :=` (a `fit` BLOB)\n"
                "- Features aligned by name; an optional `id :=` is carried through\n"
                "- Default output: `prediction` (label or value)\n"
                "- `with_proba := true` → one `proba_<class>` column per class; `output_margin := true` → "
                "a `margin` DOUBLE; `pred_leaf := true` → a `leaf` INTEGER[] (one index per tree) — the "
                "three modes are mutually exclusive"
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM xgboost.predict((SELECT * FROM xgboost.iris()), "
                    "model_name => 'iris_clf', id => 'sample_id')"
                ),
                description="Predict with the stored 'iris_clf' model",
            )
        ]

    @classmethod
    def _proba_classes(cls, meta: ModelMetadata) -> list[Any]:
        return meta.classes or []

    @classmethod
    def on_bind(cls, params: BindParams[PredictArgs]) -> BindResponse:
        a = params.args
        if not a.model_name and not a.model:
            raise ValueError("predict requires either 'model_name' (a registry name) or 'model' (a model BLOB)")
        if a.with_proba and (a.output_margin or a.pred_leaf):
            raise ValueError("with_proba cannot be combined with output_margin or pred_leaf")
        if a.output_margin and a.pred_leaf:
            raise ValueError("output_margin and pred_leaf are mutually exclusive")
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.model_name:
            try:
                meta = get_store().load_meta(a.model_name)
            except ModelNotFoundError as exc:
                raise ValueError(f"model {a.model_name!r} not found in the registry") from exc
        else:
            meta = unpack_meta(a.model)

        # Fail fast at bind if the input is missing any feature the model needs.
        # (predict selects features by name, so order doesn't matter and extra
        # columns are ignored — only missing ones are an error.)
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
        if a.pred_leaf:
            fields.append(
                sfield("leaf", pa.list_(pa.int32()), "Leaf index this row reaches in each tree.", nullable=False)
            )
        elif a.output_margin:
            fields.append(sfield("margin", pa.float64(), "Raw (untransformed) margin score.", nullable=False))
        else:
            fields.append(_prediction_field(meta.task, meta.classes))
        if a.with_proba and meta.task == CLASSIFICATION:
            for c in cls._proba_classes(meta):
                fields.append(sfield(f"proba_{c}", pa.float64(), f"P(class = {c}).", nullable=False))
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def _model(cls, params: ProcessParams[PredictArgs]) -> tuple[Any, ModelMetadata]:
        assert params.init_response is not None
        key = params.init_response.execution_id
        cached = _PREDICT_CACHE.get(key)
        if cached is None:
            a = params.args
            cached = get_store().load(a.model_name) if a.model_name else unpack_model(a.model)
            _PREDICT_CACHE[key] = cached
        return cached

    @classmethod
    def process(
        cls,
        params: ProcessParams[PredictArgs],
        state: None,
        batch: pa.RecordBatch,
        out: InOutCollector,
    ) -> None:
        a = params.args
        estimator, meta = cls._model(params)

        assert params.init_response is not None
        key = params.init_response.execution_id
        if meta.xgboost_version and meta.xgboost_version != xgboost.__version__ and key not in _VERSION_WARNED:
            _VERSION_WARNED.add(key)
            with contextlib.suppress(Exception):
                out.client_log(
                    Level.WARN,
                    f"model {(a.model_name or '<blob>')!r} was fitted with xgboost {meta.xgboost_version}, "
                    f"worker has {xgboost.__version__}; predictions may differ",
                )

        cat_mask = meta.categorical or [False] * len(meta.feature_names)
        x = build_x_predict(pa.Table.from_batches([batch]), meta.feature_names, cat_mask, meta.categories)

        columns: dict[str, list[Any]] = {}
        if a.id:
            columns[a.id] = batch.column(a.id).to_pylist()

        if a.pred_leaf:
            # pred_leaf lives on the Booster, not the scikit-learn wrapper's predict().
            booster = estimator.get_booster()
            dmat = xgboost.DMatrix(x, enable_categorical=True)
            leaves = np.atleast_2d(booster.predict(dmat, pred_leaf=True))
            columns["leaf"] = [[int(v) for v in row] for row in leaves]
        elif a.output_margin:
            margin = estimator.predict(x, output_margin=True)
            margin = np.asarray(margin)
            # multiclass margin is 2D; collapse to the chosen class's margin for a scalar column
            if margin.ndim > 1:
                margin = margin.max(axis=1)
            columns["margin"] = [float(v) for v in margin]
        else:
            preds = estimator.predict(x)
            if meta.task == CLASSIFICATION:
                columns["prediction"] = _decode_labels(preds, meta.classes)
            else:
                columns["prediction"] = [float(v) for v in preds]

        if a.with_proba and meta.task == CLASSIFICATION:
            proba = estimator.predict_proba(x)
            for j, c in enumerate(cls._proba_classes(meta)):
                columns[f"proba_{c}"] = [float(v) for v in proba[:, j]]

        out.emit(pa.RecordBatch.from_pydict(columns, schema=params.output_schema))


# ===========================================================================
# cross_val_predict (no persistence)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class CrossValArgs:
    data: Annotated[TableInput, Arg(0, doc="Training table (features + target [+ id]).")]
    estimator: Annotated[str, Arg("estimator", default="xgb_classifier", doc="Estimator name.")]
    target: Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to carry through.")]
    params: Annotated[str, Arg("params", default="", doc="JSON object of hyperparameters.")]
    cv: Annotated[int, Arg("cv", default=5, doc="Number of cross-validation folds.")]


class CrossValPredict(SinkBuffer[CrossValArgs, DrainState]):
    FunctionArguments: ClassVar[type] = CrossValArgs

    class Meta:
        name = "cross_val_predict"
        description = "Out-of-fold cross-validated predictions (no model is stored)"
        categories = ["models", "supervised", "evaluation"]
        tags = {
            "vgi.result_columns_md": columns_md_rows(
                [
                    (
                        "prediction",
                        "BIGINT or DOUBLE",
                        "Out-of-fold predicted class label (classification) or value (regression).",
                    ),
                ],
                note="If an `id` column is named, it is carried through as the first column.",
            ),
            "vgi.doc_llm": (
                "Computes out-of-fold cross-validated predictions for a training table without persisting "
                "any model. Each row is predicted by a model trained on the other folds (`cv :=` folds, "
                "default 5), so the result is an honest, leakage-free prediction per row — ideal for "
                "building a held-out prediction column to score with metrics or to stack. Name the label "
                "with `target :=`, pick the `estimator :=`, optionally carry an `id :=` through, and tune "
                "via JSON `params :=`. Classification labels of any dtype are decoded back to the original "
                "values."
            ),
            "vgi.doc_md": (
                "**Cross-validated out-of-fold predictions** — no model is stored.\n\n"
                "- Each row is predicted by a model fit on the *other* folds (`cv :=`, default 5)\n"
                "- `estimator :=`, `target :=`, optional `id :=` passthrough, JSON `params :=`\n"
                "- Returns one `prediction` per input row (label or value), leakage-free\n\n"
                "Use it to make a held-out prediction column for metrics or model stacking."
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT * FROM xgboost.cross_val_predict("
                    "(SELECT sample_id, sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target "
                    "FROM xgboost.iris()), estimator => 'xgb_classifier', target => 'target', id => 'sample_id')"
                ),
                description="5-fold out-of-fold predictions on iris",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[CrossValArgs]) -> BindResponse:
        a = params.args
        if not a.target:
            raise ValueError("cross_val_predict requires 'target' (the label column name, e.g. target := 'label')")
        task, _ = build_estimator(a.estimator, _parse_params(a.params))
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.target not in input_schema.names:
            raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")
        fields: list[pa.Field] = []
        if a.id:
            fields.append(input_schema.field(a.id))
        # For classification the label dtype isn't known until the data is read,
        # so the cross_val_predict output column is typed from the target column's
        # Arrow type (string -> VARCHAR, otherwise BIGINT).
        if task == CLASSIFICATION:
            target_type = input_schema.field(a.target).type
            label_type = (
                pa.string()
                if (pa.types.is_string(target_type) or pa.types.is_large_string(target_type))
                else pa.int64()
            )
            fields.append(sfield("prediction", label_type, "Predicted class label.", nullable=False))
        else:
            fields.append(_prediction_field(task))
        return BindResponse(output_schema=pa.schema(fields))

    @classmethod
    def initial_finalize_state(cls, finalize_state_id: bytes, params: TableBufferingParams[CrossValArgs]) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[CrossValArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        input_schema = input_schema_of(params)
        feats = _features_excluding(input_schema, a.target, a.id)
        task, estimator = build_estimator(a.estimator, _parse_params(a.params))

        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            out.emit(
                pa.RecordBatch.from_pydict({n: [] for n in params.output_schema.names}, schema=params.output_schema)
            )
            return

        x, y, _cat_mask, _categories, classes = _xy(table, feats, a.target, task)
        preds = sk_cross_val_predict(estimator, x, y, cv=a.cv)

        columns: dict[str, list[Any]] = {}
        if a.id:
            columns[a.id] = table.column(a.id).to_pylist()
        columns["prediction"] = _decode_labels(preds, classes) if task == CLASSIFICATION else [float(v) for v in preds]
        out.emit(pa.RecordBatch.from_pydict(columns, schema=params.output_schema))


# ===========================================================================
# cross_val_score (per-fold held-out scores, no persistence)
# ===========================================================================


@dataclass(slots=True, frozen=True)
class CrossValScoreArgs:
    data: Annotated[TableInput, Arg(0, doc="Training table (features + target [+ id]).")]
    estimator: Annotated[str, Arg("estimator", default="xgb_classifier", doc="Estimator name.")]
    target: Annotated[str, Arg("target", default="", doc="Name of the target/label column (required).")]
    id: Annotated[str, Arg("id", default="", doc="Optional id column to exclude from features.")]
    params: Annotated[str, Arg("params", default="", doc="JSON object of hyperparameters.")]
    cv: Annotated[int, Arg("cv", default=5, doc="Number of cross-validation folds.")]
    scoring: Annotated[str, Arg("scoring", default="", doc="Scorer name (default: the estimator's own scorer).")]


_CV_SCORE_SCHEMA = pa.schema(
    [
        sfield("fold", pa.int64(), "Cross-validation fold index (0-based).", nullable=False),
        sfield("score", pa.float64(), "Held-out score for this fold.", nullable=False),
    ]
)


class CrossValScore(SinkBuffer[CrossValScoreArgs, DrainState]):
    FunctionArguments: ClassVar[type] = CrossValScoreArgs

    class Meta:
        name = "cross_val_score"
        description = "Cross-validated held-out scores, one row per fold (no model is stored)"
        categories = ["models", "supervised", "evaluation"]
        tags = {
            "vgi.result_columns_md": columns_md(_CV_SCORE_SCHEMA),
            "vgi.doc_llm": (
                "Runs k-fold cross-validation and returns the held-out score for each fold (one `(fold, "
                "score)` row), without storing any model. Name the label with `target :=`, choose the "
                "`estimator :=`, set the number of folds with `cv :=` (default 5), and optionally pick a "
                "`scoring :=` metric (default: the estimator's own scorer — accuracy for classifiers, R^2 "
                "for regressors). Aggregate the rows (e.g. `avg(score)`) for a single cross-validated "
                "performance estimate or to compare estimators/hyperparameters."
            ),
            "vgi.doc_md": (
                "**Cross-validated fold scores** — one row per fold, no model stored.\n\n"
                "- `estimator :=`, `target :=`, `cv :=` folds (default 5), optional `scoring :=`\n"
                "- Returns `(fold, score)` — the held-out score for each fold\n\n"
                "Default scorer is the estimator's own (accuracy / R^2). Take `avg(score)` for a single "
                "cross-validated estimate."
            ),
        }
        examples = [
            FunctionExample(
                sql=(
                    "SELECT fold, score FROM xgboost.cross_val_score("
                    "(SELECT sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, target "
                    "FROM xgboost.iris()), estimator => 'xgb_classifier', target => 'target')"
                ),
                description="5-fold accuracy of an XGBoost classifier on iris",
            )
        ]

    @classmethod
    def on_bind(cls, params: BindParams[CrossValScoreArgs]) -> BindResponse:
        a = params.args
        if not a.target:
            raise ValueError("cross_val_score requires 'target' (the label column name, e.g. target := 'label')")
        build_estimator(a.estimator, _parse_params(a.params))
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.target not in input_schema.names:
            raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")
        return BindResponse(output_schema=_CV_SCORE_SCHEMA)

    @classmethod
    def initial_finalize_state(
        cls, finalize_state_id: bytes, params: TableBufferingParams[CrossValScoreArgs]
    ) -> DrainState:
        return DrainState()

    @classmethod
    def finalize(
        cls,
        params: TableBufferingParams[CrossValScoreArgs],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True

        a = params.args
        input_schema = input_schema_of(params)
        feats = _features_excluding(input_schema, a.target, a.id)
        task, estimator = build_estimator(a.estimator, _parse_params(a.params))

        table = cls.buffered_table(params, input_schema)
        if table is None or table.num_rows == 0:
            raise ValueError("cross_val_score received no training rows")

        x, y, _cat_mask, _categories, _classes = _xy(table, feats, a.target, task)
        scores = sk_cross_val_score(estimator, x, y, cv=a.cv, scoring=(a.scoring or None))
        out.emit(
            pa.RecordBatch.from_pydict(
                {"fold": list(range(len(scores))), "score": [float(s) for s in scores]},
                schema=params.output_schema,
            )
        )


# ===========================================================================
# Registry management: list_models / model_info / drop_model
# ===========================================================================

_MODEL_INFO_SCHEMA = pa.schema(
    [
        sfield("model_name", pa.string(), "Stored model name.", nullable=False),
        sfield("estimator", pa.string(), "Estimator type.", nullable=False),
        sfield("task", pa.string(), "classification or regression.", nullable=False),
        sfield("target", pa.string(), "Target column the model was trained on.", nullable=False),
        sfield("n_features", pa.int64(), "Number of features.", nullable=False),
        sfield("n_samples", pa.int64(), "Number of training rows.", nullable=False),
        sfield("n_classes", pa.int64(), "Number of classes (NULL for regression)."),
        sfield("train_score", pa.float64(), "In-sample training score."),
        sfield("xgboost_version", pa.string(), "xgboost version used to fit."),
        sfield("created_at", pa.string(), "UTC timestamp the model was stored."),
        sfield("features", pa.list_(pa.string()), "Ordered feature column names.", nullable=False),
    ]
)


def _meta_rows(metas: list[ModelMetadata]) -> dict[str, list[Any]]:
    return {
        "model_name": [m.name for m in metas],
        "estimator": [m.estimator for m in metas],
        "task": [m.task for m in metas],
        "target": [m.target for m in metas],
        "n_features": [m.n_features for m in metas],
        "n_samples": [m.n_samples for m in metas],
        "n_classes": [len(m.classes) if m.classes is not None else None for m in metas],
        "train_score": [m.train_score for m in metas],
        "xgboost_version": [m.xgboost_version for m in metas],
        "created_at": [m.created_at for m in metas],
        "features": [m.feature_names for m in metas],
    }


@dataclass(slots=True, frozen=True)
class NoArgs:
    pass


@init_single_worker
@bind_fixed_schema
class ListModels(TableFunctionGenerator[NoArgs]):
    FIXED_SCHEMA: ClassVar[pa.Schema] = _MODEL_INFO_SCHEMA

    class Meta:
        name = "list_models"
        description = "List all models in the registry"
        categories = ["models", "registry"]
        tags = {
            "vgi.result_columns_md": columns_md(_MODEL_INFO_SCHEMA),
            "vgi.doc_llm": (
                "Zero-argument table function listing every model persisted to the registry (by `fit` / "
                "`fit_<estimator>` / search with a `model_name`). One row per model: `model_name`, "
                "`estimator`, `task`, `target`, training shape (`n_features`/`n_samples`/`n_classes`), "
                "`train_score`, `xgboost_version`, `created_at`, and the ordered `features` list. Query it "
                "to discover what is available to `predict`/`explain`/`feature_importance`."
            ),
            "vgi.doc_md": (
                "**Model registry listing** — every saved model, one row each.\n\n"
                "- `model_name`, `estimator`, `task`, `target`\n"
                "- `n_features` / `n_samples` / `n_classes`, `train_score`\n"
                "- `xgboost_version`, `created_at`, `features`\n\n"
                "Takes no arguments. Use it to find models to score with `predict`."
            ),
        }
        examples = [FunctionExample(sql="SELECT * FROM xgboost.list_models()", description="List stored models")]

    @classmethod
    def cardinality(cls, params: BindParams[NoArgs]) -> TableCardinality:
        return TableCardinality(estimate=10, max=10000)

    @classmethod
    def process(cls, params: ProcessParams[NoArgs], state: None, out: OutputCollector) -> None:
        out.emit(pa.RecordBatch.from_pydict(_meta_rows(get_store().list()), schema=params.output_schema))
        out.finish()


@dataclass(slots=True, frozen=True)
class ModelInfoArgs:
    model_name: Annotated[str, Arg(0, doc="Name of a stored model.")]


@init_single_worker
@bind_fixed_schema
class ModelInfo(TableFunctionGenerator[ModelInfoArgs]):
    FIXED_SCHEMA: ClassVar[pa.Schema] = _MODEL_INFO_SCHEMA

    class Meta:
        name = "model_info"
        description = "Describe a single stored model (one row, empty if absent)"
        categories = ["models", "registry"]
        tags = {
            "vgi.result_columns_md": columns_md(_MODEL_INFO_SCHEMA),
            "vgi.doc_llm": (
                "Returns the registry metadata for one model named positionally (`model_info('my_model')`): "
                "a single row with `model_name`, `estimator`, `task`, `target`, training shape "
                "(`n_features`/`n_samples`/`n_classes`), `train_score`, `xgboost_version`, `created_at`, "
                "and the ordered `features` list. Emits zero rows if the model does not exist, so it never "
                "errors on a missing name. Use it to inspect a specific saved model before predicting."
            ),
            "vgi.doc_md": (
                "**Describe one stored model** — its registry metadata.\n\n"
                "- Call positionally: `model_info('my_model')`\n"
                "- One row: `estimator`, `task`, `target`, shape, `train_score`, `xgboost_version`, "
                "`created_at`, `features`\n\n"
                "Returns no rows if the name is absent (never errors)."
            ),
        }
        examples = [
            FunctionExample(sql="SELECT * FROM xgboost.model_info('iris_clf')", description="Show one model's metadata")
        ]

    @classmethod
    def cardinality(cls, params: BindParams[ModelInfoArgs]) -> TableCardinality:
        return TableCardinality(estimate=1, max=1)

    @classmethod
    def process(cls, params: ProcessParams[ModelInfoArgs], state: None, out: OutputCollector) -> None:
        try:
            metas = [get_store().load_meta(params.args.model_name)]
        except ModelNotFoundError:
            metas = []
        out.emit(pa.RecordBatch.from_pydict(_meta_rows(metas), schema=params.output_schema))
        out.finish()


@dataclass(slots=True, frozen=True)
class DropModelArgs:
    model_name: Annotated[str, Arg(0, doc="Name of the model to delete.")]


_DROP_SCHEMA = pa.schema(
    [
        sfield("model_name", pa.string(), "Name of the model.", nullable=False),
        sfield("dropped", pa.bool_(), "True if a model was deleted, False if it did not exist.", nullable=False),
    ]
)


@init_single_worker
@bind_fixed_schema
class DropModel(TableFunctionGenerator[DropModelArgs]):
    FIXED_SCHEMA: ClassVar[pa.Schema] = _DROP_SCHEMA

    class Meta:
        name = "drop_model"
        description = "Delete a model from the registry"
        categories = ["models", "registry"]
        tags = {
            "vgi.result_columns_md": columns_md(_DROP_SCHEMA),
            "vgi.doc_llm": (
                "Deletes a model from the registry by name (`drop_model('my_model')`) and returns one row "
                "`(model_name, dropped)` where `dropped` is true when a model was removed and false when "
                "the name did not exist. Idempotent and safe to call on an absent model. Use it to clean "
                "up models created by `fit`/`fit_<estimator>`/search with a `model_name`."
            ),
            "vgi.doc_md": (
                "**Delete a stored model** — registry cleanup.\n\n"
                "- Call positionally: `drop_model('my_model')`\n"
                "- Returns `(model_name, dropped)`; `dropped` is false if the name did not exist\n\n"
                "Idempotent — safe on a missing model."
            ),
        }
        examples = [
            FunctionExample(sql="SELECT * FROM xgboost.drop_model('iris_clf')", description="Delete a stored model")
        ]

    @classmethod
    def cardinality(cls, params: BindParams[DropModelArgs]) -> TableCardinality:
        return TableCardinality(estimate=1, max=1)

    @classmethod
    def process(cls, params: ProcessParams[DropModelArgs], state: None, out: OutputCollector) -> None:
        name = params.args.model_name
        dropped = get_store().delete(name)
        out.emit(pa.RecordBatch.from_pydict({"model_name": [name], "dropped": [dropped]}, schema=params.output_schema))
        out.finish()


MODEL_FUNCTIONS: list[type] = [
    FitModel,
    PredictModel,
    CrossValPredict,
    CrossValScore,
    ListModels,
    ModelInfo,
    DropModel,
]
