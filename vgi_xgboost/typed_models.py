"""Typed per-estimator fit functions: ``xgboost.fit_<estimator>(...)``.

These wrap the generic ``fit`` with XGBoost's common hyperparameters exposed as
**native, typed SQL named arguments** -- so they appear in the catalog and
DuckDB's autocomplete, are type-checked, and are discoverable without consulting
docs:

    SELECT * FROM xgboost.fit_xgb_classifier(
      (SELECT * FROM training), model_name := 'm', target := 'y',
      n_estimators := 300, max_depth := 6, learning_rate := 0.1);

Each function behaves exactly like ``fit``: it returns the training summary plus
the model as a BLOB, and persists to the registry when ``model_name`` is given.
The generic ``fit`` (JSON ``params``) remains the escape hatch for hyperparameters
not surfaced here. The curated set per estimator is XGBoost's most-tuned knobs --
see ``_HPARAMS`` below.

Every typed argument defaults to XGBoost's own documented default, so the values
shown in the catalog are truthful and any value is settable literally (no magic
sentinels).
"""

from __future__ import annotations

import types
from dataclasses import dataclass, make_dataclass
from dataclasses import field as dc_field
from typing import Annotated, Any

from vgi.arguments import Arg, TableInput
from vgi.invocation import BindResponse
from vgi.metadata import FunctionExample
from vgi.table_buffering_function import OutputCollector, TableBufferingParams
from vgi.table_function import BindParams

from .buffering import DrainState, SinkBuffer, input_schema_of
from .models import _ESTIMATORS, _FIT_SCHEMA, _fit_and_emit
from .registry import validate_name
from .schema_utils import columns_md

_UNSET: Any = object()

# All typed fit_<estimator> functions share the generic fit result schema.
_FIT_COLUMNS_MD = columns_md(_FIT_SCHEMA)


@dataclass(frozen=True)
class _HP:
    """One typed hyperparameter exposed as a SQL named argument."""

    name: str
    type: type
    default: Any
    doc: str
    none_if: Any = _UNSET  # if the SQL value equals this, omit the kwarg so XGBoost uses its own default
    kwarg: str | None = None  # XGBoost kwarg name, if it differs from ``name``


# XGBoost's most-tuned hyperparameters. Each defaults to XGBoost's own documented
# default, so the values shown in the catalog are truthful and always forwarded.
# ``objective`` / ``booster`` keep a ``none_if=""`` so the empty string (their SQL
# default) means "let the task/library decide" rather than passing an empty value.
_TREE_BOOSTER = [
    _HP("n_estimators", int, 100, "Number of boosting rounds."),
    _HP("max_depth", int, 6, "Max tree depth."),
    _HP("learning_rate", float, 0.3, "Boosting learning rate / eta."),
    _HP("subsample", float, 1.0, "Row subsample ratio per tree."),
    _HP("colsample_bytree", float, 1.0, "Column subsample ratio per tree."),
    _HP("min_child_weight", float, 1.0, "Min sum of instance weight in a child."),
    _HP("gamma", float, 0.0, "Min loss reduction to split."),
    _HP("reg_alpha", float, 0.0, "L1 regularization on weights."),
    _HP("reg_lambda", float, 1.0, "L2 regularization on weights."),
    _HP("objective", str, "", "Learning objective (e.g. 'binary:logistic'); '' = task default.", none_if=""),
    _HP("booster", str, "", "Booster: 'gbtree', 'gblinear', or 'dart' ('' = default).", none_if=""),
    _HP("tree_method", str, "hist", "Tree construction algorithm ('hist', 'approx', 'exact')."),
    _HP("random_state", int, 0, "Random seed."),
]

_HPARAMS: dict[str, list[_HP]] = {
    "xgb_classifier": _TREE_BOOSTER,
    "xgb_regressor": _TREE_BOOSTER,
    "xgb_rf_classifier": _TREE_BOOSTER,
    "xgb_rf_regressor": _TREE_BOOSTER,
}

# Per-estimator prose for the typed fit_<estimator> functions: a one-line task
# summary woven into the rich doc tags below so each generated function gets
# distinct, estimator-specific documentation rather than a templated string.
_ESTIMATOR_DOC: dict[str, tuple[str, str]] = {
    "xgb_classifier": (
        "gradient-boosted decision-tree **classifier** (XGBoost's `XGBClassifier`)",
        "classification",
    ),
    "xgb_regressor": (
        "gradient-boosted decision-tree **regressor** (XGBoost's `XGBRegressor`)",
        "regression",
    ),
    "xgb_rf_classifier": (
        "**random-forest classifier** built on XGBoost (`XGBRFClassifier`) — a single boosting round of bagged trees",
        "classification",
    ),
    "xgb_rf_regressor": (
        "**random-forest regressor** built on XGBoost (`XGBRFRegressor`) — a single boosting round of bagged trees",
        "regression",
    ),
}


def _typed_doc_tags(est_name: str) -> dict[str, str]:
    """Build distinct, estimator-specific ``vgi.doc_llm`` / ``vgi.doc_md`` tags."""
    blurb, task = _ESTIMATOR_DOC[est_name]
    doc_llm = (
        f"Buffers a training table and fits a {blurb} with its key hyperparameters exposed as typed, "
        f"named SQL arguments (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, "
        f"`colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`, ...), each "
        f"defaulting to XGBoost's own documented default. Equivalent to "
        f"`fit(estimator := '{est_name}', ...)` but discoverable and type-checked in the catalog instead "
        f"of a JSON `params` blob. Name the {task} label with `target :=`, optionally carry an `id :=` "
        f"through; every other column is a feature (strings/missing values handled natively). Returns the "
        f"same one-row summary with a reusable `model` BLOB, and persists to the registry when "
        f"`model_name :=` is given. Feed the BLOB to `predict`/`explain`/`feature_importance`."
    )
    doc_md = (
        f"**`fit_{est_name}`** — fit a {blurb} with typed hyperparameters.\n\n"
        f"- Input: a training table `(SELECT ...)`; name the label with `target :=`, optional `id :=` "
        f"passthrough\n"
        f"- Hyperparameters are named, typed SQL args (`n_estimators`, `max_depth`, `learning_rate`, "
        f"`subsample`, `reg_lambda`, ...) at XGBoost's real defaults\n"
        f"- Returns the standard fit summary plus a reusable `model` BLOB; `model_name :=` also persists "
        f"it\n\n"
        f"The discoverable, type-checked alternative to `fit(estimator := '{est_name}', ...)`."
    )
    return {"vgi.result_columns_md": _FIT_COLUMNS_MD, "vgi.doc_llm": doc_llm, "vgi.doc_md": doc_md}


def _estimator_kwargs(spec: list[_HP], args: Any) -> dict[str, Any]:
    """Translate the typed SQL args into XGBoost estimator kwargs, dropping sentinels."""
    kw: dict[str, Any] = {}
    for hp in spec:
        v = getattr(args, hp.name)
        if hp.none_if is not _UNSET and v == hp.none_if:
            continue  # leave the hyperparameter at XGBoost's default
        kw[hp.kwarg or hp.name] = v
    return kw


def _make_args_class(est_name: str, spec: list[_HP]) -> type:
    fields: list[Any] = [
        ("data", Annotated[TableInput, Arg(0, doc="Training table (features + target [+ id]).")]),
        (
            "model_name",
            Annotated[
                str,
                Arg("model_name", default="", doc="Optional registry name; the model is always returned as a BLOB."),
            ],
            dc_field(default=""),
        ),
        (
            "target",
            Annotated[str, Arg("target", default="", doc="Label column name (required).")],
            dc_field(default=""),
        ),
        ("id", Annotated[str, Arg("id", default="", doc="Optional id passthrough column.")], dc_field(default="")),
    ]
    for hp in spec:
        fields.append(
            (hp.name, Annotated[hp.type, Arg(hp.name, default=hp.default, doc=hp.doc)], dc_field(default=hp.default))
        )
    cls_name = "Fit" + "".join(p.title() for p in est_name.split("_")) + "Args"
    return make_dataclass(cls_name, fields, frozen=True, slots=True)


def _make_fit_function(est_name: str) -> type:
    task, est_cls, defaults = _ESTIMATORS[est_name]
    spec = _HPARAMS[est_name]
    args_cls = _make_args_class(est_name, spec)
    fn_name = f"fit_{est_name}"
    param_hint = "n_estimators := 200, max_depth := 6"

    def on_bind(cls: type, params: BindParams[Any]) -> BindResponse:
        a = params.args
        if not a.target:
            raise ValueError(f"{fn_name} requires 'target' (the label column name, e.g. target := 'label')")
        if a.model_name:
            validate_name(a.model_name)
        input_schema = params.bind_call.input_schema
        assert input_schema is not None
        if a.target not in input_schema.names:
            raise ValueError(f"target column {a.target!r} not found in input; columns: {', '.join(input_schema.names)}")
        return BindResponse(output_schema=_FIT_SCHEMA)

    def initial_finalize_state(cls: type, finalize_state_id: bytes, params: TableBufferingParams[Any]) -> DrainState:
        return DrainState()

    def finalize(
        cls: type,
        params: TableBufferingParams[Any],
        finalize_state_id: bytes,
        state: DrainState,
        out: OutputCollector,
    ) -> None:
        if state.done:
            out.finish()
            return
        state.done = True
        a = params.args
        kwargs = _estimator_kwargs(spec, a)
        estimator = est_cls(**{**defaults, **kwargs})
        table = cls.buffered_table(params, input_schema_of(params))  # type: ignore[attr-defined]
        _fit_and_emit(
            out,
            params.output_schema,
            table=table,
            input_schema=input_schema_of(params),
            estimator_label=est_name,
            task=task,
            estimator=estimator,
            model_name=a.model_name,
            target=a.target,
            id_col=a.id,
            params_dict=kwargs,
        )

    meta = type(
        "Meta",
        (),
        {
            "name": fn_name,
            "description": f"Fit a {est_name} with typed hyperparameters; returns/stores the model",
            "categories": ["models", "supervised", "typed"],
            "tags": _typed_doc_tags(est_name),
            "examples": [
                FunctionExample(
                    sql=(
                        f"SELECT * FROM xgboost.{fn_name}((SELECT * FROM xgboost.iris()), "
                        f"target := 'target', id := 'sample_id', {param_hint})"
                    ),
                    description=f"Train a {est_name} with named hyperparameters",
                )
            ],
        },
    )
    namespace = {
        "FunctionArguments": args_cls,
        "Meta": meta,
        "on_bind": classmethod(on_bind),
        "initial_finalize_state": classmethod(initial_finalize_state),
        "finalize": classmethod(finalize),
    }
    cls_name = "Fit" + "".join(p.title() for p in est_name.split("_"))
    base = SinkBuffer[args_cls, DrainState]  # type: ignore[valid-type]
    return types.new_class(cls_name, (base,), {}, lambda ns: ns.update(namespace))


TYPED_FIT_FUNCTIONS: list[type] = [_make_fit_function(name) for name in _HPARAMS]
