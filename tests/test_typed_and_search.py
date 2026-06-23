"""Unit tests for typed fit functions, the model BLOB, and hyperparameter search.

The full RPC path is covered by the SQL suite; here we test the generation /
translation helpers and the registry BLOB round-trip directly.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from xgboost import XGBClassifier

from vgi_xgboost.models import _ESTIMATORS, build_estimator
from vgi_xgboost.registry import ModelMetadata, pack_model, unpack_meta, unpack_model
from vgi_xgboost.search import _GRID_UNION, _grid_size, _member_struct, _param_grid
from vgi_xgboost.typed_models import _HPARAMS, TYPED_FIT_FUNCTIONS, _estimator_kwargs, _make_args_class


def _fitted() -> XGBClassifier:
    x = np.array([[0.0], [1.0], [0.0], [1.0], [0.2], [0.9]])
    y = np.array([0, 1, 0, 1, 0, 1])
    return XGBClassifier(n_estimators=5, random_state=0, enable_categorical=True, tree_method="hist").fit(x, y)


def _meta() -> ModelMetadata:
    return ModelMetadata(
        name="m",
        estimator="xgb_classifier",
        task="classification",
        target="y",
        feature_names=["a"],
        categorical=[False],
        categories=[None],
        classes=[0, 1],
        n_samples=6,
        n_features=1,
        train_score=1.0,
        xgboost_version="x",
        created_at="now",
    )


class TestModelBlob:
    def test_roundtrip(self) -> None:
        est = _fitted()
        blob = pack_model(est, _meta())
        # metadata can be read cheaply without loading the estimator
        meta = unpack_meta(blob)
        assert meta.estimator == "xgb_classifier"
        assert meta.feature_names == ["a"]
        # the estimator reloads and predicts identically to the original
        reloaded, meta2 = unpack_model(blob)
        x = np.array([[0.0], [1.0]])
        assert np.array_equal(reloaded.predict(x), est.predict(x))
        assert meta2.classes == [0, 1]

    def test_truncated_blob_rejected(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            unpack_meta(b"\x00\x01")


class TestTypedFit:
    def test_one_function_per_estimator(self) -> None:
        assert len(TYPED_FIT_FUNCTIONS) == len(_HPARAMS)
        names = {f.Meta.name for f in TYPED_FIT_FUNCTIONS}
        assert names == {f"fit_{e}" for e in _HPARAMS}

    def test_every_typed_param_is_valid_for_its_estimator(self) -> None:
        # Guards against exposing a hyperparameter XGBoost doesn't actually have.
        for est_name, spec in _HPARAMS.items():
            _task, cls, _defaults = _ESTIMATORS[est_name]
            valid = set(cls().get_params().keys())
            for hp in spec:
                assert (hp.kwarg or hp.name) in valid, f"{est_name}.{hp.name} not a real param"

    def test_defaults_are_xgboost_real_defaults(self) -> None:
        # The typed defaults are XGBoost's documented defaults (no magic sentinels).
        spec = {hp.name: hp for hp in _HPARAMS["xgb_classifier"]}
        assert spec["n_estimators"].default == 100
        assert spec["max_depth"].default == 6
        assert spec["learning_rate"].default == 0.3
        assert spec["subsample"].default == 1.0
        assert spec["colsample_bytree"].default == 1.0
        assert spec["min_child_weight"].default == 1.0
        assert spec["gamma"].default == 0.0
        assert spec["reg_alpha"].default == 0.0
        assert spec["reg_lambda"].default == 1.0

    def test_default_args_forward_real_values(self) -> None:
        spec = _HPARAMS["xgb_classifier"]
        args_cls = _make_args_class("xgb_classifier", spec)
        args = args_cls(data=None)  # type: ignore[call-arg]
        kw = _estimator_kwargs(spec, args)
        # numeric knobs are always forwarded at their real default (no sentinel drop)
        assert kw["n_estimators"] == 100
        assert kw["max_depth"] == 6
        assert kw["learning_rate"] == 0.3
        # objective/booster '' still means "let the task/library decide"
        assert "objective" not in kw
        assert "booster" not in kw
        assert kw["tree_method"] == "hist"

    def test_default_fit_matches_bare_estimator(self) -> None:
        spec = _HPARAMS["xgb_classifier"]
        args_cls = _make_args_class("xgb_classifier", spec)
        kw = _estimator_kwargs(spec, args_cls(data=None))  # type: ignore[call-arg]
        _task, est_cls, defaults = _ESTIMATORS["xgb_classifier"]
        typed = est_cls(**{**defaults, **kw})
        bare = est_cls(**defaults)
        x = np.array([[0.0], [1.0], [0.0], [1.0], [0.2], [0.9]])
        y = np.array([0, 1, 0, 1, 0, 1])
        assert np.array_equal(typed.fit(x, y).predict(x), bare.fit(x, y).predict(x))

    def test_explicit_values_passed_through(self) -> None:
        spec = _HPARAMS["xgb_classifier"]
        args_cls = _make_args_class("xgb_classifier", spec)
        args = args_cls(data=None, n_estimators=300, max_depth=4, learning_rate=0.1)  # type: ignore[call-arg]
        kw = _estimator_kwargs(spec, args)
        assert kw["n_estimators"] == 300
        assert kw["max_depth"] == 4
        assert kw["learning_rate"] == 0.1


class TestGridUnionType:
    def test_member_per_estimator(self) -> None:
        names = [_GRID_UNION.field(i).name for i in range(_GRID_UNION.num_fields)]
        assert set(names) == set(_HPARAMS)

    def test_members_are_structs_of_lists(self) -> None:
        idx = next(i for i in range(_GRID_UNION.num_fields) if _GRID_UNION.field(i).name == "xgb_classifier")
        member = _GRID_UNION.field(idx).type
        ne = member.field(member.get_field_index("n_estimators")).type
        assert pa.types.is_list(ne)
        assert pa.types.is_integer(ne.value_type)
        lr = member.field(member.get_field_index("learning_rate")).type
        assert pa.types.is_floating(lr.value_type)

    def test_member_struct_matches_hparams(self) -> None:
        struct = _member_struct(_HPARAMS["xgb_classifier"])
        assert [struct.field(i).name for i in range(struct.num_fields)] == [
            hp.name for hp in _HPARAMS["xgb_classifier"]
        ]


class TestParamGrid:
    def test_only_listed_params_searched(self) -> None:
        grid = _param_grid("xgb_classifier", {"n_estimators": [50, 100], "max_depth": None})
        assert grid == {"n_estimators": [50, 100]}  # max_depth (None) omitted -> estimator default

    def test_values_pass_through(self) -> None:
        # No sentinels left to translate: values forward straight through.
        grid = _param_grid("xgb_classifier", {"max_depth": [3, 5, 8]})
        assert grid["max_depth"] == [3, 5, 8]

    def test_none_if_drops_empty_objective(self) -> None:
        # objective '' (its none_if) is dropped element-wise.
        grid = _param_grid("xgb_classifier", {"objective": ["", "binary:logistic"]})
        assert grid["objective"] == ["binary:logistic"]

    def test_empty_value_is_empty_grid(self) -> None:
        assert _param_grid("xgb_classifier", None) == {}


class TestSearchHelpers:
    def test_grid_size(self) -> None:
        assert _grid_size({"a": [1, 2, 3], "b": [4, 5]}) == 6

    def test_empty_grid_is_one(self) -> None:
        assert _grid_size({}) == 1


class TestNativeSerializationDefaults:
    def test_common_defaults_enable_categorical(self) -> None:
        _task, est = build_estimator("xgb_classifier", {})
        params = est.get_params()
        assert params["enable_categorical"] is True
        assert params["tree_method"] == "hist"
