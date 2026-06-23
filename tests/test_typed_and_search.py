"""Unit tests for typed fit functions, the model BLOB, and hyperparameter search.

The full RPC path is covered by the SQL suite; here we test the generation /
translation helpers and the registry BLOB round-trip directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from xgboost import XGBClassifier

from vgi_xgboost.models import _ESTIMATORS, build_estimator
from vgi_xgboost.registry import ModelMetadata, pack_model, unpack_meta, unpack_model
from vgi_xgboost.search import _grid_size, _parse_grid
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

    def test_sentinels_drop_to_defaults(self) -> None:
        spec = _HPARAMS["xgb_classifier"]
        args_cls = _make_args_class("xgb_classifier", spec)
        # all-default args: every none_if sentinel should be dropped, leaving only tree_method + random_state
        args = args_cls(data=None)  # type: ignore[call-arg]
        kw = _estimator_kwargs(spec, args)
        assert "n_estimators" not in kw  # 0 sentinel dropped
        assert kw["tree_method"] == "hist"
        assert kw["random_state"] == 0

    def test_explicit_values_passed_through(self) -> None:
        spec = _HPARAMS["xgb_classifier"]
        args_cls = _make_args_class("xgb_classifier", spec)
        args = args_cls(data=None, n_estimators=300, max_depth=6, learning_rate=0.1)  # type: ignore[call-arg]
        kw = _estimator_kwargs(spec, args)
        assert kw["n_estimators"] == 300
        assert kw["max_depth"] == 6
        assert kw["learning_rate"] == 0.1


class TestSearchHelpers:
    def test_parse_grid_wraps_scalars(self) -> None:
        assert _parse_grid('{"n_estimators": [50, 100], "max_depth": 3}') == {
            "n_estimators": [50, 100],
            "max_depth": [3],
        }

    def test_parse_grid_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="required"):
            _parse_grid("")
        with pytest.raises(ValueError, match="non-empty JSON object"):
            _parse_grid("[]")

    def test_grid_size(self) -> None:
        assert _grid_size({"a": [1, 2, 3], "b": [4, 5]}) == 6


class TestNativeSerializationDefaults:
    def test_common_defaults_enable_categorical(self) -> None:
        _task, est = build_estimator("xgb_classifier", {})
        params = est.get_params()
        assert params["enable_categorical"] is True
        assert params["tree_method"] == "hist"
