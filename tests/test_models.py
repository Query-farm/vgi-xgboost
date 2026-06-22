"""Unit tests for the model registry and estimator catalog.

The full fit -> predict -> list -> drop lifecycle is covered end-to-end by
test/sql/xgboost_models.test; here we test the storage backend and helpers.
"""

from __future__ import annotations

import numpy as np
import pytest
from xgboost import XGBClassifier

from vgi_xgboost.models import _parse_params, build_estimator
from vgi_xgboost.registry import (
    LocalDiskStore,
    ModelMetadata,
    ModelNameError,
    ModelNotFoundError,
    validate_name,
)

_TRAIN_X = np.array([[0.0], [1.0], [0.0], [1.0], [0.2], [0.9]])
_TRAIN_Y = np.array([0, 1, 0, 1, 0, 1])


def _fitted() -> XGBClassifier:
    return XGBClassifier(n_estimators=5, random_state=0).fit(_TRAIN_X, _TRAIN_Y)


def _meta(name: str = "m") -> ModelMetadata:
    return ModelMetadata(
        name=name,
        estimator="xgb_classifier",
        task="classification",
        target="y",
        feature_names=["a"],
        classes=[0, 1],
        n_samples=4,
        n_features=1,
        train_score=1.0,
        xgboost_version="x",
        created_at="now",
    )


class TestLocalDiskStore:
    def test_roundtrip(self, tmp_path) -> None:
        original = _fitted()
        store = LocalDiskStore(tmp_path)
        store.save(original, _meta())
        assert store.exists("m")
        est, meta = store.load("m")
        assert meta.name == "m"
        assert meta.feature_names == ["a"]
        assert meta.classes == [0, 1]
        # the reloaded model predicts identically to the one that was saved
        assert np.array_equal(est.predict(_TRAIN_X), original.predict(_TRAIN_X))

    def test_list(self, tmp_path) -> None:
        store = LocalDiskStore(tmp_path)
        store.save(_fitted(), _meta("a"))
        store.save(_fitted(), _meta("b"))
        assert sorted(m.name for m in store.list()) == ["a", "b"]

    def test_delete(self, tmp_path) -> None:
        store = LocalDiskStore(tmp_path)
        store.save(_fitted(), _meta())
        assert store.delete("m") is True
        assert store.delete("m") is False
        assert not store.exists("m")

    def test_load_missing_raises(self, tmp_path) -> None:
        with pytest.raises(ModelNotFoundError):
            LocalDiskStore(tmp_path).load("nope")


class TestValidateName:
    def test_accepts_reasonable(self) -> None:
        assert validate_name("iris_clf-1.2") == "iris_clf-1.2"

    @pytest.mark.parametrize("bad", ["", "../etc", "a/b", ".hidden", "with space"])
    def test_rejects_unsafe(self, bad: str) -> None:
        with pytest.raises(ModelNameError):
            validate_name(bad)


class TestEstimatorCatalog:
    def test_build_with_params(self) -> None:
        task, est = build_estimator("xgb_classifier", {"n_estimators": 7})
        assert task == "classification"
        assert est.n_estimators == 7

    def test_regression_task(self) -> None:
        task, est = build_estimator("xgb_regressor", {})
        assert task == "regression"

    def test_unknown_estimator(self) -> None:
        with pytest.raises(ValueError, match="unknown estimator"):
            build_estimator("does_not_exist", {})

    def test_unknown_hyperparameter(self) -> None:
        with pytest.raises(ValueError, match="unknown hyperparameter"):
            build_estimator("xgb_classifier", {"nonsense": 5})


class TestParseParams:
    def test_empty(self) -> None:
        assert _parse_params("") == {}
        assert _parse_params("   ") == {}

    def test_json_object(self) -> None:
        assert _parse_params('{"n_estimators": 200, "max_depth": 4}') == {"n_estimators": 200, "max_depth": 4}

    def test_non_object_rejected(self) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            _parse_params("[1, 2, 3]")
