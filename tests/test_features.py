"""Unit tests for native categorical + missing-value feature assembly."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa

from vgi_xgboost.features import build_x_fit, build_x_predict, categorical_mask


def _table() -> pa.Table:
    return pa.table(
        {
            "num": pa.array([1.0, 2.0, None, 4.0], type=pa.float64()),
            "cat": pa.array(["x", "y", "x", "z"], type=pa.string()),
        }
    )


class TestCategoricalMask:
    def test_detects_string_columns(self) -> None:
        table = _table()
        mask = categorical_mask([table.schema.field(n).type for n in ["num", "cat"]])
        assert mask == [False, True]

    def test_dictionary_is_categorical(self) -> None:
        assert categorical_mask([pa.dictionary(pa.int8(), pa.string())]) == [True]


class TestBuildX:
    def test_fit_captures_categories_and_preserves_nan(self) -> None:
        table = _table()
        x, categories = build_x_fit(table, ["num", "cat"], [False, True])
        assert list(categories[0] or []) == []  # numeric -> None
        assert categories[0] is None
        assert sorted(categories[1] or []) == ["x", "y", "z"]
        # NaN is preserved as the missing marker for the numeric column
        assert math.isnan(float(x["num"].iloc[2]))
        assert str(x["cat"].dtype) == "category"

    def test_predict_maps_unseen_category_to_nan(self) -> None:
        table = _table()
        _x, categories = build_x_fit(table, ["num", "cat"], [False, True])
        score = pa.table({"num": pa.array([5.0]), "cat": pa.array(["UNSEEN"])})
        xp = build_x_predict(score, ["num", "cat"], [False, True], categories)
        # the unseen category becomes NaN (treated as missing), not an error
        assert xp["cat"].isna().all()

    def test_decimal_is_numeric(self) -> None:
        table = pa.table({"d": pa.array([1.5, 2.5]).cast(pa.decimal128(4, 1))})
        x, categories = build_x_fit(table, ["d"], [False])
        assert categories == [None]
        assert np.allclose(x["d"].to_numpy(), [1.5, 2.5])
