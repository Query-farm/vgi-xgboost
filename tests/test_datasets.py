"""Unit tests for the dataset table functions (in-process harness)."""

from __future__ import annotations

import pyarrow as pa

from tests.harness import invoke_table_function
from vgi_xgboost.datasets import (
    IrisFunction,
    MakeClassificationFunction,
    MakeRegressionFunction,
)


class TestToyDatasets:
    def test_iris_shape_and_schema(self) -> None:
        table = invoke_table_function(IrisFunction)
        assert table.num_rows == 150
        assert "sample_id" in table.schema.names
        assert "target" in table.schema.names
        assert "target_name" in table.schema.names
        # 4 iris measurements + sample_id + target + target_name
        assert table.num_columns == 7

    def test_iris_three_classes(self) -> None:
        table = invoke_table_function(IrisFunction)
        assert set(table.column("target").to_pylist()) == {0, 1, 2}


class TestGenerators:
    def test_make_classification_shape(self) -> None:
        table = invoke_table_function(
            MakeClassificationFunction,
            named={
                "n_samples": pa.scalar(80),
                "n_features": pa.scalar(5),
                "n_classes": pa.scalar(3),
            },
        )
        assert table.num_rows == 80
        # 5 features + sample_id + target
        assert table.num_columns == 7
        assert set(table.column("target").to_pylist()) == {0, 1, 2}

    def test_make_regression_target_is_float(self) -> None:
        table = invoke_table_function(
            MakeRegressionFunction,
            named={"n_samples": pa.scalar(50), "n_features": pa.scalar(4)},
        )
        assert table.num_rows == 50
        assert pa.types.is_floating(table.schema.field("target").type)
