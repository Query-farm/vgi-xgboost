"""Feature-matrix assembly with native categorical + missing-value support.

XGBoost's edge over plain numeric models is that it handles **categorical
features** (no one-hot needed) and **missing values** natively. To use that we
must feed XGBoost a pandas ``DataFrame`` whose categorical columns carry the
``category`` dtype (with ``enable_categorical=True``), and let NaN flow through
as the missing marker.

This module is the single place that turns an Arrow table into that DataFrame:

* ``categorical_mask`` -- which feature columns are categorical (string/dict
  Arrow types). Mirrors the per-feature mask stored in ``ModelMetadata``.
* ``build_x_fit``      -- assemble the training ``X`` and capture the ordered
  category list of each categorical column (stored in metadata).
* ``build_x_predict``  -- rebuild ``X`` for scoring using the *training*
  categories, so column dtypes line up and **unseen categories map to NaN**
  (treated as missing) instead of raising.

A model with no string columns produces an all-numeric DataFrame, exactly the
plain path; categorical support is therefore zero-cost when unused.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.types import CategoricalDtype


def is_categorical_type(t: pa.DataType) -> bool:
    """True for Arrow types we treat as categorical features (string / dictionary)."""
    if pa.types.is_dictionary(t):
        return True
    return pa.types.is_string(t) or pa.types.is_large_string(t)


def categorical_mask(types: list[pa.DataType]) -> list[bool]:
    """Per-feature categorical flag for the given (ordered) Arrow column types."""
    return [is_categorical_type(t) for t in types]


def _numeric_column(table: pa.Table, name: str) -> np.ndarray:
    field = table.schema.field(name)
    if not (
        pa.types.is_floating(field.type)
        or pa.types.is_integer(field.type)
        or pa.types.is_boolean(field.type)
        or pa.types.is_decimal(field.type)
    ):
        raise ValueError(
            f"feature column {name!r} ({field.type}) is neither numeric nor a "
            "categorical (string) column; select numeric/string features only"
        )
    column = table.column(name)
    if pa.types.is_decimal(field.type):
        column = column.cast(pa.float64())
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=float)


def build_x_fit(
    table: pa.Table, feature_names: list[str], cat_mask: list[bool]
) -> tuple[pd.DataFrame, list[list[str] | None]]:
    """Build the training ``X`` DataFrame and capture each categorical column's categories.

    Returns ``(X, categories)`` where ``categories[i]`` is the ordered category
    list for feature ``i`` (``None`` for numeric features). Numeric/boolean
    columns become float64 (NaN preserved as missing); string columns become an
    ordered ``category`` dtype.
    """
    data: dict[str, Any] = {}
    categories: list[list[str] | None] = []
    for name, is_cat in zip(feature_names, cat_mask, strict=True):
        if is_cat:
            values = [None if v is None else str(v) for v in table.column(name).to_pylist()]
            series = pd.Series(values, dtype="object").astype("category")
            data[name] = series
            categories.append([str(c) for c in series.cat.categories])
        else:
            data[name] = _numeric_column(table, name)
            categories.append(None)
    return pd.DataFrame(data), categories


def build_x_predict(
    table: pa.Table,
    feature_names: list[str],
    cat_mask: list[bool],
    categories: list[list[str] | None] | None,
) -> pd.DataFrame:
    """Rebuild ``X`` for scoring, aligned to the training categories.

    Categorical columns are coerced to the *training* category set, so unseen
    values (and NULLs) become NaN -- which XGBoost treats as missing -- instead
    of raising. This makes ``predict`` / ``explain`` robust to new categories.
    """
    cats_by_idx = categories or [None] * len(feature_names)
    data: dict[str, Any] = {}
    for i, (name, is_cat) in enumerate(zip(feature_names, cat_mask, strict=True)):
        if is_cat:
            train_cats = cats_by_idx[i]
            if train_cats is not None:
                allowed = set(train_cats)
                # Map unseen values (and NULLs) to None up front so the resulting
                # Categorical has no out-of-dtype entries (avoids a pandas warning)
                # while still landing them as NaN == missing for XGBoost.
                raw = table.column(name).to_pylist()
                values = [str(v) if (v is not None and str(v) in allowed) else None for v in raw]
                data[name] = pd.Series(values, dtype="object").astype(CategoricalDtype(categories=train_cats))
            else:
                values = [None if v is None else str(v) for v in table.column(name).to_pylist()]
                data[name] = pd.Series(values, dtype="object").astype("category")
        else:
            data[name] = _numeric_column(table, name)
    return pd.DataFrame(data)
