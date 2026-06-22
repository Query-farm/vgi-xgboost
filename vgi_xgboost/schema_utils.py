"""Shared Arrow-schema helpers for the XGBoost worker.

Keeps column-comment plumbing and name sanitisation in one place so every
dataset/model function exposes consistent, documented schemas to DuckDB.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pyarrow as pa


def field(
    name: str,
    type: pa.DataType,
    comment: str,
    *,
    nullable: bool = True,
) -> pa.Field:
    """Build a ``pa.Field`` carrying a column comment in its metadata.

    The ``comment`` metadata key is the framework's transport for column
    comments -- DuckDB surfaces it via ``duckdb_columns()`` and ``DESCRIBE``.
    """
    return pa.field(
        name,
        type,
        nullable=nullable,
        metadata={b"comment": comment.encode("utf-8")},
    )


_NON_IDENT = re.compile(r"[^0-9a-z]+")


def snake_case(name: str) -> str:
    """Normalise a feature label to a SQL-friendly column name.

    ``"sepal length (cm)"`` -> ``"sepal_length_cm"``. Collapses any run of
    non-alphanumeric characters to a single underscore and lowercases.
    """
    cleaned = _NON_IDENT.sub("_", name.strip().lower()).strip("_")
    if not cleaned:
        return "feature"
    if cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    return cleaned


def dedupe_names(names: list[str]) -> list[str]:
    """Ensure column names are unique by suffixing collisions (``_2``, ``_3`` ...)."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
    return out


@dataclass(slots=True, frozen=True, kw_only=True)
class NoArgs:
    """Empty argument set for functions that take no user-facing parameters."""
