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


def _sql_type(t: pa.DataType) -> str:
    """Map a common Arrow type to a readable DuckDB-ish type name."""
    if pa.types.is_int64(t):
        return "BIGINT"
    if pa.types.is_int32(t):
        return "INTEGER"
    if pa.types.is_int16(t):
        return "SMALLINT"
    if pa.types.is_int8(t):
        return "TINYINT"
    if pa.types.is_float64(t):
        return "DOUBLE"
    if pa.types.is_float32(t):
        return "FLOAT"
    if pa.types.is_string(t):
        return "VARCHAR"
    if pa.types.is_boolean(t):
        return "BOOLEAN"
    if pa.types.is_binary(t):
        return "BLOB"
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return f"{_sql_type(t.value_type)}[]"
    return str(t)


def columns_md(schema: pa.Schema, *, note: str | None = None) -> str:
    """Render a Markdown table of a table function's RETURN columns for the
    ``vgi.columns_md`` tag. DuckDB cannot expose a VGI table-function schema,
    so this documents it. Reads each field's ``comment`` metadata."""
    lines = ["| Column | Type | Description |", "| --- | --- | --- |"]
    for f in schema:
        desc = ""
        if f.metadata and b"comment" in f.metadata:
            desc = f.metadata[b"comment"].decode("utf-8")
        lines.append(f"| `{f.name}` | {_sql_type(f.type)} | {desc} |")
    md = "\n".join(lines)
    if note:
        md += f"\n\n{note}"
    return md


def columns_md_rows(rows: list[tuple[str, str, str]], *, note: str | None = None) -> str:
    """Same Markdown table, from explicit (column, type, description) rows --
    for functions whose output schema is computed dynamically at bind."""
    lines = ["| Column | Type | Description |", "| --- | --- | --- |"]
    lines += [f"| `{n}` | {t} | {d} |" for n, t, d in rows]
    md = "\n".join(lines)
    if note:
        md += f"\n\n{note}"
    return md


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
