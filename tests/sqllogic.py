"""A minimal sqllogictest runner for the ``test/sql/*.test`` files.

The authoritative runner is DuckDB's own ``unittest`` binary built with the VGI
extension (``make test-stdio`` / ``test-http``). That binary doesn't exist on a
stock CI runner, so this module re-runs the *same* ``.test`` files in-process
against Query Farm's ``haybarn`` DuckDB distribution (which can
``INSTALL vgi FROM community``). It supports only the subset of the sqllogictest
format these files use:

* ``require-env NAME``      -- assert an env var is set
* ``require EXT``           -- ensure a DuckDB extension is loaded (best effort)
* ``statement ok``         -- run SQL, expect success
* ``statement error`` + ``----`` + substring -- run SQL, expect the error to contain it
* ``query <types>`` + ``----`` + rows         -- run SQL, compare tab-separated rows

``${VAR}`` placeholders in SQL are substituted from the environment. Results are
compared with no reordering (the suite only uses single-row / deterministic
queries), matching DuckDB's default ``nosort`` mode.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


class SqlLogicError(AssertionError):
    """A .test assertion (statement/query) did not match the expected result."""


def _subst(text: str) -> str:
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), text)


def _fmt(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


@dataclass
class _Record:
    header: str
    body: list[str]  # lines after the header, up to the blank-line separator


def _records(lines: list[str]) -> list[_Record]:
    """Split a .test file into records separated by blank lines (comments dropped)."""
    out: list[_Record] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        block: list[str] = []
        while i < len(lines) and lines[i].strip():
            if not lines[i].lstrip().startswith("#"):
                block.append(lines[i])
            i += 1
        out.append(_Record(header=block[0].strip(), body=block[1:]))
    return out


def _split_expected(body: list[str]) -> tuple[str, list[str] | None]:
    """Return (sql, expected_lines) splitting on a ``----`` separator if present."""
    if "----" in body:
        sep = body.index("----")
        return "\n".join(body[:sep]), body[sep + 1 :]
    return "\n".join(body), None


def _execute(con: object, sql: str) -> list[tuple]:
    cur = con.execute(_subst(sql))  # type: ignore[attr-defined]
    try:
        return cur.fetchall()
    except Exception:
        return []


def run_test_file(path: str, con: object) -> None:
    """Execute one .test file against an open haybarn/duckdb connection."""
    with open(path) as fh:
        lines = fh.read().splitlines()

    for rec in _records(lines):
        head = rec.header
        if head.startswith("require-env"):
            name = head.split()[1]
            if name not in os.environ:
                raise SqlLogicError(f"{path}: require-env {name} not set")
        elif head.startswith("require"):
            ext = head.split()[1]
            for stmt in (f"LOAD {ext}", f"INSTALL {ext} FROM community", f"LOAD {ext}"):
                try:
                    con.execute(stmt)  # type: ignore[attr-defined]
                except Exception:
                    continue
        elif head == "statement ok":
            sql, _ = _split_expected(rec.body)
            try:
                _execute(con, sql)
            except Exception as exc:  # noqa: BLE001
                raise SqlLogicError(f"{path}: expected success but got error: {exc}\nSQL: {sql}") from exc
        elif head == "statement error":
            sql, expected = _split_expected(rec.body)
            needle = "\n".join(expected or []).strip()
            try:
                _execute(con, sql)
            except Exception as exc:  # noqa: BLE001
                if needle and needle not in str(exc):
                    raise SqlLogicError(f"{path}: error did not contain {needle!r}; got: {exc}") from exc
            else:
                raise SqlLogicError(f"{path}: expected an error containing {needle!r} but the statement succeeded")
        elif head.startswith("query"):
            sql, expected = _split_expected(rec.body)
            rows = _execute(con, sql)
            actual = "\n".join("\t".join(_fmt(v) for v in row) for row in rows)
            want = "\n".join(expected or []).strip()
            if actual.strip() != want:
                raise SqlLogicError(
                    f"{path}: query mismatch\nSQL: {sql}\n--- expected ---\n{want}\n--- actual ---\n{actual}"
                )
        # other directives (halt, mode, hash-threshold, ...) are not used here
