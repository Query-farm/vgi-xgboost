"""Run the authoritative ``test/sql/*.test`` files in-process via haybarn.

This is the CI-portable form of the SQL integration suite: it loads the ``vgi``
community extension into Query Farm's ``haybarn`` DuckDB build, attaches the
worker over stdio, and replays the same ``.test`` files the local ``unittest``
runner uses. Opt-in via ``VGI_SQL_HAYBARN=1`` (set in CI) so the fast offline
``make pytest`` does not spawn workers or hit the network.
"""

from __future__ import annotations

import glob
import os
import sys

import pytest

if not os.environ.get("VGI_SQL_HAYBARN"):
    pytest.skip("set VGI_SQL_HAYBARN=1 to run the haybarn SQL suite", allow_module_level=True)

haybarn = pytest.importorskip("haybarn")

from tests.sqllogic import run_test_file  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST_FILES = sorted(glob.glob(os.path.join(_REPO_ROOT, "test", "sql", "*.test")))


@pytest.fixture(scope="session", autouse=True)
def _worker_env(tmp_path_factory) -> None:
    # Point the worker at the same interpreter running pytest (deps already
    # installed) and an isolated registry so the suite is hermetic.
    os.environ["VGI_XGBOOST_WORKER"] = f"{sys.executable} xgboost_worker.py"
    os.environ["XGBOOST_MODELS_DIR"] = str(tmp_path_factory.mktemp("models"))


@pytest.mark.parametrize("test_file", _TEST_FILES, ids=lambda p: os.path.basename(p))
def test_sql_file(test_file: str) -> None:
    con = haybarn.connect()
    con.execute("INSTALL vgi FROM community")
    con.execute("LOAD vgi")
    try:
        run_test_file(test_file, con)
    finally:
        con.close()
