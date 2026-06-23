# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http,oauth]>=0.8.3",
#     "vgi-rpc[sentry]>=0.20.4",
#     "xgboost>=2.0",
#     "scikit-learn>=1.5",
#     "numpy",
#     "pandas",
# ]
# ///
"""VGI worker exposing XGBoost to DuckDB/SQL.

Assembles the per-area implementation modules in ``vgi_xgboost`` into a single
``xgboost`` catalog and runs the worker over stdio (local) or HTTP (Fly.io).

Usage:
    uv run xgboost_worker.py            # serve over stdio (DuckDB subprocess)
    python serve.py --port 8000         # serve over HTTP

    ATTACH 'xgboost' (TYPE vgi, LOCATION 'uv run xgboost_worker.py');
    SELECT * FROM xgboost.fit((SELECT * FROM xgboost.iris()), model_name => 'm', target => 'target');
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, ReadOnlyCatalogInterface, Schema
from vgi.catalog.catalog_interface import CatalogAttachResult, CatalogInfo

from vgi_xgboost import __version__
from vgi_xgboost.datasets import DATASET_FUNCTIONS
from vgi_xgboost.importance import IMPORTANCE_FUNCTIONS
from vgi_xgboost.models import MODEL_FUNCTIONS
from vgi_xgboost.search import SEARCH_FUNCTIONS
from vgi_xgboost.typed_models import TYPED_FIT_FUNCTIONS

log = logging.getLogger(__name__)

DATA_VERSION = __version__
GIT_COMMIT = os.environ.get("VGI_XGBOOST_GIT_COMMIT") or "unknown"

# Every callable the worker exposes, grouped by area.
_FUNCTIONS: list[type] = [
    *DATASET_FUNCTIONS,
    *MODEL_FUNCTIONS,
    *TYPED_FIT_FUNCTIONS,
    *SEARCH_FUNCTIONS,
    *IMPORTANCE_FUNCTIONS,
]

_XGBOOST_CATALOG = Catalog(
    name="xgboost",
    default_schema="main",
    schemas=[
        Schema(
            name="main",
            comment="XGBoost train/predict model registry, datasets, and interpretation for SQL",
            functions=list(_FUNCTIONS),
        ),
    ],
)


class XGBoostCatalog(ReadOnlyCatalogInterface):
    """Advertises the worker's data + implementation version on ATTACH."""

    catalog = _XGBOOST_CATALOG
    catalog_name = _XGBOOST_CATALOG.name

    def catalogs(self) -> list[CatalogInfo]:
        return [
            CatalogInfo(
                name=self._effective_catalog_name,
                implementation_version=GIT_COMMIT,
                data_version_spec=DATA_VERSION,
                attach_option_specs=[spec.serialize() for spec in self.attach_option_specs],
            )
        ]

    def catalog_attach(self, **kwargs: Any) -> CatalogAttachResult:
        result = super().catalog_attach(**kwargs)
        return dataclasses.replace(
            result,
            resolved_data_version=DATA_VERSION,
            resolved_implementation_version=GIT_COMMIT,
        )


class XGBoostWorker(Worker):
    """Worker process hosting the XGBoost catalog."""

    catalog = _XGBOOST_CATALOG
    catalog_interface = XGBoostCatalog


def main() -> None:
    """Run the XGBoost worker process (stdio or, via flags, HTTP)."""
    XGBoostWorker.main()


if __name__ == "__main__":
    main()
