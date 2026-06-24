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
# data_version_spec is advertised as a SemVer *range* (a packaging SpecifierSet),
# not a bare version. The worker regenerates its data each release, so it serves
# exactly the current data version — an exact-match range.
DATA_VERSION_SPEC = f"=={DATA_VERSION}"
GIT_COMMIT = os.environ.get("VGI_XGBOOST_GIT_COMMIT") or "unknown"

# Every callable the worker exposes, grouped by area.
_FUNCTIONS: list[type] = [
    *DATASET_FUNCTIONS,
    *MODEL_FUNCTIONS,
    *TYPED_FIT_FUNCTIONS,
    *SEARCH_FUNCTIONS,
    *IMPORTANCE_FUNCTIONS,
]

# Provenance / about link advertised on the catalog (VGI source_url).
SOURCE_URL = "https://github.com/query-farm/vgi-xgboost"

# Catalog-level metadata surfaced through duckdb_databases() (comment + tags).
# The description_llm/_md tags feed agent/doc consumers; author/copyright/license
# advertise provenance.
_CATALOG_COMMENT = "XGBoost train/predict model registry, datasets, and interpretation for DuckDB/SQL"
# Catalog-level description: the high-level "what this worker is".
_CATALOG_DESCRIPTION_LLM = (
    "XGBoost for SQL. Load datasets; fit gradient-boosted models (fit returns a "
    "model BLOB, predict aligns features by name and decodes string labels); run "
    "typed fit_<estimator>, grid and randomized hyperparameter search, feature "
    "importance, SHAP explanations, partial dependence, and permutation importance "
    "— all as DuckDB table functions."
)
_CATALOG_DESCRIPTION_MD = (
    "# XGBoost for SQL\n\n"
    "Exposes [XGBoost](https://xgboost.ai) to DuckDB/SQL as VGI functions:\n\n"
    "- **Datasets** — toy datasets and generators\n"
    "- **Models** — `fit`/`predict`, typed `fit_<estimator>`, cross-validation, "
    "grid/randomized search\n"
    "- **Interpretation** — feature importance, SHAP `explain`, partial dependence, "
    "permutation importance\n\n"
    "Models are stored as reusable BLOBs in a registry; native UBJSON serialization."
)
# Schema-level description: an index of what is callable in the `main` namespace.
_SCHEMA_DESCRIPTION_LLM = (
    "Functions in xgboost.main, by family: datasets (table functions); models "
    "(fit → model BLOB → predict, typed fit_<estimator>, cross-validation, grid and "
    "randomized search); and interpretation (feature importance, SHAP explain, "
    "partial dependence, permutation importance). All are DuckDB table functions."
)
_SCHEMA_DESCRIPTION_MD = (
    "# `main` schema\n\n"
    "Every XGBoost function lives here, grouped by family:\n\n"
    "- **datasets** — toy/generated data as table functions\n"
    "- **models** — `fit`/`predict`, typed `fit_<estimator>`, cross-validation, "
    "grid/randomized search\n"
    "- **interpretation** — feature importance, SHAP `explain`, partial dependence, "
    "permutation importance\n\n"
    "Fit returns a reusable model BLOB; predict aligns features by name."
)
_CATALOG_TAGS = {
    "vgi.description_llm": _CATALOG_DESCRIPTION_LLM,
    "vgi.description_md": _CATALOG_DESCRIPTION_MD,
    "vgi.author": "Query Farm <hello@query.farm>",
    "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
    "vgi.license": "MIT",
    "vgi.support_contact": f"{SOURCE_URL}/issues",
    "vgi.support_policy_url": f"{SOURCE_URL}/blob/main/SUPPORT.md",
}

_XGBOOST_CATALOG = Catalog(
    name="xgboost",
    default_schema="main",
    comment=_CATALOG_COMMENT,
    tags=_CATALOG_TAGS,
    schemas=[
        Schema(
            name="main",
            comment="XGBoost train/predict model registry, datasets, and interpretation for SQL",
            tags={
                "provider": "XGBoost",
                "domain": "machine-learning",
                "vgi.description_llm": _SCHEMA_DESCRIPTION_LLM,
                "vgi.description_md": _SCHEMA_DESCRIPTION_MD,
            },
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
                data_version_spec=DATA_VERSION_SPEC,
                source_url=SOURCE_URL,
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
