# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http,oauth]>=0.50.0",
#     "vgi-rpc[sentry]>=0.53.1",
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
import json
import logging
import os
from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, ReadOnlyCatalogInterface, Schema, Table, constraint_record
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
    "model `BLOB`, predict aligns features by name and decodes string labels); run "
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
    "(fit → model `BLOB` → predict, typed fit_<estimator>, cross-validation, grid and "
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
    "Fit returns a reusable model `BLOB`; predict aligns features by name."
)
# Guaranteed-runnable, self-contained examples advertised on the catalog
# (VGI509): each is fully schema-qualified and executes as written against a
# freshly attached worker. Multi-statement examples run in order in one session,
# so a model can be fit, stashed in a session variable (the model BLOB can't ride
# a table function's single subquery slot), and then used by predict. Kept fast
# (small/default params, cv := 3) so each runs in well under ~30s.
_CATALOG_EXECUTABLE_EXAMPLES = json.dumps(
    [
        {
            "description": "Load the built-in iris dataset.",
            "sql": "SELECT * FROM xgboost.main.iris LIMIT 5",
        },
        {
            "description": "Fit an XGBoost classifier on iris, then predict with the fitted model.",
            "sql": [
                (
                    "SET VARIABLE iris_model = ("
                    "SELECT model FROM xgboost.main.fit((SELECT * FROM xgboost.main.iris), "
                    "estimator := 'xgb_classifier', target := 'target', id := 'sample_id'))"
                ),
                (
                    "SELECT sample_id, prediction "
                    "FROM xgboost.main.predict((SELECT * FROM xgboost.main.iris), "
                    "model := getvariable('iris_model'), id := 'sample_id') LIMIT 5"
                ),
            ],
        },
        {
            "description": "Three-fold cross-validated accuracy of an XGBoost classifier on iris.",
            "sql": (
                "SELECT fold, score FROM xgboost.main.cross_val_score("
                "(SELECT * EXCLUDE (target_name) FROM xgboost.main.iris), "
                "estimator := 'xgb_classifier', target := 'target', cv := 3)"
            ),
        },
        {
            "description": "Grid-search an XGBoost classifier on iris over a tiny grid (3-fold).",
            "sql": (
                "SELECT params, mean_test_score, rank FROM xgboost.main.grid_search("
                "(SELECT * EXCLUDE (target_name) FROM xgboost.main.iris), "
                "target := 'target', cv := 3, "
                "estimator := union_value(xgb_classifier := {'n_estimators': [25, 50], 'max_depth': [3]})) "
                "ORDER BY rank"
            ),
        },
    ]
)
# Fixed agent-suitability suite run by `vgi-lint simulate`. Only each task's
# name and the `prompt` shown to the simulated analyst are published here; the
# hidden graders (`reference_sql`) live in vgi-agent-tests.yaml in the repo
# (VGI416), so the catalog never leaks the answers. Prompts name their output
# columns because grading is strict on column names + values + order. The suite
# mixes single-call smoke tests with multi-step tasks that exercise the real
# workflow — fit -> model BLOB in a session variable -> predict / interpret.
_CATALOG_AGENT_TEST_TASKS = json.dumps(
    [
        {
            "name": "breast_cancer_cv_accuracy",
            "prompt": (
                "Before deploying a tumor-screening model, I want an honest estimate of how "
                "accurately gradient boosting distinguishes malignant from benign breast tumors "
                "on unseen data. Using the built-in Wisconsin breast-cancer dataset and all 30 "
                "cell-nucleus features, run 5-fold cross-validation with an XGBoost classifier "
                "and report the mean held-out accuracy across the folds. Return a single row with"
                " one column named mean_cv_accuracy."
            ),
        },
        {
            "name": "iris_grid_search_best_accuracy",
            "prompt": (
                "I'm tuning an XGBoost classifier on Fisher's iris dataset. Run a 5-fold cross-"
                "validated grid search over n_estimators (30, 60, 90) and max_depth (2, 4, 6), "
                "and report the best mean cross-validated accuracy across all those "
                "hyperparameter combinations. Return a single row with one column named "
                "best_cv_accuracy."
            ),
        },
        {
            "name": "diabetes_predict_rmse",
            "prompt": (
                "Train a gradient-boosted regressor on the diabetes dataset (predict target from "
                "the baseline measurements), then use the fitted model to predict on those same "
                "rows and report the root-mean-squared error between the predictions and the "
                "actual target. The predictions come back keyed by sample_id, so join them back "
                "to the data by sample_id. Round to 2 decimals and return a single row with one "
                "column named train_rmse."
            ),
        },
        {
            "name": "n_features_above_mean_gain",
            "prompt": (
                "Train a gradient-boosted regressor on the diabetes dataset, compute gain-based "
                "feature importances, and count how many features have a gain importance strictly"
                " greater than the mean gain importance across all features. Return a single row "
                "with one column named n_above_mean."
            ),
        },
        {
            "name": "wine_sample_count",
            "prompt": (
                "How many samples are in the built-in wine recognition dataset? Return a single "
                "row with one column named n."
            ),
        },
        {
            "name": "california_housing_count",
            "prompt": (
                "How many districts are in the built-in California housing dataset? Return a "
                "single row with one column named n."
            ),
        },
        {
            "name": "make_classification_rows",
            "prompt": (
                "Generate a synthetic classification dataset with 120 samples, 3 classes, and "
                "random_state 42, then report how many rows it has. Return a single row with one "
                "column named n."
            ),
        },
        {
            "name": "make_regression_rows",
            "prompt": (
                "Generate a synthetic regression dataset with 90 samples and random_state 42, "
                "then report how many rows it has. Return a single row with one column named n."
            ),
        },
        {
            "name": "fit_xgb_classifier_summary",
            "prompt": (
                "Using the typed classifier fitter, train an XGBoost classifier on the iris "
                "dataset (target column 'target', id column 'sample_id') and report the training "
                "summary's sample and feature counts. Return one row with columns named n_samples"
                " and n_features."
            ),
        },
        {
            "name": "fit_xgb_regressor_summary",
            "prompt": (
                "Using the typed regressor fitter, train an XGBoost regressor on the diabetes "
                "dataset (target column 'target', id column 'sample_id') and report the training "
                "summary's sample and feature counts. Return one row with columns named n_samples"
                " and n_features."
            ),
        },
        {
            "name": "fit_xgb_rf_classifier_summary",
            "prompt": (
                "Using the typed random-forest classifier fitter, train an XGBoost RF classifier "
                "on the iris dataset (target 'target', id 'sample_id') and report the training "
                "summary's sample and feature counts. Return one row with columns named n_samples"
                " and n_features."
            ),
        },
        {
            "name": "fit_xgb_rf_regressor_summary",
            "prompt": (
                "Using the typed random-forest regressor fitter, train an XGBoost RF regressor on"
                " the diabetes dataset (target 'target', id 'sample_id') and report the training "
                "summary's sample and feature counts. Return one row with columns named n_samples"
                " and n_features."
            ),
        },
        {
            "name": "cross_val_predict_accuracy",
            "prompt": (
                "Using 5-fold out-of-fold cross-validated predictions from an XGBoost classifier "
                "on the iris dataset (target 'target', id 'sample_id'), compute the overall "
                "accuracy of those out-of-fold predictions against the true labels (join by "
                "sample_id). Round to 4 decimals and return a single column named oof_accuracy."
            ),
        },
        {
            "name": "shap_top_feature",
            "prompt": (
                "Train an XGBoost regressor on the diabetes dataset, then use SHAP contributions "
                "(via the explain function) to identify the single most influential feature "
                "overall — the one with the largest mean absolute SHAP value. Return one row with"
                " a column named feature."
            ),
        },
        {
            "name": "bmi_partial_dependence_range",
            "prompt": (
                "Train an XGBoost regressor on the diabetes dataset, then compute a partial-"
                "dependence curve for the 'bmi' feature over a 20-point grid, and report the "
                "minimum and maximum predicted value along the curve, each rounded to 2 decimals."
                " Return one row with columns named min_pd and max_pd."
            ),
        },
        {
            "name": "top_permutation_feature",
            "prompt": (
                "Train an XGBoost regressor on the diabetes dataset, then use permutation "
                "importance (5 repeats, random_state 42) to find the single most important "
                "feature by mean importance. Return one row with a column named feature."
            ),
        },
        {
            "name": "fit_iris_train_accuracy",
            "prompt": (
                "Train an XGBoost classifier on the built-in iris dataset with the generic fit "
                "function (estimator xgb_classifier, target column 'target', id column "
                "'sample_id') and report its in-sample training accuracy rounded to 4 decimals. "
                "Return a single row with one column named train_accuracy."
            ),
        },
        {
            "name": "saved_model_info",
            "prompt": (
                "Train an XGBoost classifier on the wine dataset (target 'target', id "
                "'sample_id') and save it in the model registry under the name 'wine_clf'. Then "
                "look that saved model up in the registry and report which target column it was "
                "trained on and how many features it uses. Return one row with columns named "
                "target and n_features."
            ),
        },
        {
            "name": "saved_model_listed",
            "prompt": (
                "Train an XGBoost regressor on the diabetes dataset (target 'target', id "
                "'sample_id') and save it in the model registry as 'diab_saved'. Then list the "
                "stored models and return the registry entry for that model only. Return one row "
                "with columns named model_name, estimator and task."
            ),
        },
        {
            "name": "drop_saved_model",
            "prompt": (
                "Train an XGBoost classifier on iris (target 'target', id 'sample_id'), save it "
                "in the model registry as 'scratch_clf', then delete it from the registry and "
                "report whether a model was actually removed. Return one row with one column "
                "named dropped."
            ),
        },
        {
            "name": "iris_randomized_search_best",
            "prompt": (
                "Tune an XGBoost classifier on the iris dataset with a randomized hyperparameter "
                "search: sample 4 combinations (random seed 0) from n_estimators (25, 50, 100) "
                "and max_depth (2, 3, 4), scoring each by 3-fold cross-validation, using only the"
                " four measurement columns as features. Report the best mean cross-validated "
                "accuracy, rounded to 4 decimals. Return a single row with one column named "
                "best_cv_accuracy."
            ),
        },
    ]
)
_CATALOG_TAGS = {
    "vgi.doc_llm": _CATALOG_DESCRIPTION_LLM,
    "vgi.doc_md": _CATALOG_DESCRIPTION_MD,
    "vgi.author": "Query Farm <hello@query.farm>",
    "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
    "vgi.license": "MIT",
    "vgi.support_contact": f"{SOURCE_URL}/issues",
    "vgi.support_policy_url": f"{SOURCE_URL}/blob/main/SUPPORT.md",
    "vgi.title": "XGBoost",
    "vgi.subtitle": "Train, tune, score and explain gradient-boosted tree models on your tables with plain SQL.",
    "vgi.keywords": json.dumps(
        [
            "xgboost",
            "gradient boosting",
            "machine learning",
            "models",
            "classification",
            "regression",
            "cross-validation",
            "hyperparameter search",
            "feature importance",
            "shap",
        ]
    ),
    "vgi.executable_examples": _CATALOG_EXECUTABLE_EXAMPLES,
    "vgi.agent_test_tasks": _CATALOG_AGENT_TEST_TASKS,
}

# Per-schema metadata for the single `main` schema (VGI506/124/126): title,
# keywords (JSON array), and a runnable, schema-qualified example query.
# Navigation sections for the schema (VGI413); every function and table names
# one of these in its ``vgi.category`` (see ``_apply_discovery_tags``).
_CATEGORIES = [
    {
        "name": "datasets",
        "title": "Datasets",
        "description": "Built-in toy datasets and synthetic data generators to train on.",
    },
    {
        "name": "models",
        "title": "Models",
        "description": "Fit, predict with, cross-validate and manage stored XGBoost models.",
    },
    {
        "name": "tuning",
        "title": "Hyperparameter search",
        "description": "Cross-validated grid and randomized hyperparameter search.",
    },
    {
        "name": "interpretation",
        "title": "Interpretation",
        "description": "Feature importance, SHAP contributions and partial dependence.",
    },
]

_SCHEMA_TAGS = {
    "provider": "XGBoost",
    "domain": "machine-learning",
    "vgi.title": "XGBoost functions",
    "vgi.categories": json.dumps(_CATEGORIES),
    "vgi.doc_llm": _SCHEMA_DESCRIPTION_LLM,
    "vgi.doc_md": _SCHEMA_DESCRIPTION_MD,
    "vgi.keywords": json.dumps(
        [
            "xgboost",
            "gradient boosting",
            "models",
            "datasets",
            "interpretation",
            "cross-validation",
        ]
    ),
    "vgi.example_queries": json.dumps(
        [
            {
                "description": "Fit an XGBoost classifier on iris and return the training summary",
                "sql": (
                    "SELECT estimator, task, n_samples, n_features, train_score "
                    "FROM xgboost.main.fit((SELECT * FROM xgboost.main.iris), "
                    "estimator := 'xgb_classifier', target := 'target', id := 'sample_id')"
                ),
            }
        ]
    ),
}


def _humanize(name: str) -> str:
    """Title-case a snake_case function name for a display title."""
    return name.replace("_", " ").title()


# The navigation category (a ``_CATEGORIES`` name) for each function family.
_FAMILY: dict[type, str] = {
    **dict.fromkeys(DATASET_FUNCTIONS, "datasets"),
    **dict.fromkeys((*MODEL_FUNCTIONS, *TYPED_FIT_FUNCTIONS), "models"),
    **dict.fromkeys(SEARCH_FUNCTIONS, "tuning"),
    **dict.fromkeys(IMPORTANCE_FUNCTIONS, "interpretation"),
}


def _apply_discovery_tags(functions: list[type]) -> None:
    """Inject the per-function discovery tags the catalog-quality linter expects.

    ``vgi.title`` and ``vgi.keywords`` (a JSON array of strings) are derived
    mechanically from each function's existing Meta (display name, categories).
    ``vgi.source_url`` is deliberately NOT set here — it is a catalog-only tag
    (VGI139). The richer ``vgi.doc_llm`` / ``vgi.doc_md`` tags are authored per
    function in the implementation modules and are left untouched here.
    """
    for fn in functions:
        meta = getattr(fn, "Meta", None)
        if meta is None:
            continue
        name = getattr(meta, "name", fn.__name__)
        cats = list(getattr(meta, "categories", []) or [])
        tags = dict(getattr(meta, "tags", {}) or {})
        tags.setdefault("vgi.title", _humanize(name))
        keywords = list(dict.fromkeys(cats or name.split("_")))
        tags.setdefault("vgi.keywords", json.dumps(keywords))
        tags.setdefault("vgi.category", _FAMILY[fn])
        meta.tags = tags


_apply_discovery_tags(_FUNCTIONS)


def _is_parameterless_table_fn(fn: type) -> bool:
    """True for a table function whose argument dataclass has no fields.

    Such a function always returns the same rows, so it is also exposed as a
    plain table (VGI311) — ``SELECT * FROM schema.name`` without parentheses.
    """
    args = getattr(fn, "FunctionArguments", None)
    return args is not None and dataclasses.is_dataclass(args) and not dataclasses.fields(args)


# Table-specific metadata for the parameterless functions also exposed as tables
# (VGI311). Each carries a table-oriented description, a descriptive title (not a
# restatement of the name), a primary key, and a runnable example — distinct from
# the backing function's documentation.
_TABLE_META: dict[str, dict[str, Any]] = {
    "iris": {
        "title": "Fisher's iris flowers",
        "comment": "150 iris flowers with four sepal/petal measurements (cm) and their species.",
        "keywords": json.dumps(["iris", "flowers", "classification", "toy dataset", "fisher"]),
        "primary_key": (("sample_id",),),
        "doc_llm": (
            "Fisher's classic iris table: 150 rows, one per flower. Columns are `sample_id`, four numeric "
            "measurements in centimetres (`sepal_length_cm`, `sepal_width_cm`, `petal_length_cm`, "
            "`petal_width_cm`), an integer `target` (0/1/2), and the `target_name` species "
            "(setosa/versicolor/virginica). Read it as a plain table (no parentheses) for a "
            "balanced 3-class toy dataset to feed `fit`/`predict`."
        ),
        "doc_md": (
            "### `iris` table\n\n"
            "150 iris flowers, evenly split across three species:\n\n"
            "- `sample_id` — row id\n"
            "- four measurements in **cm** — `sepal_length_cm`, `sepal_width_cm`, `petal_length_cm`, "
            "`petal_width_cm`\n"
            "- `target` (0–2) and `target_name` (species)"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Row count per species",
                    "sql": "SELECT target_name, count(*) FROM xgboost.main.iris GROUP BY target_name",
                }
            ]
        ),
    },
    "wine": {
        "title": "Wine cultivar chemistry",
        "comment": "178 wines with 13 chemical measurements and their cultivar of origin.",
        "keywords": json.dumps(["wine", "chemistry", "classification", "toy dataset", "cultivars"]),
        "primary_key": (("sample_id",),),
        "doc_llm": (
            "The wine-recognition table: 178 rows, one per wine sample, with 13 numeric chemical-analysis "
            "features (alcohol, malic acid, ash, magnesium, total phenols, flavanoids, colour intensity, "
            "hue, proline, ...), an integer `target`, and the cultivar `target_name`. A 3-class dataset "
            "for classification over continuous features."
        ),
        "doc_md": (
            "### `wine` table\n\n"
            "178 wines from three cultivars:\n\n"
            "- `sample_id` — row id\n"
            "- 13 chemical features (alcohol, phenols, colour intensity, proline, ...)\n"
            "- `target` (0–2) and `target_name` (cultivar)"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Average alcohol per cultivar",
                    "sql": "SELECT target_name, round(avg(alcohol), 2) FROM xgboost.main.wine GROUP BY target_name",
                }
            ]
        ),
    },
    "breast_cancer": {
        "title": "Breast-cancer cell diagnostics",
        "comment": "569 tumour samples with 30 cell-nucleus measurements and a benign/malignant label.",
        "keywords": json.dumps(["breast cancer", "diagnostics", "classification", "toy dataset", "binary"]),
        "primary_key": (("sample_id",),),
        "doc_llm": (
            "The Wisconsin breast-cancer table: 569 rows, one per tumour, with 30 numeric features "
            "summarising cell-nucleus geometry (mean/standard-error/worst of radius, texture, area, "
            "concavity, ...), a binary `target` (0 = malignant, 1 = benign), and `target_name`. A standard "
            "binary-classification benchmark."
        ),
        "doc_md": (
            "### `breast_cancer` table\n\n"
            "569 tumour samples, binary outcome:\n\n"
            "- `sample_id` — row id\n"
            "- 30 cell-nucleus features (mean / SE / worst of radius, texture, area, ...)\n"
            "- `target` (0 malignant, 1 benign) and `target_name`"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Class balance",
                    "sql": "SELECT target_name, count(*) FROM xgboost.main.breast_cancer GROUP BY target_name",
                }
            ]
        ),
    },
    "diabetes": {
        "title": "Diabetes progression",
        "comment": "442 patients with 10 baseline measurements and a one-year disease-progression score.",
        "keywords": json.dumps(["diabetes", "regression", "health", "toy dataset", "progression"]),
        "primary_key": (("sample_id",),),
        "doc_llm": (
            "Diabetes table: 442 rows, one per patient, with 10 mean-centred, scaled baseline features "
            "(`age`, `sex`, `bmi`, blood pressure `bp`, and six serum measurements `s1`..`s6`) and a "
            "continuous `target` quantifying disease progression one year after baseline. A small "
            "regression dataset — handy for `explain` and `partial_dependence`."
        ),
        "doc_md": (
            "### `diabetes` table\n\n"
            "442 patients (regression):\n\n"
            "- `sample_id` — row id\n"
            "- 10 scaled baseline features (`age`, `sex`, `bmi`, `bp`, `s1`–`s6`)\n"
            "- `target` — one-year disease-progression score"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Target range",
                    "sql": "SELECT min(target), max(target), round(avg(target), 1) FROM xgboost.main.diabetes",
                }
            ]
        ),
    },
    "california_housing": {
        "title": "California housing prices",
        "comment": "20640 census block groups with 8 features and the median house value (regression).",
        "keywords": json.dumps(["california housing", "regression", "prices", "toy dataset", "census"]),
        "primary_key": (("sample_id",),),
        "doc_llm": (
            "California-housing table: 20,640 rows, one per 1990 census block group, with 8 numeric "
            "features (`medinc` median income, `houseage`, average rooms/bedrooms, population, occupancy, "
            "latitude, longitude) and a continuous `target` — the median house value in $100,000s. A large "
            "regression dataset; downloaded from scikit-learn on first use."
        ),
        "doc_md": (
            "### `california_housing` table\n\n"
            "20,640 census block groups (regression):\n\n"
            "- `sample_id` — row id\n"
            "- 8 features (`medinc`, `houseage`, `averooms`, `latitude`, `longitude`, ...)\n"
            "- `target` — median house value (in $100k)"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "Average value by latitude band",
                    "sql": (
                        "SELECT round(latitude) AS lat, round(avg(target), 2) "
                        "FROM xgboost.main.california_housing GROUP BY lat ORDER BY lat"
                    ),
                }
            ]
        ),
    },
    "list_models": {
        "title": "Saved model registry",
        "comment": "One row per model persisted in the registry, with its estimator, task, shape, and score.",
        "keywords": json.dumps(["model registry", "models", "metadata", "catalog", "persistence"]),
        "primary_key": (("model_name",),),
        "doc_llm": (
            "A table of every model saved to the registry (by `fit`/`fit_<estimator>`/search with a "
            "`model_name`). One row per model keyed by `model_name`, with its `estimator`, `task`, "
            "`target`, training shape (`n_features`/`n_samples`/`n_classes`), `train_score`, "
            "`xgboost_version`, `created_at`, and the `features` list. Query it to discover what is "
            "available to `predict`/`explain`/`feature_importance`."
        ),
        "doc_md": (
            "### `list_models` table\n\n"
            "Every model in the registry, one row each:\n\n"
            "- `model_name` (key), `estimator`, `task`, `target`\n"
            "- `n_features` / `n_samples` / `n_classes`, `train_score`\n"
            "- `xgboost_version`, `created_at`, `features`"
        ),
        "example_queries": json.dumps(
            [
                {
                    "description": "How many models are saved",
                    "sql": "SELECT count(*) AS model_count FROM xgboost.main.list_models",
                }
            ]
        ),
    },
}


def _function_table(fn: type) -> Table:
    """Expose a parameterless table function as a same-named table (VGI311).

    The table carries its own table-oriented metadata from ``_TABLE_META`` — a
    description, descriptive title, primary key, and example — distinct from the
    backing function's documentation.
    """
    meta = getattr(fn, "Meta")  # noqa: B009 - Meta is a dynamic per-function class
    tm = _TABLE_META[meta.name]
    primary_key = tm["primary_key"]
    # vgi.v3 carries constraints as canonical records. The key columns are
    # inherently non-null (VGI804), so declare that explicitly as well.
    key_cols = list(dict.fromkeys(col for cols in primary_key for col in cols))
    records = (
        constraint_record(
            ["main"],
            meta.name,
            "pk",
            "PRIMARY_KEY",
            [list(cols) for cols in primary_key],
            enforcement="ENFORCED",
            validation="ALL_ROWS",
        ),
        *(
            constraint_record(
                ["main"],
                meta.name,
                f"{col}_not_null",
                "NOT_NULL",
                [[col]],
                enforcement="ENFORCED",
                validation="ALL_ROWS",
            )
            for col in key_cols
        ),
    )
    return Table(
        name=meta.name,
        function=fn,
        comment=tm["comment"],
        constraints=records,
        tags={
            "provider": "XGBoost",
            "domain": "machine-learning",
            "vgi.title": tm["title"],
            "vgi.category": _FAMILY[fn],
            "vgi.keywords": tm["keywords"],
            "vgi.doc_llm": tm["doc_llm"],
            "vgi.doc_md": tm["doc_md"],
            "vgi.example_queries": tm["example_queries"],
        },
    )


_TABLES = [_function_table(fn) for fn in _FUNCTIONS if _is_parameterless_table_fn(fn)]

_XGBOOST_CATALOG = Catalog(
    name="xgboost",
    default_schema="main",
    comment=_CATALOG_COMMENT,
    tags=_CATALOG_TAGS,
    schemas=[
        Schema(
            path=["main"],
            comment="XGBoost train/predict model registry, datasets, and interpretation for SQL",
            tags=_SCHEMA_TAGS,
            tables=_TABLES,
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
