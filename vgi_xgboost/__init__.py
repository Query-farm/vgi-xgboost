"""XGBoost as a VGI worker: a supervised train/predict model registry for DuckDB/SQL.

XGBoost's value is gradient-boosted train/predict, so the worker is built around
a model registry rather than the broad datasets/metrics/transforms surface of a
general ML library. The implementation is split by area:

- ``datasets``   -- a few reference datasets + generators, so demos and the SQL
  tests are self-contained (reuses scikit-learn's bundled data)
- ``models``     -- supervised ``fit`` / ``predict`` / ``cross_val_predict`` and
  the model registry, with XGBoost estimators
- ``importance`` -- XGBoost-specific extras: ``feature_importance`` and SHAP
  ``explain`` (per-feature prediction contributions)
- ``registry``   -- pluggable model store (local disk now, S3/R2 later)

``xgboost_worker.py`` at the repo root assembles these into the ``xgboost``
catalog and runs the worker.
"""

from __future__ import annotations

__version__ = "0.1.0"
