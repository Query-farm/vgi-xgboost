"""XGBoost as a VGI worker: a supervised train/predict model registry for DuckDB/SQL.

XGBoost's value is gradient-boosted train/predict, so the worker is built around
a model registry rather than the broad datasets/metrics/transforms surface of a
general ML library. The implementation is split by area:

- ``datasets``     -- a few reference datasets + generators, so demos and the SQL
  tests are self-contained (reuses scikit-learn's bundled data)
- ``models``       -- supervised ``fit`` / ``predict`` / ``cross_val_predict`` /
  ``cross_val_score`` and the model registry, with XGBoost estimators
- ``typed_models`` -- generated ``fit_<estimator>`` functions exposing XGBoost's
  hyperparameters as native typed SQL arguments
- ``search``       -- ``grid_search`` / ``randomized_search`` over an estimator's
  hyperparameters (cross-validated leaderboard + refit best model)
- ``features``     -- native categorical + missing-value feature assembly
- ``importance``   -- XGBoost-specific extras: ``feature_importance``, SHAP
  ``explain`` (long format, multiclass), ``permutation_importance``, and
  ``partial_dependence``
- ``registry``     -- pluggable model store (local disk now, S3/R2 later); models
  serialize with XGBoost's native ``save_model`` (not pickle)

``xgboost_worker.py`` at the repo root assembles these into the ``xgboost``
catalog and runs the worker.
"""

from __future__ import annotations

import os as _os

# One thread per call by default. XGBoost (OpenMP) and numpy/scipy (BLAS) otherwise size
# their pools to every core, and DuckDB already runs worker calls in parallel, so all-core
# pools oversubscribe the machine and make small fits slower. VGI_XGBOOST_THREADS raises it.
_THREADS = max(1, int(_os.environ.get("VGI_XGBOOST_THREADS", "1") or "1"))
_os.environ.setdefault("OMP_NUM_THREADS", str(_THREADS))  # read when xgboost loads OpenMP

from threadpoolctl import threadpool_limits as _threadpool_limits  # noqa: E402

_threadpool_limits(limits=_THREADS)  # BLAS pools that are already loaded (vgi imports numpy)


__version__ = "0.1.0"
