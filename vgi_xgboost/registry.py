"""Model registry: persist fitted estimators behind a swappable storage backend.

Ships a local-disk store (``XGBOOST_MODELS_DIR``, default ``./models``). The
``ModelStore`` interface is the seam where an S3/R2 backend drops in later
without touching ``models.py``.

Each model is two artifacts:
* ``<name>.ubj``  -- the estimator serialized with XGBoost's **native**
  ``Booster.save_model`` (UBJSON). This is XGBoost's own, forward-compatible
  format -- not pickle -- so models survive library upgrades far better and load
  without executing arbitrary code. The estimator class is recorded in metadata
  so the right scikit-learn wrapper is reconstructed on load.
* ``<name>.json`` -- ``ModelMetadata`` (estimator type, ordered feature names,
  target, classes, per-feature categorical mask, hyperparameters, train score,
  library versions, timestamp)

A model can also flow through SQL as a single self-describing BLOB (estimator +
metadata) -- see ``pack_model`` / ``unpack_model`` -- so a fitted model can live
inside a DuckDB table without touching the on-disk registry.
"""

from __future__ import annotations

import json
import os
import re
import struct
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import xgboost

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Estimator-name -> scikit-learn API class, for reconstructing the right wrapper
# when loading a natively-serialized model. Kept here (not in models.py) to avoid
# an import cycle; models.py validates the same names against its own catalog.
_ESTIMATOR_CLASSES: dict[str, type] = {
    "xgb_classifier": xgboost.XGBClassifier,
    "xgb_regressor": xgboost.XGBRegressor,
    "xgb_rf_classifier": xgboost.XGBRFClassifier,
    "xgb_rf_regressor": xgboost.XGBRFRegressor,
}


def _estimator_class(name: str) -> type:
    return _ESTIMATOR_CLASSES.get(name, xgboost.XGBClassifier)


def _native_dumps(estimator: Any) -> bytes:
    """Serialize a fitted scikit-learn-API XGBoost estimator to native UBJSON bytes."""
    with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as fh:
        path = fh.name
    try:
        estimator.save_model(path)
        return Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)


def _native_loads(data: bytes, estimator_name: str) -> Any:
    """Reconstruct the scikit-learn-API estimator from native UBJSON bytes."""
    est = _estimator_class(estimator_name)()
    with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as fh:
        path = fh.name
        fh.write(data)
    try:
        est.load_model(path)
        return est
    finally:
        Path(path).unlink(missing_ok=True)


class ModelNameError(ValueError):
    """Raised for model names that are empty or unsafe as a filename."""


class ModelNotFoundError(KeyError):
    """Raised when a requested model is not in the registry."""


def validate_name(name: str) -> str:
    if not name or not _NAME_RE.match(name) or "/" in name or ".." in name:
        raise ModelNameError(
            f"invalid model name {name!r}: use letters, digits, '_', '-', '.' and do not start with a separator"
        )
    return name


@dataclass(kw_only=True)
class ModelMetadata:
    """Everything needed to score new data and describe a stored model."""

    name: str
    estimator: str
    task: str  # "classification" | "regression"
    target: str
    feature_names: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    categorical: list[bool] | None = None  # per-feature: True where the feature is categorical
    categories: list[list[str] | None] | None = None  # per-feature ordered category list (None for numeric)
    classes: list[Any] | None = None
    n_samples: int = 0
    n_features: int = 0
    train_score: float | None = None
    xgboost_version: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelMetadata:
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        return cls(**{k: v for k, v in d.items() if k in known})


class ModelStore:
    """Abstract model store. Implementations persist (estimator, metadata) by name."""

    def save(self, estimator: Any, meta: ModelMetadata) -> None:
        raise NotImplementedError

    def load(self, name: str) -> tuple[Any, ModelMetadata]:
        raise NotImplementedError

    def load_meta(self, name: str) -> ModelMetadata:
        raise NotImplementedError

    def list(self) -> list[ModelMetadata]:
        raise NotImplementedError

    def delete(self, name: str) -> bool:
        raise NotImplementedError

    def exists(self, name: str) -> bool:
        raise NotImplementedError


class LocalDiskStore(ModelStore):
    """Stores models as ``<root>/<name>.ubj`` + ``<root>/<name>.json``.

    The estimator is serialized with XGBoost's native ``save_model`` (UBJSON),
    not pickle, so models are forward-compatible across library upgrades and load
    without arbitrary code execution.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def _paths(self, name: str) -> tuple[Path, Path]:
        validate_name(name)
        return self.root / f"{name}.ubj", self.root / f"{name}.json"

    def save(self, estimator: Any, meta: ModelMetadata) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        model_path, meta_path = self._paths(meta.name)
        model_path.write_bytes(_native_dumps(estimator))
        meta_path.write_text(json.dumps(meta.to_dict(), indent=2, default=str))

    def load(self, name: str) -> tuple[Any, ModelMetadata]:
        model_path, _ = self._paths(name)
        if not model_path.exists():
            raise ModelNotFoundError(name)
        meta = self.load_meta(name)
        return _native_loads(model_path.read_bytes(), meta.estimator), meta

    def load_meta(self, name: str) -> ModelMetadata:
        _, meta_path = self._paths(name)
        if not meta_path.exists():
            raise ModelNotFoundError(name)
        return ModelMetadata.from_dict(json.loads(meta_path.read_text()))

    def list(self) -> list[ModelMetadata]:
        if not self.root.exists():
            return []
        out: list[ModelMetadata] = []
        for meta_path in sorted(self.root.glob("*.json")):
            try:
                out.append(ModelMetadata.from_dict(json.loads(meta_path.read_text())))
            except (json.JSONDecodeError, OSError):
                continue
        return out

    def delete(self, name: str) -> bool:
        model_path, meta_path = self._paths(name)
        existed = model_path.exists() or meta_path.exists()
        model_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        return existed

    def exists(self, name: str) -> bool:
        model_path, _ = self._paths(name)
        return model_path.exists()


_store: ModelStore | None = None


def get_store() -> ModelStore:
    """Return the process-wide model store, configured from the environment.

    ``XGBOOST_MODELS_DIR`` selects the local-disk root (default ``./models``).
    A future S3/R2 backend would be selected here behind the same interface.
    """
    global _store
    if _store is None:
        root = os.environ.get("XGBOOST_MODELS_DIR", "models")
        _store = LocalDiskStore(root)
    return _store


def set_store(store: ModelStore | None) -> None:
    """Override the process-wide store (used by tests)."""
    global _store
    _store = store


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Self-contained model BLOB (estimator + metadata in one value)
#
# Layout: 4-byte big-endian metadata-JSON length || metadata JSON || native
# UBJSON bytes. Lets a fitted model flow through SQL as a single BLOB column and
# live inside a DuckDB table instead of (or alongside) the on-disk registry.
# DuckDB BLOB values are capped near 2 GB, so very large ensembles may not fit.
# ---------------------------------------------------------------------------


def pack_model(estimator: Any, meta: ModelMetadata) -> bytes:
    """Serialize ``(estimator, metadata)`` into one self-describing BLOB."""
    est_bytes = _native_dumps(estimator)
    meta_bytes = json.dumps(meta.to_dict(), default=str).encode("utf-8")
    return struct.pack(">I", len(meta_bytes)) + meta_bytes + est_bytes


def _split_blob(blob: bytes) -> tuple[bytes, bytes]:
    if len(blob) < 4:
        raise ValueError("not a valid xgboost model BLOB (too short)")
    (n,) = struct.unpack(">I", blob[:4])
    if len(blob) < 4 + n:
        raise ValueError("not a valid xgboost model BLOB (truncated metadata)")
    return blob[4 : 4 + n], blob[4 + n :]


def unpack_meta(blob: bytes) -> ModelMetadata:
    """Read just the metadata from a model BLOB (cheap; no estimator load)."""
    meta_bytes, _ = _split_blob(blob)
    return ModelMetadata.from_dict(json.loads(meta_bytes))


def unpack_model(blob: bytes) -> tuple[Any, ModelMetadata]:
    """Read both estimator and metadata from a model BLOB."""
    meta_bytes, est_bytes = _split_blob(blob)
    meta = ModelMetadata.from_dict(json.loads(meta_bytes))
    return _native_loads(est_bytes, meta.estimator), meta
