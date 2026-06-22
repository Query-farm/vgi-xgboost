"""Model registry: persist fitted estimators behind a swappable storage backend.

Ships a local-disk store (``XGBOOST_MODELS_DIR``, default ``./models``). The
``ModelStore`` interface is the seam where an S3/R2 backend drops in later
without touching ``models.py``.

Each model is two artifacts:
* ``<name>.joblib`` -- the pickled XGBoost (scikit-learn API) estimator
* ``<name>.json``   -- ``ModelMetadata`` (estimator type, ordered feature names,
  target, classes, hyperparameters, train score, library versions, timestamp)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


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
    """Stores models as ``<root>/<name>.joblib`` + ``<root>/<name>.json``."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def _paths(self, name: str) -> tuple[Path, Path]:
        validate_name(name)
        return self.root / f"{name}.joblib", self.root / f"{name}.json"

    def save(self, estimator: Any, meta: ModelMetadata) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        model_path, meta_path = self._paths(meta.name)
        joblib.dump(estimator, model_path)
        meta_path.write_text(json.dumps(meta.to_dict(), indent=2, default=str))

    def load(self, name: str) -> tuple[Any, ModelMetadata]:
        model_path, _ = self._paths(name)
        if not model_path.exists():
            raise ModelNotFoundError(name)
        return joblib.load(model_path), self.load_meta(name)

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
