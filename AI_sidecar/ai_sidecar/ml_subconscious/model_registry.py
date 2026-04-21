from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from uuid import uuid4

from ai_sidecar.contracts.ml_subconscious import MLModelFamilyView, MLModelVersionView, MLModelsResponse, ModelFamily

logger = logging.getLogger(__name__)

try:
    import joblib
except Exception:  # pragma: no cover - optional dependency fallback
    joblib = None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _flatten_features(value: object, *, prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    numeric = _coerce_float(value)
    if numeric is not None:
        flat[prefix or "value"] = numeric
        return flat

    if isinstance(value, str):
        key = prefix or "text"
        flat[f"{key}.len"] = float(len(value))
        return flat

    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_features(item, prefix=child_prefix))
        return flat

    if isinstance(value, list):
        for idx, item in enumerate(value[:32]):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            flat.update(_flatten_features(item, prefix=child_prefix))
        return flat

    key = prefix or "raw"
    flat[f"{key}.hash"] = float(int.from_bytes(hashlib.sha256(str(value).encode("utf-8")).digest()[:2], "little") / 65535.0)
    return flat


def vectorize_state_features(state_features: dict[str, object], *, dims: int = 128) -> tuple[list[float], dict[str, float]]:
    flat = _flatten_features(state_features)
    vector = [0.0] * max(32, int(dims))
    contributions: dict[str, float] = {}
    for key, value in flat.items():
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % len(vector)
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        contrib = sign * float(value)
        vector[idx] += contrib
        contributions[key] = float(value)
    return vector, contributions


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


@dataclass(slots=True)
class ModelRegistry:
    workspace_root: Path
    base_path: Path | None = None
    vector_dims: int = 128
    _lock: RLock = field(default_factory=RLock)
    _index: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.base_path is None:
            self.base_path = self.workspace_root / "AI_sidecar" / "data" / "ml_subconscious" / "models"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._load_index()

    @property
    def _index_file(self) -> Path:
        assert self.base_path is not None
        return self.base_path / "registry_index.json"

    def _load_index(self) -> None:
        with self._lock:
            if not self._index_file.exists():
                self._index = {}
                return
            try:
                payload = json.loads(self._index_file.read_text(encoding="utf-8"))
                self._index = payload if isinstance(payload, dict) else {}
            except Exception:
                logger.exception("ml_registry_index_load_failed", extra={"event": "ml_registry_index_load_failed"})
                self._index = {}

    def _save_index(self) -> None:
        with self._lock:
            self._index_file.write_text(json.dumps(self._index, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    def _family_dir(self, family: ModelFamily) -> Path:
        assert self.base_path is not None
        path = self.base_path / family.value
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _artifact_path(self, family: ModelFamily, version: str) -> Path:
        suffix = ".joblib" if joblib is not None else ".pkl"
        return self._family_dir(family) / f"{version}{suffix}"

    def save_model(
        self,
        *,
        family: ModelFamily,
        package: dict[str, object],
        metrics: dict[str, float],
        activate: bool = False,
    ) -> str:
        version = f"v{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
        artifact = self._artifact_path(family, version)
        if joblib is not None:
            joblib.dump(package, artifact)
        else:
            artifact.write_bytes(pickle.dumps(package))

        with self._lock:
            family_entry = self._index.setdefault(
                family.value,
                {
                    "active_version": None,
                    "versions": [],
                    "ab": {"champion": None, "challenger": None},
                },
            )
            versions = list(family_entry.get("versions") or [])
            versions.append(
                {
                    "version": version,
                    "created_at": datetime.now(UTC).isoformat(),
                    "metrics": dict(metrics),
                    "path": str(artifact),
                }
            )
            family_entry["versions"] = versions[-40:]

            current_active = family_entry.get("active_version")
            family_entry["ab"] = {
                "champion": current_active,
                "challenger": version,
            }
            if activate or current_active is None:
                family_entry["active_version"] = version
                family_entry["ab"] = {"champion": version, "challenger": None}
        self._save_index()
        return version

    def load_model(self, *, family: ModelFamily, version: str | None = None) -> dict[str, object] | None:
        with self._lock:
            family_entry = self._index.get(family.value, {})
            selected = version or family_entry.get("active_version")
            if not selected:
                return None
            versions = list(family_entry.get("versions") or [])
            row = next((item for item in versions if item.get("version") == selected), None)
            if not isinstance(row, dict):
                return None
            artifact_path = Path(str(row.get("path") or ""))
        if not artifact_path.exists():
            return None
        try:
            if joblib is not None and artifact_path.suffix == ".joblib":
                loaded = joblib.load(artifact_path)
            else:
                loaded = pickle.loads(artifact_path.read_bytes())
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            logger.exception(
                "ml_registry_load_model_failed",
                extra={"event": "ml_registry_load_model_failed", "family": family.value, "version": selected},
            )
        return None

    def activate_version(self, *, family: ModelFamily, version: str) -> bool:
        with self._lock:
            family_entry = self._index.get(family.value)
            if not isinstance(family_entry, dict):
                return False
            versions = list(family_entry.get("versions") or [])
            if not any(item.get("version") == version for item in versions if isinstance(item, dict)):
                return False
            family_entry["active_version"] = version
            family_entry["ab"] = {"champion": version, "challenger": None}
        self._save_index()
        return True

    def active_version(self, *, family: ModelFamily) -> str | None:
        with self._lock:
            family_entry = self._index.get(family.value, {})
            value = family_entry.get("active_version")
            return str(value) if isinstance(value, str) and value else None

    def list_models(self) -> MLModelsResponse:
        rows: list[MLModelFamilyView] = []
        with self._lock:
            index_snapshot = json.loads(json.dumps(self._index))
        for family in ModelFamily:
            family_entry = index_snapshot.get(family.value, {})
            active_version = family_entry.get("active_version")
            versions_payload = list(family_entry.get("versions") or [])
            versions: list[MLModelVersionView] = []
            for item in versions_payload:
                if not isinstance(item, dict):
                    continue
                created_raw = item.get("created_at")
                try:
                    created = datetime.fromisoformat(str(created_raw)) if created_raw else datetime.now(UTC)
                except Exception:
                    created = datetime.now(UTC)
                versions.append(
                    MLModelVersionView(
                        version=str(item.get("version") or ""),
                        created_at=created,
                        active=str(item.get("version") or "") == str(active_version or ""),
                        metrics={k: float(v) for k, v in dict(item.get("metrics") or {}).items()},
                        path=str(item.get("path") or ""),
                    )
                )
            rows.append(MLModelFamilyView(family=family, active_version=str(active_version) if active_version else None, versions=versions))
        return MLModelsResponse(ok=True, models=rows)

    def ab_state(self, *, family: ModelFamily) -> dict[str, object]:
        with self._lock:
            family_entry = self._index.get(family.value, {})
            ab = dict(family_entry.get("ab") or {})
            return {
                "family": family.value,
                "champion": ab.get("champion"),
                "challenger": ab.get("challenger"),
                "active_version": family_entry.get("active_version"),
            }

