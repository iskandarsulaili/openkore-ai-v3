from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from pathlib import Path

from ai_sidecar.contracts.control_domain import ControlArtifactIdentity, ControlArtifactRecord, ControlOwnerScope
from ai_sidecar.domain.control_parser import ControlParser


@dataclass(slots=True)
class ControlStorage:
    workspace_root: Path
    parser: ControlParser

    def read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def checksum(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def resolve_control_path(self, *, bot_id: str, profile: str | None, target_path: str) -> Path:
        if target_path.startswith("/"):
            return self.workspace_root / target_path.lstrip("/")
        normalized = str(target_path).strip()
        if normalized.startswith("openkore-ai-v3/"):
            normalized = normalized[len("openkore-ai-v3/") :]
        if normalized.startswith("AI_sidecar/"):
            normalized = normalized[len("AI_sidecar/") :]
        if profile:
            return self.workspace_root / "profiles" / profile / normalized
        return self.workspace_root / normalized

    def snapshot(
        self,
        *,
        identity: ControlArtifactIdentity,
        owner: ControlOwnerScope,
        path: Path,
        version: str,
        metadata: dict[str, object] | None = None,
    ) -> ControlArtifactRecord:
        content = self.read_text(path)
        checksum = self.checksum(content)
        return ControlArtifactRecord(
            identity=identity,
            owner=owner,
            checksum=checksum,
            version=version,
            updated_at=datetime.now(UTC),
            metadata=dict(metadata or {}),
        )

    def write_config(self, *, path: Path, values: dict[str, str]) -> None:
        content = self.parser.render(values)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
