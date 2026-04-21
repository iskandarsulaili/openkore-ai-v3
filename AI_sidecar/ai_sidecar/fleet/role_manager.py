from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass(slots=True)
class RoleManager:
    bot_id: str
    role: str | None = None
    confidence: float = 0.0
    expires_at: datetime | None = None
    source: str = "local"

    def update(self, *, role: str | None, confidence: float, ttl_seconds: int, source: str) -> None:
        self.role = role
        self.confidence = float(confidence)
        self.expires_at = datetime.now(UTC) + timedelta(seconds=max(5, int(ttl_seconds)))
        self.source = source

    def current(self) -> dict[str, object]:
        now = datetime.now(UTC)
        expired = self.expires_at is not None and self.expires_at <= now
        if expired:
            return {
                "role": None,
                "confidence": 0.0,
                "expires_at": self.expires_at,
                "source": self.source,
            }
        return {
            "role": self.role,
            "confidence": self.confidence,
            "expires_at": self.expires_at,
            "source": self.source,
        }

