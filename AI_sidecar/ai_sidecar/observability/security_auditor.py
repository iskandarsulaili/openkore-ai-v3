from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from threading import RLock

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{10,}"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+"),
)


@dataclass(slots=True)
class SecurityViolation:
    timestamp: datetime
    kind: str
    source: str
    bot_id: str
    detail: str
    severity: str = "warning"


class SecurityAuditor:
    def __init__(self, *, doctrine_denylist: list[str] | None = None, max_records: int = 5000) -> None:
        self._lock = RLock()
        self._denylist = {item.strip().lower() for item in list(doctrine_denylist or []) if item.strip()}
        self._max_records = max(100, int(max_records))
        self._records: list[SecurityViolation] = []

    def sanitize_text(self, text: str) -> str:
        output = text or ""
        for pattern in _SECRET_PATTERNS:
            output = pattern.sub("[REDACTED]", output)
        return output

    def sanitize_payload(self, payload: dict[str, object] | None) -> dict[str, object]:
        def _clean(value: object) -> object:
            if isinstance(value, str):
                return self.sanitize_text(value)
            if isinstance(value, dict):
                return {str(k): _clean(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_clean(item) for item in value]
            return value

        return _clean(dict(payload or {})) if payload is not None else {}

    def validate_social_response(self, *, message: str) -> tuple[bool, str]:
        lowered = (message or "").lower()
        disallowed = ["real money", "paypal", "discord.gg", "botting service", "exploit", "dupe"]
        for token in disallowed:
            if token in lowered:
                return False, f"social_policy_blocked:{token}"
        return True, "allowed"

    def validate_doctrine(self, *, doctrine: dict[str, object]) -> tuple[bool, str]:
        if not self._denylist:
            return True, "allowed"
        text = str(doctrine).lower()
        for token in sorted(self._denylist):
            if token in text:
                return False, f"doctrine_denylist_violation:{token}"
        return True, "allowed"

    def validate_macro_policy(self, *, macro_lines: list[str], automacro_conditions: list[str]) -> tuple[bool, str]:
        blocked = ["eval ", "shell ", "system(", "exec ", "wget ", "curl ", "perl "]
        all_lines = [str(item or "") for item in list(macro_lines) + list(automacro_conditions)]
        for line in all_lines:
            norm = line.lower()
            for token in blocked:
                if token in norm:
                    return False, f"macro_policy_blocked:{token.strip()}"
        return True, "allowed"

    def record(self, *, kind: str, source: str, bot_id: str, detail: str, severity: str = "warning") -> None:
        row = SecurityViolation(
            timestamp=datetime.now(UTC),
            kind=kind,
            source=source,
            bot_id=bot_id,
            detail=self.sanitize_text(detail),
            severity=severity,
        )
        with self._lock:
            self._records.append(row)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records :]

    def recent(self, *, limit: int = 100) -> list[dict[str, object]]:
        with self._lock:
            rows = list(self._records)
        rows = rows[-max(1, min(int(limit), 2000)) :]
        return [asdict(item) for item in reversed(rows)]

