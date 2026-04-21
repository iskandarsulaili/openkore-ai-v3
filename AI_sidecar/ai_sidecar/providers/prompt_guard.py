from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PromptGuard:
    max_prompt_chars: int = 32000
    max_log_chars: int = 384

    _SECRET_PATTERNS = (
        re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
        re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{10,}"),
        re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+"),
    )

    def ensure_prompt_safe(self, value: str, *, field: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError(f"{field}_empty")
        if len(text) > self.max_prompt_chars:
            raise ValueError(f"{field}_too_large:{len(text)}>{self.max_prompt_chars}")
        return text

    def redact(self, value: str) -> str:
        text = value
        for pattern in self._SECRET_PATTERNS:
            text = pattern.sub("[REDACTED]", text)
        return text

    def preview(self, value: str) -> str:
        redacted = self.redact(value)
        if len(redacted) <= self.max_log_chars:
            return redacted
        return redacted[: self.max_log_chars] + "…"

    def parse_json_object(self, value: str) -> dict[str, Any] | None:
        text = (value or "").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return None
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return None
            candidate = text[start : end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    def validate_schema(self, payload: dict[str, Any], schema: dict[str, Any]) -> None:
        required = schema.get("required")
        if isinstance(required, list):
            missing = [item for item in required if item not in payload]
            if missing:
                raise ValueError(f"schema_required_missing:{','.join(str(item) for item in missing)}")

