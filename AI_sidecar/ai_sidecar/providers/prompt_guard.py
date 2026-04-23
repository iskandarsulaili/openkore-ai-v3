from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PromptGuard:
    max_prompt_chars: int = 32000
    max_log_chars: int = 384
    max_schema_depth: int = 24

    _SECRET_PATTERNS = (
        re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
        re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{10,}"),
        re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+"),
    )
    _UNSAFE_OBJECT_KEYS = {"__proto__", "prototype", "constructor"}

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
        text = self._strip_code_fence((value or "").strip())
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return None
        except Exception:
            candidate = self._extract_first_json_object(text)
            if not candidate:
                return None
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    def normalize_for_schema(self, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(schema, dict):
            return payload
        normalized = self._normalize_value(payload, schema=schema, path="$", depth=0)
        if isinstance(normalized, dict):
            return normalized
        return payload

    def validate_schema(self, payload: dict[str, Any], schema: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("schema_payload_not_object")
        if not isinstance(schema, dict):
            raise ValueError("schema_invalid_definition")
        self._validate_against_schema(payload, schema, path="$", depth=0)

    def _validate_against_schema(self, value: Any, schema: dict[str, Any], *, path: str, depth: int) -> None:
        if depth > self.max_schema_depth:
            raise ValueError(f"schema_depth_exceeded:{path}")

        expected_type = schema.get("type")
        if expected_type is not None:
            if isinstance(expected_type, list):
                allowed_types = [str(item) for item in expected_type if isinstance(item, str)]
                if not allowed_types:
                    raise ValueError(f"schema_invalid_type_definition:{path}")
                if not any(self._matches_type(value, item) for item in allowed_types):
                    raise ValueError(f"schema_type_mismatch:{path}:expected={','.join(allowed_types)}")
            elif isinstance(expected_type, str):
                if not self._matches_type(value, expected_type):
                    raise ValueError(f"schema_type_mismatch:{path}:expected={expected_type}")
            else:
                raise ValueError(f"schema_invalid_type_definition:{path}")

        if "const" in schema and value != schema.get("const"):
            raise ValueError(f"schema_const_mismatch:{path}")

        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and enum_values and value not in enum_values:
            raise ValueError(f"schema_enum_mismatch:{path}")

        if isinstance(value, str):
            self._validate_string(value, schema, path=path)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            self._validate_number(float(value), schema, path=path)
        elif isinstance(value, list):
            self._validate_array(value, schema, path=path, depth=depth)
        elif isinstance(value, dict):
            self._validate_object(value, schema, path=path, depth=depth)

    def _validate_string(self, value: str, schema: dict[str, Any], *, path: str) -> None:
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        if isinstance(min_length, int) and len(value) < min_length:
            raise ValueError(f"schema_string_too_short:{path}:{len(value)}<{min_length}")
        if isinstance(max_length, int) and len(value) > max_length:
            raise ValueError(f"schema_string_too_long:{path}:{len(value)}>{max_length}")
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                if not re.search(pattern, value):
                    raise ValueError(f"schema_pattern_mismatch:{path}")
            except re.error as exc:
                logger.warning(
                    "schema_pattern_invalid",
                    extra={"event": "schema_pattern_invalid", "path": path, "error": type(exc).__name__},
                )
                raise ValueError(f"schema_pattern_invalid:{path}") from exc

    def _validate_number(self, value: float, schema: dict[str, Any], *, path: str) -> None:
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < float(minimum):
            raise ValueError(f"schema_number_below_minimum:{path}:{value}<{float(minimum)}")
        if isinstance(maximum, (int, float)) and value > float(maximum):
            raise ValueError(f"schema_number_above_maximum:{path}:{value}>{float(maximum)}")

    def _validate_array(self, value: list[Any], schema: dict[str, Any], *, path: str, depth: int) -> None:
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            raise ValueError(f"schema_array_too_short:{path}:{len(value)}<{min_items}")
        if isinstance(max_items, int) and len(value) > max_items:
            raise ValueError(f"schema_array_too_long:{path}:{len(value)}>{max_items}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                self._validate_against_schema(item, item_schema, path=f"{path}[{idx}]", depth=depth + 1)

    def _validate_object(self, value: dict[str, Any], schema: dict[str, Any], *, path: str, depth: int) -> None:
        for key in value:
            self._validate_safe_key(key=key, path=path)

        required = schema.get("required")
        if isinstance(required, list):
            missing = [item for item in required if isinstance(item, str) and item not in value]
            if missing:
                raise ValueError(f"schema_required_missing:{path}:{','.join(missing)}")

        properties = schema.get("properties")
        known_properties = properties if isinstance(properties, dict) else {}

        additional_allowed = schema.get("additionalProperties", True)
        if additional_allowed is False:
            extras = [key for key in value if key not in known_properties]
            if extras:
                raise ValueError(f"schema_additional_properties_forbidden:{path}:{','.join(extras)}")

        for key, nested_schema in known_properties.items():
            if key not in value:
                continue
            if isinstance(nested_schema, dict):
                self._validate_against_schema(
                    value[key],
                    nested_schema,
                    path=f"{path}.{key}",
                    depth=depth + 1,
                )

        if isinstance(additional_allowed, dict):
            for key, nested_value in value.items():
                if key in known_properties:
                    continue
                self._validate_against_schema(
                    nested_value,
                    additional_allowed,
                    path=f"{path}.{key}",
                    depth=depth + 1,
                )

    def _validate_safe_key(self, *, key: str, path: str) -> None:
        lowered = key.strip().lower()
        if lowered in self._UNSAFE_OBJECT_KEYS or lowered.startswith("$"):
            raise ValueError(f"schema_unsafe_key:{path}.{key}")

    def _strip_code_fence(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned.startswith("```"):
            return cleaned
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
        return cleaned

    def _extract_first_json_object(self, text: str) -> str:
        start = text.find("{")
        if start < 0:
            return ""
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return ""

    def _normalize_value(self, value: Any, *, schema: dict[str, Any], path: str, depth: int) -> Any:
        if depth > self.max_schema_depth:
            return value

        expected_type = schema.get("type")
        type_name = ""
        if isinstance(expected_type, list):
            candidate_types = [str(item).strip().lower() for item in expected_type if isinstance(item, str)]
            matched = [item for item in candidate_types if self._matches_type(value, item)]
            if matched:
                type_name = matched[0]
            elif "null" in candidate_types:
                type_name = candidate_types[0]
            elif candidate_types:
                type_name = candidate_types[0]
        elif isinstance(expected_type, str):
            type_name = expected_type.strip().lower()

        normalized = value
        if type_name == "object":
            normalized = self._normalize_object(value, schema=schema, path=path, depth=depth)
        elif type_name == "array":
            normalized = self._normalize_array(value, schema=schema, path=path, depth=depth)
        elif type_name == "string":
            normalized = self._normalize_string(value, schema=schema)
        elif type_name == "number":
            normalized = self._normalize_number(value, schema=schema, integer=False)
        elif type_name == "integer":
            normalized = self._normalize_number(value, schema=schema, integer=True)
        elif type_name == "boolean":
            normalized = self._normalize_bool(value)
        elif type_name == "null":
            normalized = None

        if "const" in schema:
            normalized = schema.get("const")

        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            if normalized not in enum_values:
                if isinstance(normalized, str):
                    lowered = normalized.strip().lower()
                    found = next((item for item in enum_values if isinstance(item, str) and item.lower() == lowered), None)
                    normalized = found if found is not None else enum_values[0]
                else:
                    normalized = enum_values[0]
        return normalized

    def _normalize_object(self, value: Any, *, schema: dict[str, Any], path: str, depth: int) -> dict[str, Any]:
        obj = value if isinstance(value, dict) else {}
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        additional_allowed = schema.get("additionalProperties", True)

        normalized: dict[str, Any] = {}
        for key, nested_schema in properties.items():
            if key not in obj or not isinstance(nested_schema, dict):
                continue
            normalized[key] = self._normalize_value(obj.get(key), schema=nested_schema, path=f"{path}.{key}", depth=depth + 1)

        for key, nested_value in obj.items():
            if not isinstance(key, str):
                continue
            try:
                self._validate_safe_key(key=key, path=path)
            except Exception:
                continue
            if key in normalized or key in properties:
                continue
            if additional_allowed is False:
                continue
            if isinstance(additional_allowed, dict):
                normalized[key] = self._normalize_value(
                    nested_value,
                    schema=additional_allowed,
                    path=f"{path}.{key}",
                    depth=depth + 1,
                )
            else:
                normalized[key] = nested_value

        required = schema.get("required")
        if isinstance(required, list):
            for item in required:
                if not isinstance(item, str) or item in normalized:
                    continue
                nested_schema = properties.get(item) if isinstance(properties, dict) else None
                if isinstance(nested_schema, dict):
                    normalized[item] = self._default_for_schema(nested_schema)
                else:
                    normalized[item] = ""
        return normalized

    def _normalize_array(self, value: Any, *, schema: dict[str, Any], path: str, depth: int) -> list[Any]:
        if isinstance(value, list):
            rows = list(value)
        elif value is None:
            rows = []
        else:
            rows = [value]

        item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else None
        if item_schema is not None:
            rows = [
                self._normalize_value(item, schema=item_schema, path=f"{path}[{idx}]", depth=depth + 1)
                for idx, item in enumerate(rows)
            ]

        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and max_items >= 0:
            rows = rows[:max_items]

        min_items = schema.get("minItems")
        if isinstance(min_items, int) and min_items > 0:
            while len(rows) < min_items:
                rows.append(self._default_for_schema(item_schema if isinstance(item_schema, dict) else {}))
        return rows

    def _normalize_string(self, value: Any, *, schema: dict[str, Any]) -> str:
        text = "" if value is None else str(value)
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and max_length >= 0:
            text = text[:max_length]
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and min_length > 0 and len(text) < min_length:
            text = (text + ("x" * min_length))[:min_length]
        return text

    def _normalize_number(self, value: Any, *, schema: dict[str, Any], integer: bool) -> float | int:
        out: float
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out = float(value)
        elif isinstance(value, str):
            try:
                out = float(value.strip())
            except Exception:
                out = 0.0
        else:
            out = 0.0

        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)):
            out = max(out, float(minimum))
        if isinstance(maximum, (int, float)):
            out = min(out, float(maximum))
        if integer:
            return int(round(out))
        return out

    def _normalize_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off", ""}:
                return False
        return bool(value)

    def _default_for_schema(self, schema: dict[str, Any]) -> Any:
        expected_type = schema.get("type")
        type_name = ""
        if isinstance(expected_type, list):
            candidates = [str(item).strip().lower() for item in expected_type if isinstance(item, str)]
            type_name = next((item for item in candidates if item != "null"), candidates[0] if candidates else "")
        elif isinstance(expected_type, str):
            type_name = expected_type.strip().lower()

        if "const" in schema:
            return schema.get("const")
        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            return enum_values[0]

        if type_name == "array":
            return []
        if type_name == "object":
            return {}
        if type_name == "string":
            min_length = schema.get("minLength")
            if isinstance(min_length, int) and min_length > 0:
                return "x" * min_length
            return ""
        if type_name == "integer":
            return int(self._normalize_number(0, schema=schema, integer=True))
        if type_name == "number":
            return float(self._normalize_number(0, schema=schema, integer=False))
        if type_name == "boolean":
            return False
        return None

    def _matches_type(self, value: Any, expected_type: str) -> bool:
        normalized = expected_type.strip().lower()
        if normalized == "string":
            return isinstance(value, str)
        if normalized == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if normalized == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if normalized == "boolean":
            return isinstance(value, bool)
        if normalized == "object":
            return isinstance(value, dict)
        if normalized == "array":
            return isinstance(value, list)
        if normalized == "null":
            return value is None
        return True
