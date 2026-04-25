from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ControlParseResult:
    values: dict[str, str]


class ControlParser:
    def parse(self, content: str) -> ControlParseResult:
        values: dict[str, str] = {}
        for raw in content.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
            if " " not in line:
                key = line.strip()
                if key:
                    values[key] = ""
                continue
            key, value = line.split(None, 1)
            key = key.strip()
            value = value.strip()
            if key:
                values[key] = value
        return ControlParseResult(values=values)

    def render(self, values: dict[str, str]) -> str:
        lines: list[str] = []
        for key in sorted(values.keys()):
            value = values[key]
            if value == "":
                lines.append(key)
            else:
                lines.append(f"{key} {value}")
        return "\n".join(lines) + "\n"

