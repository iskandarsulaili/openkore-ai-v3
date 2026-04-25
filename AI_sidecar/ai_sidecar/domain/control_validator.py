from __future__ import annotations

from dataclasses import dataclass

from ai_sidecar.contracts.control_domain import ControlChangeItem
from ai_sidecar.domain.control_storage import ControlStorage


@dataclass(slots=True)
class ControlValidator:
    storage: ControlStorage

    def drift(self, *, path: str, expected: dict[str, str]) -> list[ControlChangeItem]:
        target_path = self.storage.workspace_root / path
        current_text = self.storage.read_text(target_path)
        parsed = self.storage.parser.parse(current_text)
        drift: list[ControlChangeItem] = []
        for key, value in expected.items():
            current_value = parsed.values.get(key, "")
            if current_value != value:
                drift.append(
                    ControlChangeItem(
                        key=key,
                        previous=current_value,
                        updated=value,
                        reason="drift",
                    )
                )
        return drift

