"""InstanceState — active instances and time remaining."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InstanceEntry(BaseModel):
    """An active instance with remaining time and state."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    instance_id: int | None = None
    remaining_ms: int = 0  # Remaining time in milliseconds
    total_ms: int = 0  # Total duration in milliseconds
    state: str = "active"  # active | idle | expire_soon
    map_name: str | None = None
    mob_count: int = 0  # Remaining monsters
    max_mob_count: int | None = None

    @property
    def remaining_seconds(self) -> float:
        return self.remaining_ms / 1000.0

    @property
    def fraction_left(self) -> float:
        if self.total_ms <= 0:
            return 0.0
        return self.remaining_ms / self.total_ms

    @property
    def is_expiring(self) -> bool:
        return self.remaining_seconds < 300  # < 5 minutes


class InstanceState(BaseModel):
    """Current instance dungeon state."""

    model_config = ConfigDict(extra="ignore")

    active_instances: list[InstanceEntry] = Field(default_factory=list)
    total_instances: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


def collect_instances(signals: dict[str, Any]) -> InstanceState:
    """Parse instance information from the bridge signal dict.

    Handles:
      - ``signals['instances']`` — list of instance dicts
      - ``signals['instance']`` — single instance dict
      - ``signals['instance_name']``, ``signals['instance_time']`` — flat keys
    """
    raw_instances: list[dict] = list(signals.get("instances", []) or [])
    single_instance: dict[str, Any] = signals.get("instance") or {}

    instances: list[InstanceEntry] = []

    for raw in raw_instances:
        if isinstance(raw, str):
            instances.append(InstanceEntry(name=raw))
        elif isinstance(raw, dict):
            instances.append(
                InstanceEntry(
                    name=str(raw.get("name", raw.get("instance_name", ""))),
                    instance_id=int(raw.get("instance_id", 0)) or None,
                    remaining_ms=int(raw.get("remaining_ms", raw.get("remaining", 0))),
                    total_ms=int(raw.get("total_ms", raw.get("duration", raw.get("remaining", 0)))),
                    state=str(raw.get("state", "active")),
                    map_name=str(raw.get("map_name", "")) or None,
                    mob_count=int(raw.get("mob_count", 0)),
                    max_mob_count=int(raw.get("max_mob_count", 0)) or None,
                )
            )

    # Also handle single instance dict
    if single_instance and not instances:
        instances.append(
            InstanceEntry(
                name=str(single_instance.get("name", signals.get("instance_name", ""))),
                instance_id=int(single_instance.get("instance_id", 0)) or None,
                remaining_ms=int(single_instance.get("remaining_ms", signals.get("instance_time", 0))),
            )
        )

    return InstanceState(
        active_instances=instances,
        total_instances=len(instances),
    )
