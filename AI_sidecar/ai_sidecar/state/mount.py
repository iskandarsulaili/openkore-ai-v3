"""MountState — mount info (Peco Peco, Grand Pecos, etc.)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MountState(BaseModel):
    """Active mount / riding information."""

    model_config = ConfigDict(extra="ignore")

    active: bool = False
    mount_type: str | None = None  # e.g. "peco_peco", "grand_pecos"
    name: str | None = None
    level: int | None = None
    hp: int | None = None
    hp_max: int | None = None
    sp: int | None = None
    sp_max: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_alive(self) -> bool:
        return self.hp is not None and self.hp > 0

    @property
    def hp_ratio(self) -> float:
        if self.hp is None or not self.hp_max:
            return 0.0
        return self.hp / self.hp_max


def collect_mount(signals: dict[str, Any]) -> MountState:
    """Parse mount/riding information from the bridge signal dict.

    Handles:
      - ``signals['mount']`` — dict with mount info
      - ``signals['mount_type']``, ``signals['mount_name']`` — flat keys
      - ``signals['is_riding']`` — boolean flag for riding state
    """
    m_dict: dict[str, Any] = signals.get("mount") or {}

    is_riding = bool(signals.get("is_riding", False) or m_dict)
    if not is_riding:
        mount_type = signals.get("mount_type")
        if not mount_type:
            return MountState(active=False)
        # Flat key fallback
        return MountState(
            active=True,
            mount_type=str(mount_type),
            name=str(signals.get("mount_name", "")) or None,
            level=int(signals.get("mount_level", 0)) or None,
        )

    return MountState(
        active=True,
        mount_type=str(m_dict.get("type", m_dict.get("mount_type", ""))) or None,
        name=str(m_dict.get("name", m_dict.get("mount_name", ""))) or None,
        level=int(m_dict.get("level", 0)) or None,
        hp=int(m_dict.get("hp", 0)) or None,
        hp_max=int(m_dict.get("hp_max", 0)) or None,
        sp=int(m_dict.get("sp", 0)) or None,
        sp_max=int(m_dict.get("sp_max", 0)) or None,
    )
