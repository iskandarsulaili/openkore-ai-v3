"""MercenaryState — mercenary info, stats, lifetime."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MercenaryState(BaseModel):
    """Active mercenary / mercenary soldier information."""

    model_config = ConfigDict(extra="ignore")

    active: bool = False
    name: str | None = None
    mercenary_id: int | None = None
    mer_class: int | None = Field(default=None, serialization_alias="class")
    level: int | None = None
    hp: int | None = None
    hp_max: int | None = None
    sp: int | None = None
    sp_max: int | None = None
    faith: int = 0  # Mercenary faith / loyalty
    lifetime_ms: int = 0  # Remaining lifetime in milliseconds
    kills: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_alive(self) -> bool:
        return self.hp is not None and self.hp > 0

    @property
    def lifetime_remaining_seconds(self) -> float:
        return self.lifetime_ms / 1000.0

    @property
    def hp_ratio(self) -> float:
        if self.hp is None or not self.hp_max:
            return 0.0
        return self.hp / self.hp_max


def collect_mercenary(signals: dict[str, Any]) -> MercenaryState:
    """Parse mercenary information from the bridge signal dict.

    Handles:
      - ``signals['mercenary']`` — dict with mercenary info
      - ``signals['mercenary_name']``, ``signals['has_mercenary']`` — flat keys
    """
    m_dict: dict[str, Any] = signals.get("mercenary") or {}

    has_m = bool(signals.get("has_mercenary", False) or m_dict)
    if not has_m and not m_dict:
        m_name = signals.get("mercenary_name")
        if not m_name:
            return MercenaryState(active=False)
        # Flat key fallback
        return MercenaryState(
            active=True,
            name=str(m_name),
            level=int(signals.get("mercenary_level", 0)) or None,
            hp=int(signals.get("mercenary_hp", 0)) or None,
            hp_max=int(signals.get("mercenary_hp_max", 0)) or None,
            sp=int(signals.get("mercenary_sp", 0)) or None,
            sp_max=int(signals.get("mercenary_sp_max", 0)) or None,
            faith=int(signals.get("mercenary_faith", 0)),
            lifetime_ms=int(signals.get("mercenary_lifetime", 0)),
            kills=int(signals.get("mercenary_kills", 0)),
        )

    return MercenaryState(
        active=True,
        name=str(m_dict.get("name", m_dict.get("mercenary_name", ""))) or None,
        mercenary_id=int(m_dict.get("mercenary_id", 0)) or None,
        mer_class=int(m_dict.get("class", 0) or m_dict.get("mer_class", 0)) or None,
        level=int(m_dict.get("level", 0)) or None,
        hp=int(m_dict.get("hp", 0)) or None,
        hp_max=int(m_dict.get("hp_max", 0)) or None,
        sp=int(m_dict.get("sp", 0)) or None,
        sp_max=int(m_dict.get("sp_max", 0)) or None,
        faith=int(m_dict.get("faith", m_dict.get("loyalty", 0))),
        lifetime_ms=int(m_dict.get("lifetime_ms", m_dict.get("lifeTime", 0))),
        kills=int(m_dict.get("kills", 0)),
    )
