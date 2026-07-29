"""BuffState — active buffs, timers, and status effects."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ActiveBuff(BaseModel):
    """A single active buff or status effect."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    skill_id: str | None = None
    level: int = 1
    remaining_ms: int = 0  # milliseconds remaining
    total_ms: int = 0  # total duration in ms
    caster: str | None = None
    is_debuff: bool = False

    @property
    def remaining_seconds(self) -> float:
        return self.remaining_ms / 1000.0

    @property
    def fraction_left(self) -> float:
        if self.total_ms <= 0:
            return 0.0
        return self.remaining_ms / self.total_ms


class BuffState(BaseModel):
    """Active buffs and status effects on the character."""

    model_config = ConfigDict(extra="ignore")

    buffs: list[ActiveBuff] = Field(default_factory=list)
    debuffs: list[ActiveBuff] = Field(default_factory=list)
    total_buffs: int = 0
    total_debuffs: int = 0
    is_endowed: bool = False  # Elemental weapon endowment
    raw: dict[str, Any] = Field(default_factory=dict)


# ── Well-known debuff skill IDs (partial) ──
_DEBUFF_IDS: set[str] = {
    "NPC_CURSEATTACK", "NPC_POISON", "NPC_BLINDATTACK",
    "NPC_SILENCEATTACK", "NPC_STUNATTACK", "NPC_FREEZEATTACK",
    "NPC_PETRIFYATTACK", "NPC_BURNINGATTACK",
}


def collect_buffs(signals: dict[str, Any]) -> BuffState:
    """Parse active buffs/debuffs from the bridge signal dict.

    Handles:
      - ``signals['buffs']`` — list of active buff dicts with
        ``name``, ``remaining_ms``, ``total_ms``, ``caster``, etc.
      - ``signals['status_effects']`` — alternative flat list
      - ``signals['is_endowed']`` — endowment flag
    """
    raw_buffs: list[dict] = list(signals.get("buffs", signals.get("status_effects", [])) or [])
    buffs: list[ActiveBuff] = []
    debuffs: list[ActiveBuff] = []

    for raw in raw_buffs:
        if isinstance(raw, str):
            # Simple string buff name
            buffs.append(ActiveBuff(name=raw))
            continue
        if not isinstance(raw, dict):
            continue

        name = str(raw.get("name", ""))
        skill_id = str(raw.get("skill_id", "")) or None
        is_debuff = raw.get("is_debuff", False) or skill_id in _DEBUFF_IDS if skill_id else False

        buff = ActiveBuff(
            name=name,
            skill_id=skill_id,
            level=int(raw.get("level", 1)),
            remaining_ms=int(raw.get("remaining_ms", raw.get("remaining", 0))),
            total_ms=int(raw.get("total_ms", raw.get("duration", raw.get("remaining", 0)))),
            caster=raw.get("caster") or None,
            is_debuff=is_debuff,
        )
        if is_debuff:
            debuffs.append(buff)
        else:
            buffs.append(buff)

    # Check for endowment (elemental weapon buff)
    is_endowed = bool(signals.get("is_endowed", False))
    if not is_endowed:
        is_endowed = any(
            b.name.lower().startswith(("endow", "elemental", "weapon"))
            for b in buffs
        )

    return BuffState(
        buffs=buffs,
        debuffs=debuffs,
        total_buffs=len(buffs),
        total_debuffs=len(debuffs),
        is_endowed=is_endowed,
    )
