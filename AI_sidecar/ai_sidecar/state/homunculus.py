"""HomunculusState — homunculus name, level, stats, skills, intimacy."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HomunculusSkill(BaseModel):
    """A skill known by the homunculus."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    level: int = 0
    max_level: int = 0


class HomunculusState(BaseModel):
    """Active homunculus (artificial lifeform) information."""

    model_config = ConfigDict(extra="ignore")

    active: bool = False
    name: str | None = None
    homunculus_id: int | None = None
    hom_class: int | None = Field(default=None, serialization_alias="class")
    level: int | None = None
    hp: int | None = None
    hp_max: int | None = None
    sp: int | None = None
    sp_max: int | None = None
    exp: int = 0
    exp_max: int | None = None
    intimacy: int = 0
    hunger: int = 0
    skills: list[HomunculusSkill] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_alive(self) -> bool:
        return self.hp is not None and self.hp > 0

    @property
    def hp_ratio(self) -> float:
        if self.hp is None or not self.hp_max:
            return 0.0
        return self.hp / self.hp_max


def collect_homunculus(signals: dict[str, Any]) -> HomunculusState:
    """Parse homunculus information from the bridge signal dict.

    Handles:
      - ``signals['homunculus']`` — dict with homunculus info
      - ``signals['homunculus_name']``, ``signals['homunculus_level']`` — flat keys
      - ``signals['has_homunculus']`` — boolean indicator
    """
    h_dict: dict[str, Any] = signals.get("homunculus") or {}

    has_h = bool(signals.get("has_homunculus", False) or h_dict)
    if not has_h and not h_dict:
        h_name = signals.get("homunculus_name")
        if not h_name:
            return HomunculusState(active=False)
        # Flat key fallback
        raw_skills: list[dict] = list(signals.get("homunculus_skills", []) or [])
        skills = _build_homunculus_skills(raw_skills)
        return HomunculusState(
            active=True,
            name=str(h_name),
            level=int(signals.get("homunculus_level", 0)) or None,
            hp=int(signals.get("homunculus_hp", 0)) or None,
            hp_max=int(signals.get("homunculus_hp_max", 0)) or None,
            sp=int(signals.get("homunculus_sp", 0)) or None,
            sp_max=int(signals.get("homunculus_sp_max", 0)) or None,
            intimacy=int(signals.get("homunculus_intimacy", 0)),
            hunger=int(signals.get("homunculus_hunger", 0)),
            exp=int(signals.get("homunculus_exp", 0)),
            exp_max=int(signals.get("homunculus_exp_max", 0)) or None,
            skills=skills,
        )

    raw_skills: list[dict] = list(h_dict.get("skills", signals.get("homunculus_skills", [])) or [])
    skills = _build_homunculus_skills(raw_skills)

    return HomunculusState(
        active=True,
        name=str(h_dict.get("name", h_dict.get("homunculus_name", ""))) or None,
        homunculus_id=int(h_dict.get("homunculus_id", 0)) or None,
        hom_class=int(h_dict.get("class", 0) or h_dict.get("hom_class", 0)) or None,
        level=int(h_dict.get("level", 0)) or None,
        hp=int(h_dict.get("hp", 0)) or None,
        hp_max=int(h_dict.get("hp_max", h_dict.get("maxHp", 0))) or None,
        sp=int(h_dict.get("sp", 0)) or None,
        sp_max=int(h_dict.get("sp_max", h_dict.get("maxSp", 0))) or None,
        exp=int(h_dict.get("exp", h_dict.get("experience", 0))),
        exp_max=int(h_dict.get("exp_max", 0)) or None,
        intimacy=int(h_dict.get("intimacy", h_dict.get("intimate", 0))),
        hunger=int(h_dict.get("hunger", h_dict.get("hungry", 0))),
        skills=skills,
    )


def _build_homunculus_skills(raw_skills: list[dict]) -> list[HomunculusSkill]:
    """Build HomunculusSkill list from raw dicts."""
    result: list[HomunculusSkill] = []
    for s in raw_skills:
        if isinstance(s, dict):
            filtered = {k: v for k, v in s.items() if k in HomunculusSkill.model_fields}
            result.append(HomunculusSkill(**filtered))
    return result
