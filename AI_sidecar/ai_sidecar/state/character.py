"""CharacterState — core character vitals, progression, and stats."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CharacterState(BaseModel):
    """Core character state: HP, SP, level, job, weight, zeny, and stats."""

    model_config = ConfigDict(extra="ignore")

    # ── Vitals ──
    hp: int = 0
    hp_max: int = 1
    sp: int = 0
    sp_max: int = 1
    hp_ratio: float = 0.0
    sp_ratio: float = 0.0

    # ── Progression ──
    base_level: int = 1
    job_level: int = 1
    job_id: int | None = None
    job_name: str = "novice"
    base_exp: int = 0
    base_exp_max: int | None = None
    job_exp: int = 0
    job_exp_max: int | None = None
    zeny: int = 0
    weight: int = 0
    weight_max: int = 2000
    weight_ratio: float = 0.0

    # ── Stat points ──
    stat_points: int = 0
    skill_points: int = 0
    str_stat: int = 1
    agi_stat: int = 1
    vit_stat: int = 1
    int_stat: int = 1
    dex_stat: int = 1
    luk_stat: int = 1

    # ── Raw signal extras ──
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    @property
    def is_sitting(self) -> bool:
        return self.hp_ratio < 1.0 and self.hp == self.hp_max and self.sp == self.sp_max


def collect_character(signals: dict[str, Any]) -> CharacterState:
    """Parse character vitals and progression from the bridge signal dict.

    Handles the flat signal format emitted by the bridge plugin:
      - Top-level keys: hp, hp_max, sp, sp_max, hp_ratio, sp_ratio
      - Progression keys: base_level, job_level, job_name, zeny
      - Stat keys: stat_points, skill_points, str, agi, vit, int, dex, luk
      - Weight keys: weight, weight_max, weight_ratio
      - Also reads from signals.get('vitals', {}) and signals.get('progression', {})
        for the structured BotStateSnapshot format.
    """
    # Try structured sub-dicts first, then fall back to flat keys
    vitals = signals.get("vitals") or {}
    progression = signals.get("progression") or {}

    def _get_val(*keys: str, default: Any = 0) -> Any:
        for key in keys:
            val = signals.get(key)
            if val is not None:
                return val
            val = vitals.get(key) if isinstance(vitals, dict) else None
            if val is not None:
                return val
            val = progression.get(key) if isinstance(progression, dict) else None
            if val is not None:
                return val
        return default

    hp = int(_get_val("hp", default=0))
    hp_max = int(_get_val("hp_max", "maxHp", default=1))
    sp = int(_get_val("sp", default=0))
    sp_max = int(_get_val("sp_max", "maxSp", default=1))

    return CharacterState(
        hp=hp,
        hp_max=hp_max,
        sp=sp,
        sp_max=sp_max,
        hp_ratio=float(_get_val("hp_ratio", default=hp / max(hp_max, 1))),
        sp_ratio=float(_get_val("sp_ratio", default=sp / max(sp_max, 1))),
        base_level=int(_get_val("base_level", "level", default=1)),
        job_level=int(_get_val("job_level", default=1)),
        job_id=int(_get_val("job_id", default=0)) or None,
        job_name=str(_get_val("job_name", "job", default="novice")).lower(),
        base_exp=int(_get_val("base_exp", "exp", default=0)),
        base_exp_max=int(_get_val("base_exp_max", default=0)) or None,
        job_exp=int(_get_val("job_exp", default=0)),
        job_exp_max=int(_get_val("job_exp_max", default=0)) or None,
        zeny=int(_get_val("zeny", default=0)),
        weight=int(_get_val("weight", default=0)),
        weight_max=int(_get_val("weight_max", default=2000)),
        weight_ratio=float(_get_val("weight_ratio", default=0.0)),
        stat_points=int(_get_val("stat_points", default=0)),
        skill_points=int(_get_val("skill_points", default=0)),
        str_stat=int(_get_val("str", "str_stat", default=1)),
        agi_stat=int(_get_val("agi", "agi_stat", default=1)),
        vit_stat=int(_get_val("vit", "vit_stat", default=1)),
        int_stat=int(_get_val("int", "int_stat", default=1)),
        dex_stat=int(_get_val("dex", "dex_stat", default=1)),
        luk_stat=int(_get_val("luk", "luk_stat", default=1)),
        raw={
            k: v
            for k, v in signals.items()
            if k
            not in {
                "hp", "hp_max", "sp", "sp_max", "hp_ratio", "sp_ratio",
                "base_level", "job_level", "job_name", "zeny",
                "stat_points", "skill_points", "weight", "weight_max",
            }
        },
    )
