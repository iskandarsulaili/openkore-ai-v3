"""Monster AI Overrides — per-monster behavior flags for intelligent targeting.

RO monsters have unique AI behaviors that a pro player exploits:
- Some flee at low HP (Poring, Lunatic, Fabre) — use ranged or one-shot
- Some call friends when attacked (Smokie, Marionette, Golem) — isolate first
- Some teleport when hit (Whisper, Deviling, Ghostring) — use stun/freeze
- Some are aggressive only at night (Bathory, Isis, Mummy) — avoid at night
- Some use skills (Magnolia heals, Arclouse uses provok) — interrupt priority
- MVP monsters have special mechanics (teleport, heal, summon minions)
"""
from __future__ import annotations

from typing import Any

# ── Monster AI behavior flags ─────────────────────────────────────────────

MONSTER_AI_OVERRIDES: dict[str, dict[str, Any]] = {
    # ── Flee at low HP ──
    "Poring": {"flees_at_low_hp": True, "flee_hp_pct": 0.25, "tactic": "one_shot_or_ranged"},
    "Lunatic": {"flees_at_low_hp": True, "flee_hp_pct": 0.30, "tactic": "one_shot_or_ranged"},
    "Fabre": {"flees_at_low_hp": True, "flee_hp_pct": 0.20, "tactic": "one_shot_or_ranged"},
    "Pupa": {"flees_at_low_hp": False, "tactic": "stationary_tank"},
    "Chonchon": {"flees_at_low_hp": True, "flee_hp_pct": 0.15, "tactic": "fast_melee"},
    "Steel Chonchon": {"flees_at_low_hp": True, "flee_hp_pct": 0.15, "tactic": "fast_melee"},
    "Hunter Fly": {"flees_at_low_hp": True, "flee_hp_pct": 0.20, "tactic": "ranged_kite"},
    "Drainliar": {"flees_at_low_hp": True, "flee_hp_pct": 0.20, "tactic": "ranged_kite"},
    "Poporing": {"flees_at_low_hp": True, "flee_hp_pct": 0.25, "tactic": "one_shot_or_ranged"},
    "Metaller": {"flees_at_low_hp": True, "flee_hp_pct": 0.20, "tactic": "ranged"},
    "Plankton": {"flees_at_low_hp": True, "flee_hp_pct": 0.15, "tactic": "ranged"},
    "Marina": {"flees_at_low_hp": True, "flee_hp_pct": 0.20, "tactic": "ranged"},
    "Kukre": {"flees_at_low_hp": True, "flee_hp_pct": 0.20, "tactic": "ranged"},
    "Vadon": {"flees_at_low_hp": True, "flee_hp_pct": 0.20, "tactic": "ranged"},

    # ── Call friends when attacked ──
    "Smokie": {"calls_friends": True, "call_range": 7, "call_count": 2, "tactic": "isolate_or_one_shot"},
    "Marionette": {"calls_friends": True, "call_range": 10, "call_count": 3, "tactic": "isolate_or_one_shot"},
    "Golem": {"calls_friends": False, "tactic": "slow_tank"},
    "Myst Case": {"calls_friends": True, "call_range": 8, "call_count": 2, "tactic": "isolate"},
    "Arclouse": {"calls_friends": True, "call_range": 6, "call_count": 1, "tactic": "isolate"},
    "Ancient Worm": {"calls_friends": True, "call_range": 12, "call_count": 4, "tactic": "isolate_or_avoid"},
    "Maya": {"calls_friends": True, "call_range": 15, "call_count": 5, "tactic": "mvp_preparation"},
    "Maya Purple": {"calls_friends": True, "call_range": 15, "call_count": 5, "tactic": "mvp_preparation"},
    "Phreeoni": {"calls_friends": True, "call_range": 12, "call_count": 3, "tactic": "mvp_preparation"},
    "Eddga": {"calls_friends": True, "call_range": 10, "call_count": 3, "tactic": "mvp_preparation"},

    # ── Teleport when hit ──
    "Whisper": {"teleports": True, "teleport_hp_pct": 0.50, "tactic": "stun_or_freeze_first"},
    "Deviling": {"teleports": True, "teleport_hp_pct": 0.60, "tactic": "stun_or_freeze_first"},
    "Ghostring": {"teleports": True, "teleport_hp_pct": 0.40, "tactic": "stun_or_freeze_first"},
    "Angeling": {"teleports": True, "teleport_hp_pct": 0.30, "tactic": "stun_or_freeze_first"},
    "Archangeling": {"teleports": True, "teleport_hp_pct": 0.30, "tactic": "stun_or_freeze_first"},
    "Mastering": {"teleports": True, "teleport_hp_pct": 0.25, "tactic": "stun_or_freeze_first"},
    "Mistress": {"teleports": True, "teleport_hp_pct": 0.50, "tactic": "mvp_stun_or_freeze"},
    "Moonlight Flower": {"teleports": True, "teleport_hp_pct": 0.40, "tactic": "mvp_stun_or_freeze"},
    "Doppelganger": {"teleports": True, "teleport_hp_pct": 0.30, "tactic": "mvp_stun_or_freeze"},

    # ── Aggressive only at night ──
    "Bathory": {"aggro_night_only": True, "tactic": "avoid_at_night"},
    "Isis": {"aggro_night_only": True, "tactic": "avoid_at_night"},
    "Mummy": {"aggro_night_only": True, "tactic": "avoid_at_night"},
    "Mummy Guard": {"aggro_night_only": True, "tactic": "avoid_at_night"},
    "Evil Druid": {"aggro_night_only": True, "tactic": "avoid_at_night"},
    "Wraith": {"aggro_night_only": True, "tactic": "avoid_at_night"},
    "Wraith Dead": {"aggro_night_only": True, "tactic": "avoid_at_night"},
    "Nightmare": {"aggro_night_only": True, "tactic": "avoid_at_night"},
    "Nightmare Terror": {"aggro_night_only": True, "tactic": "avoid_at_night"},

    # ── Use skills (interrupt priority) ──
    "Magnolia": {"uses_skills": True, "skills": ["heal"], "tactic": "interrupt_healer"},
    "Arclouse": {"uses_skills": True, "skills": ["provoke"], "tactic": "interrupt_provoker"},
    "Peco Peco": {"uses_skills": True, "skills": ["charge"], "tactic": "kite"},
    "Peco Peco Egg": {"uses_skills": False, "tactic": "stationary"},
    "Orc Warrior": {"uses_skills": True, "skills": ["bash"], "tactic": "tank_or_kite"},
    "Orc Archer": {"uses_skills": True, "skills": ["double_strafe"], "tactic": "rush_down"},
    "Orc Lady": {"uses_skills": True, "skills": ["heal"], "tactic": "interrupt_healer"},
    "Succubus": {"uses_skills": True, "skills": ["soul_drain"], "tactic": "ranged_kite"},
    "Incubus": {"uses_skills": True, "skills": ["soul_drain"], "tactic": "ranged_kite"},
    "Medusa": {"uses_skills": True, "skills": ["stone_curse"], "tactic": "ranged_or_anti_stone"},
    "Strouf": {"uses_skills": True, "skills": ["waterball"], "tactic": "ranged_kite"},
    "Obeaune": {"uses_skills": True, "skills": ["waterball"], "tactic": "ranged_kite"},

    # ── MVP monsters ──
    "MVP": {
        "is_mvp": True,
        "tactic": "mvp_preparation",
        "recommended_party": True,
        "recommended_buffs": ["blessing", "agnus_dei", "gloria", "magnificat", "assumptio"],
    },
    "Golden Thief Bug": {
        "is_mvp": True, "teleports": True, "calls_friends": True,
        "tactic": "mvp_preparation", "element": "neutral", "size": "large",
    },
    "Osiris": {
        "is_mvp": True, "teleports": True, "calls_friends": True,
        "tactic": "mvp_preparation", "element": "undead", "size": "medium",
        "weakness": "holy",
    },
    "Dracula": {
        "is_mvp": True, "teleports": True, "uses_skills": True,
        "skills": ["soul_drain", "teleport"],
        "tactic": "mvp_preparation", "element": "dark", "size": "large",
        "weakness": "holy",
    },
    "Baphomet": {
        "is_mvp": True, "teleports": True, "calls_friends": True,
        "tactic": "mvp_preparation", "element": "dark", "size": "large",
        "weakness": "holy",
    },
    "Lord of the Dead": {
        "is_mvp": True, "calls_friends": True, "uses_skills": True,
        "skills": ["hell_judgment"],
        "tactic": "mvp_preparation", "element": "undead", "size": "large",
        "weakness": "holy",
    },
    "Dark Lord": {
        "is_mvp": True, "calls_friends": True, "teleports": True,
        "tactic": "mvp_preparation", "element": "undead", "size": "large",
        "weakness": "holy",
    },
    "Orc Hero": {
        "is_mvp": True, "uses_skills": True,
        "skills": ["bash", "provoke"],
        "tactic": "mvp_preparation", "element": "earth", "size": "large",
    },
    "Orc Lord": {
        "is_mvp": True, "uses_skills": True, "calls_friends": True,
        "skills": ["bash", "provoke", "critical_slash"],
        "tactic": "mvp_preparation", "element": "earth", "size": "large",
    },
    "Pharaoh": {
        "is_mvp": True, "teleports": True,
        "tactic": "mvp_preparation", "element": "neutral", "size": "medium",
    },
    "Knight of Windstorm": {
        "is_mvp": True, "uses_skills": True,
        "skills": ["brandish_spear", "spear_boomerang"],
        "tactic": "mvp_preparation", "element": "wind", "size": "large",
    },
    "Detardeurus": {
        "is_mvp": True, "uses_skills": True,
        "skills": ["fire_breath"],
        "tactic": "mvp_preparation", "element": "fire", "size": "large",
    },
    "Atroce": {
        "is_mvp": True, "teleports": True, "calls_friends": True,
        "tactic": "mvp_preparation", "element": "dark", "size": "large",
        "weakness": "holy",
    },
    "Garm": {
        "is_mvp": True, "uses_skills": True,
        "skills": ["storm_gust", "ice_breath"],
        "tactic": "mvp_preparation", "element": "water", "size": "large",
    },
    "Ktullanux": {
        "is_mvp": True, "uses_skills": True,
        "skills": ["waterball", "ice_breath"],
        "tactic": "mvp_preparation", "element": "water", "size": "large",
    },
    "Valkyrie Randgris": {
        "is_mvp": True, "teleports": True, "uses_skills": True,
        "skills": ["thunder_storm", "heal"],
        "tactic": "mvp_preparation", "element": "holy", "size": "large",
    },
    "Ifrit": {
        "is_mvp": True, "uses_skills": True, "calls_friends": True,
        "skills": ["fire_breath", "meteor_storm"],
        "tactic": "mvp_preparation", "element": "fire", "size": "large",
    },
    "Beelzebub": {
        "is_mvp": True, "teleports": True, "calls_friends": True,
        "tactic": "mvp_preparation", "element": "dark", "size": "large",
        "weakness": "holy",
    },
    "Turtle General": {
        "is_mvp": True, "uses_skills": True,
        "skills": ["ground_shatter"],
        "tactic": "mvp_preparation", "element": "earth", "size": "large",
    },
    "Mao Guai": {
        "is_mvp": True, "teleports": True,
        "tactic": "mvp_preparation", "element": "wind", "size": "medium",
    },
    "Evil Snake Lord": {
        "is_mvp": True, "calls_friends": True,
        "tactic": "mvp_preparation", "element": "poison", "size": "large",
    },
    "Tao Gunka": {
        "is_mvp": True, "uses_skills": True,
        "skills": ["full_recovery"],
        "tactic": "mvp_preparation", "element": "neutral", "size": "large",
    },
    "Leak": {
        "is_mvp": True, "teleports": True,
        "tactic": "mvp_preparation", "element": "water", "size": "medium",
    },
    "Gioia": {
        "is_mvp": True, "calls_friends": True,
        "tactic": "mvp_preparation", "element": "holy", "size": "large",
    },
    "Skeggiold": {
        "is_mvp": True, "uses_skills": True,
        "skills": ["dark_breath"],
        "tactic": "mvp_preparation", "element": "dark", "size": "large",
        "weakness": "holy",
    },
    "Bacsojin": {
        "is_mvp": True, "teleports": True,
        "tactic": "mvp_preparation", "element": "poison", "size": "medium",
    },
    "Kiel-D-01": {
        "is_mvp": True, "uses_skills": True,
        "skills": ["soul_drain", "teleport"],
        "tactic": "mvp_preparation", "element": "dark", "size": "medium",
        "weakness": "holy",
    },
    "Egnigem Cenia": {
        "is_mvp": True, "uses_skills": True,
        "skills": ["fire_bolt", "meteor_storm"],
        "tactic": "mvp_preparation", "element": "fire", "size": "medium",
    },
    "Memory of Thanatos": {
        "is_mvp": True, "teleports": True, "calls_friends": True,
        "tactic": "mvp_preparation", "element": "neutral", "size": "large",
    },
    "Randgris": {
        "is_mvp": True, "uses_skills": True,
        "skills": ["thunder_storm", "heal"],
        "tactic": "mvp_preparation", "element": "holy", "size": "large",
    },
    "Flail": {
        "is_mvp": True, "calls_friends": True,
        "tactic": "mvp_preparation", "element": "neutral", "size": "large",
    },
    "Gloom Under Night": {
        "is_mvp": True, "teleports": True, "uses_skills": True,
        "skills": ["dark_breath", "soul_drain"],
        "tactic": "mvp_preparation", "element": "dark", "size": "large",
        "weakness": "holy",
    },
}


def get_monster_ai(name: str) -> dict[str, Any] | None:
    """Get AI override data for a monster by name.

    Returns None if no override exists (use default behavior).
    """
    return MONSTER_AI_OVERRIDES.get(name)


def get_monster_tactic(name: str) -> str:
    """Get the recommended tactic for a monster.

    Returns 'default' if no specific tactic is defined.
    """
    override = MONSTER_AI_OVERRIDES.get(name)
    if override:
        return override.get("tactic", "default")
    return "default"


def should_one_shot(name: str) -> bool:
    """Check if a monster should be one-shot to prevent fleeing."""
    override = MONSTER_AI_OVERRIDES.get(name)
    if override and override.get("flees_at_low_hp"):
        return True
    return False


def should_isolate(name: str) -> bool:
    """Check if a monster should be isolated before attacking (calls friends)."""
    override = MONSTER_AI_OVERRIDES.get(name)
    if override and override.get("calls_friends"):
        return True
    return False


def should_stun_first(name: str) -> bool:
    """Check if a monster should be stunned/frozen first (teleports)."""
    override = MONSTER_AI_OVERRIDES.get(name)
    if override and override.get("teleports"):
        return True
    return False


def is_night_aggro(name: str) -> bool:
    """Check if a monster is aggressive only at night."""
    override = MONSTER_AI_OVERRIDES.get(name)
    if override and override.get("aggro_night_only"):
        return True
    return False


def is_mvp(name: str) -> bool:
    """Check if a monster is an MVP."""
    override = MONSTER_AI_OVERRIDES.get(name)
    if override and override.get("is_mvp"):
        return True
    return False


def get_mvp_weakness(name: str) -> str | None:
    """Get the elemental weakness of an MVP monster."""
    override = MONSTER_AI_OVERRIDES.get(name)
    if override:
        return override.get("weakness")
    return None


def get_recommended_buffs(name: str) -> list[str]:
    """Get recommended buffs for fighting a specific monster."""
    override = MONSTER_AI_OVERRIDES.get(name)
    if override:
        return override.get("recommended_buffs", [])
    return []
