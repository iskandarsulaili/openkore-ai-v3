"""CombatAgent — element/size/race/MVP/PVP/GVG/party combat decisions."""

from __future__ import annotations

from typing import Any

from ai_sidecar.fleet.behavior_profiles import BehaviorProfile

_ELEMENT_WEAKNESS = {
    "fire": "earth", "earth": "wind", "wind": "water", "water": "fire",
    "poison": "neutral", "holy": "shadow", "shadow": "ghost",
    "ghost": "poison", "undead": "holy",
}
_ELEMENT_ADVANTAGE = {
    "neutral": 1.0, "water": 1.0, "earth": 1.0, "fire": 1.0, "wind": 1.0,
    "poison": 1.0, "holy": 1.75, "shadow": 1.0, "ghost": 0.75, "undead": 1.25,
}

_WEAPON_VS_SIZE = {
    "dagger": {"small": 1.0, "medium": 0.75, "large": 0.5},
    "sword": {"small": 0.75, "medium": 1.0, "large": 0.75},
    "two_hand_sword": {"small": 0.6, "medium": 0.75, "large": 1.0},
    "spear": {"small": 0.75, "medium": 1.0, "large": 1.0},
    "mace": {"small": 1.0, "medium": 1.0, "large": 1.0},
    "axe": {"small": 0.6, "medium": 0.8, "large": 1.0},
    "bow": {"small": 0.75, "medium": 1.0, "large": 1.0},
    "staff": {"small": 1.0, "medium": 1.0, "large": 1.0},
    "knuckle": {"small": 1.0, "medium": 0.75, "large": 0.5},
    "whip": {"small": 1.0, "medium": 0.75, "large": 0.5},
    "instrument": {"small": 1.0, "medium": 0.75, "large": 0.5},
}


class CombatAgent(BehaviorProfile):
    """Handles element, size, race, MVP, PVP, GVG, party combat, and skill rotation."""

    def best_weapon_for(self, monster_size: str, monster_element: str, monster_race: str) -> dict[str, Any]:
        strong_element = _ELEMENT_WEAKNESS.get(monster_element, "neutral")
        best_weapon, best_dmg = "", 0.0
        for weapon, size_map in _WEAPON_VS_SIZE.items():
            size_mod = size_map.get(monster_size, 0.75)
            element_mod = _ELEMENT_ADVANTAGE.get(strong_element, 1.0)
            dmg = size_mod * element_mod
            if dmg > best_dmg:
                best_dmg, best_weapon = dmg, weapon
        return {"recommended_weapon": best_weapon, "damage_multiplier": best_dmg,
                "element_advantage": strong_element}

    def choose_mvp_strategy(self, mvp_hp_pct: float, party_healers: int) -> str:
        if mvp_hp_pct > 0.75:
            return "debuff_first"  # Strip, lex divina, break
        if mvp_hp_pct > 0.3:
            return "sustained_dps"
        if party_healers > 0:
            return "burst_with_heals"
        return "kite_and_skill"

    def pvp_action(self, target_hp_pct: float, my_hp_pct: float) -> str:
        if my_hp_pct < 0.3:
            return "flee_or_teleport"
        if target_hp_pct < 0.4:
            return "burst_skill"
        return "pvp_engage"

    def gvg_action(self, target_is_emperium: bool, allies_nearby: int) -> str:
        if target_is_emperium:
            return "attack_emperium_skill"
        if allies_nearby >= 3:
            return "aoe_offensive"
        return "support_allies"

    def party_combat(self, role: str, tank_hp_pct: float, mob_count: int) -> str:
        if role == "tank" and tank_hp_pct < 0.5:
            return "use_defensive_skill"
        if role == "tank":
            return "provoke_and_hold"
        if role == "healer" and any(self._signals.get("ally_low_hp", [])):
            return "heal_party"
        if role == "buffer":
            return "buff_party"
        if role in ("dps_melee", "dps_ranged", "dps_magic"):
            return "attack_optimal_target"
        return "attack"

    def decide_skill_rotation(self, sp_pct: float, sp_regen_rate: float) -> list[str]:
        best, score = self.best_action("combat")
        if best and score > 0.6 and sp_pct > 0.3:
            return [best, "basic_attack"]
        if sp_pct > 0.5:
            return ["high_damage_skill", "utility_skill", "basic_attack"]
        if sp_regen_rate > 10:
            return ["medium_skill", "basic_attack"]
        return ["basic_attack"]

    def record_outcome(self, action: str, success: bool, damage: float = 0.0) -> None:
        self._record_experience("combat", action, success, reward=damage, damage_dealt=damage)
