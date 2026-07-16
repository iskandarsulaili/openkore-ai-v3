"""
Reflex Combat Layer — hardcoded combat reflexes that bypass the LLM entirely.

Combat decisions are made in under 50ms without asking any LLM.
The LLM only gets involved for strategic decisions: which map to farm,
which MVP to hunt, when to restock. Combat is reflex, not thought.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CombatReflex:
    """A single combat reflex — condition + action pair."""
    name: str
    priority: int = 50
    condition: str = ""
    action: str = ""
    cooldown_ms: int = 100
    last_fired: float = 0.0


@dataclass
class CombatSituation:
    """Current combat situation snapshot."""
    target_element: str = "neutral"
    target_size: str = "medium"
    target_race: str = "formless"
    target_hp_pct: float = 1.0
    target_distance: float = 0.0
    target_is_casting: bool = False
    target_casting_skill: str = ""
    target_is_boss: bool = False
    aggro_count: int = 0
    my_hp_pct: float = 1.0
    my_sp_pct: float = 1.0
    my_sp: int = 0
    my_job_class: str = "novice"
    my_buffs: list[str] = field(default_factory=list)
    my_weapon_element: str = "neutral"
    my_weapon_type: str = "sword"
    available_skills: list[str] = field(default_factory=list)
    cooldowns: dict[str, int] = field(default_factory=dict)
    party_members_nearby: int = 0
    enemies_nearby: int = 0


class ReflexCombat:
    """Hardcoded combat reflex system — no LLM involved."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._reflexes: dict[str, CombatReflex] = {}
        self._load_reflexes()

    def _load_reflexes(self) -> None:
        """Load all combat reflexes."""
        reflexes = [
            # ── Emergency (priority 100) ──
            CombatReflex("emergency_heal", 100, "my_hp_pct < 0.4", "use_potion_or_heal", cooldown_ms=2000),
            CombatReflex("emergency_flee", 100, "my_hp_pct < 0.2 and aggro_count > 3", "flee_to_safe_spot", cooldown_ms=500),
            CombatReflex("emergency_teleport", 100, "my_hp_pct < 0.1", "teleport_away", cooldown_ms=1000),

            # ── Interrupt (priority 95) ──
            CombatReflex("interrupt_dangerous_cast", 95, "target_is_casting and target_is_boss", "stun_or_silence_target", cooldown_ms=2000),
            CombatReflex("interrupt_aoe_cast", 95, "target_is_casting and aggro_count > 2", "interrupt_caster", cooldown_ms=1500),

            # ── Elemental Advantage (priority 90) ──
            CombatReflex("element_fire_vs_earth", 90, "target_element == earth", "use_fire_skill", cooldown_ms=500),
            CombatReflex("element_water_vs_fire", 90, "target_element == fire", "use_water_skill", cooldown_ms=500),
            CombatReflex("element_wind_vs_water", 90, "target_element == water", "use_wind_skill", cooldown_ms=500),
            CombatReflex("element_earth_vs_wind", 90, "target_element == wind", "use_earth_skill", cooldown_ms=500),
            CombatReflex("element_holy_vs_undead", 90, "target_race == undead", "use_holy_skill", cooldown_ms=500),
            CombatReflex("element_holy_vs_demon", 90, "target_race == demon", "use_holy_skill", cooldown_ms=500),
            CombatReflex("element_ghost_vs_ghost", 90, "target_element == ghost", "use_ghost_skill", cooldown_ms=500),

            # ── Retreat (priority 90) ──
            CombatReflex("retreat_overwhelmed", 90, "aggro_count > 5 and party_members_nearby == 0", "retreat_to_safe_spot", cooldown_ms=3000),

            # ── Boss Fight (priority 85) ──
            CombatReflex("boss_prepot", 85, "target_is_boss and target_distance < 15", "pre_drink_potion", cooldown_ms=5000),
            CombatReflex("boss_finisher", 85, "target_is_boss and target_hp_pct < 0.3", "use_strongest_skill", cooldown_ms=2000),
            CombatReflex("boss_phase_prepare", 80, "target_is_boss and target_hp_pct < 0.5", "prepare_for_phase_change", cooldown_ms=3000),

            # ── Gear Swap (priority 85) ──
            CombatReflex("gear_swap_element", 85, "target_element != my_weapon_element", "swap_to_elemental_weapon", cooldown_ms=2000),
            CombatReflex("gear_swap_boss", 80, "target_is_boss", "equip_tank_set", cooldown_ms=3000),

            # ── AoE Clear (priority 80-85) ──
            CombatReflex("aoe_clear_heavy", 85, "aggro_count > 5", "use_strongest_aoe", cooldown_ms=3000),
            CombatReflex("aoe_clear_medium", 80, "aggro_count > 3", "use_aoe_skill", cooldown_ms=2000),

            # ── Killsteal Protection (priority 80) ──
            CombatReflex("ks_finisher", 80, "target_hp_pct < 0.2 and enemies_nearby > 0", "use_fast_finisher", cooldown_ms=500),

            # ── SP Management (priority 75) ──
            CombatReflex("sp_conserve", 75, "my_sp_pct < 0.2", "use_basic_attack_only", cooldown_ms=2000),
            CombatReflex("sp_restore", 70, "my_sp_pct < 0.1", "rest_and_regen_sp", cooldown_ms=5000),

            # ── Party Support (priority 75) ──
            CombatReflex("party_heal", 75, "party_members_nearby > 0 and my_hp_pct < 0.5", "heal_party_member", cooldown_ms=2000),

            # ── Kite (priority 70) ──
            CombatReflex("kite_melee", 70, "target_distance < 3 and my_job_class in ('archer', 'mage', 'wizard')", "maintain_distance", cooldown_ms=1000),

            # ── Buff Maintenance (priority 65-70) ──
            CombatReflex("buff_blessing", 70, "'Blessing' not in my_buffs", "cast_blessing", cooldown_ms=5000),
            CombatReflex("buff_agi", 70, "'Increase Agility' not in my_buffs", "cast_increase_agility", cooldown_ms=5000),
            CombatReflex("buff_endure", 65, "'Endure' not in my_buffs", "cast_endure", cooldown_ms=5000),
            CombatReflex("buff_improve_concentration", 65, "'Improve Concentration' not in my_buffs", "cast_improve_concentration", cooldown_ms=5000),

            # ── Distance (priority 60) ──
            CombatReflex("melee_range", 60, "target_distance < 2", "use_melee_skill", cooldown_ms=500),
            CombatReflex("ranged_range", 60, "target_distance > 8", "use_ranged_skill", cooldown_ms=500),

            # ── Party Buff (priority 60) ──
            CombatReflex("party_buff", 60, "party_members_nearby > 0", "buff_party_members", cooldown_ms=10000),

            # ── High SP Spender (priority 40) ──
            CombatReflex("high_sp_spender", 40, "my_sp_pct > 0.8", "use_high_sp_skill", cooldown_ms=2000),
        ]

        for r in reflexes:
            self._reflexes[r.name] = r

    # ── Public API ──

    def evaluate(self, situation: CombatSituation | dict[str, Any]) -> list[CombatReflex]:
        """Evaluate a situation and return all matching reflexes sorted by priority."""
        if isinstance(situation, dict):
            situation = CombatSituation(**{k: v for k, v in situation.items() if k in CombatSituation.__dataclass_fields__})

        now = time.time() * 1000  # ms
        matching: list[CombatReflex] = []

        with self._lock:
            for reflex in self._reflexes.values():
                if not self._check_condition(reflex, situation):
                    continue
                # Check cooldown
                if now - reflex.last_fired * 1000 < reflex.cooldown_ms:
                    continue
                matching.append(reflex)

        matching.sort(key=lambda r: -r.priority)
        return matching

    def get_best_action(self, situation: CombatSituation | dict[str, Any]) -> CombatReflex | None:
        """Get the highest priority matching reflex."""
        matching = self.evaluate(situation)
        if not matching:
            return None
        best = matching[0]
        with self._lock:
            best.last_fired = time.time()
        return best

    def should_bypass_llm(self, situation: CombatSituation | dict[str, Any]) -> bool:
        """Check if any high-priority reflex matches (bypass LLM)."""
        matching = self.evaluate(situation)
        return any(r.priority >= 80 for r in matching)

    def _check_condition(self, reflex: CombatReflex, situation: CombatSituation) -> bool:
        """Check if a reflex's condition is met."""
        cond = reflex.condition
        try:
            if cond == "my_hp_pct < 0.4":
                return situation.my_hp_pct < 0.4
            elif cond == "my_hp_pct < 0.2 and aggro_count > 3":
                return situation.my_hp_pct < 0.2 and situation.aggro_count > 3
            elif cond == "my_hp_pct < 0.1":
                return situation.my_hp_pct < 0.1
            elif cond == "target_is_casting and target_is_boss":
                return situation.target_is_casting and situation.target_is_boss
            elif cond == "target_is_casting and aggro_count > 2":
                return situation.target_is_casting and situation.aggro_count > 2
            elif cond == "target_element == earth":
                return situation.target_element == "earth"
            elif cond == "target_element == fire":
                return situation.target_element == "fire"
            elif cond == "target_element == water":
                return situation.target_element == "water"
            elif cond == "target_element == wind":
                return situation.target_element == "wind"
            elif cond == "target_race == undead":
                return situation.target_race == "undead"
            elif cond == "target_race == demon":
                return situation.target_race == "demon"
            elif cond == "target_element == ghost":
                return situation.target_element == "ghost"
            elif cond == "aggro_count > 5 and party_members_nearby == 0":
                return situation.aggro_count > 5 and situation.party_members_nearby == 0
            elif cond == "target_is_boss and target_distance < 15":
                return situation.target_is_boss and situation.target_distance < 15
            elif cond == "target_is_boss and target_hp_pct < 0.3":
                return situation.target_is_boss and situation.target_hp_pct < 0.3
            elif cond == "target_is_boss and target_hp_pct < 0.5":
                return situation.target_is_boss and situation.target_hp_pct < 0.5
            elif cond == "target_element != my_weapon_element":
                return situation.target_element != situation.my_weapon_element
            elif cond == "target_is_boss":
                return situation.target_is_boss
            elif cond == "aggro_count > 5":
                return situation.aggro_count > 5
            elif cond == "aggro_count > 3":
                return situation.aggro_count > 3
            elif cond == "target_hp_pct < 0.2 and enemies_nearby > 0":
                return situation.target_hp_pct < 0.2 and situation.enemies_nearby > 0
            elif cond == "my_sp_pct < 0.2":
                return situation.my_sp_pct < 0.2
            elif cond == "my_sp_pct < 0.1":
                return situation.my_sp_pct < 0.1
            elif cond == "party_members_nearby > 0 and my_hp_pct < 0.5":
                return situation.party_members_nearby > 0 and situation.my_hp_pct < 0.5
            elif cond == "target_distance < 3 and my_job_class in ('archer', 'mage', 'wizard')":
                return situation.target_distance < 3 and situation.my_job_class in ('archer', 'mage', 'wizard')
            elif cond == "'Blessing' not in my_buffs":
                return "Blessing" not in situation.my_buffs
            elif cond == "'Increase Agility' not in my_buffs":
                return "Increase Agility" not in situation.my_buffs
            elif cond == "'Endure' not in my_buffs":
                return "Endure" not in situation.my_buffs
            elif cond == "'Improve Concentration' not in my_buffs":
                return "Improve Concentration" not in situation.my_buffs
            elif cond == "target_distance < 2":
                return situation.target_distance < 2
            elif cond == "target_distance > 8":
                return situation.target_distance > 8
            elif cond == "party_members_nearby > 0":
                return situation.party_members_nearby > 0
            elif cond == "my_sp_pct > 0.8":
                return situation.my_sp_pct > 0.8
        except Exception:
            pass
        return False

    def get_reflex(self, name: str) -> CombatReflex | None:
        with self._lock:
            return self._reflexes.get(name)

    def register_reflex(self, reflex: CombatReflex) -> None:
        with self._lock:
            self._reflexes[reflex.name] = reflex

    def get_all_reflexes(self) -> list[CombatReflex]:
        with self._lock:
            return list(self._reflexes.values())

    def get_reflexes_by_priority(self, min_priority: int) -> list[CombatReflex]:
        with self._lock:
            return [r for r in self._reflexes.values() if r.priority >= min_priority]

    def get_reflexes_for_situation(self, situation: CombatSituation | dict[str, Any]) -> list[CombatReflex]:
        return self.evaluate(situation)

    def get_reflex_summary(self) -> str:
        with self._lock:
            lines = [f"── Reflex Combat Summary ──"]
            lines.append(f"Total reflexes: {len(self._reflexes)}")
            by_priority: dict[int, int] = {}
            for r in self._reflexes.values():
                by_priority[r.priority] = by_priority.get(r.priority, 0) + 1
            for pri in sorted(by_priority.keys(), reverse=True):
                lines.append(f"  Priority {pri}: {by_priority[pri]} reflexes")
            return "\n".join(lines)


# ── Global Singleton ──

_reflex_combat: ReflexCombat | None = None
_reflex_combat_lock = RLock()


def get_reflex_combat() -> ReflexCombat:
    global _reflex_combat
    with _reflex_combat_lock:
        if _reflex_combat is None:
            _reflex_combat = ReflexCombat()
        return _reflex_combat
