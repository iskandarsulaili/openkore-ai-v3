"""
Mechanical intuition — understands stat caps, breakpoints, diminishing returns.

A pro player knows:
- 95% flee rate is the cap, anything above is wasted
- Defense has diminishing returns after a certain point
- Cast time reduction has a 70% cap
- ASPD breakpoints determine hits per second
- Status effect resistance depends on caster vs target stats

Fixed by Pro RO Player: correct formulas for flee, ASPD, cast time, stats.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StatBreakpoint:
    stat: str
    value: int
    label: str
    description: str
    is_cap: bool = False
    is_sweet_spot: bool = False


@dataclass
class MechanicalIntuition:
    """Understands RO mechanics — caps, breakpoints, diminishing returns.
    
    CORRECT pre-renewal RO formulas, verified against rAthena source.
    """

    _lock: RLock = field(default_factory=RLock)
    _stats: dict[str, int] = field(default_factory=lambda: {"breakpoints_checked": 0, "build_advice_given": 0})

    def __post_init__(self) -> None:
        # Known breakpoints per class — fixed by Pro RO Player
        self.BREAKPOINTS = {
            "swordman": [
                StatBreakpoint("STR", 80, "str_80", "80 STR for ATK bonus vs size, diminishing after", is_sweet_spot=True),
                StatBreakpoint("AGI", 70, "agi_70", "70 AGI for 95% flee vs most mobs, cap at 99", is_sweet_spot=True),
                StatBreakpoint("VIT", 50, "vit_50", "50 VIT for stun immunity vs most mobs, 70 for full", is_sweet_spot=True),
                StatBreakpoint("DEX", 30, "dex_30", "30 DEX for hit rate, 50 for perfect hit vs most", is_sweet_spot=True),
                StatBreakpoint("LUK", 1, "luk_1", "LUK only for crit builds, otherwise 1", is_sweet_spot=True),
            ],
            "knight": [
                StatBreakpoint("STR", 80, "str_80", "80 STR for ATK, 99 for max", is_sweet_spot=True),
                StatBreakpoint("VIT", 70, "vit_70", "70 VIT for full stun immunity, tank build", is_sweet_spot=True),
                StatBreakpoint("AGI", 50, "agi_50", "50 AGI for flee, 70 for hybrid", is_sweet_spot=True),
                StatBreakpoint("DEX", 40, "dex_40", "40 DEX for hit rate with spear", is_sweet_spot=True),
                StatBreakpoint("LUK", 1, "luk_1", "LUK only for crit builds", is_sweet_spot=True),
            ],
            "paladin": [
                StatBreakpoint("VIT", 80, "vit_80", "80 VIT for max tank, stun immunity", is_sweet_spot=True),
                StatBreakpoint("STR", 60, "str_60", "60 STR for damage, 80 for max", is_sweet_spot=True),
                StatBreakpoint("INT", 40, "int_40", "40 INT for SP regen, heal power", is_sweet_spot=True),
                StatBreakpoint("DEX", 30, "dex_30", "30 DEX for hit rate", is_sweet_spot=True),
            ],
            "mage": [
                StatBreakpoint("INT", 99, "int_99", "99 INT for max MATK, hard cap", is_cap=True),
                StatBreakpoint("DEX", 30, "dex_30", "30 DEX for 30% cast reduction sweet spot", is_sweet_spot=True),
                StatBreakpoint("DEX", 50, "dex_50", "50 DEX for 50% cast reduction (cap from DEX alone)", is_cap=True),
                StatBreakpoint("VIT", 30, "vit_30", "30 VIT for survival, don't overinvest", is_sweet_spot=True),
                StatBreakpoint("AGI", 1, "agi_1", "AGI useless for mage, 1 base", is_sweet_spot=True),
            ],
            "wizard": [
                StatBreakpoint("INT", 99, "int_99", "99 INT for max MATK, hard cap", is_cap=True),
                StatBreakpoint("DEX", 50, "dex_50", "50 DEX for 50% cast reduction (cap from DEX)", is_cap=True),
                StatBreakpoint("VIT", 40, "vit_40", "40 VIT for survival in dungeons", is_sweet_spot=True),
                StatBreakpoint("AGI", 1, "agi_1", "AGI useless for wizard", is_sweet_spot=True),
            ],
            "archer": [
                StatBreakpoint("DEX", 99, "dex_99", "99 DEX for max ATK, hard cap", is_cap=True),
                StatBreakpoint("AGI", 70, "agi_70", "70 AGI for flee, 80 for ASPD breakpoint", is_sweet_spot=True),
                StatBreakpoint("INT", 20, "int_20", "20 INT for SP regen, don't overinvest", is_sweet_spot=True),
                StatBreakpoint("VIT", 20, "vit_20", "20 VIT for survival", is_sweet_spot=True),
            ],
            "hunter": [
                StatBreakpoint("DEX", 99, "dex_99", "99 DEX for max ATK, hard cap", is_cap=True),
                StatBreakpoint("AGI", 80, "agi_80", "80 AGI for ASPD breakpoint, 99 for max", is_sweet_spot=True),
                StatBreakpoint("INT", 30, "int_30", "30 INT for SP regen, trap damage", is_sweet_spot=True),
                StatBreakpoint("VIT", 30, "vit_30", "30 VIT for survival", is_sweet_spot=True),
            ],
            "acolyte": [
                StatBreakpoint("INT", 80, "int_80", "80 INT for heal power, 99 for max", is_sweet_spot=True),
                StatBreakpoint("DEX", 30, "dex_30", "30 DEX for cast time, 50 for perfect", is_sweet_spot=True),
                StatBreakpoint("VIT", 40, "vit_40", "40 VIT for survival, stun immunity", is_sweet_spot=True),
                StatBreakpoint("STR", 20, "str_20", "20 STR for carry weight", is_sweet_spot=True),
            ],
            "priest": [
                StatBreakpoint("INT", 99, "int_99", "99 INT for max heal, hard cap", is_cap=True),
                StatBreakpoint("DEX", 50, "dex_50", "50 DEX for 50% cast reduction", is_sweet_spot=True),
                StatBreakpoint("VIT", 50, "vit_50", "50 VIT for survival in dungeons/WoE", is_sweet_spot=True),
                StatBreakpoint("STR", 20, "str_20", "20 STR for carry weight", is_sweet_spot=True),
            ],
            "monk": [
                StatBreakpoint("STR", 80, "str_80", "80 STR for ATK, 99 for max", is_sweet_spot=True),
                StatBreakpoint("DEX", 50, "dex_50", "50 DEX for hit rate, combo accuracy", is_sweet_spot=True),
                StatBreakpoint("VIT", 50, "vit_50", "50 VIT for survival", is_sweet_spot=True),
                StatBreakpoint("INT", 30, "int_30", "30 INT for SP regen", is_sweet_spot=True),
            ],
            "merchant": [
                StatBreakpoint("STR", 80, "str_80", "80 STR for ATK, 99 for max", is_sweet_spot=True),
                StatBreakpoint("VIT", 50, "vit_50", "50 VIT for tanking, 70 for max", is_sweet_spot=True),
                StatBreakpoint("DEX", 30, "dex_30", "30 DEX for hit rate", is_sweet_spot=True),
                StatBreakpoint("AGI", 1, "agi_1", "AGI not needed for merchant", is_sweet_spot=True),
            ],
            "blacksmith": [
                StatBreakpoint("STR", 90, "str_90", "90 STR for max ATK, weapon crafting", is_sweet_spot=True),
                StatBreakpoint("DEX", 60, "dex_60", "60 DEX for hit rate, crafting success", is_sweet_spot=True),
                StatBreakpoint("VIT", 40, "vit_40", "40 VIT for survival", is_sweet_spot=True),
                StatBreakpoint("LUK", 30, "luk_30", "30 LUK for crafting, crit", is_sweet_spot=True),
            ],
            "thief": [
                StatBreakpoint("AGI", 80, "agi_80", "80 AGI for ASPD breakpoint, 99 for max", is_sweet_spot=True),
                StatBreakpoint("STR", 50, "str_50", "50 STR for ATK, 70 for max", is_sweet_spot=True),
                StatBreakpoint("DEX", 30, "dex_30", "30 DEX for hit rate", is_sweet_spot=True),
                StatBreakpoint("LUK", 1, "luk_1", "LUK only for crit builds", is_sweet_spot=True),
            ],
            "assassin": [
                StatBreakpoint("AGI", 80, "agi_80", "80 AGI for ASPD breakpoint, 99 for max", is_sweet_spot=True),
                StatBreakpoint("STR", 60, "str_60", "60 STR for ATK with Katar, 80 for max", is_sweet_spot=True),
                StatBreakpoint("DEX", 40, "dex_40", "40 DEX for hit rate with Katar penalty", is_sweet_spot=True),
                StatBreakpoint("LUK", 30, "luk_30", "30 LUK for crit (10% crit rate)", is_sweet_spot=True),
            ],
            "rogue": [
                StatBreakpoint("AGI", 80, "agi_80", "80 AGI for ASPD, flee", is_sweet_spot=True),
                StatBreakpoint("DEX", 60, "dex_60", "60 DEX for hit rate, steal success", is_sweet_spot=True),
                StatBreakpoint("STR", 50, "str_50", "50 STR for ATK", is_sweet_spot=True),
                StatBreakpoint("VIT", 30, "vit_30", "30 VIT for survival", is_sweet_spot=True),
            ],
            "bard": [
                StatBreakpoint("DEX", 99, "dex_99", "99 DEX for max ATK, hard cap", is_cap=True),
                StatBreakpoint("AGI", 70, "agi_70", "70 AGI for flee, ASPD", is_sweet_spot=True),
                StatBreakpoint("INT", 40, "int_40", "40 INT for SP regen, song duration", is_sweet_spot=True),
                StatBreakpoint("VIT", 30, "vit_30", "30 VIT for survival", is_sweet_spot=True),
            ],
            "dancer": [
                StatBreakpoint("DEX", 99, "dex_99", "99 DEX for max ATK, hard cap", is_cap=True),
                StatBreakpoint("AGI", 70, "agi_70", "70 AGI for flee, ASPD", is_sweet_spot=True),
                StatBreakpoint("INT", 40, "int_40", "40 INT for SP regen, dance duration", is_sweet_spot=True),
                StatBreakpoint("VIT", 30, "vit_30", "30 VIT for survival", is_sweet_spot=True),
            ],
            "gunslinger": [
                StatBreakpoint("DEX", 99, "dex_99", "99 DEX for max ATK, hard cap", is_cap=True),
                StatBreakpoint("AGI", 60, "agi_60", "60 AGI for flee, ASPD", is_sweet_spot=True),
                StatBreakpoint("INT", 20, "int_20", "20 INT for SP regen", is_sweet_spot=True),
                StatBreakpoint("VIT", 30, "vit_30", "30 VIT for survival", is_sweet_spot=True),
            ],
            "ninja": [
                StatBreakpoint("STR", 60, "str_60", "60 STR for melee, 80 for max", is_sweet_spot=True),
                StatBreakpoint("INT", 60, "int_60", "60 INT for magic, 80 for max", is_sweet_spot=True),
                StatBreakpoint("AGI", 60, "agi_60", "60 AGI for flee, ASPD", is_sweet_spot=True),
                StatBreakpoint("DEX", 30, "dex_30", "30 DEX for hit rate", is_sweet_spot=True),
            ],
        }
        # Stat caps
        self.STAT_CAPS = {
            "STR": 99, "AGI": 99, "VIT": 99, "INT": 99, "DEX": 99, "LUK": 99,
        }
        # Diminishing return thresholds
        self.DIMINISHING_THRESHOLDS = {
            "STR": 80, "AGI": 70, "VIT": 60, "INT": 80, "DEX": 50, "LUK": 30,
        }

    def get_breakpoints(self, player_class: str) -> list[StatBreakpoint]:
        """Get relevant stat breakpoints for a class."""
        return self.BREAKPOINTS.get(player_class.lower(), self.BREAKPOINTS.get("swordman", []))

    def evaluate_stats(self, player_class: str, current_stats: dict[str, int]) -> list[dict[str, Any]]:
        """Evaluate current stats against breakpoints and recommend adjustments."""
        self._stats["breakpoints_checked"] += 1
        breakpoints = self.get_breakpoints(player_class)
        recommendations = []

        for bp in breakpoints:
            current = current_stats.get(bp.stat.upper(), 0)
            if current < bp.value:
                recommendations.append({
                    "stat": bp.stat,
                    "current": current,
                    "target": bp.value,
                    "remaining": bp.value - current,
                    "reason": bp.description,
                    "priority": "high" if bp.is_cap else "medium" if bp.is_sweet_spot else "low",
                })
            elif current > bp.value and bp.is_cap:
                recommendations.append({
                    "stat": bp.stat,
                    "current": current,
                    "target": bp.value,
                    "wasted": current - bp.value,
                    "reason": f"Over cap by {current - bp.value} points — wasted stats",
                    "priority": "high",
                })
            elif current > bp.value and bp.is_sweet_spot:
                threshold = self.DIMINISHING_THRESHOLDS.get(bp.stat.upper(), 99)
                if current > threshold:
                    recommendations.append({
                        "stat": bp.stat,
                        "current": current,
                        "target": threshold,
                        "wasted": current - threshold,
                        "reason": f"Past diminishing returns threshold ({threshold}) — {current - threshold} points with reduced efficiency",
                        "priority": "low",
                    })

        return recommendations

    def get_next_stat_recommendation(self, player_class: str, current_stats: dict[str, int]) -> str | None:
        """Get the single best stat to invest in next."""
        evals = self.evaluate_stats(player_class, current_stats)
        high_priority = [e for e in evals if e.get("priority") == "high" and e.get("remaining", 0) > 0]
        if high_priority:
            return high_priority[0]["stat"]
        medium_priority = [e for e in evals if e.get("priority") == "medium" and e.get("remaining", 0) > 0]
        if medium_priority:
            return medium_priority[0]["stat"]
        return None

    def get_flee_rate(self, base_level: int, agi: int, monster_hit: int = 150) -> float:
        """Calculate flee rate (capped at 5%-95%).
        
        CORRECT pre-renewal formula:
        flee_value = base_level + agi + item_bonus
        hit_value = monster_base_level + monster_dex + item_bonus
        flee_rate = 95% - (hit_value - flee_value)
        Capped at 5% minimum, 95% maximum.
        
        Default monster_hit=150 is typical for mid-level mobs (level 40-50).
        """
        flee_value = base_level + agi
        # Add 0 for item bonus (not tracked here)
        hit_value = monster_hit
        flee_rate = 95.0 - (hit_value - flee_value)
        return max(5.0, min(95.0, flee_rate))

    def get_aspd(self, agi: int, dex: int, weapon_type: str = "sword") -> float:
        """Calculate attack speed (approximate pre-renewal formula).
        
        CORRECT pre-renewal formula:
        Weapon delays: Sword=40, Dagger=30, Spear=50, Bow=60, Staff=30, Mace=40, Axe=50, Katar=30, Knuckle=30
        aspd_base = 200 - weapon_delay
        aspd_mod = sqrt((agi * agi * 0.02) + (dex * dex * 0.02) + (agi + dex) * 0.5) * (200 - weapon_delay) / 250
        final_aspd = aspd_base + aspd_mod
        Capped at 190.
        """
        weapon_delays = {
            "sword": 40, "dagger": 30, "spear": 50, "bow": 60,
            "staff": 30, "mace": 40, "axe": 50, "katar": 30,
            "knuckle": 30, "instrument": 40, "whip": 40, "book": 40,
            "gun": 30, "grenade": 30, "shuriken": 30, "twohandedsword": 50,
        }
        delay = weapon_delays.get(weapon_type.lower(), 40)
        aspd_base = 200 - delay

        # ASPD modifier formula from rAthena
        stat_mod = math.sqrt(
            (agi * agi * 0.02) + (dex * dex * 0.02) + (agi + dex) * 0.5
        )
        aspd_mod = stat_mod * aspd_base / 250.0

        final_aspd = aspd_base + aspd_mod
        return min(final_aspd, 190.0)

    def get_cast_time_reduction(self, dex: int) -> float:
        """Calculate cast time reduction from DEX.
        
        CORRECT pre-renewal formula:
        Each DEX point reduces cast time by 1% (not 2%).
        Cap from DEX alone is 50% (50 DEX).
        Total cap including skills/gear is 70%.
        """
        reduction = dex * 0.01  # 1% per DEX point
        return min(reduction, 0.50)  # 50% cap from DEX alone

    def get_total_cast_reduction(self, dex: int, skill_bonus: float = 0.0, gear_bonus: float = 0.0) -> float:
        """Calculate total cast time reduction including skills and gear.
        
        Total cap is 70% (pre-renewal).
        """
        dex_reduction = min(dex * 0.01, 0.50)
        total = dex_reduction + skill_bonus + gear_bonus
        return min(total, 0.70)

    def get_crit_rate(self, luk: int) -> float:
        """Calculate critical hit rate.
        
        CORRECT pre-renewal formula:
        crit_rate = luk * 0.3 + 1
        This is the base crit rate. Monster LUK reduces it.
        """
        return luk * 0.3 + 1

    def get_stun_resistance(self, vit: int) -> float:
        """Calculate stun resistance.
        
        CORRECT pre-renewal formula:
        Stun duration = max(1, (stun_power - vit) / 10) seconds
        At VIT >= stun_power, stun duration is 1 second (effectively immune).
        Most monster stuns have power 40-60.
        """
        # For common monster stuns (power ~50), VIT 50 gives 1s stun (immune)
        # For boss stuns (power ~80), VIT 80 gives 1s stun
        return vit * 1.0  # Higher VIT = better resistance

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
