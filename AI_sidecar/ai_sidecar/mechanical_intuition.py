"""
Mechanical intuition — understands stat caps, breakpoints, diminishing returns.

A pro player knows:
- 95% flee rate is the cap, anything above is wasted
- Defense has diminishing returns after a certain point
- Cast time reduction has a 70% cap
- ASPD breakpoints determine hits per second
- Status effect resistance depends on caster vs target stats
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
    """Understands RO mechanics — caps, breakpoints, diminishing returns."""
    
    _lock: RLock = field(default_factory=RLock)
    _stats: dict[str, int] = field(default_factory=lambda: {"breakpoints_checked": 0, "build_advice_given": 0})
    
    def __post_init__(self) -> None:
        # Known breakpoints per class (initialized here to avoid mutable default issues)
        self.BREAKPOINTS = {
        "swordman": [
            StatBreakpoint("STR", 80, "str_80", "80 STR for ATK bonus vs size, diminishing after", is_sweet_spot=True),
            StatBreakpoint("AGI", 70, "agi_70", "70 AGI for 95% flee vs most mobs, cap at 99", is_sweet_spot=True),
            StatBreakpoint("VIT", 40, "vit_40", "40 VIT for stun immunity, diminishing after 60", is_sweet_spot=True),
            StatBreakpoint("DEX", 30, "dex_30", "30 DEX for hit rate, 50 for perfect hit vs most", is_sweet_spot=True),
            StatBreakpoint("LUK", 1, "luk_1", "LUK only for crit builds, otherwise 1", is_sweet_spot=True),
        ],
        "mage": [
            StatBreakpoint("INT", 99, "int_99", "99 INT for max MATK, hard cap", is_cap=True),
            StatBreakpoint("DEX", 30, "dex_30", "30 DEX for cast time reduction sweet spot", is_sweet_spot=True),
            StatBreakpoint("DEX", 60, "dex_60", "60 DEX for 70% cast reduction cap", is_cap=True),
            StatBreakpoint("VIT", 30, "vit_30", "30 VIT for survival, don't overinvest", is_sweet_spot=True),
            StatBreakpoint("AGI", 1, "agi_1", "AGI useless for mage, 1 base", is_sweet_spot=True),
        ],
        "archer": [
            StatBreakpoint("DEX", 99, "dex_99", "99 DEX for max ATK, hard cap", is_cap=True),
            StatBreakpoint("AGI", 70, "agi_70", "70 AGI for flee, 80 for ASPD breakpoint", is_sweet_spot=True),
            StatBreakpoint("INT", 20, "int_20", "20 INT for SP regen, don't overinvest", is_sweet_spot=True),
            StatBreakpoint("VIT", 20, "vit_20", "20 VIT for survival", is_sweet_spot=True),
        ],
        "acolyte": [
            StatBreakpoint("INT", 80, "int_80", "80 INT for heal power, 99 for max", is_sweet_spot=True),
            StatBreakpoint("DEX", 30, "dex_30", "30 DEX for cast time, 50 for perfect", is_sweet_spot=True),
            StatBreakpoint("VIT", 40, "vit_40", "40 VIT for survival, stun immunity", is_sweet_spot=True),
            StatBreakpoint("STR", 20, "str_20", "20 STR for carry weight", is_sweet_spot=True),
        ],
        "merchant": [
            StatBreakpoint("STR", 80, "str_80", "80 STR for ATK, 99 for max", is_sweet_spot=True),
            StatBreakpoint("VIT", 50, "vit_50", "50 VIT for tanking, 70 for max", is_sweet_spot=True),
            StatBreakpoint("DEX", 30, "dex_30", "30 DEX for hit rate", is_sweet_spot=True),
            StatBreakpoint("AGI", 1, "agi_1", "AGI not needed for merchant", is_sweet_spot=True),
        ],
        "thief": [
            StatBreakpoint("AGI", 80, "agi_80", "80 AGI for ASPD breakpoint, 99 for max", is_sweet_spot=True),
            StatBreakpoint("STR", 50, "str_50", "50 STR for ATK, 70 for max", is_sweet_spot=True),
            StatBreakpoint("DEX", 30, "dex_30", "30 DEX for hit rate", is_sweet_spot=True),
            StatBreakpoint("LUK", 30, "luk_30", "30 LUK for crit if Katar build", is_sweet_spot=True),
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
    
    def get_flee_cap(self, base_level: int, agi: int) -> int:
        """Calculate flee rate (capped at 95%)."""
        flee = base_level + agi
        return min(flee, 95)  # 95% is the hard cap
    
    def get_aspd(self, agi: int, dex: int, weapon_type: str = "sword") -> float:
        """Calculate attack speed (approximate)."""
        base_aspd = {"sword": 140, "dagger": 150, "spear": 130, "bow": 120, "staff": 140, "mace": 140, "axe": 130, "katar": 150}
        base = base_aspd.get(weapon_type, 140)
        aspd = base + (agi * 0.5) + (dex * 0.3)
        return min(aspd, 190)  # 190 is the hard cap
    
    def get_cast_time_reduction(self, dex: int) -> float:
        """Calculate cast time reduction (capped at 70%)."""
        reduction = dex * 0.02  # ~2% per DEX point
        return min(reduction, 0.7)  # 70% is the hard cap
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)
