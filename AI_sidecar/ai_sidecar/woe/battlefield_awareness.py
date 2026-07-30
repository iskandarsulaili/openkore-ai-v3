"""Battlefield Awareness — threat assessment, class-based priority, escape economy.

In WoE, knowing when to fight and when to run is the difference between
winning and feeding. This module provides:
- Battlefield threat assessment with class-based priority
- Escape scroll economy (Fly Wing vs Butterfly Wing vs Escape Scroll vs die)
- Enemy count thresholds (3 enemies = run)
- Class-specific kill priority (Priest first, Wizard first, Paladin ignore)
- Assassin cloaking awareness
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from threading import RLock
from typing import Any

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

# Threat priority: lower number = higher priority (kill first)
class ThreatPriority(IntEnum):
    HEALER = 10       # Priest/Arch Bishop — kill first
    AOE_CASTER = 20   # Wizard/High Wizard/Warlock — kill second
    BREAKER = 30      # Assassin/Champion — kill third
    RANGED = 40       # Hunter/Sniper/Gunslinger — kill fourth
    SUPPORT = 50      # Bard/Dancer/Sage — kill fifth
    TANK = 60         # Paladin/Crusader — ignore (unkillable without Alchemist)
    UNKNOWN = 100     # Unknown class — default


# Class threat priorities
CLASS_THREAT_PRIORITY: dict[str, ThreatPriority] = {
    # Healers — highest priority
    "priest": ThreatPriority.HEALER,
    "high_priest": ThreatPriority.HEALER,
    "arch_bishop": ThreatPriority.HEALER,
    "acolyte": ThreatPriority.HEALER,
    # AoE casters
    "wizard": ThreatPriority.AOE_CASTER,
    "high_wizard": ThreatPriority.AOE_CASTER,
    "warlock": ThreatPriority.AOE_CASTER,
    "sage": ThreatPriority.AOE_CASTER,
    "professor": ThreatPriority.AOE_CASTER,
    "sorcerer": ThreatPriority.AOE_CASTER,
    # Breakers
    "assassin": ThreatPriority.BREAKER,
    "assassin_cross": ThreatPriority.BREAKER,
    "guillotine_cross": ThreatPriority.BREAKER,
    "monk": ThreatPriority.BREAKER,
    "champion": ThreatPriority.BREAKER,
    "sura": ThreatPriority.BREAKER,
    "rogue": ThreatPriority.BREAKER,
    "stalker": ThreatPriority.BREAKER,
    "shadow_chaser": ThreatPriority.BREAKER,
    # Ranged
    "hunter": ThreatPriority.RANGED,
    "sniper": ThreatPriority.RANGED,
    "ranger": ThreatPriority.RANGED,
    "gunslinger": ThreatPriority.RANGED,
    "rebellion": ThreatPriority.RANGED,
    # Support
    "bard": ThreatPriority.SUPPORT,
    "clown": ThreatPriority.SUPPORT,
    "minstrel": ThreatPriority.SUPPORT,
    "dancer": ThreatPriority.SUPPORT,
    "gypsy": ThreatPriority.SUPPORT,
    "wanderer": ThreatPriority.SUPPORT,
    "soul_linker": ThreatPriority.SUPPORT,
    "alchemist": ThreatPriority.SUPPORT,
    "creator": ThreatPriority.SUPPORT,
    "genetic": ThreatPriority.SUPPORT,
    # Tanks — lowest priority
    "swordman": ThreatPriority.TANK,
    "knight": ThreatPriority.TANK,
    "lord_knight": ThreatPriority.TANK,
    "rune_knight": ThreatPriority.TANK,
    "crusader": ThreatPriority.TANK,
    "paladin": ThreatPriority.TANK,
    "royal_guard": ThreatPriority.TANK,
    "merchant": ThreatPriority.TANK,
    "blacksmith": ThreatPriority.TANK,
    "whitesmith": ThreatPriority.TANK,
    "mechanic": ThreatPriority.TANK,
}

# Enemy count thresholds
RUN_THRESHOLD = 3           # 3+ enemies on screen = run
CAUTION_THRESHOLD = 2       # 2 enemies = cautious
SAFE_THRESHOLD = 1          # 1 enemy = engage if favorable

# Escape item costs (approximate market prices)
ESCAPE_COSTS: dict[str, int] = {
    "fly_wing": 100,         # 100z — random teleport
    "butterfly_wing": 500,   # 500z — return to save point
    "escape_scroll": 5000,   # 5000z — respawn at save point
}

# EXP loss on death (1% per death)
DEATH_EXP_LOSS_PCT: float = 0.01

# Escape item IDs
FLY_WING_ID = 601
BUTTERFLY_WING_ID = 602


# ── Data models ───────────────────────────────────────────────────────────

@dataclass
class EnemyThreat:
    """Threat assessment for a single enemy."""
    name: str = ""
    class_name: str = ""
    guild_name: str = ""
    hp_pct: float = 1.0
    sp_pct: float = 1.0
    distance: float = 0.0
    threat_priority: ThreatPriority = ThreatPriority.UNKNOWN
    is_cloaked: bool = False
    is_known_ally: bool = False
    has_emperium: bool = False

    @property
    def threat_score(self) -> float:
        """Calculate numeric threat score (higher = more threatening)."""
        score = float(self.threat_priority)
        # Closer enemies are more threatening
        if self.distance > 0:
            score += max(0, 20 - self.distance)
        # Low HP enemies are less threatening
        if self.hp_pct < 0.3:
            score -= 10
        # Cloaked enemies are more threatening (unknown position)
        if self.is_cloaked:
            score += 15
        return score


@dataclass
class BattlefieldAssessment:
    """Complete battlefield assessment."""
    total_enemies: int = 0
    total_allies: int = 0
    threat_priority_list: list[EnemyThreat] = field(default_factory=list)
    should_engage: bool = False
    should_retreat: bool = False
    retreat_reason: str = ""
    primary_target: EnemyThreat | None = None
    escape_item: str = ""
    escape_reason: str = ""


# ── Battlefield Awareness Engine ─────────────────────────────────────────

class BattlefieldAwareness:
    """Battlefield threat assessment and escape economy.

    Features:
      - Class-based threat priority (Priest > Wizard > Assassin > Paladin)
      - Enemy count thresholds (3+ = run, 2 = cautious, 1 = engage)
      - Escape scroll economy (Fly Wing vs Butterfly Wing vs Escape Scroll vs die)
      - Cloaked Assassin detection
      - Ally/enemy ratio assessment
      - Primary target selection
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._last_assessment: BattlefieldAssessment | None = None
        self._last_assess_time: float = 0.0
        self._escape_item_used: dict[str, float] = {}  # item_name -> last_use_time

    # ── Public API ────────────────────────────────────────────────────

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Run battlefield awareness assessment."""
        if not signals:
            return

        with self._lock:
            assessment = self._assess_battlefield(signals, bot_id)
            self._last_assessment = assessment
            self._last_assess_time = time.time()

            # Emit actions based on assessment
            if assessment.should_retreat:
                self._emit_retreat(assessment, actions, bot_id, signals)
            elif assessment.should_engage and assessment.primary_target:
                self._emit_engage(assessment, actions, bot_id)
            else:
                self._emit_cautious(assessment, actions, bot_id)

    def get_last_assessment(self) -> BattlefieldAssessment | None:
        """Get the last battlefield assessment."""
        with self._lock:
            return self._last_assessment

    def get_threat_priority(
        self, class_name: str
    ) -> ThreatPriority:
        """Get threat priority for a class."""
        return CLASS_THREAT_PRIORITY.get(class_name.lower(), ThreatPriority.UNKNOWN)

    def recommend_escape_item(
        self,
        hp_pct: float,
        has_fly_wings: bool,
        has_butterfly_wings: bool,
        has_escape_scrolls: bool,
        zeny: int,
        in_woe: bool,
    ) -> tuple[str, str]:
        """Recommend the best escape item based on situation.

        Returns (item_name, reason).
        """
        if in_woe:
            # In WoE: dying loses 1% EXP and respawns at save point
            # Sometimes better than using an expensive scroll
            if hp_pct < 0.15:
                # Critical HP — use whatever is cheapest
                if has_fly_wings:
                    return ("fly_wing", "Critical HP in WoE — Fly Wing to random position")
                elif has_butterfly_wings:
                    return ("butterfly_wing", "Critical HP in WoE — Butterfly Wing to save point")
                elif has_escape_scrolls:
                    return ("escape_scroll", "Critical HP in WoE — Escape Scroll to save point")
                else:
                    return ("die", "No escape items — dying is cheaper than buying")
            elif hp_pct < 0.30:
                # Low HP — use Fly Wing if available
                if has_fly_wings:
                    return ("fly_wing", "Low HP in WoE — Fly Wing to safety")
                elif has_butterfly_wings:
                    return ("butterfly_wing", "Low HP in WoE — Butterfly Wing to save point")
                else:
                    return ("stay", "Low HP but no escape items — hide and regen")
            else:
                return ("stay", "HP acceptable — no escape needed")
        else:
            # Not in WoE: use cheapest option
            if hp_pct < 0.20:
                if has_fly_wings:
                    return ("fly_wing", "Low HP — Fly Wing escape (100z)")
                elif has_butterfly_wings:
                    return ("butterfly_wing", "Low HP — Butterfly Wing to town (500z)")
                elif has_escape_scrolls:
                    return ("escape_scroll", "Low HP — Escape Scroll (5000z)")
                else:
                    return ("die", "No escape items — death is cheaper")
            else:
                return ("stay", "HP acceptable — no escape needed")

    # ── Internal ─────────────────────────────────────────────────────

    def _assess_battlefield(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> BattlefieldAssessment:
        """Assess the current battlefield situation."""
        players = signals.get("players", []) or []
        my_name = str(signals.get("name", "") or "")
        my_guild = str(signals.get("guild_name", "") or "")
        my_hp_pct = signals.get("hp_ratio", 1.0) or 1.0
        my_class = str(signals.get("job_name", "novice") or "novice").lower()
        map_name = str(signals.get("map", "") or "").lower()
        in_woe = "gld_" in map_name

        enemies: list[EnemyThreat] = []
        allies = 0

        for p in players:
            if not isinstance(p, dict):
                continue
            p_name = str(p.get("name", "") or "")
            p_guild = str(p.get("guild_name", "") or "")
            p_class = str(p.get("job", "") or "novice").lower()
            p_hp = p.get("hp", 1) or 1
            p_max_hp = p.get("max_hp", 1) or 1

            if p_name == my_name:
                continue

            # Determine if enemy or ally
            is_ally = p_guild and p_guild == my_guild
            if is_ally:
                allies += 1
                continue

            # This is an enemy
            threat = EnemyThreat(
                name=p_name,
                class_name=p_class,
                guild_name=p_guild,
                hp_pct=p_hp / max(p_max_hp, 1),
                distance=p.get("distance", 0) or 0,
                threat_priority=self.get_threat_priority(p_class),
                is_cloaked=p.get("cloaked", False),
            )
            enemies.append(threat)

        # Sort by threat score (highest first)
        enemies.sort(key=lambda e: e.threat_score, reverse=True)

        # Determine engagement
        total_enemies = len(enemies)
        should_engage = False
        should_retreat = False
        retreat_reason = ""
        primary_target = enemies[0] if enemies else None
        escape_item = ""
        escape_reason = ""

        if total_enemies >= RUN_THRESHOLD:
            should_retreat = True
            retreat_reason = f"{total_enemies} enemies on screen — retreating"
            # Recommend escape item
            has_fw = signals.get("item_count", {}).get(str(FLY_WING_ID), 0) > 0
            has_bw = signals.get("item_count", {}).get(str(BUTTERFLY_WING_ID), 0) > 0
            has_es = signals.get("item_count", {}).get("escape_scroll", 0) > 0
            zeny = signals.get("zeny", 0) or 0
            escape_item, escape_reason = self.recommend_escape_item(
                my_hp_pct, has_fw, has_bw, has_es, zeny, in_woe
            )
        elif total_enemies >= CAUTION_THRESHOLD:
            # 2 enemies — cautious
            if my_hp_pct < 0.50:
                should_retreat = True
                retreat_reason = f"2 enemies + low HP ({my_hp_pct:.0%}) — retreating"
            else:
                should_engage = True
        elif total_enemies == SAFE_THRESHOLD and primary_target:
            # 1 enemy — engage if favorable
            if primary_target.threat_priority == ThreatPriority.TANK:
                # Paladin — ignore (unkillable without Alchemist)
                should_retreat = True
                retreat_reason = f"Enemy is {primary_target.class_name} (Paladin) — ignore, unkillable without Alchemist"
            elif my_hp_pct < 0.30:
                should_retreat = True
                retreat_reason = f"1 enemy but HP too low ({my_hp_pct:.0%}) — retreating"
            else:
                should_engage = True
        else:
            # No enemies — safe
            pass

        return BattlefieldAssessment(
            total_enemies=total_enemies,
            total_allies=allies,
            threat_priority_list=enemies,
            should_engage=should_engage,
            should_retreat=should_retreat,
            retreat_reason=retreat_reason,
            primary_target=primary_target,
            escape_item=escape_item,
            escape_reason=escape_reason,
        )

    def _emit_retreat(
        self,
        assessment: BattlefieldAssessment,
        actions: list[HeuristicAction],
        bot_id: str,
        signals: dict[str, Any],
    ) -> None:
        """Emit retreat actions."""
        # Use escape item if recommended
        if assessment.escape_item == "fly_wing":
            actions.append(HeuristicAction(
                kind="command",
                command=f"use {FLY_WING_ID}",
                confidence=0.99,
                domain="pvp",
                reason=assessment.escape_reason,
                metadata={"action": "retreat", "method": "fly_wing", "enemies": assessment.total_enemies},
            ))
        elif assessment.escape_item == "butterfly_wing":
            actions.append(HeuristicAction(
                kind="command",
                command=f"use {BUTTERFLY_WING_ID}",
                confidence=0.99,
                domain="pvp",
                reason=assessment.escape_reason,
                metadata={"action": "retreat", "method": "butterfly_wing", "enemies": assessment.total_enemies},
            ))
        elif assessment.escape_item == "escape_scroll":
            actions.append(HeuristicAction(
                kind="command",
                command="use_escape_scroll",
                confidence=0.95,
                domain="pvp",
                reason=assessment.escape_reason,
                metadata={"action": "retreat", "method": "escape_scroll", "enemies": assessment.total_enemies},
            ))
        else:
            # Just move away
            actions.append(HeuristicAction(
                kind="command",
                command="retreat",
                confidence=0.95,
                domain="pvp",
                reason=assessment.retreat_reason,
                metadata={"action": "retreat", "method": "move", "enemies": assessment.total_enemies},
            ))

        # Log threat details
        for enemy in assessment.threat_priority_list[:3]:
            actions.append(HeuristicAction(
                kind="log",
                command=f"threat_{enemy.name}",
                confidence=0.80,
                domain="pvp",
                reason=f"Threat: {enemy.name} ({enemy.class_name}) — priority {enemy.threat_priority.name}",
                metadata={
                    "enemy_name": enemy.name,
                    "enemy_class": enemy.class_name,
                    "threat_priority": enemy.threat_priority.name,
                    "distance": enemy.distance,
                    "hp_pct": enemy.hp_pct,
                },
            ))

    def _emit_engage(
        self,
        assessment: BattlefieldAssessment,
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Emit engage actions."""
        target = assessment.primary_target
        if not target:
            return

        # Determine tactic based on target class
        tactic = "attack"
        if target.threat_priority == ThreatPriority.HEALER:
            tactic = "interrupt_healer"
        elif target.threat_priority == ThreatPriority.AOE_CASTER:
            tactic = "rush_caster"
        elif target.is_cloaked:
            tactic = "reveal_cloaked"

        actions.append(HeuristicAction(
            kind="command",
            command=f"attack {target.name}",
            confidence=0.90,
            domain="pvp",
            reason=f"Engaging {target.name} ({target.class_name}) — {tactic} priority",
            metadata={
                "target": target.name,
                "target_class": target.class_name,
                "threat_priority": target.threat_priority.name,
                "tactic": tactic,
                "enemies_total": assessment.total_enemies,
                "allies_total": assessment.total_allies,
            },
        ))

    def _emit_cautious(
        self,
        assessment: BattlefieldAssessment,
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Emit cautious actions (no clear engage/retreat)."""
        if assessment.total_enemies > 0:
            actions.append(HeuristicAction(
                kind="log",
                command="battlefield_cautious",
                confidence=0.70,
                domain="pvp",
                reason=f"{assessment.total_enemies} enemies, {assessment.total_allies} allies — holding position",
                metadata={
                    "enemies": assessment.total_enemies,
                    "allies": assessment.total_allies,
                },
            ))


# ── Singleton factory ─────────────────────────────────────────────────────

_battlefield_awareness: BattlefieldAwareness | None = None


def get_battlefield_awareness() -> BattlefieldAwareness:
    """Get or create the singleton BattlefieldAwareness."""
    global _battlefield_awareness
    if _battlefield_awareness is None:
        _battlefield_awareness = BattlefieldAwareness()
    return _battlefield_awareness
