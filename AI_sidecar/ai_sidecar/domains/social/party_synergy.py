"""Party Synergy Engine — class combo system for coordinated party play.

A pro party in RO doesn't just share EXP — it chains class-specific skills
for multiplicative damage, crowd control, and survivability.

This module provides:
- ClassComboSystem: Detects available combos from party composition
- Combo execution coordination with timing windows
- Party composition recommendations
- Class-vs-class counter strategies for WoE
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from ai_sidecar.actions import HeuristicAction
from ai_sidecar.domains.social.class_combos import (
    CLASS_COMBOS,
    ClassCombo,
    ComboCategory,
    get_combos_for_classes,
    get_combos_by_category,
    get_class_vs_class_counter,
    find_combo_by_name,
)

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────

@dataclass
class ComboExecution:
    """Tracks the execution state of an active combo."""
    combo: ClassCombo
    prep_caster: str
    main_caster: str
    target_id: int = 0
    target_x: int = 0
    target_y: int = 0
    started_at: float = 0.0
    prep_cast_at: float = 0.0
    main_cast_at: float = 0.0
    completed: bool = False
    failed: bool = False
    failure_reason: str = ""

    @property
    def is_timed_out(self) -> bool:
        """Check if the combo window has expired."""
        if self.completed or self.failed:
            return True
        if self.prep_cast_at > 0:
            elapsed = time.time() - self.prep_cast_at
            return elapsed > self.combo.window_s + self.combo.latency_buffer
        return False

    @property
    def is_ready_for_main(self) -> bool:
        """Check if it's time for the main caster to act."""
        if self.completed or self.failed:
            return False
        if self.prep_cast_at == 0:
            return False
        elapsed = time.time() - self.prep_cast_at
        return self.combo.prep_time_s <= elapsed <= (self.combo.window_s + self.combo.latency_buffer)


@dataclass
class PartyComposition:
    """Describes the current party composition."""
    member_count: int = 0
    classes: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    levels: list[int] = field(default_factory=list)
    roles: dict[str, str] = field(default_factory=dict)  # name -> role

    def has_class(self, class_name: str) -> bool:
        """Check if party has a specific class."""
        return any(class_name.lower() in c.lower() for c in self.classes)

    def get_members_by_class(self, class_name: str) -> list[str]:
        """Get member names with a specific class."""
        return [
            self.names[i] for i, c in enumerate(self.classes)
            if class_name.lower() in c.lower()
        ]

    def get_available_combos(self) -> list[ClassCombo]:
        """Get all combos that are possible with current party composition."""
        available: list[ClassCombo] = []
        for combo in CLASS_COMBOS:
            if self.has_class(combo.prep_class) and self.has_class(combo.main_class):
                available.append(combo)
        return available


# ── Party Synergy Engine ──────────────────────────────────────────────────

class PartySynergyEngine:
    """Class combo system for coordinated party play.

    Features:
      - Detects available combos from party composition
      - Coordinates combo execution with timing windows
      - Recommends party compositions for optimal synergy
      - Tracks combo cooldowns and execution state
      - Provides class-vs-class counter strategies
      - Handles Sage + Wizard elemental synergy
      - Handles Priest + Hunter defensive/offensive combos
      - Handles Dancer + Bard AoE stun and EXP combos
      - Handles Alchemist + anyone Acid Demonstration
      - Handles Monk + Priest Lex Aeterna → Asura Strike instakill
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._active_executions: dict[str, ComboExecution] = {}  # combo_name -> execution
        self._combo_cooldowns: dict[str, float] = {}  # combo_name -> ready_at
        self._combo_history: list[dict[str, Any]] = []
        self._last_party_composition: PartyComposition | None = None
        self._last_composition_check: float = 0.0

    # ── Public API ────────────────────────────────────────────────────

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Run party synergy assessment — call every PDCA cycle."""
        if not signals:
            return

        with self._lock:
            party = signals.get("party", {}) or {}
            party_members = party.get("members", []) if isinstance(party, dict) else []
            if len(party_members) < 1:
                return

            # Build current party composition
            composition = self._build_composition(signals, party_members)
            if not composition or composition.member_count < 2:
                return

            self._last_party_composition = composition

            # 1. Detect available combos
            available_combos = composition.get_available_combos()
            if not available_combos:
                return

            # 2. Check for combo opportunities
            my_job = str(signals.get("job_name", "novice") or "novice").lower()
            my_name = str(signals.get("name", "") or "")

            for combo in available_combos:
                # Check cooldown
                if combo.name in self._combo_cooldowns:
                    if time.time() < self._combo_cooldowns[combo.name]:
                        continue

                # Check if this bot is the prep caster
                if combo.prep_class in my_job:
                    self._try_execute_prep(combo, composition, my_name, actions, signals)

                # Check if this bot is the main caster
                if combo.main_class in my_job:
                    self._try_execute_main(combo, composition, my_name, actions, signals)

            # 3. Clean up timed-out executions
            self._cleanup_executions()

    def get_available_combos(self, signals: dict[str, Any]) -> list[ClassCombo]:
        """Get all combos available with current party."""
        with self._lock:
            party = signals.get("party", {}) or {}
            party_members = party.get("members", []) if isinstance(party, dict) else []
            composition = self._build_composition(signals, party_members)
            if composition:
                return composition.get_available_combos()
            return []

    def get_active_executions(self) -> list[ComboExecution]:
        """Get all active combo executions."""
        with self._lock:
            return list(self._active_executions.values())

    def get_combo_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent combo execution history."""
        with self._lock:
            return self._combo_history[-limit:]

    def recommend_party_composition(
        self,
        my_class: str,
        available_classes: list[str],
    ) -> list[str]:
        """Recommend which classes to invite for optimal synergy."""
        my_class_lower = my_class.lower()
        recommendations: list[str] = []

        # Priests are always valuable
        if my_class_lower not in ("priest", "high_priest", "acolyte"):
            if "priest" in available_classes or "acolyte" in available_classes:
                recommendations.append("priest")

        # Sage + Wizard combo
        if my_class_lower in ("wizard", "high_wizard", "sage", "professor"):
            if "sage" in available_classes or "wizard" in available_classes:
                recommendations.append("sage" if my_class_lower == "wizard" else "wizard")

        # Priest + Hunter combo
        if my_class_lower in ("hunter", "sniper"):
            if "priest" in available_classes:
                recommendations.append("priest")

        # Dancer + Bard combo
        if my_class_lower in ("dancer", "gypsy"):
            if "bard" in available_classes or "clown" in available_classes:
                recommendations.append("bard")
        if my_class_lower in ("bard", "clown"):
            if "dancer" in available_classes or "gypsy" in available_classes:
                recommendations.append("dancer")

        # Monk + Priest instakill
        if my_class_lower in ("monk", "champion", "sura"):
            if "priest" in available_classes:
                recommendations.append("priest")

        # Alchemist for WoE
        if my_class_lower in ("alchemist", "creator", "genetic"):
            recommendations.append("any_dps")

        return recommendations

    def get_class_counter(self, attacker_class: str, defender_class: str) -> str | None:
        """Get counter strategy for class-vs-class in WoE."""
        return get_class_vs_class_counter(attacker_class, defender_class)

    # ── Internal ─────────────────────────────────────────────────────

    def _build_composition(
        self,
        signals: dict[str, Any],
        party_members: list[Any],
    ) -> PartyComposition | None:
        """Build PartyComposition from signals and party members."""
        my_name = str(signals.get("name", "") or "")
        my_class = str(signals.get("job_name", "novice") or "novice")
        my_level = signals.get("base_level", 1) or 1

        classes: list[str] = [my_class]
        names: list[str] = [my_name]
        levels: list[int] = [my_level]

        for m in party_members:
            if isinstance(m, dict):
                m_name = str(m.get("name", "") or "")
                m_class = str(m.get("job", "") or "novice")
                m_level = m.get("level", 1) or 1
                if m_name and m_name != my_name:
                    classes.append(m_class)
                    names.append(m_name)
                    levels.append(m_level)
            elif isinstance(m, str):
                if m and m != my_name:
                    classes.append("unknown")
                    names.append(m)
                    levels.append(1)

        if len(names) < 2:
            return None

        return PartyComposition(
            member_count=len(names),
            classes=classes,
            names=names,
            levels=levels,
        )

    def _try_execute_prep(
        self,
        combo: ClassCombo,
        composition: PartyComposition,
        my_name: str,
        actions: list[HeuristicAction],
        signals: dict[str, Any],
    ) -> None:
        """Try to execute the prep phase of a combo."""
        # Find a main caster in the party
        main_casters = composition.get_members_by_class(combo.main_class)
        if not main_casters:
            return

        # Don't start if already executing this combo
        if combo.name in self._active_executions:
            return

        # Check if we have the skill
        known_skills = signals.get("skills", []) or []
        if combo.prep_skill not in known_skills:
            return

        # Check SP
        sp = signals.get("sp", 0) or 0
        if sp < 20:  # Minimum SP threshold
            return

        # Start execution
        target_id = signals.get("target_id", 0) or 0
        target_x = signals.get("target_x", 0) or 0
        target_y = signals.get("target_y", 0) or 0

        execution = ComboExecution(
            combo=combo,
            prep_caster=my_name,
            main_caster=main_casters[0],
            target_id=target_id,
            target_x=target_x,
            target_y=target_y,
            started_at=time.time(),
        )
        self._active_executions[combo.name] = execution

        # Emit prep action
        target_str = f" {target_id}" if target_id > 0 else ""
        actions.append(HeuristicAction(
            kind="command",
            command=f"use_skill {combo.prep_skill}{target_str}",
            confidence=0.85,
            domain="party",
            reason=f"Combo prep: {combo.name} — {combo.description}",
            metadata={
                "combo": combo.name,
                "phase": "prep",
                "prep_caster": my_name,
                "main_caster": main_casters[0],
                "target_id": target_id,
            },
        ))
        execution.prep_cast_at = time.time()
        logger.info("[Combo] %s: prep phase started for %s (main: %s)", my_name, combo.name, main_casters[0])

    def _try_execute_main(
        self,
        combo: ClassCombo,
        composition: PartyComposition,
        my_name: str,
        actions: list[HeuristicAction],
        signals: dict[str, Any],
    ) -> None:
        """Try to execute the main phase of a combo (follow-up)."""
        execution = self._active_executions.get(combo.name)
        if not execution:
            return

        # Check if we're the main caster
        if execution.main_caster != my_name:
            return

        # Check if it's time for main cast
        if not execution.is_ready_for_main:
            return

        # Check if we have the skill
        known_skills = signals.get("skills", []) or []
        if combo.main_skill not in known_skills and combo.main_skill != "attack":
            return

        # Execute main phase
        target_str = f" {execution.target_id}" if execution.target_id > 0 else ""
        skill_cmd = combo.main_skill if combo.main_skill != "attack" else "attack"

        actions.append(HeuristicAction(
            kind="command",
            command=f"use_skill {skill_cmd}{target_str}",
            confidence=0.90,
            domain="party",
            reason=f"Combo main: {combo.name} — follow-up after {combo.prep_skill}",
            metadata={
                "combo": combo.name,
                "phase": "main",
                "prep_caster": execution.prep_caster,
                "main_caster": my_name,
            },
        ))
        execution.main_cast_at = time.time()
        execution.completed = True

        # Set cooldown
        self._combo_cooldowns[combo.name] = time.time() + 30.0  # 30s cooldown

        # Record history
        self._combo_history.append({
            "combo": combo.name,
            "prep_caster": execution.prep_caster,
            "main_caster": my_name,
            "completed_at": time.time(),
            "success": True,
        })

        # Remove from active
        del self._active_executions[combo.name]
        logger.info("[Combo] %s: main phase completed for %s (prep: %s)", my_name, combo.name, execution.prep_caster)

    def _cleanup_executions(self) -> None:
        """Remove timed-out or failed executions."""
        now = time.time()
        failed: list[str] = []
        for name, execution in self._active_executions.items():
            if execution.is_timed_out:
                execution.failed = True
                execution.failure_reason = "timed_out"
                self._combo_history.append({
                    "combo": name,
                    "prep_caster": execution.prep_caster,
                    "main_caster": execution.main_caster,
                    "failed_at": now,
                    "success": False,
                    "reason": "timed_out",
                })
                failed.append(name)

        for name in failed:
            del self._active_executions[name]
            # Short cooldown on failure
            self._combo_cooldowns[name] = time.time() + 10.0


# ── Singleton factory ─────────────────────────────────────────────────────

_party_synergy_engine: PartySynergyEngine | None = None


def get_party_synergy_engine() -> PartySynergyEngine:
    """Get or create the singleton PartySynergyEngine."""
    global _party_synergy_engine
    if _party_synergy_engine is None:
        _party_synergy_engine = PartySynergyEngine()
    return _party_synergy_engine
