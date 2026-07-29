"""Character progression lifecycle state machine — NOVICE → ENDGAME.

Provides:
  - LifecyclePhase enum: NOVICE, FIRST_JOB, SECOND_JOB, TRANS_CLASS, ENDGAME
  - PhaseConfig: stat build, skill priorities, gear targets, map preferences per phase
  - LifecycleStateMachine: tracks current phase, transitions, and generates
    progression-appropriate HeuristicActions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ai_sidecar.actions import HeuristicAction

logger = __import__("logging").getLogger(__name__)


# ── Phase definitions ────────────────────────────────────────────────────

class LifecyclePhase(str, Enum):
    """Character lifecycle phase mapped to RO job progression.

    NOVICE       (1–10)    → farm Porings, buy basic gear
    FIRST_JOB    (10–40)   → swordsman/thief/archer/mage/acolyte
    SECOND_JOB   (40–70)   → knight/assassin/hunter/wizard/priest
    TRANS_CLASS  (70–99)   → transcended / 2-2 classes
    ENDGAME      (99+)     → rebirth, MVP hunting, endgame gear
    """
    NOVICE = "novice"
    FIRST_JOB = "first_job"
    SECOND_JOB = "second_job"
    TRANS_CLASS = "trans_class"
    ENDGAME = "endgame"


# ── Phase configuration ──────────────────────────────────────────────────

@dataclass
class PhaseConfig:
    """Configuration for a single lifecycle phase."""
    phase: LifecyclePhase
    level_range: tuple[int, int]        # (min_level, max_level) inclusive

    # Stat allocation priorities (stat → points per level allocated)
    stat_priorities: dict[str, int] = field(default_factory=dict)

    # Skill priorities (skill_name → priority 1=highest)
    skill_priorities: dict[str, int] = field(default_factory=dict)

    # Gear targets (slot → item_id/name)
    gear_targets: dict[str, str] = field(default_factory=dict)

    # Preferred hunting maps
    preferred_maps: list[str] = field(default_factory=list)

    # Minimum HP ratio before retreat
    safe_hp_ratio: float = 0.40

    # Auto-attack config
    attack_distance: int = 5
    attack_max_distance: int = 20
    teleport_min_aggro: int = 8

    # Monsters to ignore during this phase
    ignore_monsters: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"<PhaseConfig:{self.phase.value} lv{self.level_range}>"


# ── Built-in phase configurations ────────────────────────────────────────

NOVICE_CONFIG = PhaseConfig(
    phase=LifecyclePhase.NOVICE,
    level_range=(1, 10),
    stat_priorities={"str": 3, "dex": 2, "agi": 1},
    skill_priorities={},  # Novices have no skills
    gear_targets={
        "weapon": "1201",   # Knife
        "armor": "2301",    # Adventurer's Suit
    },
    preferred_maps=["prt_fild05", "prt_fild01", "pay_fild01"],
    safe_hp_ratio=0.60,
    attack_distance=3,
    attack_max_distance=12,
    teleport_min_aggro=6,
    ignore_monsters=[
        "Thief Bug Egg", "Pupa", "Thief Bug",
        "Lunatic", "Fabre", "Condor",
    ],
)

SWORDMAN_CONFIG = PhaseConfig(
    phase=LifecyclePhase.FIRST_JOB,
    level_range=(10, 40),
    stat_priorities={"str": 5, "vit": 3, "dex": 2, "agi": 1},
    skill_priorities={
        "increase_agility": 1,
        "magnum_break": 2,
        "bash": 3,
        "provoke": 4,
    },
    gear_targets={
        "weapon": "1901",   # Katana
        "armor": "2302",    # Chain Mail
        "shield": "2101",   # Guard
    },
    preferred_maps=["prt_fild05", "mjolnir_04", "pay_fild01"],
    safe_hp_ratio=0.40,
    attack_distance=5,
    attack_max_distance=20,
    teleport_min_aggro=8,
)

THIEF_CONFIG = PhaseConfig(
    phase=LifecyclePhase.FIRST_JOB,
    level_range=(10, 40),
    stat_priorities={"agi": 5, "dex": 3, "str": 2, "luk": 1},
    skill_priorities={
        "double_attack": 1,
        "improve_dodge": 2,
        "hide": 3,
        "detoxify": 4,
    },
    gear_targets={
        "weapon": "1301",   # Dagger — will be overridden
        "armor": "2301",    # Adventurer's Suit
    },
    preferred_maps=["mjolnir_04", "pay_fild01", "prt_fild05"],
    safe_hp_ratio=0.35,
    attack_distance=3,
    attack_max_distance=15,
    teleport_min_aggro=6,
)

ARCHER_CONFIG = PhaseConfig(
    phase=LifecyclePhase.FIRST_JOB,
    level_range=(10, 40),
    stat_priorities={"dex": 5, "agi": 3, "luk": 2, "str": 1},
    skill_priorities={
        "improve_concentration": 1,
        "owl_eye": 2,
        "double_strafing": 3,
        "arrow_shower": 4,
    },
    gear_targets={
        "weapon": "1701",   # Bow
        "armor": "2301",
    },
    preferred_maps=["pay_fild01", "pay_fild02", "prt_fild05"],
    safe_hp_ratio=0.40,
    attack_distance=10,
    attack_max_distance=30,
    teleport_min_aggro=4,
)

MAGE_CONFIG = PhaseConfig(
    phase=LifecyclePhase.FIRST_JOB,
    level_range=(10, 40),
    stat_priorities={"int": 5, "dex": 3, "luk": 2, "vit": 1},
    skill_priorities={
        "fire_bolt": 1,
        "cold_bolt": 2,
        "lightning_bolt": 3,
        "napalm_beat": 4,
    },
    gear_targets={
        "weapon": "1601",   # Rod
        "armor": "2301",
    },
    preferred_maps=["pay_fild01", "moc_fild01", "gef_fild01"],
    safe_hp_ratio=0.45,
    attack_distance=8,
    attack_max_distance=25,
    teleport_min_aggro=3,
)

ACOLYTE_CONFIG = PhaseConfig(
    phase=LifecyclePhase.FIRST_JOB,
    level_range=(10, 40),
    stat_priorities={"int": 5, "dex": 3, "vit": 2, "str": 1},
    skill_priorities={
        "heal": 1,
        "blessing": 2,
        "increase_agility": 3,
        "teleport": 4,
    },
    gear_targets={
        "weapon": "1501",   # Mace
        "armor": "2301",
    },
    preferred_maps=["pay_fild01", "prt_fild05", "moc_fild01"],
    safe_hp_ratio=0.35,  # Heal means less retreat
    attack_distance=7,
    attack_max_distance=25,
    teleport_min_aggro=5,
)

SECOND_JOB_CONFIG = PhaseConfig(
    phase=LifecyclePhase.SECOND_JOB,
    level_range=(40, 70),
    stat_priorities={"str": 4, "agi": 3, "vit": 2, "dex": 2},
    skill_priorities={
        "two_hand_quicken": 1,
        "auto_counter": 2,
        "parrying": 3,
        "brandish_spear": 4,
    },
    gear_targets={
        "weapon": "1130",   # Broad Sword (placeholder)
        "armor": "2310",    # Chain Mail lv3
    },
    preferred_maps=["ra_fild01", "moc_fild05", "cmd_fild01"],
    safe_hp_ratio=0.35,
    attack_distance=5,
    attack_max_distance=20,
    teleport_min_aggro=10,
)

TRANS_CLASS_CONFIG = PhaseConfig(
    phase=LifecyclePhase.TRANS_CLASS,
    level_range=(70, 99),
    stat_priorities={"str": 3, "agi": 3, "vit": 2, "dex": 2, "int": 1},
    skill_priorities={
        "mountainous_fissure": 1,
        "spiral_pierce": 2,
        "concentration": 3,
        "shield_spell": 4,
    },
    gear_targets={
        "weapon": "1470",   # Orcish Axe (placeholder)
        "armor": "2350",
    },
    preferred_maps=["ra_fild08", "gefg_dun01", "moc_fild07"],
    safe_hp_ratio=0.30,
    attack_distance=5,
    attack_max_distance=22,
    teleport_min_aggro=12,
)

ENDGAME_CONFIG = PhaseConfig(
    phase=LifecyclePhase.ENDGAME,
    level_range=(99, 999),
    stat_priorities={"str": 3, "agi": 3, "vit": 2, "dex": 2, "int": 1, "luk": 1},
    skill_priorities={
        "spiral_pierce": 1,
        "pressure": 2,
        "mille": 3,
    },
    gear_targets={
        "weapon": "2701",   # MVP weapon (placeholder)
        "armor": "2370",
    },
    preferred_maps=["gefg_dun02", "thor_v", "bif_fild01"],
    safe_hp_ratio=0.25,
    attack_distance=5,
    attack_max_distance=22,
    teleport_min_aggro=15,
)

# Map of first-job class -> appropriate FIRST_JOB config
FIRST_JOB_CONFIGS: dict[str, PhaseConfig] = {
    "swordman": SWORDMAN_CONFIG,
    "knight": SECOND_JOB_CONFIG,
    "thief": THIEF_CONFIG,
    "assassin": SECOND_JOB_CONFIG,
    "archer": ARCHER_CONFIG,
    "hunter": SECOND_JOB_CONFIG,
    "mage": MAGE_CONFIG,
    "wizard": SECOND_JOB_CONFIG,
    "acolyte": ACOLYTE_CONFIG,
    "priest": SECOND_JOB_CONFIG,
    "merchant": SWORDMAN_CONFIG,  # fallback
    "blacksmith": SECOND_JOB_CONFIG,
    "alchemist": SECOND_JOB_CONFIG,
    # 2-2 classes also use SECOND_JOB_CONFIG
    "rogue": SECOND_JOB_CONFIG,
    "bard": SECOND_JOB_CONFIG,
    "dancer": SECOND_JOB_CONFIG,
    "sage": SECOND_JOB_CONFIG,
    "crusader": SECOND_JOB_CONFIG,
    "monk": SECOND_JOB_CONFIG,
    # Trans / advanced
    "lord knight": TRANS_CLASS_CONFIG,
    "paladin": TRANS_CLASS_CONFIG,
    "rune knight": TRANS_CLASS_CONFIG,
    "royal guard": TRANS_CLASS_CONFIG,
    "guillotine cross": TRANS_CLASS_CONFIG,
    "shadow chaser": TRANS_CLASS_CONFIG,
    "minstrel": TRANS_CLASS_CONFIG,
    "wanderer": TRANS_CLASS_CONFIG,
    "sniper": TRANS_CLASS_CONFIG,
    "warlock": TRANS_CLASS_CONFIG,
    "sorcerer": TRANS_CLASS_CONFIG,
    "arch bishop": TRANS_CLASS_CONFIG,
    "sura": TRANS_CLASS_CONFIG,
    "genetic": TRANS_CLASS_CONFIG,
    "mechanic": TRANS_CLASS_CONFIG,
    "ranger": TRANS_CLASS_CONFIG,
    "rogue (2-2)": SECOND_JOB_CONFIG,
    "alchemist (2-2)": SECOND_JOB_CONFIG,
    "bard (2-2)": SECOND_JOB_CONFIG,
    "dancer (2-2)": SECOND_JOB_CONFIG,
    "sage (2-2)": SECOND_JOB_CONFIG,
    "crusader (2-2)": SECOND_JOB_CONFIG,
    "monk (2-2)": SECOND_JOB_CONFIG,
}


# ── Lifecycle state machine ──────────────────────────────────────────────

class LifecycleStateMachine:
    """Tracks a bot's progression phase and emits phase-appropriate actions.

    State transitions:
      NOVICE (1-10) → FIRST_JOB (10-40) → SECOND_JOB (40-70)
      → TRANS_CLASS (70-99) → ENDGAME (99+)
    """

    def __init__(self) -> None:
        self._phases: dict[str, LifecyclePhase] = {}  # bot_id → current phase
        self._phase_cache: dict[str, PhaseConfig] = {}

    # ── Public API ────────────────────────────────────────────────────

    def get_phase(self, bot_id: str) -> LifecyclePhase:
        """Return the current lifecycle phase for *bot_id*."""
        return self._phases.get(bot_id, LifecyclePhase.NOVICE)

    def get_config(self, bot_id: str, job_name: str = "") -> PhaseConfig:
        """Return the PhaseConfig for the current lifecycle phase."""
        phase = self.get_phase(bot_id)
        return self._resolve_config(phase, job_name)

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Evaluate progression state and emit phase-appropriate actions."""
        base_level = int(signals.get("base_level", 1) or 1)
        job_level = int(signals.get("job_level", 1) or 1)
        job_name = str(signals.get("job_name", "novice") or "novice").lower()
        map_name = str(signals.get("map", "") or "").lower().replace(".gat", "")
        zeny = int(signals.get("zeny", 0) or 0)

        # Update phase based on base level
        new_phase = self._calculate_phase(base_level, job_name)
        current_phase = self._phases.get(bot_id)
        if new_phase != current_phase:
            logger.info(
                "[lifecycle] %s: %s → %s (lv%d, %s)",
                bot_id, current_phase.value if current_phase else "?",
                new_phase.value, base_level, job_name,
            )
            self._phases[bot_id] = new_phase

        config = self._resolve_config(new_phase, job_name)

        # ── Phase-appropriate stat allocation ──
        self._emit_stat_allocation(actions, bot_id, config, signals)

        # ── Phase-appropriate combat config ──
        self._emit_combat_config(actions, bot_id, config, signals)

        # ── Phase-appropriate map preference ──
        self._emit_map_preference(actions, bot_id, config, map_name)

        # ── Phase-appropriate gear targets ──
        self._emit_gear_targets(actions, bot_id, config, zeny, signals)

        # ── Phase-appropriate mon_control ──
        if config.ignore_monsters:
            self._emit_ignore_monsters(actions, bot_id, config)

        # ── Skill training ──
        if config.skill_priorities:
            self._emit_skill_training(actions, bot_id, config, signals)

    # ── Phase calculation ─────────────────────────────────────────────

    @staticmethod
    def _calculate_phase(base_level: int, job_name: str) -> LifecyclePhase:
        """Determine the correct lifecycle phase from level and job."""
        if base_level >= 99:
            return LifecyclePhase.ENDGAME
        if base_level >= 70:
            return LifecyclePhase.TRANS_CLASS
        if base_level >= 40:
            return LifecyclePhase.SECOND_JOB
        if base_level >= 10 and job_name != "novice":
            return LifecyclePhase.FIRST_JOB

        # Level 1–10, or still Novice past 10
        if base_level >= 10 and job_name == "novice":
            return LifecyclePhase.NOVICE

        return LifecyclePhase.NOVICE

    # ── Config resolution ─────────────────────────────────────────────

    def _resolve_config(
        self,
        phase: LifecyclePhase,
        job_name: str,
    ) -> PhaseConfig:
        """Get the appropriate PhaseConfig for the given phase + job."""
        cache_key = f"{phase.value}-{job_name}"
        if cache_key in self._phase_cache:
            return self._phase_cache[cache_key]

        config: PhaseConfig
        if phase == LifecyclePhase.NOVICE:
            config = NOVICE_CONFIG
        elif phase == LifecyclePhase.FIRST_JOB:
            config = FIRST_JOB_CONFIGS.get(job_name, SWORDMAN_CONFIG)
        elif phase == LifecyclePhase.SECOND_JOB:
            config = SECOND_JOB_CONFIG
        elif phase == LifecyclePhase.TRANS_CLASS:
            config = TRANS_CLASS_CONFIG
        else:
            config = ENDGAME_CONFIG

        self._phase_cache[cache_key] = config
        return config

    # ── Action emitters ───────────────────────────────────────────────

    def _emit_stat_allocation(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        config: PhaseConfig,
        signals: dict[str, Any],
    ) -> None:
        """Emit stat point allocation commands based on phase priorities."""
        status_points = int(signals.get("status_points", 0) or 0)
        if status_points < 5:
            return  # Not enough points to bother

        # Build stat command from priorities
        total_weight = sum(config.stat_priorities.values())
        if total_weight == 0:
            return

        stat_allocs: list[str] = []
        for stat, weight in config.stat_priorities.items():
            points = max(1, int(status_points * weight / total_weight))
            stat_allocs.append(f"{stat} {points}")

        stat_cmd = "stat_add " + " ".join(stat_allocs)
        actions.append(HeuristicAction(
            kind="command",
            command=stat_cmd,
            confidence=0.90,
            domain="progression",
            reason=f"Lifecycle stat allocation ({config.phase.value} phase): "
                   f"{', '.join(stat_allocs)}",
            metadata={"phase": config.phase.value, "status_points": status_points},
        ))

    def _emit_combat_config(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        config: PhaseConfig,
        signals: dict[str, Any],
    ) -> None:
        """Set phase-appropriate attack / teleport config."""
        hp_ratio = float(signals.get("hp_ratio", 1.0) or 1.0)

        actions.append(HeuristicAction(
            kind="command",
            command=f"set attackDistance {config.attack_distance}",
            confidence=0.90,
            domain="progression",
            reason=f"Phase {config.phase.value}: attackDistance={config.attack_distance}",
        ))
        actions.append(HeuristicAction(
            kind="command",
            command=f"set attackMaxDistance {config.attack_max_distance}",
            confidence=0.85,
            domain="progression",
            reason=f"Phase {config.phase.value}: max chase distance",
        ))
        actions.append(HeuristicAction(
            kind="command",
            command=f"set teleportAuto_minAggressives {config.teleport_min_aggro}",
            confidence=0.85,
            domain="progression",
            reason=f"Phase {config.phase.value}: teleport at {config.teleport_min_aggro}+ mobs",
        ))

        # Phase-scaled attackAuto
        aa = "2" if config.phase == LifecyclePhase.NOVICE else "3"
        actions.append(HeuristicAction(
            kind="command",
            command=f"set attackAuto {aa}",
            confidence=0.90,
            domain="progression",
            reason=f"Phase {config.phase.value}: attackAuto={aa}",
        ))

    def _emit_map_preference(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        config: PhaseConfig,
        current_map: str,
    ) -> None:
        """Suggest a better hunting map if the current one isn't appropriate."""
        if not config.preferred_maps:
            return

        best_map = config.preferred_maps[0]
        if current_map not in config.preferred_maps:
            actions.append(HeuristicAction(
                kind="command",
                command=f"move {best_map}",
                confidence=0.80,
                domain="progression",
                reason=f"Phase {config.phase.value}: preferred map is {best_map} "
                       f"(currently in {current_map})",
                metadata={"phase": config.phase.value, "target_map": best_map},
            ))

    def _emit_gear_targets(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        config: PhaseConfig,
        zeny: int,
        signals: dict[str, Any],
    ) -> None:
        """Recommend gear upgrades for the current phase."""
        inventory = signals.get("inventory_items", []) or []
        inv_str = " ".join(str(i).lower() for i in inventory)

        for slot, item_id in config.gear_targets.items():
            # Only recommend if we don't already have it
            if item_id.lower() in inv_str:
                continue
            # Only if we can afford it (est. 1000z per item)
            if zeny < 1000:
                continue

            actions.append(HeuristicAction(
                kind="command",
                command=f"buy {item_id} 1",
                confidence=0.75,
                domain="progression",
                reason=f"Phase {config.phase.value}: buy {slot} item {item_id}",
                metadata={"phase": config.phase.value, "slot": slot, "item_id": item_id},
            ))

    def _emit_ignore_monsters(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        config: PhaseConfig,
    ) -> None:
        """Emit mon_control ignore commands for this phase."""
        for monster in config.ignore_monsters:
            actions.append(HeuristicAction(
                kind="command",
                command=f"mon_control {monster}\\t-1 0 0",
                confidence=0.95,
                domain="progression",
                reason=f"Phase {config.phase.value}: ignore {monster}",
            ))

    def _emit_skill_training(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        config: PhaseConfig,
        signals: dict[str, Any],
    ) -> None:
        """Emit skill training commands for priority skills."""
        skill_points = int(signals.get("skill_points", 0) or 0)
        if skill_points < 1:
            return

        known_skills = [str(s).lower() for s in (signals.get("skills", []) or [])]
        skill_levels = signals.get("skill_levels", {}) or {}

        # Find the highest-priority skill that isn't maxed
        sorted_skills = sorted(
            config.skill_priorities.items(),
            key=lambda kv: kv[1],  # lower number = higher priority
        )
        for skill_name, _ in sorted_skills:
            sk = skill_name.lower()
            current_lv = skill_levels.get(sk, 0) if isinstance(skill_levels, dict) else 0
            if current_lv < 10 and skill_points > 0:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"use_skill {sk}",
                    confidence=0.85,
                    domain="progression",
                    reason=f"Phase {config.phase.value}: train {sk} "
                           f"(priority #{_}, current lv{current_lv})",
                    metadata={"phase": config.phase.value, "skill": sk, "priority": _},
                ))
                break


# ── Convenience ──────────────────────────────────────────────────────────

def create_lifecycle_machine() -> LifecycleStateMachine:
    """Factory for a new lifecycle state machine instance."""
    return LifecycleStateMachine()
