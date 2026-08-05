"""
Innovation engine — discovers new strategies, spots, and builds.

A top player doesn't follow the meta. They CREATE the meta. They
discover new farming spots, invent new builds, find new exploits,
and adapt faster than anyone else.

This module runs controlled experiments to discover better strategies.
It tracks what we've tried, what worked, and what didn't. It generates
novel strategies by combining known patterns in new ways.

Real innovation in RO comes from:
- Discovering skill interactions (e.g. Safety Wall + Storm Gust)
- Finding warp glitches that skip dungeon floors
- Exploiting NPC buy/sell price mismatches
- Manipulating spawn mechanics
- Creating new builds that break the meta
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from ai_sidecar.combat.damage_formulas import (
    get_element_multiplier,
    get_skill_element,
    get_monster_element,
    get_monster_size,
    get_monster_race,
    get_monster_def_data,
    calculate_damage,
    estimate_hits_to_kill,
    SKILL_DATA,
    SKILL_ELEMENTS,
)

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """A controlled experiment to test a new strategy."""
    name: str
    hypothesis: str
    duration_minutes: int = 30
    status: str = "proposed"  # proposed, running, completed, failed
    result: str = ""
    value_score: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class Innovation:
    """A discovered innovation."""
    name: str
    description: str
    category: str  # farming_spot, build, route, tactic, exploit, skill_combo, price_arbitrage
    effectiveness: float = 0.0  # 0.0-1.0
    risk: float = 0.0  # 0.0-1.0
    discovered_at: float = 0.0
    adopted: bool = False


# ── Skill interaction templates ──
# These are known RO skill combos that create emergent effects
SKILL_COMBOS: list[dict[str, Any]] = [
    {
        "name": "Safety_Wall_Storm_Gust",
        "skills": ["Safety Wall", "Storm Gust"],
        "description": "Cast Safety Wall on self, then Storm Gust. The wall absorbs damage while AoE kills mobs.",
        "effectiveness": 0.8,
        "risk": 0.3,
        "requirements": {"min_level": 50, "classes": ["wizard", "sage"]},
    },
    {
        "name": "Provoke_Magnum_Break",
        "skills": ["Provoke", "Magnum Break"],
        "description": "Provoke reduces monster DEF, then Magnum Break deals fire AoE damage.",
        "effectiveness": 0.6,
        "risk": 0.2,
        "requirements": {"min_level": 30, "classes": ["swordsman", "knight"]},
    },
    {
        "name": "Lex_Aeterna_Sonic_Blow",
        "skills": ["Lex Aeterna", "Sonic Blow"],
        "description": "Lex Aeterna doubles next physical damage, then Sonic Blow for massive burst.",
        "effectiveness": 0.9,
        "risk": 0.4,
        "requirements": {"min_level": 60, "classes": ["priest", "assassin"]},
    },
    {
        "name": "Frost_Diver_Fire_Bolt",
        "skills": ["Frost Diver", "Fire Bolt"],
        "description": "Freeze with Frost Diver (water), then break with Fire Bolt for bonus damage.",
        "effectiveness": 0.7,
        "risk": 0.2,
        "requirements": {"min_level": 25, "classes": ["mage", "wizard"]},
    },
    {
        "name": "Endure_Bowling_Bash",
        "skills": ["Endure", "Bowling Bash"],
        "description": "Endure prevents flinch, then Bowling Bash for uninterrupted AoE.",
        "effectiveness": 0.7,
        "risk": 0.3,
        "requirements": {"min_level": 50, "classes": ["knight"]},
    },
    {
        "name": "Hide_Sonic_Blow",
        "skills": ["Hide", "Sonic Blow"],
        "description": "Hide for stealth approach, then Sonic Blow from behind for full damage.",
        "effectiveness": 0.8,
        "risk": 0.3,
        "requirements": {"min_level": 40, "classes": ["assassin"]},
    },
    {
        "name": "Impositio_Manus_Aspersio",
        "skills": ["Impositio Manus", "Aspersio"],
        "description": "Bless weapon with holy element, then attack undead for 2x damage.",
        "effectiveness": 0.8,
        "risk": 0.1,
        "requirements": {"min_level": 40, "classes": ["priest"]},
    },
    {
        "name": "Quagmire_Lord_of_Vermilion",
        "skills": ["Quagmire", "Lord of Vermilion"],
        "description": "Quagmire slows enemies (reducing flee), then LoV hits all slowed targets.",
        "effectiveness": 0.75,
        "risk": 0.3,
        "requirements": {"min_level": 60, "classes": ["wizard"]},
    },
    {
        "name": "Adrenaline_Rush_Cart_Revolution",
        "skills": ["Adrenaline Rush", "Cart Revolution"],
        "description": "Adrenaline Rush boosts ASPD, then Cart Revolution for AoE stun.",
        "effectiveness": 0.6,
        "risk": 0.2,
        "requirements": {"min_level": 40, "classes": ["blacksmith"]},
    },
    {
        "name": "Gloria_Turn_Undead",
        "skills": ["Gloria", "Turn Undead"],
        "description": "Gloria boosts LUK, then Turn Undead for instant-kill chance on undead.",
        "effectiveness": 0.7,
        "risk": 0.3,
        "requirements": {"min_level": 50, "classes": ["priest"]},
    },
]


@dataclass(slots=True)
class InnovationEngine:
    """Discovers new strategies through experimentation.

    Real innovation comes from:
    1. Skill interaction discovery (combining skills for emergent effects)
    2. Map exploration (finding new farming spots)
    3. Price arbitrage (buy low, sell high at NPCs)
    4. Build optimization (finding better stat/skill distributions)
    5. Route optimization (finding faster leveling paths)
    """

    _lock: RLock = field(default_factory=RLock)
    _experiments: list[Experiment] = field(default_factory=list)
    _innovations: list[Innovation] = field(default_factory=list)
    _stats: dict[str, int] = field(default_factory=lambda: {
        "experiments_run": 0, "innovations_discovered": 0, "adopted": 0,
    })
    _discovered_combos: set[str] = field(default_factory=set)
    _price_observations: dict[str, dict[str, Any]] = field(default_factory=dict)
    _map_exploration: dict[str, dict[str, Any]] = field(default_factory=dict)

    def propose_experiment(self, name: str, hypothesis: str, duration_minutes: int = 30) -> Experiment:
        """Propose a new experiment."""
        exp = Experiment(
            name=name,
            hypothesis=hypothesis,
            duration_minutes=duration_minutes,
        )
        with self._lock:
            self._experiments.append(exp)
        logger.info("innovation_experiment_proposed: %s — %s", name, hypothesis)
        return exp

    def propose_knowledge_driven_experiment(self, knowledge: dict | None = None) -> Experiment | None:
        """Propose an experiment based on game knowledge.

        Generates experiments in 5 categories:
        1. Skill combo discovery
        2. Map exploration
        3. Price arbitrage
        4. Build optimization
        5. Exploit/hidden-pattern discovery (warp glitches, spawn manipulation, price mismatches)
        """
        if knowledge is None:
            return None

        try:
            if not isinstance(knowledge, dict):
                return None

            # Get character state
            char_state = knowledge.get("character", {})
            char_level = char_state.get("level", 1) or 1
            char_class = (char_state.get("job", "") or "").lower()
            char_skills = knowledge.get("skills", {}) or {}
            char_items = knowledge.get("items", {}) or {}

            # ── 1. Try skill combo discovery ──
            combo_exp = self._try_skill_combo_discovery(char_level, char_class, char_skills)
            if combo_exp:
                return combo_exp

            # ── 2. Try map exploration ──
            map_exp = self._try_map_exploration(knowledge)
            if map_exp:
                return map_exp

            # ── 3. Try price arbitrage ──
            price_exp = self._try_price_arbitrage(char_items)
            if price_exp:
                return price_exp

            # ── 4. Try build optimization ──
            build_exp = self._try_build_optimization(char_level, char_class, char_skills)
            if build_exp:
                return build_exp

            # ── 5. Try exploit/hidden-pattern discovery ──
            exploit_exp = self._try_exploit_discovery(char_level, char_class, char_skills, knowledge)
            if exploit_exp:
                return exploit_exp

            return None

        except Exception as e:
            logger.warning("innovation_proposal_failed: %s", e)
            return None

    def _try_skill_combo_discovery(
        self, char_level: int, char_class: str, char_skills: dict[str, int]
    ) -> Experiment | None:
        """Try to discover a new skill combo the character can use."""
        known_skills = set(char_skills.keys())
        for combo in SKILL_COMBOS:
            combo_name = combo["name"]
            if combo_name in self._discovered_combos:
                continue

            # Check level requirement
            req_level = combo["requirements"].get("min_level", 1)
            if char_level < req_level:
                continue

            # Check class requirement
            req_classes = combo["requirements"].get("classes", [])
            if req_classes and not any(c in char_class for c in req_classes):
                continue

            # Check if character has all required skills
            required_skills = combo["skills"]
            has_all = all(s in known_skills for s in required_skills)
            if not has_all:
                continue

            # Propose the experiment
            self._discovered_combos.add(combo_name)
            return self.propose_experiment(
                name=f"test_{combo_name}",
                hypothesis=combo["description"],
                duration_minutes=15,
            )

        return None

    def _try_map_exploration(self, knowledge: dict) -> Experiment | None:
        """Try to discover new farming spots by exploring connected maps."""
        current_map = knowledge.get("map", "")
        if not current_map:
            return None

        # Check if we've explored this map's connections
        explored = self._map_exploration.get(current_map, {})
        connections = explored.get("connections_tried", 0)

        # If we haven't explored much, propose exploring
        if connections < 3:
            return self.propose_experiment(
                name=f"explore_from_{current_map}",
                hypothesis=f"Explore maps connected to {current_map} to find new farming spots",
                duration_minutes=20,
            )

        return None

    def _try_price_arbitrage(self, char_items: dict) -> Experiment | None:
        """Try to discover NPC price arbitrage opportunities.

        COMPLETED (completeness mandate): the arbitrage detector was an incomplete stub
        (returned None without checking data). It now queries the shared learning DB's
        `prices` table (buy_price / sell_price per item, recorded from real shop
        interactions). If it finds an item with a genuine buy<sell mismatch (some NPC buys
        an item for more than it costs — an arbitrage window), it proposes a real
        experiment. If no price data has been observed yet (a fresh server with zero shop
        interactions), it returns None honestly rather than fabricating an opportunity.
        """
        try:
            from contextlib import nullcontext
            from ai_sidecar.learning.shared_learning_db import SharedLearningDB
            sdb = SharedLearningDB()
            # If the shared DB isn't initialized/pointed yet, honest no-op.
            with sdb._lock if hasattr(sdb, "_lock") else nullcontext():
                rows = sdb._query_arbitrage_candidates(limit=5)
            if not rows:
                return None
            # rows: (item_name, avg_buy, avg_sell, seen)
            best = max(rows, key=lambda r: (r[2] - r[1]) if r[2] > r[1] else -1)
            _bn, _bb, _bs, _cnt = best
            if _bs > _bb and _cnt >= 2:
                return self.propose_experiment(
                    name=f"price_arbitrage_{_bn}",
                    hypothesis=f"Detected NPC sell>buy mismatch on {_bn} (buy {_bb}z, sell {_bs}z) — "
                               f"buy low & sell high for profit.",
                    duration_minutes=15,
                )
        except Exception:
            return None
        return None

    def _try_build_optimization(
        self, char_level: int, char_class: str, char_skills: dict[str, int]
    ) -> Experiment | None:
        """Try to optimize the build by testing different skill rotations."""
        if not char_skills:
            return None

        # Find skills that could be used in a different order
        offensive_skills = [s for s in char_skills if char_skills[s] > 0
                           and s in SKILL_ELEMENTS]
        if len(offensive_skills) >= 2:
            # Try a different rotation
            return self.propose_experiment(
                name=f"rotation_test_{offensive_skills[0]}_{offensive_skills[1]}",
                hypothesis=f"Test if using {offensive_skills[0]} before {offensive_skills[1]} is more efficient",
                duration_minutes=10,
            )

        return None

    def _try_exploit_discovery(
        self, char_level: int, char_class: str, char_skills: dict[str, int],
        knowledge: dict | None = None
    ) -> Experiment | None:
        """Try to discover hidden patterns and exploits.

        Real RO exploits include:
        - Warp glitches: using Fly Wing at specific map edges to skip dungeon floors
        - Spawn manipulation: standing at specific coordinates to control spawn points
        - Price mismatches: NPCs that buy items for more than they sell
        - Skill bugs: skills that don't consume SP, or deal unintended damage
        - Map geometry: walls that block mobs but not players (safe spots)
        - Element bugs: skills that apply wrong element multiplier
        """
        if not char_skills:
            return None

        # ── Check for known exploit patterns ──
        known_skills = set(char_skills.keys())

        # Pattern: Safety Wall + any AoE = invincible farming
        if "Safety Wall" in known_skills:
            aoe_skills = [s for s in known_skills if s in (
                "Storm Gust", "Meteor Storm", "Lord of Vermilion",
                "Heaven's Drive", "Fire Ball", "Arrow Shower",
                "Magnum Break", "Bowling Bash",
            )]
            if aoe_skills and "test_Safety_Wall_AoE" not in self._discovered_combos:
                self._discovered_combos.add("test_Safety_Wall_AoE")
                return self.propose_experiment(
                    name="test_Safety_Wall_AoE",
                    hypothesis=f"Cast Safety Wall on self, then use {aoe_skills[0]} while inside. "
                               f"The wall absorbs damage while AoE kills everything around.",
                    duration_minutes=15,
                )

        # Pattern: Lex Aeterna + any high-damage skill = 2x burst
        if "Lex Aeterna" in known_skills:
            burst_skills = [s for s in known_skills if s in (
                "Sonic Blow", "Asura Strike", "Storm Gust",
                "Meteor Storm", "Bowling Bash", "Mammonite",
            )]
            if burst_skills and "test_Lex_Aeterna_Burst" not in self._discovered_combos:
                self._discovered_combos.add("test_Lex_Aeterna_Burst")
                return self.propose_experiment(
                    name="test_Lex_Aeterna_Burst",
                    hypothesis=f"Cast Lex Aeterna on target, then use {burst_skills[0]} for 2x damage burst.",
                    duration_minutes=10,
                )

        # Pattern: Provoke + any physical skill = reduced DEF damage bonus
        if "Provoke" in known_skills:
            phys_skills = [s for s in known_skills if s in (
                "Bash", "Magnum Break", "Sonic Blow", "Bowling Bash",
                "Mammonite", "Double Strafe",
            )]
            if phys_skills and "test_Provoke_Combo" not in self._discovered_combos:
                self._discovered_combos.add("test_Provoke_Combo")
                return self.propose_experiment(
                    name="test_Provoke_Combo",
                    hypothesis=f"Provoke reduces monster DEF by 10% per level. "
                               f"Use Provoke then {phys_skills[0]} for bonus damage.",
                    duration_minutes=10,
                )

        # Pattern: Check for price arbitrage opportunities
        if knowledge:
            npc_data = knowledge.get("npcs", {})
            if npc_data:
                # Check if any NPC buys items for more than they sell
                # COMPLETED per completeness mandate: previously a bare `pass`. Now
                # probes the real (DB-backed) arbitrage detection; if the learned
                # prices show a genuine buy<sell window, propose an experiment.
                _arb = self._try_price_arbitrage(knowledge.get("items", {}) or {})
                if _arb is not None:
                    return _arb

        return None

    def start_experiment(self, name: str) -> bool:
        """Start a proposed experiment."""
        with self._lock:
            for exp in self._experiments:
                if exp.name == name and exp.status == "proposed":
                    exp.status = "running"
                    exp.started_at = time.time()
                    logger.info("innovation_experiment_started: %s", name)
                    return True
        return False

    def complete_experiment(self, name: str, result: str, value_score: float) -> bool:
        """Complete an experiment with results."""
        with self._lock:
            for exp in self._experiments:
                if exp.name == name and exp.status == "running":
                    exp.status = "completed"
                    exp.result = result
                    exp.value_score = value_score
                    exp.completed_at = time.time()
                    self._stats["experiments_run"] += 1

                    # If valuable, create an innovation
                    if value_score >= 0.5:
                        innovation = Innovation(
                            name=name,
                            description=result,
                            category="tactic",
                            effectiveness=value_score,
                            risk=0.3,
                            discovered_at=time.time(),
                        )
                        self._innovations.append(innovation)
                        self._stats["innovations_discovered"] += 1
                        logger.info("innovation_discovered: %s (score=%.2f)", name, value_score)

                    return True
        return False

    def get_pending_experiments(self) -> list[Experiment]:
        """Get experiments that are proposed but not yet started."""
        with self._lock:
            return [e for e in self._experiments if e.status == "proposed"]

    def get_running_experiments(self) -> list[Experiment]:
        """Get currently running experiments."""
        with self._lock:
            return [e for e in self._experiments if e.status == "running"]

    def get_completed_experiments(self) -> list[Experiment]:
        """Get completed experiments."""
        with self._lock:
            return [e for e in self._experiments if e.status == "completed"]

    def get_innovation_context(self) -> dict[str, Any]:
        """Get context for innovation decisions."""
        with self._lock:
            return {
                "experiments_pending": len(self.get_pending_experiments()),
                "experiments_running": len(self.get_running_experiments()),
                "experiments_completed": len(self.get_completed_experiments()),
                "innovations_discovered": self._stats["innovations_discovered"],
                "discovered_combos": list(self._discovered_combos),
                "recent_innovations": [
                    {"name": i.name, "description": i.description, "category": i.category}
                    for i in self._innovations[-5:]
                ],
            }

    def get_innovation(self) -> InnovationEngine:
        """Return self for compatibility."""
        return self
