"""
Innovation engine — discovers new strategies, spots, and builds.

A top player doesn't follow the meta. They CREATE the meta. They
discover new farming spots, invent new builds, find new exploits,
and adapt faster than anyone else.

This module runs controlled experiments to discover better strategies.
It tracks what we've tried, what worked, and what didn't. It generates
novel strategies by combining known patterns in new ways.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any

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
    category: str  # farming_spot, build, route, tactic, exploit
    effectiveness: float = 0.0  # 0.0-1.0
    risk: float = 0.0  # 0.0-1.0
    discovered_at: float = 0.0
    adopted: bool = False


@dataclass(slots=True)
class InnovationEngine:
    """Discovers new strategies through experimentation."""
    
    _lock: RLock = field(default_factory=RLock)
    _experiments: list[Experiment] = field(default_factory=list)
    _innovations: list[Innovation] = field(default_factory=list)
    _stats: dict[str, int] = field(default_factory=lambda: {
        "experiments_run": 0, "innovations_discovered": 0, "adopted": 0,
    })
    
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
        """Propose an experiment based on game knowledge, not random walk.
        
        Uses knowledge.json data to find promising farming spots, mobs with
        good drop rates, and efficient leveling routes.
        """
        if knowledge is None:
            return None
        
        try:
            items = knowledge.get("items", {}).get("all", [])
            mobs = knowledge.get("mobs", {}).get("all", [])
            
            # Find mobs with valuable drops
            valuable_mobs = []
            for mob in mobs:
                drops = mob.get("drops", []) or mob.get("MvP_Drops", []) or []
                for drop in drops:
                    drop_id = drop.get("Id", drop.get("id", 0))
                    drop_rate = drop.get("Rate", drop.get("rate", 0))
                    if drop_rate > 0 and drop_rate < 1000:  # Rare drops (under 10%)
                        valuable_mobs.append({
                            "name": mob.get("Name", mob.get("name", "unknown")),
                            "map": mob.get("Map", mob.get("map", "unknown")),
                            "level": mob.get("Level", mob.get("level", 0)),
                            "drop_id": drop_id,
                            "drop_rate": drop_rate,
                        })
            
            if valuable_mobs:
                # Pick the most promising mob to farm
                target = valuable_mobs[0]
                exp = self.propose_experiment(
                    name=f"farm_{target['name']}",
                    hypothesis=f"Farm {target['name']} on {target['map']} for rare drops",
                    duration_minutes=30,
                )
                exp.metadata = {"map": target["map"], "mob": target["name"]}
                return exp
            
            # Fallback: find high-density mob areas
            mob_maps = {}
            for mob in mobs:
                m = mob.get("Map", mob.get("map", "unknown"))
                if m != "unknown":
                    mob_maps[m] = mob_maps.get(m, 0) + 1
            
            if mob_maps:
                best_map = max(mob_maps, key=mob_maps.get)
                exp = self.propose_experiment(
                    name=f"density_test_{best_map.replace('_', '')}",
                    hypothesis=f"Test farming density on {best_map} ({mob_maps[best_map]} mob types)",
                    duration_minutes=30,
                )
                exp.metadata = {"map": best_map}
                return exp
        except Exception as e:
            logger.warning("innovation_knowledge_proposal_failed: %s", e)
        
        return None
    
    def start_experiment(self, name: str) -> bool:
        """Start a proposed experiment."""
        with self._lock:
            for exp in self._experiments:
                if exp.name == name and exp.status == "proposed":
                    exp.status = "running"
                    exp.started_at = time.time()
                    self._stats["experiments_run"] += 1
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
                    logger.info("innovation_experiment_completed: %s score=%.1f", name, value_score)
                    
                    # If valuable, record as innovation
                    if value_score > 50:
                        innovation = Innovation(
                            name=name,
                            description=result,
                            category=self._categorize_experiment(name),
                            effectiveness=min(1.0, value_score / 100),
                            risk=0.3,
                            discovered_at=time.time(),
                        )
                        self._innovations.append(innovation)
                        self._stats["innovations_discovered"] += 1
                        logger.info("innovation_discovered: %s", name)
                    
                    return True
        return False
    
    def _categorize_experiment(self, name: str) -> str:
        name_lower = name.lower()
        if "farm" in name_lower or "spot" in name_lower or "map" in name_lower:
            return "farming_spot"
        if "build" in name_lower or "stat" in name_lower or "skill" in name_lower:
            return "build"
        if "route" in name_lower or "path" in name_lower or "travel" in name_lower:
            return "route"
        if "tactic" in name_lower or "strategy" in name_lower:
            return "tactic"
        return "general"
    
    def get_pending_experiments(self) -> list[Experiment]:
        """Get experiments ready to run."""
        with self._lock:
            return [e for e in self._experiments if e.status == "proposed"]
    
    def get_innovation_context(self) -> str:
        """Get formatted innovation context for LLM prompts."""
        with self._lock:
            lines = ["── Innovation Engine ──"]
            
            adopted = [i for i in self._innovations if i.adopted]
            pending = [i for i in self._innovations if not i.adopted]
            
            if adopted:
                lines.append("  Adopted innovations:")
                for i in adopted[-3:]:
                    lines.append(f"    ✓ {i.name} (eff={i.effectiveness:.1f})")
            
            if pending:
                lines.append("  Pending innovations to try:")
                for i in pending[-3:]:
                    lines.append(f"    ○ {i.name} (eff={i.effectiveness:.1f})")
            
            experiments = self.get_pending_experiments()
            if experiments:
                lines.append("  Proposed experiments:")
                for e in experiments[-3:]:
                    lines.append(f"    ? {e.name}: {e.hypothesis}")
            
            return "\n".join(lines)
    
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global instance ──

_innovation: InnovationEngine | None = None
_innovation_lock = RLock()


def get_innovation() -> InnovationEngine:
    global _innovation
    with _innovation_lock:
        if _innovation is None:
            _innovation = InnovationEngine()
        return _innovation
