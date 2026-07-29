"""Homunculus stat/skill management and lifecycle."""
from __future__ import annotations

import logging
from typing import Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HomunculusState:
    """Track homunculus state."""
    name: str = ""
    homun_id: str = "homun_s"
    level: int = 1
    intimacy: int = 0
    hungry: bool = False
    is_alive: bool = False
    evolution_stage: int = 1
    known_skills: list[dict] = field(default_factory=list)
    skill_points: int = 0


_HOMUNCULUS_SKILL_BUILDS: dict[str, dict[int, list[tuple[str, int]]]] = {
    "homun_s": {
        1: [("HLIF_BRAIN", 5), ("HLIF_SIGHT", 3)],
        2: [("HLIF_BRAIN", 10), ("HLIF_SIGHT", 5), ("HLIF_BLAST", 5)],
        3: [("HLIF_BRAIN", 10), ("HLIF_SIGHT", 10), ("HLIF_BLAST", 10), ("HLIF_FLAMES", 10)],
    },
    "homun_d": {
        1: [("HLIF_BRAIN", 5), ("HLIF_SIGHT", 3)],
        2: [("HLIF_BRAIN", 10), ("HLIF_SIGHT", 5), ("HLIF_BLAST", 5)],
        3: [("HLIF_BRAIN", 10), ("HLIF_SIGHT", 10), ("HLIF_BLAST", 10), ("HLIF_FLAMES", 10)],
    },
}


class HomunculusManager:
    """Manage homunculus stat/skill allocation and lifecycle."""

    FEED_INTERVAL = 600
    INTIMACY_EVOLVE_THRESHOLD = 800

    def __init__(self, db: Any = None) -> None:
        self._homun_states: dict[str, HomunculusState] = {}
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def assess_homunculus(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Check homunculus state and recommend actions.

        Returns list of action dicts.
        """
        actions: list[dict] = []
        homun_info = signals.get("homunculus", {}) or {}
        now = __import__("time").time()

        hstate = self._homun_states.get(bot_id)
        if not homun_info and not hstate:
            return actions

        if homun_info and not hstate:
            hstate = HomunculusState()
            self._homun_states[bot_id] = hstate

        if homun_info:
            hstate.name = str(homun_info.get("name", hstate.name) or "")
            hstate.level = int(homun_info.get("level", hstate.level) or 1)
            hstate.intimacy = int(homun_info.get("intimacy", hstate.intimacy) or 0)
            hstate.hungry = bool(homun_info.get("hungry", False))
            hstate.is_alive = bool(homun_info.get("is_alive", True))
            hstate.skill_points = int(homun_info.get("skill_points", hstate.skill_points) or 0)

        if not hstate.is_alive:
            actions.append({
                "type": "revive_homunculus",
                "priority": 8,
                "reason": f"Homunculus {hstate.name} is dead — needs revival",
            })

        if hstate.hungry:
            actions.append({
                "type": "feed_homunculus",
                "priority": 7,
                "reason": f"Homunculus {hstate.name} is hungry (intimacy: {hstate.intimacy})",
            })

        if hstate.skill_points > 0:
            skill_builds = _HOMUNCULUS_SKILL_BUILDS.get(
                hstate.homun_id, _HOMUNCULUS_SKILL_BUILDS["homun_s"],
            )
            stage_build = skill_builds.get(hstate.evolution_stage, skill_builds[1])

            for skill_id, target_lv in stage_build:
                current_lv = next(
                    (int(s.get("level", 0) if isinstance(s, dict) else 0)
                     for s in hstate.known_skills
                     if (s.get("id", "") if isinstance(s, dict) else "") == skill_id),
                    0,
                )
                if current_lv < target_lv and hstate.skill_points > 0:
                    actions.append({
                        "type": "train_homunculus_skill",
                        "priority": 6,
                        "reason": f"Train homunculus skill {skill_id} to level {target_lv}",
                        "skill": skill_id,
                        "target_level": target_lv,
                    })
                    break

        if hstate.evolution_stage < 3 and hstate.intimacy >= self.INTIMACY_EVOLVE_THRESHOLD:
            actions.append({
                "type": "evolve_homunculus",
                "priority": 5,
                "reason": f"Homunculus {hstate.name} ready for evolution (stage {hstate.evolution_stage})",
            })

        return actions

    def get_feed_command(self) -> str:
        return "homunculus feed"

    def get_skill_command(self, skill_id: str) -> str:
        return f"homunculus skill {skill_id}"

    def get_homunculus_info_command(self) -> str:
        return "homunculus info"

    def record_feed(self, bot_id: str) -> None:
        hstate = self._homun_states.setdefault(bot_id, HomunculusState())
        hstate.hungry = False
        hstate.intimacy = min(1000, hstate.intimacy + 5)

    def record_evolution(self, bot_id: str) -> None:
        hstate = self._homun_states.get(bot_id)
        if hstate:
            hstate.evolution_stage = min(3, hstate.evolution_stage + 1)

    def cleanup_bot(self, bot_id: str) -> None:
        self._homun_states.pop(bot_id, None)
