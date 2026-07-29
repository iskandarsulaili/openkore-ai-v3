"""Instance lifecycle management: enter -> navigate -> complete -> exit."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ai_sidecar.domains.instances.registry import InstanceRegistry

logger = logging.getLogger(__name__)


@dataclass
class InstanceSession:
    """Tracks an in-progress instance run."""
    instance_id: str = ""
    current_stage: int = 0
    total_stages: int = 0
    monsters_killed: int = 0
    items_collected: list[str] = field(default_factory=list)
    started_at: float = 0.0
    completed: bool = False
    failed: bool = False
    rewards_claimed: bool = False


class InstanceCoordinator:
    """Manage instance lifecycle: enter -> navigate -> complete -> exit."""

    def __init__(
        self,
        registry: InstanceRegistry | None = None,
        db: Any = None,
    ) -> None:
        self.registry = registry or InstanceRegistry()
        self._sessions: dict[str, InstanceSession] = {}
        self._last_run: dict[str, dict[str, float]] = {}
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def assess_instance_opportunities(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Check if bot should enter or progress an instance.

        Returns list of action dicts.
        """
        actions: list[dict] = []
        base_level = int(signals.get("base_level", 1) or 1)
        map_name = str(signals.get("map", "") or "").lower().replace(".gat", "")
        zeny = int(signals.get("zeny", 0) or 0)

        # Check active session
        session = self._sessions.get(bot_id)
        if session and not session.completed and not session.failed:
            actions.append({
                "type": "progress_instance",
                "priority": 8,
                "reason": f"Instance: {session.instance_id} stage {session.current_stage + 1}/{session.total_stages}",
                "instance_id": session.instance_id,
                "stage": session.current_stage,
            })
            return actions

        # Check if we're near an instance entrance
        available = self.registry.get_available_instances(base_level)
        for inst_info in available:
            entrance_map = inst_info["entrance_map"]
            if entrance_map not in map_name:
                continue

            entry_cost = self.registry.get_entry_cost(inst_info["instance_id"])
            if zeny < entry_cost:
                continue

            # Check cooldown
            bot_runs = self._last_run.setdefault(bot_id, {})
            last_run = bot_runs.get(inst_info["instance_id"], 0)
            cooldown_status = self.registry.get_cooldown_status(
                inst_info["instance_id"], last_run,
            )

            if cooldown_status["is_available"]:
                actions.append({
                    "type": "enter_instance",
                    "priority": 7,
                    "reason": f"Instance {inst_info['name']} available — enter at {inst_info['entrance_npc']}",
                    "instance_id": inst_info["instance_id"],
                    "entrance_npc": inst_info["entrance_npc"],
                    "entrance_map": entrance_map,
                    "cost": entry_cost,
                })
            else:
                hours_left = cooldown_status["time_until_available"] / 3600
                logger.debug(
                    "[instances] %s: %s on cooldown — %.1f hours left",
                    bot_id, inst_info["name"], hours_left,
                )

        return actions

    def start_instance(self, bot_id: str, instance_id: str) -> str | None:
        """Start a new instance run.

        Returns the entrance talk command, or None if instance unknown.
        """
        inst = self.registry.get_instance(instance_id)
        if not inst:
            logger.warning("[instances] %s: unknown instance '%s'", bot_id, instance_id)
            return None

        self._sessions[bot_id] = InstanceSession(
            instance_id=instance_id,
            current_stage=0,
            total_stages=len(inst["stages"]),
            started_at=time.time(),
        )
        self._last_run.setdefault(bot_id, {})[instance_id] = time.time()

        logger.info("[instances] %s: starting instance '%s'", bot_id, inst["name"])
        return f"talk @{inst['entrance_npc']}@"

    def advance_stage(self, bot_id: str) -> None:
        """Advance to the next stage of the current instance."""
        session = self._sessions.get(bot_id)
        if session and not session.completed:
            session.current_stage += 1
            logger.info(
                "[instances] %s: advanced to stage %d/%d",
                bot_id, session.current_stage, session.total_stages,
            )
            if session.current_stage >= session.total_stages:
                session.completed = True
                logger.info("[instances] %s: instance complete!", bot_id)

    def record_monster_kill(self, bot_id: str, monster_name: str) -> None:
        """Record a monster kill inside an instance."""
        session = self._sessions.get(bot_id)
        if session:
            session.monsters_killed += 1

    def complete_instance(self, bot_id: str) -> list[str]:
        """Finalize a completed instance and get reward commands.

        Returns list of reward claim commands.
        """
        session = self._sessions.get(bot_id)
        if not session:
            return []

        session.completed = True
        inst_def = self.registry.get_instance(session.instance_id)
        rewards = inst_def["rewards"] if inst_def else {}
        inst = self.registry.get_instance(session.instance_id)

        commands: list[str] = []
        if rewards.get("items"):
            logger.info(
                "[instances] %s: claimed rewards from %s",
                bot_id, session.instance_id,
            )
            commands.append(f"talk @{inst['entrance_npc']}@ reward")

        return commands

    def fail_instance(self, bot_id: str, reason: str = "") -> None:
        """Mark the current instance as failed."""
        session = self._sessions.get(bot_id)
        if session:
            session.failed = True
            logger.warning(
                "[instances] %s: instance '%s' failed: %s",
                bot_id, session.instance_id, reason,
            )

    def exit_instance(self, bot_id: str) -> str:
        """Generate commands to exit the instance."""
        session = self._sessions.get(bot_id)
        if session:
            session.completed = True
        return "talk @instance_exit@"

    def is_in_instance(self, bot_id: str) -> bool:
        """Check if bot is currently inside an instance."""
        session = self._sessions.get(bot_id)
        if not session:
            return False
        return not session.completed and not session.failed

    def get_instance_progress(self, bot_id: str) -> dict | None:
        """Get progress info for the current instance."""
        session = self._sessions.get(bot_id)
        if not session:
            return None
        return {
            "instance_id": session.instance_id,
            "current_stage": session.current_stage,
            "total_stages": session.total_stages,
            "monsters_killed": session.monsters_killed,
            "completed": session.completed,
            "failed": session.failed,
        }

    def cleanup_bot(self, bot_id: str) -> None:
        """Remove instance state for a bot."""
        self._sessions.pop(bot_id, None)
        self._last_run.pop(bot_id, None)
