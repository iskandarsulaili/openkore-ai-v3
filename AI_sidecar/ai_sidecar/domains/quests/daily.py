"""Daily quest identification and rotation management."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# RO daily quest definitions
_DAILY_QUESTS: dict[str, dict[str, Any]] = {
    "prontera": {
        "board_npc": "Quest Board",
        "quests": [
            {"name": "Monster Subjugation (Porings)", "monster": "Poring", "count": 30,
             "reward_zeny": 5000, "reward_exp": 5000},
            {"name": "Monster Subjugation (Fabre)", "monster": "Fabre", "count": 30,
             "reward_zeny": 5000, "reward_exp": 5000},
            {"name": "Material Collection", "item": "Jellopy", "count": 20,
             "reward_zeny": 8000, "reward_exp": 3000},
        ],
    },
    "morocc": {
        "board_npc": "Quest Board",
        "quests": [
            {"name": "Dessert Monster Hunt", "monster": "Desert Wolf", "count": 20,
             "reward_zeny": 8000, "reward_exp": 5000},
            {"name": "Evil Spirit Extermination", "monster": "Zombie", "count": 25,
             "reward_zeny": 10000, "reward_exp": 6000},
        ],
    },
    "payon": {
        "board_npc": "Quest Board",
        "quests": [
            {"name": "Leaf Cat Hunt", "monster": "Leaf Cat", "count": 20,
             "reward_zeny": 8000, "reward_exp": 6000},
            {"name": "Goblin Extermination", "monster": "Goblin", "count": 30,
             "reward_zeny": 12000, "reward_exp": 8000},
        ],
    },
}


@dataclass
class DailyQuestState:
    """State for a single daily quest."""
    quest_name: str = ""
    monster: str = ""
    item: str = ""
    target_count: int = 0
    current_count: int = 0
    completed: bool = False
    reward_zeny: int = 0
    reward_exp: int = 0
    location: str = ""
    npc: str = ""


@dataclass
class DailyState:
    """Overall daily rotation state."""
    day_start: float = 0.0
    active_quests: list[DailyQuestState] = field(default_factory=list)
    completed_quest_ids: set[str] = field(default_factory=set)
    last_board_map: str = ""
    board_visited: bool = False


class DailyQuestManager:
    """Manage daily quest identification, acceptance, and rotation."""

    # 24-hour cooldown for dailies
    DAILY_COOLDOWN = 86400  # 24 hours in seconds

    def __init__(self, db: Any = None) -> None:
        self._daily_states: dict[str, DailyState] = {}
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def assess_daily_opportunities(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Check if dailies should be picked up or progressed.

        Returns list of action dicts.
        """
        actions: list[dict] = []
        base_level = int(signals.get("base_level", 1) or 1)
        map_name = str(signals.get("map", "") or "").lower().replace(".gat", "")
        now = time.time()

        daily_state = self._daily_states.setdefault(bot_id, DailyState())

        # Check cooldown expiry
        if daily_state.day_start > 0:
            elapsed = now - daily_state.day_start
            if elapsed < self.DAILY_COOLDOWN:
                remaining_hours = (self.DAILY_COOLDOWN - elapsed) / 3600
                logger.debug(
                    "[dailies] %s: %d hours until daily reset",
                    bot_id, remaining_hours,
                )
                return actions  # Dailies still on cooldown

        # Find dailies for current map
        map_dailies = _DAILY_QUESTS.get(map_name)
        if not map_dailies:
            # Try to find nearest town with dailies
            for town_name, town_info in _DAILY_QUESTS.items():
                if town_name in map_name:
                    map_dailies = town_info
                    break
        if not map_dailies:
            return actions

        # If we haven't visited the board yet this cycle
        if not daily_state.board_visited or daily_state.last_board_map != map_name:
            actions.append({
                "type": "visit_daily_board",
                "priority": "medium",
                "reason": f"Visit {map_dailies['board_npc']} for daily quests",
                "npc": map_dailies["board_npc"],
                "map": map_name,
            })

        # Progress active dailies
        for dq in daily_state.active_quests:
            if dq.completed:
                continue
            if dq.monster:
                actions.append({
                    "type": "progress_daily",
                    "priority": "medium",
                    "reason": f"Daily: {dq.quest_name} ({dq.current_count}/{dq.target_count})",
                    "monster": dq.monster,
                    "target_count": dq.target_count,
                    "current_count": dq.current_count,
                })
            elif dq.item:
                actions.append({
                    "type": "progress_daily_collect",
                    "priority": "medium",
                    "reason": f"Daily: collect {dq.item} ({dq.current_count}/{dq.target_count})",
                    "item": dq.item,
                    "target_count": dq.target_count,
                    "current_count": dq.current_count,
                })

        return actions

    def activate_daily(self, bot_id: str, map_name: str, quest_idx: int = 0) -> DailyQuestState | None:
        """Activate a daily quest for the bot.

        Args:
            bot_id: Bot identifier
            map_name: Current map name
            quest_idx: Index of which daily to activate (rotates if not specified)

        Returns:
            The activated DailyQuestState, or None if no dailies available
        """
        map_dailies = _DAILY_QUESTS.get(map_name)
        if not map_dailies:
            return None

        daily_state = self._daily_states.setdefault(bot_id, DailyState())
        daily_state.day_start = time.time()
        daily_state.board_visited = True
        daily_state.last_board_map = map_name

        available_quests = map_dailies["quests"]
        if quest_idx >= len(available_quests):
            quest_idx = 0

        quest_info = available_quests[quest_idx]
        dq = DailyQuestState(
            quest_name=quest_info["name"],
            monster=quest_info.get("monster", ""),
            item=quest_info.get("item", ""),
            target_count=quest_info.get("count", 1),
            current_count=0,
            reward_zeny=quest_info.get("reward_zeny", 0),
            reward_exp=quest_info.get("reward_exp", 0),
            location=map_name,
            npc=map_dailies["board_npc"],
        )
        daily_state.active_quests.append(dq)
        logger.info("[dailies] %s: activated daily '%s'", bot_id, dq.quest_name)
        return dq

    def progress_monster_kill(
        self,
        bot_id: str,
        monster_name: str,
        count: int = 1,
    ) -> None:
        """Progress daily quest by monster kill count."""
        daily_state = self._daily_states.get(bot_id)
        if not daily_state:
            return
        for dq in daily_state.active_quests:
            if dq.completed:
                continue
            if dq.monster and dq.monster.lower() == monster_name.lower():
                dq.current_count += count
                if dq.current_count >= dq.target_count:
                    dq.completed = True
                    logger.info("[dailies] %s: daily '%s' completed!", bot_id, dq.quest_name)

    def get_daily_rewards(self, bot_id: str) -> list[dict]:
        """Get pending daily quest rewards."""
        daily_state = self._daily_states.get(bot_id)
        if not daily_state:
            return []
        rewards = []
        for dq in daily_state.active_quests:
            if dq.completed:
                rewards.append({
                    "quest_name": dq.quest_name,
                    "zeny": dq.reward_zeny,
                    "exp": dq.reward_exp,
                })
        return rewards

    def claim_daily_rewards(self, bot_id: str) -> list[str]:
        """Generate commands to claim completed daily rewards."""
        commands = []
        rewards = self.get_daily_rewards(bot_id)
        for reward in rewards:
            commands.append(f"talk @{reward['quest_name']}@ reward")
            self._daily_states[bot_id].completed_quest_ids.add(reward["quest_name"])
        return commands

    def is_daily_reset_available(self, bot_id: str) -> bool:
        """Check if dailies have reset (24h cooldown expired)."""
        daily_state = self._daily_states.get(bot_id)
        if not daily_state or daily_state.day_start == 0:
            return True
        return (time.time() - daily_state.day_start) >= self.DAILY_COOLDOWN

    def cleanup_bot(self, bot_id: str) -> None:
        """Remove daily state for a bot."""
        self._daily_states.pop(bot_id, None)
