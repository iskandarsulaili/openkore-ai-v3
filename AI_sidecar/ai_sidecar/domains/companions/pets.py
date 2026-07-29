"""Pet feeding, intimacy optimization, and pet management."""
from __future__ import annotations

import logging
from typing import Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


_PET_FOOD: dict[str, dict[str, Any]] = {
    "poring": {"food": "Apple", "food_id": "512", "intimacy_gain": 5},
    "lunatic": {"food": "Carrot", "food_id": "515", "intimacy_gain": 5},
    "picky": {"food": "Meat", "food_id": "517", "intimacy_gain": 5},
    "poporing": {"food": "Apple", "food_id": "512", "intimacy_gain": 5},
    "drops": {"food": "Jellopy", "food_id": "909", "intimacy_gain": 2},
    "chuk": {"food": "Banana", "food_id": "513", "intimacy_gain": 5},
}


@dataclass
class PetState:
    """Track a bot's pet state."""
    pet_name: str = ""
    pet_type: str = ""
    intimacy: int = 0  # 0-1000
    hungry: bool = False
    is_alive: bool = False
    level: int = 1
    last_fed: float = 0.0
    last_caressed: float = 0.0
    evolution_stage: int = 1


class PetManager:
    """Handle pet feeding, intimacy optimization, and evolution."""

    INTIMACY_VERY_HUNGRY = 100
    INTIMACY_HUNGRY = 300
    INTIMACY_NORMAL = 500
    INTIMACY_HAPPY = 700
    INTIMACY_LOYAL = 900

    FEED_INTERVAL = 600
    CARESS_INTERVAL = 300

    def __init__(self, db: Any = None) -> None:
        self._pet_states: dict[str, PetState] = {}
        self._gk_db = db
        if self._gk_db is None:
            from ai_sidecar.game_knowledge_db import GameKnowledgeDB
            self._gk_db = GameKnowledgeDB()

    def assess_pet(
        self,
        signals: dict[str, Any],
        bot_id: str,
    ) -> list[dict]:
        """Check pet state and recommend actions.

        Returns list of action dicts.
        """
        actions: list[dict] = []
        inventory = signals.get("inventory", []) or []
        pet_info = signals.get("pet", {}) or {}
        zeny = int(signals.get("zeny", 0) or 0)
        now = __import__("time").time()

        pet_state = self._pet_states.get(bot_id)
        if not pet_state and not pet_info:
            return actions

        if pet_info and not pet_state:
            pet_state = PetState()
            self._pet_states[bot_id] = pet_state

        if pet_info:
            pet_state.pet_name = str(pet_info.get("name", pet_state.pet_name) or "")
            pet_state.pet_type = str(pet_info.get("type", pet_state.pet_type) or "").lower()
            pet_state.intimacy = int(pet_info.get("intimacy", pet_state.intimacy) or 0)
            pet_state.hungry = bool(pet_info.get("hungry", False))
            pet_state.is_alive = bool(pet_info.get("is_alive", True))

        if not pet_state.is_alive:
            return actions

        pet_type = pet_state.pet_type
        time_since_fed = now - pet_state.last_fed

        if pet_state.hungry or time_since_fed > self.FEED_INTERVAL:
            food_info = _PET_FOOD.get(pet_type)
            if food_info:
                food_name = food_info["food"]
                food_id = food_info["food_id"]
                has_food = any(
                    food_name.lower() in (item.get("name", "") or "").lower()
                    or food_id == item.get("id", "")
                    for item in inventory
                )
                if has_food:
                    actions.append({
                        "type": "feed_pet",
                        "priority": 6,
                        "reason": f"Feed {pet_state.pet_name} ({food_name}) — intimacy: {pet_state.intimacy}",
                        "food": food_name,
                        "food_id": food_id,
                    })
                elif zeny > 1000:
                    actions.append({
                        "type": "buy_pet_food",
                        "priority": 5,
                        "reason": f"Buy {food_name} for {pet_state.pet_name}",
                        "food": food_name,
                        "quantity": 5,
                    })

        time_since_caress = now - pet_state.last_caressed
        if time_since_caress > self.CARESS_INTERVAL and pet_state.intimacy < self.INTIMACY_LOYAL:
            actions.append({
                "type": "pet_caress",
                "priority": 4,
                "reason": f"Caress {pet_state.pet_name} (intimacy: {pet_state.intimacy})",
            })

        if pet_state.evolution_stage < 3 and pet_state.intimacy >= self.INTIMACY_LOYAL:
            actions.append({
                "type": "evolve_pet",
                "priority": 7,
                "reason": f"{pet_state.pet_name} ready for evolution (intimacy: {pet_state.intimacy})",
            })

        return actions

    def get_feed_command(self) -> str:
        return "pet feed"

    def get_caress_command(self) -> str:
        return "pet caress"

    def get_pet_info_command(self) -> str:
        return "pet info"

    def record_feed(self, bot_id: str) -> None:
        now = __import__("time").time()
        state = self._pet_states.setdefault(bot_id, PetState())
        state.last_fed = now
        state.hungry = False
        state.intimacy = min(1000, state.intimacy + 5)

    def record_caress(self, bot_id: str) -> None:
        now = __import__("time").time()
        state = self._pet_states.setdefault(bot_id, PetState())
        state.last_caressed = now
        state.intimacy = min(1000, state.intimacy + 3)

    def get_pet_summary(self, bot_id: str) -> str | None:
        state = self._pet_states.get(bot_id)
        if not state:
            return None
        return (
            f"Pet: {state.pet_name} ({state.pet_type}) | "
            f"Intimacy: {state.intimacy}/1000 | "
            f"Hungry: {state.hungry} | "
            f"Stage: {state.evolution_stage}/3"
        )

    def cleanup_bot(self, bot_id: str) -> None:
        self._pet_states.pop(bot_id, None)
