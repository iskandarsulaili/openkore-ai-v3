"""
Progression Driver — queues concrete actions for the bridge to execute.

This is the bridge between the conscious engine's decisions and the action queue.
It produces real commands that the bridge executes:
- learn_skill <skill_name>  → learn a skill
- stat_add <stat> <points>  → add stat points
- buy <item> <qty>          → buy from NPC shop
- sell <item> <qty>         → sell to NPC shop
- move <map_name>           → move to a map
- sit                        → sit to regen
- tele                      → teleport to save point
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Any, Callable

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal

logger = logging.getLogger(__name__)

# ── Skill Names (matching OpenKore internal names) ──
SKILL_NAMES = {
    "NV_BASIC": "Basic Skill",
    "NV_FIRSTAID": "First Aid",
    "NV_TRICKDEAD": "Trick Dead",
    "SM_SWORD": "Sword Mastery",
    "SM_RECOVERY": "HP Recovery",
    "SM_BASH": "Bash",
    "SM_MAGNUM": "Magnum Break",
    "MG_SRECOVERY": "SP Recovery",
    "MG_FIREBOLT": "Fire Bolt",
    "MG_COLDBOLT": "Cold Bolt",
    "MG_LIGHTNINGBOLT": "Lightning Bolt",
    "AL_HEAL": "Heal",
    "AL_INCAGI": "Increase AGI",
    "AL_BLESSING": "Blessing",
    "AL_TELEPORT": "Teleport",
    "AC_OWL": "Owl's Eye",
    "AC_VULTURE": "Vulture's Eye",
    "AC_DOUBLE": "Double Strafe",
    "AC_SHOWER": "Arrow Shower",
    "TF_DOUBLE": "Double Attack",
    "TF_HIDE": "Hide",
    "TF_STEAL": "Steal",
    "TF_POISON": "Envenom",
    "MC_PUSHCART": "Pushcart",
    "MC_DISCOUNT": "Discount",
    "MC_OVERCHARGE": "Overcharge",
    "MC_MAMMONITE": "Mammonite",
}

# ── NPC Shop Data (Prontera) ──
NPC_SHOPS = {
    "prontera": {
        "potion_npc": {"name": "Kafra Employee", "x": 158, "y": 89},
        "tool_npc": {"name": "Tool Dealer", "x": 147, "y": 88},
        "weapon_npc": {"name": "Weapon Dealer", "x": 165, "y": 95},
    },
    "prt_fild08": {
        "potion_npc": None,  # No NPC in field maps
    },
}


class ProgressionDriver:
    """Queues concrete actions for bot progression."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._queue_fn: Callable | None = None
        self._last_action: dict[str, float] = {}  # bot_id -> time of last action
        self._action_cooldown: float = 5.0  # Minimum seconds between actions per bot
        self._learned_skills: dict[str, set[str]] = {}  # bot_id -> set of skill names
        self._restocked_items: dict[str, dict[str, float]] = {}  # bot_id -> {item: last_time}

    def set_queue_fn(self, fn: Callable) -> None:
        """Set the function to queue actions."""
        self._queue_fn = fn

    def _queue(self, bot_id: str, command: str, kind: str = "command",
               priority: ActionPriorityTier = ActionPriorityTier.tactical,
               conflict_key: str | None = None) -> bool:
        """Queue an action for the bridge to execute."""
        if self._queue_fn is None:
            return False

        now = time.time()
        # Check cooldown
        last = self._last_action.get(bot_id, 0)
        if now - last < self._action_cooldown:
            return False
        self._last_action[bot_id] = now

        proposal = ActionProposal(
            action_id=f"progression_{bot_id}_{int(now * 1000)}",
            kind=kind,
            command=command,
            priority_tier=priority,
            conflict_key=conflict_key or f"progression_{bot_id}",
            source="reflex",
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=30),
            idempotency_key=f"progression_{bot_id}_{command}_{int(now)}",
        )
        try:
            accepted, status, action_id, reason = self._queue_fn(
                proposal=proposal, bot_id=bot_id
            )
            if accepted:
                logger.info(
                    "progression_queued: bot=%s cmd=%s action_id=%s",
                    bot_id, command, action_id,
                )
                return True
            else:
                logger.debug(
                    "progression_rejected: bot=%s cmd=%s reason=%s",
                    bot_id, command, reason,
                )
                return False
        except Exception as e:
            logger.warning("progression_error: bot=%s cmd=%s err=%s", bot_id, command, e)
            return False

    def learn_skill(self, bot_id: str, skill_name: str) -> bool:
        """Queue a skill learning action."""
        learned = self._learned_skills.setdefault(bot_id, set())
        if skill_name in learned:
            return False  # Already learned

        display_name = SKILL_NAMES.get(skill_name, skill_name)
        ok = self._queue(
            bot_id=bot_id,
            command=f"learn_skill {skill_name}",
            conflict_key=f"skill_{bot_id}",
        )
        if ok:
            learned.add(skill_name)
            logger.info("progression_learn_skill: bot=%s skill=%s", bot_id, skill_name)
        return ok

    def add_stat(self, bot_id: str, stat: str, points: int = 1) -> bool:
        """Queue a stat addition action."""
        return self._queue(
            bot_id=bot_id,
            command=f"stat_add {stat} {points}",
            conflict_key=f"stat_{bot_id}",
        )

    def buy_item(self, bot_id: str, item: str, qty: int) -> bool:
        """Queue a buy action."""
        restocked = self._restocked_items.setdefault(bot_id, {})
        now = time.time()
        last = restocked.get(item, 0)
        if now - last < 300:  # Don't restock same item more than once per 5 min
            return False
        restocked[item] = now

        return self._queue(
            bot_id=bot_id,
            command=f"buy {item} {qty}",
            conflict_key=f"restock_{bot_id}",
        )

    def sell_item(self, bot_id: str, item: str, qty: int) -> bool:
        """Queue a sell action."""
        return self._queue(
            bot_id=bot_id,
            command=f"sell {item} {qty}",
            conflict_key=f"restock_{bot_id}",
        )

    def move_map(self, bot_id: str, map_name: str) -> bool:
        """Queue a map movement action."""
        return self._queue(
            bot_id=bot_id,
            command=f"move {map_name}",
            conflict_key=f"move_{bot_id}",
        )

    def sit_regen(self, bot_id: str) -> bool:
        """Queue a sit-to-regen action."""
        return self._queue(
            bot_id=bot_id,
            command="sit",
            conflict_key=f"regen_{bot_id}",
        )

    def teleport_save(self, bot_id: str) -> bool:
        """Queue a teleport to save point action."""
        return self._queue(
            bot_id=bot_id,
            command="tele",
            conflict_key=f"teleport_{bot_id}",
        )

    def process_decisions(self, bot_id: str, state: dict[str, Any]) -> None:
        """Process conscious engine decisions and queue actions."""
        inventory = state.get("inventory", {})
        skills = state.get("skills", [])
        hp_pct = state.get("hp_pct", 1.0)
        sp_pct = state.get("sp_pct", 1.0)
        base_level = state.get("base_level", 1)
        job_name = state.get("job_name", "novice").lower()
        zeny = state.get("zeny", 0)
        map_name = state.get("map", "")
        weight_ratio = state.get("weight_ratio", 0.0)

        # 1. Learn skills if possible
        if "NV_BASIC" not in skills:
            self.learn_skill(bot_id, "NV_BASIC")
            return  # One action at a time

        if "NV_FIRSTAID" not in skills:
            self.learn_skill(bot_id, "NV_FIRSTAID")
            return

        # 2. Emergency sit to regen (if HP is low)
        if hp_pct < 0.50 and hp_pct > 0.10:
            self.sit_regen(bot_id)
            return

        # 3. Restock potions if missing
        has_potion = any("Potion" in k for k in inventory.keys())
        if not has_potion and zeny > 500:
            self.buy_item(bot_id, "White Potion", 30)
            return

        # 4. Sell junk if overweight
        if weight_ratio > 0.80:
            # Sell common junk items
            for junk in ["Jellopy", "Apple", "Knife", "Boots"]:
                if junk in inventory and inventory[junk] > 0:
                    self.sell_item(bot_id, junk, inventory[junk])
                    return

        # 5. Move to better farming map if level is appropriate
        if base_level >= 10 and "prt_fild" in map_name and map_name == "prt_fild08":
            self.move_map(bot_id, "prt_fild04")
            return

        # 6. Buy fly wings if poor
        if not has_potion and zeny < 500:
            # Farm more zeny
            pass

    def get_summary(self, bot_id: str) -> str:
        """Get a summary of progression state."""
        learned = self._learned_skills.get(bot_id, set())
        restocked = self._restocked_items.get(bot_id, {})
        lines = [f"── Progression Driver: {bot_id} ──"]
        lines.append(f"  Skills learned: {len(learned)}")
        for s in sorted(learned):
            lines.append(f"    - {s}")
        lines.append(f"  Items restocked: {len(restocked)}")
        return "\n".join(lines)


# Global singleton
_driver: ProgressionDriver | None = None
_driver_lock = RLock()


def get_progression_driver() -> ProgressionDriver:
    global _driver
    with _driver_lock:
        if _driver is None:
            _driver = ProgressionDriver()
        return _driver