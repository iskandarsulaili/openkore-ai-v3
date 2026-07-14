"""
Conscious Decision Engine — makes all high-level decisions for bot progression.

This is the bot's "conscious brain" that decides:
1. Skill learning order (based on job class, level, and build)
2. Stat distribution (based on build and level)
3. Equipment goals (what to aim for at each level)
4. Item restocking (what to buy, when, and how many)
5. Map selection (where to farm based on level and gear)
6. Party coordination (when to party, what role to play)

Runs in the sidecar's PDCA loop and produces actions for the bridge to execute.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

_KNOWLEDGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "knowledge", "knowledge.json"
)


@dataclass
class BuildPlan:
    """A complete build plan for a character."""
    job_class: str
    primary_stat: str
    secondary_stat: str
    stat_priority: list[str]
    skill_learn_order: list[dict[str, Any]]
    equipment_goals: list[dict[str, Any]]
    farming_maps: list[dict[str, Any]]
    consumables: list[dict[str, Any]]


@dataclass
class Decision:
    """A decision the engine has made."""
    domain: str  # skills, stats, equipment, restock, map, party
    action: str  # learn_skill, add_stat, buy_item, move_map, etc.
    target: str  # skill name, stat name, item name, map name
    priority: int  # 1=immediate, 5=when convenient
    reason: str
    params: dict[str, Any] = field(default_factory=dict)


# ── Build definitions for all job classes ──
BUILDS: dict[str, BuildPlan] = {
    "novice": BuildPlan(
        job_class="Novice",
        primary_stat="dex",
        secondary_stat="agi",
        stat_priority=["dex", "agi", "vit", "luk", "str", "int"],
        skill_learn_order=[
            {"name": "NV_BASIC", "max_level": 9, "reason": "Required for sitting, trading, and other basic actions"},
            {"name": "NV_FIRSTAID", "max_level": 1, "reason": "Emergency self-heal (45 HP)"},
            {"name": "NV_TRICKDEAD", "max_level": 1, "reason": "Fake death to avoid aggro"},
        ],
        equipment_goals=[
            {"item": "Apple", "qty": 10, "reason": "Cheap healing food"},
            {"item": "Red Potion", "qty": 30, "reason": "Basic healing potion"},
            {"item": "White Potion", "qty": 20, "reason": "Better healing potion"},
            {"item": "Fly Wing", "qty": 10, "reason": "Emergency escape"},
        ],
        farming_maps=[
            {"map": "prt_fild08", "min_level": 1, "max_level": 20, "reason": "Poring, Lunatic, Drops — easy mobs"},
            {"map": "prt_fild04", "min_level": 10, "max_level": 30, "reason": "Fabre, Peco Peco — more exp"},
        ],
        consumables=[
            {"item": "Red Potion", "min_stock": 10, "buy_qty": 30, "max_price": 500},
            {"item": "Fly Wing", "min_stock": 3, "buy_qty": 10, "max_price": 500},
        ],
    ),
    "swordsman": BuildPlan(
        job_class="Swordsman",
        primary_stat="str",
        secondary_stat="dex",
        stat_priority=["str", "dex", "vit", "agi", "luk", "int"],
        skill_learn_order=[
            {"name": "SM_SWORD", "max_level": 10, "reason": "Sword mastery — increases damage"},
            {"name": "SM_RECOVERY", "max_level": 10, "reason": "HP recovery skill"},
            {"name": "SM_BASH", "max_level": 10, "reason": "Bash — main attack skill"},
            {"name": "SM_MAGNUM", "max_level": 10, "reason": "Magnum Break — AoE attack"},
        ],
        equipment_goals=[
            {"item": "White Potion", "qty": 50, "reason": "Main healing potion"},
            {"item": "Red Potion", "qty": 30, "reason": "Backup healing"},
            {"item": "Fly Wing", "qty": 20, "reason": "Emergency escape"},
        ],
        farming_maps=[
            {"map": "prt_fild08", "min_level": 1, "max_level": 25, "reason": "Easy mobs for leveling"},
            {"map": "pay_fild04", "min_level": 20, "max_level": 40, "reason": "Savage, Munak — good exp"},
        ],
        consumables=[
            {"item": "White Potion", "min_stock": 20, "buy_qty": 50, "max_price": 500},
            {"item": "Fly Wing", "min_stock": 5, "buy_qty": 20, "max_price": 500},
        ],
    ),
    "mage": BuildPlan(
        job_class="Mage",
        primary_stat="int",
        secondary_stat="dex",
        stat_priority=["int", "dex", "vit", "agi", "luk", "str"],
        skill_learn_order=[
            {"name": "MG_SRECOVERY", "max_level": 10, "reason": "SP recovery — essential for casting"},
            {"name": "MG_FIREBOLT", "max_level": 10, "reason": "Fire Bolt — main attack"},
            {"name": "MG_COLDBOLT", "max_level": 10, "reason": "Cold Bolt — water element attack"},
            {"name": "MG_LIGHTNINGBOLT", "max_level": 10, "reason": "Lightning Bolt — wind element attack"},
        ],
        equipment_goals=[
            {"item": "White Potion", "qty": 30, "reason": "Healing potion"},
            {"item": "Blue Potion", "qty": 20, "reason": "SP recovery potion"},
            {"item": "Fly Wing", "qty": 20, "reason": "Emergency escape"},
        ],
        farming_maps=[
            {"map": "prt_fild08", "min_level": 1, "max_level": 25, "reason": "Easy mobs"},
            {"map": "moc_fild17", "min_level": 20, "max_level": 40, "reason": "Drainliar, Flora — good for mages"},
        ],
        consumables=[
            {"item": "White Potion", "min_stock": 10, "buy_qty": 30, "max_price": 500},
            {"item": "Blue Potion", "min_stock": 5, "buy_qty": 20, "max_price": 2000},
            {"item": "Fly Wing", "min_stock": 5, "buy_qty": 20, "max_price": 500},
        ],
    ),
    "archer": BuildPlan(
        job_class="Archer",
        primary_stat="dex",
        secondary_stat="agi",
        stat_priority=["dex", "agi", "vit", "luk", "str", "int"],
        skill_learn_order=[
            {"name": "AC_OWL", "max_level": 1, "reason": "Owl's Eye — increases DEX"},
            {"name": "AC_VULTURE", "max_level": 1, "reason": "Vulture's Eye — increases range"},
            {"name": "AC_DOUBLE", "max_level": 10, "reason": "Double Strafe — main attack"},
            {"name": "AC_SHOWER", "max_level": 10, "reason": "Arrow Shower — AoE attack"},
        ],
        equipment_goals=[
            {"item": "White Potion", "qty": 30, "reason": "Healing potion"},
            {"item": "Arrow", "qty": 1000, "reason": "Ammunition"},
            {"item": "Fly Wing", "qty": 20, "reason": "Emergency escape"},
        ],
        farming_maps=[
            {"map": "prt_fild08", "min_level": 1, "max_level": 25, "reason": "Easy mobs"},
            {"map": "pay_fild04", "min_level": 20, "max_level": 40, "reason": "Good for archers"},
        ],
        consumables=[
            {"item": "White Potion", "min_stock": 10, "buy_qty": 30, "max_price": 500},
            {"item": "Arrow", "min_stock": 200, "buy_qty": 1000, "max_price": 2},
            {"item": "Fly Wing", "min_stock": 5, "buy_qty": 20, "max_price": 500},
        ],
    ),
    "acolyte": BuildPlan(
        job_class="Acolyte",
        primary_stat="int",
        secondary_stat="dex",
        stat_priority=["int", "dex", "vit", "agi", "luk", "str"],
        skill_learn_order=[
            {"name": "AL_HEAL", "max_level": 10, "reason": "Heal — primary healing skill"},
            {"name": "AL_INCAGI", "max_level": 10, "reason": "Increase AGI — party buff"},
            {"name": "AL_BLESSING", "max_level": 10, "reason": "Blessing — party buff"},
            {"name": "AL_TELEPORT", "max_level": 1, "reason": "Teleport — emergency escape"},
        ],
        equipment_goals=[
            {"item": "White Potion", "qty": 20, "reason": "Backup healing"},
            {"item": "Blue Potion", "qty": 30, "reason": "SP recovery for healing"},
            {"item": "Fly Wing", "qty": 10, "reason": "Emergency escape"},
        ],
        farming_maps=[
            {"map": "prt_fild08", "min_level": 1, "max_level": 25, "reason": "Easy mobs"},
            {"map": "moc_fild17", "min_level": 20, "max_level": 40, "reason": "Undead — heal deals damage"},
        ],
        consumables=[
            {"item": "White Potion", "min_stock": 10, "buy_qty": 20, "max_price": 500},
            {"item": "Blue Potion", "min_stock": 10, "buy_qty": 30, "max_price": 2000},
            {"item": "Fly Wing", "min_stock": 5, "buy_qty": 10, "max_price": 500},
        ],
    ),
    "merchant": BuildPlan(
        job_class="Merchant",
        primary_stat="str",
        secondary_stat="dex",
        stat_priority=["str", "dex", "vit", "agi", "luk", "int"],
        skill_learn_order=[
            {"name": "MC_PUSHCART", "max_level": 10, "reason": "Pushcart — carry more items"},
            {"name": "MC_DISCOUNT", "max_level": 10, "reason": "Discount — cheaper purchases"},
            {"name": "MC_OVERCHARGE", "max_level": 10, "reason": "Overcharge — sell for more"},
            {"name": "MC_MAMMONITE", "max_level": 10, "reason": "Mammonite — zeny attack"},
        ],
        equipment_goals=[
            {"item": "White Potion", "qty": 50, "reason": "Healing potion"},
            {"item": "Red Potion", "qty": 30, "reason": "Backup healing"},
            {"item": "Fly Wing", "qty": 20, "reason": "Emergency escape"},
        ],
        farming_maps=[
            {"map": "prt_fild08", "min_level": 1, "max_level": 25, "reason": "Easy mobs"},
            {"map": "pay_fild04", "min_level": 20, "max_level": 40, "reason": "Good exp"},
        ],
        consumables=[
            {"item": "White Potion", "min_stock": 20, "buy_qty": 50, "max_price": 500},
            {"item": "Fly Wing", "min_stock": 5, "buy_qty": 20, "max_price": 500},
        ],
    ),
    "thief": BuildPlan(
        job_class="Thief",
        primary_stat="agi",
        secondary_stat="dex",
        stat_priority=["agi", "dex", "vit", "luk", "str", "int"],
        skill_learn_order=[
            {"name": "TF_DOUBLE", "max_level": 10, "reason": "Double Attack — passive damage boost"},
            {"name": "TF_HIDE", "max_level": 10, "reason": "Hide — stealth and escape"},
            {"name": "TF_STEAL", "max_level": 10, "reason": "Steal — steal from monsters"},
            {"name": "TF_POISON", "max_level": 10, "reason": "Envenom — poison attack"},
        ],
        equipment_goals=[
            {"item": "White Potion", "qty": 30, "reason": "Healing potion"},
            {"item": "Red Potion", "qty": 20, "reason": "Backup healing"},
            {"item": "Fly Wing", "qty": 20, "reason": "Emergency escape"},
        ],
        farming_maps=[
            {"map": "prt_fild08", "min_level": 1, "max_level": 25, "reason": "Easy mobs"},
            {"map": "pay_dun00", "min_level": 20, "max_level": 40, "reason": "Dungeon — good for thieves"},
        ],
        consumables=[
            {"item": "White Potion", "min_stock": 10, "buy_qty": 30, "max_price": 500},
            {"item": "Fly Wing", "min_stock": 5, "buy_qty": 20, "max_price": 500},
        ],
    ),
}


class ConsciousDecisionEngine:
    """Makes all high-level decisions for bot progression."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._bot_state: dict[str, dict[str, Any]] = {}
        self._decisions_made: dict[str, list[str]] = defaultdict(list)
        self._last_evaluation: float = 0.0
        self._evaluation_interval: float = 10.0  # Evaluate every 10 seconds
        self._knowledge: dict[str, Any] = {}
        self._load_knowledge()

    def _load_knowledge(self) -> None:
        """Load knowledge database."""
        try:
            with open(_KNOWLEDGE_PATH) as f:
                self._knowledge = json.load(f)
            logger.info("knowledge_loaded: %d keys", len(self._knowledge))
        except Exception as e:
            logger.warning("Failed to load knowledge: %s", e)

    def update_from_snapshot(self, bot_id: str, snapshot: Any) -> None:
        """Update bot state from snapshot."""
        with self._lock:
            state = self._bot_state.setdefault(bot_id, {})
            state["last_seen"] = time.time()

            if hasattr(snapshot, "vitals"):
                v = snapshot.vitals
                state["hp"] = getattr(v, "hp", 0)
                state["max_hp"] = getattr(v, "hp_max", 1)
                state["hp_pct"] = getattr(v, "hp_ratio", 1.0)
                state["sp"] = getattr(v, "sp", 0)
                state["max_sp"] = getattr(v, "sp_max", 1)
                state["sp_pct"] = getattr(v, "sp_ratio", 1.0)
                state["base_level"] = getattr(v, "base_level", 1)
                state["job_level"] = getattr(v, "job_level", 1)
                state["job_name"] = str(getattr(v, "job_name", "novice")).lower()
                state["zeny"] = getattr(v, "zeny", 0)
                state["weight_ratio"] = getattr(v, "weight_ratio", 0.0)

            if hasattr(snapshot, "position"):
                pos = snapshot.position
                state["map"] = str(getattr(pos, "map", ""))

            if hasattr(snapshot, "inventory_items"):
                state["inventory"] = {
                    str(getattr(item, "name", "")): int(getattr(item, "amount", 0))
                    for item in (snapshot.inventory_items or [])
                }

            if hasattr(snapshot, "skills"):
                state["skills"] = [
                    str(getattr(s, "name", ""))
                    for s in (snapshot.skills or [])
                ]

            if hasattr(snapshot, "stats"):
                st = snapshot.stats
                state["stats"] = {
                    "str": getattr(st, "str", 0),
                    "agi": getattr(st, "agi", 0),
                    "vit": getattr(st, "vit", 0),
                    "int": getattr(st, "int", 0),
                    "dex": getattr(st, "dex", 0),
                    "luk": getattr(st, "luk", 0),
                }

    def _get_build(self, job_name: str) -> BuildPlan:
        """Get the build plan for a job class."""
        job = job_name.lower()
        if job in BUILDS:
            return BUILDS[job]
        # Try to find by partial match
        for key, build in BUILDS.items():
            if key in job or job in key:
                return build
        return BUILDS["novice"]

    def evaluate(self, bot_id: str) -> list[Decision]:
        """Evaluate all decisions for a bot."""
        with self._lock:
            now = time.time()
            if now - self._last_evaluation < self._evaluation_interval:
                return []
            self._last_evaluation = now

            state = self._bot_state.get(bot_id, {})
            if not state:
                return []

            decisions: list[Decision] = []
            build = self._get_build(state.get("job_name", "novice"))
            made = self._decisions_made.setdefault(bot_id, [])
            inventory = state.get("inventory", {})
            skills = state.get("skills", [])
            stats = state.get("stats", {})
            base_level = state.get("base_level", 1)
            job_level = state.get("job_level", 1)
            zeny = state.get("zeny", 0)
            hp_pct = state.get("hp_pct", 1.0)
            sp_pct = state.get("sp_pct", 1.0)

            # ── 1. Skill Learning Decisions ──
            for skill in build.skill_learn_order:
                name = skill["name"]
                if name not in skills:
                    # Check if prerequisites are met
                    decisions.append(Decision(
                        domain="skills",
                        action="learn_skill",
                        target=name,
                        priority=1,
                        reason=skill["reason"],
                        params={"max_level": skill["max_level"]},
                    ))
                    made.append(f"learn_{name}")
                    break  # Learn one skill at a time

            # ── 2. Stat Distribution Decisions ──
            if stats:
                for stat in build.stat_priority:
                    current = stats.get(stat, 0)
                    # Check if we have stat points to distribute
                    # (base_level * 2 + job_level - current_total = available)
                    total_stats = sum(stats.values())
                    expected = base_level * 2 + job_level
                    if total_stats < expected:
                        # We have unassigned stat points
                        target_value = base_level * 2 // len(build.stat_priority)
                        if current < target_value:
                            decisions.append(Decision(
                                domain="stats",
                                action="add_stat",
                                target=stat,
                                priority=2,
                                reason=f"Build priority: {stat} (current: {current}, target: {target_value})",
                                params={"points": 1},
                            ))
                            made.append(f"stat_{stat}")
                            break

            # ── 3. Restock Decisions ──
            for consumable in build.consumables:
                item = consumable["item"]
                min_stock = consumable["min_stock"]
                current_stock = inventory.get(item, 0)
                if current_stock < min_stock:
                    decisions.append(Decision(
                        domain="restock",
                        action="buy_item",
                        target=item,
                        priority=2 if "Potion" in item else 3,
                        reason=f"Low on {item} ({current_stock}/{min_stock})",
                        params={
                            "qty": consumable["buy_qty"],
                            "max_price": consumable["max_price"],
                        },
                    ))
                    made.append(f"restock_{item}")

            # ── 4. Emergency Restock (HP < 50% and no potions) ──
            if hp_pct < 0.50:
                has_heal = any(
                    "Potion" in k or "Apple" in k or "Herb" in k
                    for k in inventory.keys()
                )
                if not has_heal:
                    decisions.append(Decision(
                        domain="restock",
                        action="emergency_restock",
                        target="White Potion",
                        priority=1,
                        reason=f"HP at {hp_pct:.0%} with no healing items — emergency restock needed",
                        params={"qty": 30, "max_price": 500},
                    ))

            # ── 5. Map Selection Decision ──
            current_map = state.get("map", "")
            for farm_map in build.farming_maps:
                if farm_map["min_level"] <= base_level <= farm_map["max_level"]:
                    if current_map != farm_map["map"]:
                        decisions.append(Decision(
                            domain="map",
                            action="move_map",
                            target=farm_map["map"],
                            priority=3,
                            reason=farm_map["reason"],
                        ))
                    break

            # ── 6. Party Coordination Decision ──
            if hasattr(state, "nearby_party") and state.get("nearby_party", 0) > 0:
                if hp_pct < 0.60 and "AL_HEAL" in skills:
                    decisions.append(Decision(
                        domain="party",
                        action="request_heal",
                        target="party",
                        priority=2,
                        reason=f"HP at {hp_pct:.0%}, party members nearby — request heal",
                    ))

            # Sort by priority
            decisions.sort(key=lambda d: d.priority)
            return decisions

    def get_summary(self, bot_id: str) -> str:
        """Get a human-readable summary of decisions."""
        with self._lock:
            state = self._bot_state.get(bot_id, {})
            build = self._get_build(state.get("job_name", "novice"))
            decisions = self.evaluate(bot_id)

            lines = [f"── Conscious Decisions for {bot_id} ──"]
            lines.append(f"  Job: {state.get('job_name', '?')}  Level: {state.get('base_level', 1)}/{state.get('job_level', 1)}")
            lines.append(f"  Build: {build.primary_stat}/{build.secondary_stat}")
            lines.append(f"  HP: {state.get('hp_pct', 1.0):.0%}  SP: {state.get('sp_pct', 1.0):.0%}  Zeny: {state.get('zeny', 0)}z")
            lines.append(f"  Map: {state.get('map', '?')}")
            lines.append("")

            if decisions:
                lines.append("  Decisions:")
                for d in decisions:
                    lines.append(f"    [{d.priority}] {d.domain}.{d.action}({d.target}): {d.reason}")
            else:
                lines.append("  No decisions needed.")

            return "\n".join(lines)


from collections import defaultdict

# Global singleton
_engine: ConsciousDecisionEngine | None = None
_engine_lock = RLock()


def get_conscious_engine() -> ConsciousDecisionEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = ConsciousDecisionEngine()
        return _engine
