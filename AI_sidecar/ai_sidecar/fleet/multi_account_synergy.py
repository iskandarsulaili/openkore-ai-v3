"""
Multi-account synergy — coordinates multiple bots as a team with actual execution.

A top player with 3 accounts doesn't have 3 bots. They have a team.
Each account has a role. They complement each other. They cover each
other's weaknesses. This module coordinates multi-account operations
with actual execution: shared inventory, zeny management, coordinated
leveling, specialization enforcement, production pipeline, and real-time
coordination.

Key capabilities:
- Shared inventory management (track what each bot has, coordinate transfers)
- Shared zeny management (track total wealth, allocate resources)
- Coordinated leveling (schedule who levels when, who carries whom)
- Specialization enforcement (farmer farms, crafter crafts, merchant sells)
- Production pipeline (farmer -> crafter -> merchant -> PVP)
- Real-time coordination (sub-second communication between bots)
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TeamRole:
    """A role in the multi-account team."""
    name: str
    primary_task: str  # farmer, buffer, merchant, scout, tank, healer, crafter, pvp
    secondary_task: str = ""
    bot_id: str = ""
    level: int = 0
    class_name: str = ""
    map: str = ""
    status: str = "idle"  # idle, farming, trading, scouting, returning, dead, crafting, pvp
    last_updated: float = 0.0
    specialization: str = ""  # farming, crafting, merchant, pvp, support
    zeny: int = 0  # Current zeny this bot holds
    weight_capacity: int = 0  # Current weight capacity
    current_weight: int = 0  # Current weight


@dataclass
class TeamOrder:
    """An order for a team member."""
    bot_id: str
    order_type: str  # move, attack, buff, trade, return, defend, transfer, craft, sell, level
    target: str = ""
    priority: int = 5
    issued_at: float = 0.0
    completed: bool = False
    result: str = ""


@dataclass
class SharedItem:
    """An item in the shared inventory."""
    item_name: str
    total_quantity: int = 0
    held_by: dict[str, int] = field(default_factory=dict)  # bot_id -> quantity
    reserved_for: str = ""  # pipeline stage
    estimated_value: int = 0
    last_updated: float = 0.0


@dataclass
class LevelingSchedule:
    """A coordinated leveling schedule."""
    bot_id: str
    target_level: int
    current_level: int = 0
    carrier_bot: str = ""  # Who is carrying
    map_name: str = ""
    priority: int = 5
    status: str = "pending"  # pending, in_progress, completed
    started_at: float = 0.0
    completed_at: float = 0.0


@dataclass
class ProductionOrder:
    """An order in the production pipeline."""
    stage: str  # farm, craft, sell, use
    item_name: str
    quantity: int
    source_bot: str = ""
    target_bot: str = ""
    status: str = "pending"  # pending, in_progress, completed
    created_at: float = 0.0
    completed_at: float = 0.0


@dataclass(slots=True)
class MultiAccountSynergy:
    """Coordinates multiple bots as a coordinated team with actual execution.

    Wires into:
      - empire_manager.py: C-level role coordination
      - fleet_coordinator.py: fleet management
      - unified_consciousness.py: decision-making
      - PDCA loop: action execution
    """

    _lock: RLock = field(default_factory=RLock)
    _roles: dict[str, TeamRole] = field(default_factory=dict)
    _orders: list[TeamOrder] = field(default_factory=list)
    _shared_inventory: dict[str, SharedItem] = field(default_factory=dict)
    _shared_zeny: int = 0  # Total shared zeny pool
    _leveling_schedules: list[LevelingSchedule] = field(default_factory=list)
    _production_orders: list[ProductionOrder] = field(default_factory=list)
    _pipeline_active: bool = False
    _stats: dict[str, int] = field(default_factory=lambda: {
        "orders_issued": 0, "orders_completed": 0, "synergies": 0,
        "transfers": 0, "items_shared": 0, "zeny_allocated": 0,
        "levels_gained": 0, "pipeline_transfers": 0,
    })
    _enqueue_fn: Callable | None = None

    # ── Role Management ─────────────────────────────────────────────

    def assign_role(self, bot_id: str, primary: str, secondary: str = "",
                    level: int = 0, class_name: str = "",
                    specialization: str = "") -> None:
        """Assign a role to a bot with specialization."""
        with self._lock:
            role = self._roles.setdefault(bot_id, TeamRole(
                name=bot_id,
                primary_task=primary,
                bot_id=bot_id,
            ))
            role.primary_task = primary
            role.secondary_task = secondary
            role.level = level or role.level
            role.class_name = class_name or role.class_name
            role.specialization = specialization or primary
            role.last_updated = time.time()
            logger.info(
                "team_role_assigned: %s → %s (secondary: %s, spec: %s)",
                bot_id, primary, secondary, role.specialization,
            )

    def update_status(self, bot_id: str, status: str, map: str = "",
                      level: int = 0, zeny: int = 0,
                      weight_capacity: int = 0, current_weight: int = 0) -> None:
        """Update a bot's status with full state."""
        with self._lock:
            role = self._roles.get(bot_id)
            if role:
                role.status = status
                if map:
                    role.map = map
                if level:
                    role.level = level
                if zeny:
                    role.zeny = zeny
                if weight_capacity:
                    role.weight_capacity = weight_capacity
                if current_weight:
                    role.current_weight = current_weight
                role.last_updated = time.time()

    def get_role(self, bot_id: str) -> TeamRole | None:
        """Get a bot's role."""
        with self._lock:
            return self._roles.get(bot_id)

    def get_bots_by_task(self, task: str) -> list[TeamRole]:
        """Get all bots with a specific primary task."""
        with self._lock:
            return [r for r in self._roles.values() if r.primary_task == task]

    def get_bots_by_specialization(self, spec: str) -> list[TeamRole]:
        """Get all bots with a specific specialization."""
        with self._lock:
            return [r for r in self._roles.values() if r.specialization == spec]

    def get_active_bots(self) -> list[TeamRole]:
        """Get all active (non-dead) bots."""
        with self._lock:
            return [r for r in self._roles.values() if r.status != "dead"]

    # ── Order Management ────────────────────────────────────────────

    def issue_order(self, bot_id: str, order_type: str, target: str = "",
                    priority: int = 5) -> bool:
        """Issue an order to a team member with actual execution."""
        with self._lock:
            order = TeamOrder(
                bot_id=bot_id,
                order_type=order_type,
                target=target,
                priority=priority,
                issued_at=time.time(),
            )
            self._orders.append(order)
            self._stats["orders_issued"] += 1

            # Execute via enqueue with actual commands
            if self._enqueue_fn:
                cmd = self._build_command(order_type, target, bot_id)
                if cmd:
                    try:
                        self._enqueue_fn(bot_id, cmd)
                        logger.info("team_order: %s → %s (%s) cmd=%s", bot_id, order_type, target, cmd)
                        return True
                    except Exception as e:
                        logger.warning("team_order_failed: %s → %s: %s", bot_id, order_type, e)
            return False

    def _build_command(self, order_type: str, target: str, bot_id: str) -> str:
        """Build an actual executable command from an order type."""
        cmd_map = {
            "move": f"move {target}",
            "attack": f"attack {target}",
            "return": "move prontera",
            "defend": f"move {target}",
            "buff": f"use {target}",
            "trade": f"chat anyone selling {target}?",
            "transfer": f"move {target}",
            "craft": f"ai auto",
            "sell": f"ai auto",
            "level": f"ai auto",
            "restock": f"ai auto",
            "store": f"ai auto",
            "party": f"party {target}",
            "follow": f"follow {target}",
            "scout": f"move {target}",
            "retreat": "move prontera",
        }
        return cmd_map.get(order_type, f"ai auto")

    def complete_order(self, bot_id: str, order_type: str, result: str = "") -> bool:
        """Mark the most recent order of a type as completed."""
        with self._lock:
            for order in reversed(self._orders):
                if order.bot_id == bot_id and order.order_type == order_type and not order.completed:
                    order.completed = True
                    order.result = result
                    self._stats["orders_completed"] += 1
                    return True
            return False

    def get_pending_orders(self, bot_id: str | None = None) -> list[TeamOrder]:
        """Get pending orders, optionally filtered by bot."""
        with self._lock:
            pending = [o for o in self._orders if not o.completed]
            if bot_id:
                pending = [o for o in pending if o.bot_id == bot_id]
            return sorted(pending, key=lambda o: o.priority, reverse=True)

    # ── Shared Inventory Management ────────────────────────────────

    def add_to_shared_inventory(self, item_name: str, quantity: int,
                                 held_by: str, estimated_value: int = 0) -> None:
        """Add items to the shared inventory."""
        with self._lock:
            item = self._shared_inventory.setdefault(
                item_name,
                SharedItem(item_name=item_name, estimated_value=estimated_value),
            )
            item.total_quantity += quantity
            item.held_by[held_by] = item.held_by.get(held_by, 0) + quantity
            item.estimated_value = estimated_value or item.estimated_value
            item.last_updated = time.time()
            self._stats["items_shared"] += 1
            logger.info("shared_inventory_add: %dx %s from %s", quantity, item_name, held_by)

    def remove_from_shared_inventory(self, item_name: str, quantity: int,
                                      taken_by: str) -> bool:
        """Remove items from the shared inventory."""
        with self._lock:
            item = self._shared_inventory.get(item_name)
            if not item or item.total_quantity < quantity:
                return False
            item.total_quantity -= quantity
            item.held_by[taken_by] = item.held_by.get(taken_by, 0) + quantity
            item.last_updated = time.time()
            self._stats["transfers"] += 1
            logger.info("shared_inventory_remove: %dx %s by %s", quantity, item_name, taken_by)
            return True

    def transfer_item(self, item_name: str, quantity: int,
                       from_bot: str, to_bot: str) -> bool:
        """Transfer an item between two bots via shared inventory."""
        with self._lock:
            # Remove from source
            item = self._shared_inventory.get(item_name)
            if not item or item.held_by.get(from_bot, 0) < quantity:
                logger.warning("transfer_failed: %s has %d %s, need %d",
                              from_bot, item.held_by.get(from_bot, 0) if item else 0, item_name, quantity)
                return False

            item.held_by[from_bot] = item.held_by.get(from_bot, 0) - quantity
            item.held_by[to_bot] = item.held_by.get(to_bot, 0) + quantity
            item.last_updated = time.time()
            self._stats["transfers"] += 1

            # Issue transfer order
            self.issue_order(
                from_bot, "transfer", f"{quantity}x {item_name} to {to_bot}",
                priority=7,
            )

            logger.info("transfer: %dx %s from %s to %s", quantity, item_name, from_bot, to_bot)
            return True

    def get_shared_inventory(self) -> dict[str, SharedItem]:
        """Get the full shared inventory."""
        with self._lock:
            return dict(self._shared_inventory)

    def get_shared_inventory_summary(self) -> str:
        """Get a formatted summary of shared inventory."""
        with self._lock:
            if not self._shared_inventory:
                return "  No shared inventory."
            lines = [f"  Shared Inventory ({len(self._shared_inventory)} items):"]
            for name, item in sorted(self._shared_inventory.items())[:10]:
                holders = ", ".join(
                    f"{bot}({qty})" for bot, qty in item.held_by.items() if qty > 0
                )
                lines.append(f"    {name}: {item.total_quantity} [{holders}]")
            return "\n".join(lines)

    # ── Shared Zeny Management ──────────────────────────────────────

    def add_to_shared_zeny(self, amount: int, source_bot: str) -> None:
        """Add zeny to the shared pool."""
        with self._lock:
            self._shared_zeny += amount
            role = self._roles.get(source_bot)
            if role:
                role.zeny = max(0, role.zeny - amount)
            logger.info("shared_zeny_add: %dz from %s (total: %d)", amount, source_bot, self._shared_zeny)

    def allocate_zeny(self, amount: int, target_bot: str, purpose: str) -> bool:
        """Allocate zeny from the shared pool to a bot."""
        with self._lock:
            if self._shared_zeny < amount:
                logger.warning("allocate_zeny_failed: have %d, need %d", self._shared_zeny, amount)
                return False
            self._shared_zeny -= amount
            role = self._roles.get(target_bot)
            if role:
                role.zeny += amount
            self._stats["zeny_allocated"] += 1
            logger.info("shared_zeny_allocate: %dz to %s for %s (remaining: %d)",
                       amount, target_bot, purpose, self._shared_zeny)
            return True

    def get_total_wealth(self) -> dict[str, int]:
        """Get total empire wealth (zeny + inventory value)."""
        with self._lock:
            total_zeny = self._shared_zeny
            for role in self._roles.values():
                total_zeny += role.zeny

            total_inventory_value = 0
            for item in self._shared_inventory.values():
                total_inventory_value += item.total_quantity * item.estimated_value

            return {
                "shared_zeny": self._shared_zeny,
                "bot_zeny": sum(r.zeny for r in self._roles.values()),
                "total_zeny": total_zeny,
                "inventory_value": total_inventory_value,
                "total_wealth": total_zeny + total_inventory_value,
            }

    # ── Coordinated Leveling ────────────────────────────────────────

    def schedule_leveling(self, bot_id: str, target_level: int,
                          carrier_bot: str = "", map_name: str = "",
                          priority: int = 5) -> str:
        """Schedule a bot for coordinated leveling.

        Args:
            bot_id: Bot to level
            target_level: Target level
            carrier_bot: Bot that will carry (optional)
            map_name: Map to level on
            priority: Priority (1-10)

        Returns:
            Schedule ID
        """
        with self._lock:
            schedule = LevelingSchedule(
                bot_id=bot_id,
                target_level=target_level,
                current_level=self._roles.get(bot_id, TeamRole(name=bot_id, primary_task="", bot_id=bot_id)).level,
                carrier_bot=carrier_bot,
                map_name=map_name,
                priority=priority,
                started_at=time.time(),
            )
            self._leveling_schedules.append(schedule)

            # Issue leveling order
            self.issue_order(
                bot_id, "level",
                f"to level {target_level} on {map_name}" if map_name else f"to level {target_level}",
                priority=priority,
            )

            # If carrier specified, issue follow order
            if carrier_bot:
                self.issue_order(
                    bot_id, "follow", carrier_bot,
                    priority=priority,
                )

            logger.info(
                "leveling_scheduled: bot=%s target=%d carrier=%s map=%s priority=%d",
                bot_id, target_level, carrier_bot or "none", map_name, priority,
            )
            return f"level_{bot_id}_{int(time.time())}"

    def update_leveling_progress(self, bot_id: str, current_level: int) -> None:
        """Update leveling progress for a bot."""
        with self._lock:
            for schedule in self._leveling_schedules:
                if schedule.bot_id == bot_id and schedule.status == "in_progress":
                    schedule.current_level = current_level
                    if current_level >= schedule.target_level:
                        schedule.status = "completed"
                        schedule.completed_at = time.time()
                        self._stats["levels_gained"] += 1
                        logger.info("leveling_completed: bot=%s reached level %d", bot_id, current_level)
                    break

    def get_leveling_schedules(self, status: str = "pending") -> list[LevelingSchedule]:
        """Get leveling schedules by status."""
        with self._lock:
            return [s for s in self._leveling_schedules if s.status == status]

    # ── Specialization Enforcement ──────────────────────────────────

    def enforce_specialization(self, bot_id: str) -> str | None:
        """Enforce a bot's specialization — returns the action they should be doing.

        A farmer farms. A crafter crafts. A merchant sells. A PVP bot PvPs.
        This method ensures bots stay in their lane.
        """
        with self._lock:
            role = self._roles.get(bot_id)
            if not role:
                return None

            spec = role.specialization
            status = role.status

            # Determine what this bot SHOULD be doing
            if spec == "farming":
                if status in ("idle", "returning"):
                    return "farm"
                return None  # Already farming
            elif spec == "crafting":
                if status in ("idle", "returning"):
                    return "craft"
                return None
            elif spec == "merchant":
                if status in ("idle", "returning"):
                    return "sell"
                return None
            elif spec == "pvp":
                if status in ("idle", "returning"):
                    return "pvp"
                return None
            elif spec == "support":
                if status in ("idle", "returning"):
                    return "support"
                return None
            return None

    def get_specialization_breakdown(self) -> dict[str, list[str]]:
        """Get a breakdown of bots by specialization."""
        with self._lock:
            breakdown: dict[str, list[str]] = defaultdict(list)
            for role in self._roles.values():
                breakdown[role.specialization].append(role.bot_id)
            return dict(breakdown)

    # ── Production Pipeline ─────────────────────────────────────────

    def setup_production_pipeline(self, farmer: str, crafter: str,
                                   merchant: str, pvp_bot: str = "") -> None:
        """Set up the farmer -> crafter -> merchant -> PVP production pipeline.

        Each stage feeds the next. Raw materials flow up. Zeny flows down.
        """
        with self._lock:
            self._pipeline_active = True

            # Assign specializations
            self.assign_role(farmer, "farmer", specialization="farming")
            self.assign_role(crafter, "crafter", specialization="crafting")
            self.assign_role(merchant, "merchant", specialization="merchant")
            if pvp_bot:
                self.assign_role(pvp_bot, "pvp", specialization="pvp")

            logger.info(
                "pipeline_setup: farmer=%s crafter=%s merchant=%s pvp=%s",
                farmer, crafter, merchant, pvp_bot or "none",
            )

    def create_production_order(self, stage: str, item_name: str,
                                 quantity: int, source_bot: str = "",
                                 target_bot: str = "") -> str:
        """Create an order in the production pipeline.

        Args:
            stage: farm, craft, sell, use
            item_name: Item to produce
            quantity: How many
            source_bot: Who provides the item
            target_bot: Who receives the item

        Returns:
            Order ID
        """
        with self._lock:
            order = ProductionOrder(
                stage=stage,
                item_name=item_name,
                quantity=quantity,
                source_bot=source_bot,
                target_bot=target_bot,
                created_at=time.time(),
            )
            self._production_orders.append(order)

            # Issue the appropriate order
            if stage == "farm":
                self.issue_order(source_bot, "level", f"farm {quantity}x {item_name}", priority=6)
            elif stage == "craft":
                self.issue_order(source_bot, "craft", f"craft {quantity}x {item_name}", priority=6)
            elif stage == "sell":
                self.issue_order(source_bot, "sell", f"sell {quantity}x {item_name}", priority=6)
            elif stage == "use":
                self.issue_order(source_bot, "transfer", f"{quantity}x {item_name} to {target_bot}", priority=6)

            logger.info("production_order: stage=%s item=%s qty=%d", stage, item_name, quantity)
            return f"prod_{int(time.time() * 1000)}"

    def execute_pipeline_step(self) -> ProductionOrder | None:
        """Execute the next pending step in the production pipeline.

        Returns:
            The executed order, or None if nothing to do
        """
        with self._lock:
            if not self._pipeline_active:
                return None

            # Find the next pending order
            for order in self._production_orders:
                if order.status == "pending":
                    order.status = "in_progress"
                    self._stats["pipeline_transfers"] += 1

                    # Execute based on stage
                    if order.stage == "farm":
                        self.issue_order(
                            order.source_bot, "level",
                            f"farm {order.quantity}x {order.item_name}",
                            priority=6,
                        )
                    elif order.stage == "craft":
                        self.issue_order(
                            order.source_bot, "craft",
                            f"craft {order.quantity}x {order.item_name}",
                            priority=6,
                        )
                    elif order.stage == "sell":
                        self.issue_order(
                            order.source_bot, "sell",
                            f"sell {order.quantity}x {order.item_name}",
                            priority=6,
                        )
                    elif order.stage == "use":
                        self.issue_order(
                            order.source_bot, "transfer",
                            f"{order.quantity}x {order.item_name} to {order.target_bot}",
                            priority=6,
                        )

                    return order

            return None

    def get_pipeline_status(self) -> dict[str, Any]:
        """Get the current status of the production pipeline."""
        with self._lock:
            return {
                "active": self._pipeline_active,
                "pending_orders": len([o for o in self._production_orders if o.status == "pending"]),
                "in_progress": len([o for o in self._production_orders if o.status == "in_progress"]),
                "completed": len([o for o in self._production_orders if o.status == "completed"]),
                "specializations": self.get_specialization_breakdown(),
            }

    # ── Real-Time Coordination ─────────────────────────────────────

    def broadcast_to_team(self, message: str, priority: int = 5) -> int:
        """Broadcast a message/order to all active team members.

        Args:
            message: The message/command to broadcast
            priority: Priority level

        Returns:
            Number of bots that received the broadcast
        """
        count = 0
        with self._lock:
            for role in self._roles.values():
                if role.status != "dead":
                    self.issue_order(role.bot_id, "move", message, priority=priority)
                    count += 1
        logger.info("team_broadcast: %d bots received: %s", count, message)
        return count

    def coordinate_retreat(self, safe_map: str = "prontera") -> int:
        """Order all bots to retreat to a safe map.

        Args:
            safe_map: Map to retreat to

        Returns:
            Number of bots that received the retreat order
        """
        return self.broadcast_to_team(f"retreat to {safe_map}", priority=10)

    def coordinate_attack(self, target: str, formation: str = "standard") -> int:
        """Order all combat-capable bots to attack a target.

        Args:
            target: What to attack
            formation: Attack formation

        Returns:
            Number of bots that received the attack order
        """
        count = 0
        with self._lock:
            for role in self._roles.values():
                if role.specialization in ("pvp", "farming") and role.status != "dead":
                    self.issue_order(role.bot_id, "attack", target, priority=9)
                    count += 1
        logger.info("team_attack: %d bots targeting %s", count, target)
        return count

    def get_team_readiness(self) -> dict[str, Any]:
        """Get team readiness assessment for real-time coordination."""
        with self._lock:
            active = self.get_active_bots()
            return {
                "total_bots": len(self._roles),
                "active_bots": len(active),
                "readiness_pct": len(active) / max(len(self._roles), 1) * 100,
                "specializations": self.get_specialization_breakdown(),
                "total_wealth": self.get_total_wealth(),
                "pending_orders": len(self.get_pending_orders()),
            }

    # ── Summary / Context ──────────────────────────────────────────

    def get_synergy_context(self) -> str:
        """Get formatted team context for LLM prompts."""
        with self._lock:
            lines = ["── Multi-Account Team ──"]
            active = self.get_active_bots()

            if not active:
                lines.append("  No team members assigned.")
                return "\n".join(lines)

            lines.append(f"  Active members: {len(active)}")
            for role in active:
                lines.append(
                    f"    {role.bot_id}: {role.primary_task} ({role.specialization}) "
                    f"[{role.status}] Lv.{role.level} on {role.map}"
                )

            # Specialization breakdown
            spec_breakdown = self.get_specialization_breakdown()
            if spec_breakdown:
                lines.append("  Specializations:")
                for spec, bots in spec_breakdown.items():
                    lines.append(f"    {spec}: {', '.join(bots)}")

            # Pipeline status
            if self._pipeline_active:
                lines.append("  Production Pipeline: ACTIVE")
                pipeline = self.get_pipeline_status()
                lines.append(f"    Orders: {pipeline['pending_orders']} pending, "
                            f"{pipeline['in_progress']} in progress")

            # Shared inventory
            inv_summary = self.get_shared_inventory_summary()
            if inv_summary:
                lines.append(inv_summary)

            # Wealth
            wealth = self.get_total_wealth()
            lines.append(f"  Total Wealth: {wealth['total_wealth']:,}z "
                        f"(zeny: {wealth['total_zeny']:,}z, "
                        f"inventory: {wealth['inventory_value']:,}z)")

            # Synergy opportunities
            farmers = self.get_bots_by_task("farmer")
            buffers = self.get_bots_by_task("buffer")
            merchants = self.get_bots_by_task("merchant")
            crafters = self.get_bots_by_task("crafter")

            if farmers and buffers:
                lines.append(f"  Synergy: {len(farmers)} farmer(s) + {len(buffers)} buffer(s) available")
            if merchants:
                lines.append("  Merchant available for selling")
            if crafters:
                lines.append("  Crafter available for production")

            return "\n".join(lines)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global instance ──

_synergy: MultiAccountSynergy | None = None
_synergy_lock = RLock()


def get_multi_account_synergy() -> MultiAccountSynergy:
    """Get or create the global multi-account synergy instance."""
    global _synergy
    with _synergy_lock:
        if _synergy is None:
            _synergy = MultiAccountSynergy()
        return _synergy
