"""
Empire Manager — complete multi-account empire management with C-level roles.

A top player with 5+ accounts doesn't have 5 bots. They have an empire.
Each account has a C-level role. They execute a production pipeline.
They coordinate like a corporation.

Roles:
  CEO: Builds relationships, negotiates deals, leads WOE, sets strategic direction
  CFO: Manages economy, controls prices, runs market, tracks total wealth
  COO: Oversees farming, optimizes routes, allocates bots to maps
  CTO: Produces potions, elemental weapons, rare items, manages crafting pipeline
  HR: Recruits party members, builds guild alliances, manages reputation
  R&D: Maps dungeons, discovers MVP spawns, tracks patch changes, researches builds
  Security: Protects territory, eliminates competition, gathers intel, detects threats
  Logistics: Moves items between characters, manages shared inventory, coordinates transfers

Production Pipeline:
  Farmer -> Crafter -> Merchant -> PVP Character
  Each stage feeds the next. Raw materials flow up. Zeny flows down.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class EmpireRole:
    """A C-level role in the empire."""
    title: str  # CEO, CFO, COO, CTO, HR, R&D, Security, Logistics
    bot_id: str
    class_name: str = ""
    level: int = 0
    status: str = "active"  # active, idle, busy, offline
    last_action: float = 0.0
    performance_score: float = 0.0  # 0.0-1.0
    specialization: str = ""  # e.g., "farming", "crafting", "merchant", "pvp"


@dataclass
class EmpireDirective:
    """A strategic directive issued by a C-level role."""
    directive_id: str
    issued_by: str  # role title
    target_bot: str
    action: str
    reason: str
    priority: int  # 1-10
    issued_at: float
    deadline: float = 0.0
    completed: bool = False
    result: str = ""


@dataclass
class ProductionPipeline:
    """The farmer -> crafter -> merchant -> PVP production pipeline."""
    farmer_bot_id: str = ""
    crafter_bot_id: str = ""
    merchant_bot_id: str = ""
    pvp_bot_id: str = ""
    raw_materials: dict[str, int] = field(default_factory=dict)  # item_name -> quantity
    crafted_goods: dict[str, int] = field(default_factory=dict)
    inventory_for_sale: dict[str, int] = field(default_factory=dict)
    zeny_reserve: int = 0
    pipeline_active: bool = False
    last_transfer: float = 0.0


@dataclass
class TerritoryClaim:
    """A claimed territory (map) for exclusive farming."""
    map_name: str
    claimed_by: str  # bot_id
    claimed_at: float
    priority: int  # 1-10, how important this territory is
    competition_level: int = 0  # 0-10
    last_defended: float = 0.0
    zeny_per_hour: int = 0


@dataclass
class Alliance:
    """An alliance with another player or guild."""
    entity_name: str
    entity_type: str  # player, guild
    relationship: str  # ally, trade_partner, mutual_defense, non_aggression
    formed_at: float
    last_contact: float = 0.0
    trust_level: float = 0.5  # 0.0-1.0
    benefits: list[str] = field(default_factory=list)
    obligations: list[str] = field(default_factory=list)


@dataclass
class SharedInventory:
    """Shared inventory across all empire bots."""
    item_name: str
    total_quantity: int = 0
    allocated_to: dict[str, int] = field(default_factory=dict)  # bot_id -> quantity
    reserved_for: str = ""  # pipeline stage
    last_updated: float = 0.0


@dataclass
class EmpireReport:
    """A periodic empire status report."""
    timestamp: float
    total_bots: int
    total_zeny: int
    total_wealth: int  # zeny + inventory value
    active_directives: int
    pipeline_stage: str
    territories_held: int
    alliances: int
    threats_detected: int
    production_metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EmpireManager
# ---------------------------------------------------------------------------


class EmpireManager:
    """Complete empire management system with C-level roles and production pipeline.

    Wires into:
      - multi_account_synergy.py: team coordination
      - fleet_coordinator.py: fleet management
      - competitive_intelligence.py: threat assessment
      - crisis_manager.py: crisis handling
      - social_engine.py: relationship building
      - market_engine.py: economic operations
      - PDCA loop: strategic decision-making
    """

    def __init__(
        self,
        multi_account_synergy: Any = None,
        fleet_coordinator: Any = None,
        competitive_intelligence: Any = None,
        crisis_manager: Any = None,
        social_engine: Any = None,
        market_engine: Any = None,
        enqueue_fn: Callable | None = None,
    ) -> None:
        self._lock = RLock()

        # Wired dependencies
        self._multi_account_synergy = multi_account_synergy
        self._fleet_coordinator = fleet_coordinator
        self._competitive_intelligence = competitive_intelligence
        self._crisis_manager = crisis_manager
        self._social_engine = social_engine
        self._market_engine = market_engine
        self._enqueue_fn = enqueue_fn

        # C-level roles: title -> EmpireRole
        self._roles: dict[str, EmpireRole] = {}

        # Active directives
        self._directives: list[EmpireDirective] = []

        # Production pipeline
        self._pipeline: ProductionPipeline = ProductionPipeline()

        # Territory claims: map_name -> TerritoryClaim
        self._territories: dict[str, TerritoryClaim] = {}

        # Alliances: entity_name -> Alliance
        self._alliances: dict[str, Alliance] = {}

        # Shared inventory: item_name -> SharedInventory
        self._shared_inventory: dict[str, SharedInventory] = {}

        # Empire reports (last 100)
        self._reports: deque = deque(maxlen=100)

        # Stats
        self._stats: dict[str, int] = {
            "directives_issued": 0,
            "directives_completed": 0,
            "territories_claimed": 0,
            "alliances_formed": 0,
            "transfers_executed": 0,
            "pipeline_transfers": 0,
            "reports_generated": 0,
            "threats_handled": 0,
        }

        # Empire name
        self._empire_name: str = "Unnamed Empire"

    # ── Role Management ──────────────────────────────────────────────

    def assign_role(
        self,
        bot_id: str,
        title: str,
        class_name: str = "",
        level: int = 0,
        specialization: str = "",
    ) -> bool:
        """Assign a C-level role to a bot.

        Valid titles: CEO, CFO, COO, CTO, HR, R&D, Security, Logistics
        """
        valid_titles = {"CEO", "CFO", "COO", "CTO", "HR", "R&D", "Security", "Logistics"}
        if title not in valid_titles:
            logger.warning("empire_invalid_title: %s", title)
            return False

        with self._lock:
            # Remove existing role for this bot
            for existing_title, existing_role in list(self._roles.items()):
                if existing_role.bot_id == bot_id:
                    del self._roles[existing_title]
                    logger.info("empire_role_removed: %s from %s", existing_title, bot_id)

            role = EmpireRole(
                title=title,
                bot_id=bot_id,
                class_name=class_name,
                level=level,
                specialization=specialization or title.lower(),
                last_action=time.time(),
            )
            self._roles[title] = role
            logger.info(
                "empire_role_assigned: %s → %s (%s Lv.%d spec=%s)",
                bot_id, title, class_name, level, role.specialization,
            )

            # Wire into multi-account synergy
            if self._multi_account_synergy is not None:
                try:
                    self._multi_account_synergy.assign_role(
                        bot_id=bot_id,
                        primary=title.lower(),
                        secondary=specialization,
                        level=level,
                        class_name=class_name,
                    )
                except Exception:
                    pass

            # Wire into fleet coordinator
            if self._fleet_coordinator is not None:
                try:
                    self._fleet_coordinator.register_bot(
                        bot_id=bot_id,
                        role=title.lower(),
                        class_name=class_name,
                        level=level,
                    )
                except Exception:
                    pass

            return True

    def get_role(self, title: str) -> EmpireRole | None:
        """Get a C-level role by title."""
        with self._lock:
            return self._roles.get(title)

    def get_bot_role(self, bot_id: str) -> EmpireRole | None:
        """Get the C-level role for a specific bot."""
        with self._lock:
            for role in self._roles.values():
                if role.bot_id == bot_id:
                    return role
            return None

    def get_all_roles(self) -> list[EmpireRole]:
        """Get all C-level roles."""
        with self._lock:
            return list(self._roles.values())

    def update_role_status(self, title: str, status: str) -> None:
        """Update a role's status."""
        with self._lock:
            role = self._roles.get(title)
            if role:
                role.status = status
                role.last_action = time.time()

    def update_role_performance(self, title: str, score: float) -> None:
        """Update a role's performance score (0.0-1.0)."""
        with self._lock:
            role = self._roles.get(title)
            if role:
                role.performance_score = max(0.0, min(1.0, score))

    # ── Strategic Directives ────────────────────────────────────────

    def issue_directive(
        self,
        issued_by: str,
        target_bot: str,
        action: str,
        reason: str,
        priority: int = 5,
        deadline: float = 0.0,
    ) -> str:
        """Issue a strategic directive from a C-level role to a bot."""
        directive_id = f"dir_{int(time.time() * 1000)}_{len(self._directives)}"
        with self._lock:
            directive = EmpireDirective(
                directive_id=directive_id,
                issued_by=issued_by,
                target_bot=target_bot,
                action=action,
                reason=reason,
                priority=priority,
                issued_at=time.time(),
                deadline=deadline or (time.time() + 3600),
            )
            self._directives.append(directive)
            self._stats["directives_issued"] += 1

        logger.info(
            "empire_directive: %s → %s: %s (reason=%s priority=%d)",
            issued_by, target_bot, action, reason, priority,
        )

        # Execute via enqueue
        if self._enqueue_fn:
            try:
                self._enqueue_fn(target_bot, action)
            except Exception:
                pass

        return directive_id

    def complete_directive(self, directive_id: str, result: str = "") -> bool:
        """Mark a directive as completed."""
        with self._lock:
            for directive in self._directives:
                if directive.directive_id == directive_id and not directive.completed:
                    directive.completed = True
                    directive.result = result
                    self._stats["directives_completed"] += 1
                    return True
            return False

    def get_pending_directives(self, target_bot: str | None = None) -> list[EmpireDirective]:
        """Get pending directives, optionally filtered by target bot."""
        with self._lock:
            now = time.time()
            pending = [
                d for d in self._directives
                if not d.completed and d.deadline > now
            ]
            if target_bot:
                pending = [d for d in pending if d.target_bot == target_bot]
            return sorted(pending, key=lambda d: d.priority, reverse=True)

    # ── CEO: Strategic Direction ─────────────────────────────────────

    def ceo_set_strategy(self, empire_name: str, primary_goal: str) -> str:
        """CEO sets the strategic direction for the empire.

        Args:
            empire_name: Name of the empire
            primary_goal: e.g., "dominate_mvp", "control_economy", "pvp_supremacy", "wealth_accumulation"

        Returns:
            Directive ID
        """
        with self._lock:
            self._empire_name = empire_name

        # Issue directives to all roles based on strategy
        directives_issued = []
        for title, role in self._roles.items():
            if title == "CFO" and primary_goal in ("control_economy", "wealth_accumulation"):
                did = self.issue_directive(
                    "CEO", role.bot_id,
                    "ai auto",
                    f"CFO: Execute {primary_goal} strategy - control market prices and maximize wealth",
                    priority=10,
                )
                directives_issued.append(did)
            elif title == "COO" and primary_goal in ("dominate_mvp", "wealth_accumulation"):
                did = self.issue_directive(
                    "CEO", role.bot_id,
                    "ai auto",
                    f"COO: Execute {primary_goal} strategy - optimize farming routes and allocate bots",
                    priority=9,
                )
                directives_issued.append(did)
            elif title == "CTO":
                did = self.issue_directive(
                    "CEO", role.bot_id,
                    "ai auto",
                    f"CTO: Support {primary_goal} strategy - produce potions, elemental weapons, and crafted goods",
                    priority=8,
                )
                directives_issued.append(did)
            elif title == "Security":
                did = self.issue_directive(
                    "CEO", role.bot_id,
                    "ai auto",
                    f"Security: Protect empire assets during {primary_goal} strategy - detect threats, eliminate competition",
                    priority=8,
                )
                directives_issued.append(did)
            elif title == "HR":
                did = self.issue_directive(
                    "CEO", role.bot_id,
                    "ai auto",
                    f"HR: Support {primary_goal} strategy - recruit party members, build alliances",
                    priority=7,
                )
                directives_issued.append(did)
            elif title == "R&D":
                did = self.issue_directive(
                    "CEO", role.bot_id,
                    "ai auto",
                    f"R&D: Support {primary_goal} strategy - research builds, map dungeons, track MVP spawns",
                    priority=6,
                )
                directives_issued.append(did)
            elif title == "Logistics":
                did = self.issue_directive(
                    "CEO", role.bot_id,
                    "ai auto",
                    f"Logistics: Support {primary_goal} strategy - manage shared inventory, coordinate transfers",
                    priority=6,
                )
                directives_issued.append(did)

        logger.info(
            "empire_strategy_set: name=%s goal=%s directives=%d",
            empire_name, primary_goal, len(directives_issued),
        )
        return directives_issued[0] if directives_issued else ""

    def ceo_negotiate_deal(self, target_player: str, deal_type: str, terms: str) -> str:
        """CEO negotiates a deal with another player.

        Args:
            target_player: Player name
            deal_type: trade, alliance, non_aggression, information
            terms: What we offer and what we want
        """
        directive_id = self.issue_directive(
            "CEO", self._roles.get("CEO", EmpireRole(title="CEO", bot_id="")).bot_id,
            f"ai auto",
            f"CEO: Negotiate {deal_type} deal with {target_player}: {terms}",
            priority=9,
        )

        # Log the negotiation attempt
        logger.info(
            "empire_negotiation: target=%s type=%s terms=%s",
            target_player, deal_type, terms,
        )

        return directive_id

    def ceo_lead_woe(self, woe_map: str, objective: str) -> str:
        """CEO leads WOE operations.

        Args:
            woe_map: Castle/map to attack or defend
            objective: break_emperium, defend_castle, disrupt_enemy, support_allies
        """
        # Issue directives to all combat-capable bots
        directives_issued = []
        for title, role in self._roles.items():
            if title in ("CEO", "Security"):
                did = self.issue_directive(
                    "CEO", role.bot_id,
                    f"ai auto",
                    f"WOE: {objective} on {woe_map} - coordinate with empire",
                    priority=10,
                )
                directives_issued.append(did)

        # Notify logistics to prepare consumables
        logistics = self._roles.get("Logistics")
        if logistics:
            self.issue_directive(
                "CEO", logistics.bot_id,
                "ai auto",
                f"Logistics: Prepare WOE consumables for {woe_map} - potions, elemental converters, fly wings",
                priority=9,
            )

        logger.info(
            "empire_woe: map=%s objective=%s directives=%d",
            woe_map, objective, len(directives_issued),
        )
        return directives_issued[0] if directives_issued else ""

    # ── CFO: Economic Management ────────────────────────────────────

    def cfo_set_prices(self, item_name: str, min_price: int, max_price: int) -> str:
        """CFO sets price controls for an item.

        Args:
            item_name: Item to control
            min_price: Minimum sell price
            max_price: Maximum buy price
        """
        directive_id = self.issue_directive(
            "CFO",
            self._roles.get("CFO", EmpireRole(title="CFO", bot_id="")).bot_id,
            "ai auto",
            f"CFO: Set price controls for {item_name} - sell at {min_price}+, buy at {max_price}-",
            priority=9,
        )

        # Update market engine if wired
        if self._market_engine is not None:
            try:
                if hasattr(self._market_engine, 'set_price_control'):
                    self._market_engine.set_price_control(item_name, min_price, max_price)
            except Exception:
                pass

        logger.info(
            "empire_price_control: item=%s min=%d max=%d",
            item_name, min_price, max_price,
        )
        return directive_id

    def cfo_track_wealth(self) -> dict[str, int]:
        """CFO tracks total empire wealth.

        Returns:
            Dict with total_zeny, total_inventory_value, total_wealth
        """
        with self._lock:
            total_zeny = 0
            total_inventory_value = 0

            # Sum up shared inventory value
            for item_name, inv in self._shared_inventory.items():
                # Estimate item value (simplified)
                estimated_value = self._estimate_item_value(item_name)
                total_inventory_value += inv.total_quantity * estimated_value

            # Sum up pipeline zeny
            total_zeny = self._pipeline.zeny_reserve

            return {
                "total_zeny": total_zeny,
                "total_inventory_value": total_inventory_value,
                "total_wealth": total_zeny + total_inventory_value,
            }

    def _estimate_item_value(self, item_name: str) -> int:
        """Estimate the value of an item based on market data."""
        if self._market_engine is not None:
            try:
                if hasattr(self._market_engine, 'get_item_value'):
                    return self._market_engine.get_item_value(item_name)
            except Exception:
                pass
        # Default estimates
        value_map = {
            "Red Potion": 50, "White Potion": 500, "Blue Potion": 2000,
            "Fly Wing": 500, "Butterfly Wing": 1000,
            "Empty Bottle": 50, "Poison Bottle": 1000,
            "Oridecon": 50000, "Elunium": 30000,
            "Poring Card": 10000, "Thief Bug Card": 5000,
        }
        return value_map.get(item_name, 100)

    def cfo_allocate_budget(self, bot_id: str, amount: int, purpose: str) -> str:
        """CFO allocates zeny to a bot for a specific purpose.

        Args:
            bot_id: Target bot
            amount: Zeny amount
            purpose: restock, gear_upgrade, crafting, trading
        """
        with self._lock:
            if self._pipeline.zeny_reserve < amount:
                logger.warning("empire_insufficient_zeny: have=%d need=%d", self._pipeline.zeny_reserve, amount)
                return ""

            self._pipeline.zeny_reserve -= amount

        directive_id = self.issue_directive(
            "CFO", bot_id,
            f"ai auto",
            f"CFO: Allocated {amount}z for {purpose} - use wisely",
            priority=8,
        )

        logger.info(
            "empire_budget_allocated: bot=%s amount=%d purpose=%s",
            bot_id, amount, purpose,
        )
        return directive_id

    # ── COO: Operations Management ──────────────────────────────────

    def coo_allocate_bot(self, bot_id: str, map_name: str, task: str) -> str:
        """COO allocates a bot to a specific map and task.

        Args:
            bot_id: Bot to allocate
            map_name: Target map
            task: farming, scouting, guarding, patrolling
        """
        directive_id = self.issue_directive(
            "COO", bot_id,
            f"ai auto",
            f"COO: Allocate to {map_name} for {task} - optimize route for maximum efficiency",
            priority=8,
        )

        # Update fleet coordinator
        if self._fleet_coordinator is not None:
            try:
                self._fleet_coordinator.update_bot_status(
                    bot_id=bot_id,
                    map=map_name,
                    status="farming" if task == "farming" else "scouting",
                )
            except Exception:
                pass

        logger.info(
            "empire_bot_allocated: bot=%s map=%s task=%s",
            bot_id, map_name, task,
        )
        return directive_id

    def coo_optimize_routes(self) -> list[dict[str, Any]]:
        """COO analyzes and optimizes all farming routes.

        Returns:
            List of optimization recommendations
        """
        recommendations = []
        with self._lock:
            for title, role in self._roles.items():
                if role.specialization == "farming" and role.status == "active":
                    # Check if this bot has a territory
                    bot_territories = [
                        t for t in self._territories.values()
                        if t.claimed_by == role.bot_id
                    ]
                    if not bot_territories:
                        recommendations.append({
                            "bot_id": role.bot_id,
                            "issue": "no_territory",
                            "recommendation": "Assign a farming territory",
                            "priority": 7,
                        })

        return recommendations

    def coo_claim_territory(self, map_name: str, bot_id: str, priority: int = 5) -> bool:
        """COO claims a territory (map) for exclusive farming.

        Args:
            map_name: Map to claim
            bot_id: Bot that will farm it
            priority: How important this territory is (1-10)
        """
        with self._lock:
            # Check if already claimed
            existing = self._territories.get(map_name)
            if existing:
                logger.warning("empire_territory_already_claimed: %s by %s", map_name, existing.claimed_by)
                return False

            claim = TerritoryClaim(
                map_name=map_name,
                claimed_by=bot_id,
                claimed_at=time.time(),
                priority=priority,
            )
            self._territories[map_name] = claim
            self._stats["territories_claimed"] += 1

        logger.info("empire_territory_claimed: map=%s bot=%s priority=%d", map_name, bot_id, priority)

        # Issue directive to the bot
        self.issue_directive(
            "COO", bot_id,
            f"ai auto",
            f"COO: Territory claimed - {map_name}. Farm and defend this territory.",
            priority=priority,
        )
        return True

    def coo_get_territories(self) -> list[TerritoryClaim]:
        """Get all claimed territories."""
        with self._lock:
            return list(self._territories.values())

    # ── CTO: Production Management ──────────────────────────────────

    def cto_produce_item(self, item_name: str, quantity: int, priority: int = 5) -> str:
        """CTO produces items through the crafting pipeline.

        Args:
            item_name: Item to produce
            quantity: How many to produce
            priority: Production priority (1-10)
        """
        crafter = self._roles.get("CTO")
        if not crafter:
            logger.warning("empire_no_cto: cannot produce %s", item_name)
            return ""

        directive_id = self.issue_directive(
            "CTO", crafter.bot_id,
            f"ai auto",
            f"CTO: Produce {quantity}x {item_name} - check raw materials, craft, and store",
            priority=priority,
        )

        # Update pipeline
        with self._lock:
            if item_name in self._pipeline.raw_materials:
                # This is a raw material being produced
                self._pipeline.raw_materials[item_name] = \
                    self._pipeline.raw_materials.get(item_name, 0) + quantity
            else:
                # This is a crafted good
                self._pipeline.crafted_goods[item_name] = \
                    self._pipeline.crafted_goods.get(item_name, 0) + quantity

        logger.info(
            "empire_production: item=%s qty=%d priority=%d",
            item_name, quantity, priority,
        )
        return directive_id

    def cto_manage_crafting_pipeline(self) -> dict[str, Any]:
        """CTO reviews and manages the crafting pipeline.

        Returns:
            Dict with pipeline status and recommendations
        """
        with self._lock:
            status = {
                "raw_materials": dict(self._pipeline.raw_materials),
                "crafted_goods": dict(self._pipeline.crafted_goods),
                "inventory_for_sale": dict(self._pipeline.inventory_for_sale),
                "recommendations": [],
            }

            # Check if raw materials need replenishing
            for item, qty in self._pipeline.raw_materials.items():
                if qty < 10:
                    status["recommendations"].append(
                        f"Low on {item} ({qty}) - farmer should collect more"
                    )

            # Check if crafted goods need to be moved to merchant
            for item, qty in self._pipeline.crafted_goods.items():
                if qty > 0:
                    status["recommendations"].append(
                        f"{qty}x {item} ready for sale - move to merchant"
                    )

            return status

    # ── HR: People Management ───────────────────────────────────────

    def hr_recruit_party(self, target_players: list[str], role_needed: str) -> str:
        """HR recruits players for party content.

        Args:
            target_players: Players to recruit
            role_needed: tank, healer, dps, support
        """
        hr_role = self._roles.get("HR")
        if not hr_role:
            return ""

        directive_id = self.issue_directive(
            "HR", hr_role.bot_id,
            "ai auto",
            f"HR: Recruit {target_players} for {role_needed} role - build party synergy",
            priority=8,
        )

        logger.info(
            "empire_recruitment: targets=%s role=%s",
            target_players, role_needed,
        )
        return directive_id

    def hr_build_alliance(self, entity_name: str, entity_type: str, relationship: str) -> str:
        """HR builds an alliance with another player or guild.

        Args:
            entity_name: Player or guild name
            entity_type: player, guild
            relationship: ally, trade_partner, mutual_defense, non_aggression
        """
        with self._lock:
            alliance = Alliance(
                entity_name=entity_name,
                entity_type=entity_type,
                relationship=relationship,
                formed_at=time.time(),
                last_contact=time.time(),
            )
            self._alliances[entity_name] = alliance
            self._stats["alliances_formed"] += 1

        logger.info(
            "empire_alliance_formed: entity=%s type=%s relationship=%s",
            entity_name, entity_type, relationship,
        )

        # Notify social engine
        if self._social_engine is not None:
            try:
                if hasattr(self._social_engine, 'mark_relationship'):
                    self._social_engine.mark_relationship(
                        player_name=entity_name if entity_type == "player" else "",
                        relationship_type=relationship,
                    )
            except Exception:
                pass

        return self.issue_directive(
            "HR",
            self._roles.get("HR", EmpireRole(title="HR", bot_id="")).bot_id,
            "ai auto",
            f"HR: Alliance formed with {entity_name} ({relationship}) - maintain relationship",
            priority=7,
        )

    def hr_get_alliances(self) -> list[Alliance]:
        """Get all active alliances."""
        with self._lock:
            return list(self._alliances.values())

    # ── R&D: Research and Development ───────────────────────────────

    def rd_discover_mvp_spawn(self, map_name: str, mvp_name: str, spawn_time: float) -> str:
        """R&D records an MVP spawn discovery.

        Args:
            map_name: Map where MVP spawns
            mvp_name: MVP monster name
            spawn_time: When it was observed spawning
        """
        directive_id = self.issue_directive(
            "R&D",
            self._roles.get("R&D", EmpireRole(title="R&D", bot_id="")).bot_id,
            "ai auto",
            f"R&D: MVP spawn recorded - {mvp_name} on {map_name} at {spawn_time}",
            priority=7,
        )

        logger.info(
            "empire_mvp_discovered: mvp=%s map=%s time=%s",
            mvp_name, map_name, spawn_time,
        )
        return directive_id

    def rd_research_build(self, class_name: str, build_type: str, notes: str) -> str:
        """R&D researches a new build.

        Args:
            class_name: Job class
            build_type: vit, agi, int, hybrid
            notes: Research findings
        """
        directive_id = self.issue_directive(
            "R&D",
            self._roles.get("R&D", EmpireRole(title="R&D", bot_id="")).bot_id,
            "ai auto",
            f"R&D: Research {class_name} {build_type} build - {notes}",
            priority=6,
        )

        logger.info(
            "empire_build_researched: class=%s build=%s",
            class_name, build_type,
        )
        return directive_id

    def rd_track_patch_changes(self, patch_notes: str) -> str:
        """R&D tracks game patch changes and adapts strategy.

        Args:
            patch_notes: Description of patch changes
        """
        directive_id = self.issue_directive(
            "R&D",
            self._roles.get("R&D", EmpireRole(title="R&D", bot_id="")).bot_id,
            "ai auto",
            f"R&D: Patch change detected - {patch_notes} - adapt strategy accordingly",
            priority=8,
        )

        logger.info("empire_patch_tracked: %s", patch_notes)
        return directive_id

    # ── Security: Threat Management ─────────────────────────────────

    def security_detect_threats(self) -> list[dict[str, Any]]:
        """Security scans for threats to the empire.

        Returns:
            List of detected threats with severity
        """
        threats = []

        # Check competitive intelligence for threats
        if self._competitive_intelligence is not None:
            try:
                ci_threats = self._competitive_intelligence.get_threats(min_score=50)
                for threat in ci_threats:
                    threats.append({
                        "type": "player_threat",
                        "player": threat.player_name,
                        "class": threat.job_class,
                        "level": threat.base_level,
                        "threat_score": threat.threat_score,
                        "severity": "high" if threat.threat_score >= 70 else "medium",
                    })
            except Exception:
                pass

        # Check territory competition
        with self._lock:
            for map_name, claim in self._territories.items():
                if claim.competition_level >= 7:
                    threats.append({
                        "type": "territory_competition",
                        "map": map_name,
                        "claimed_by": claim.claimed_by,
                        "competition_level": claim.competition_level,
                        "severity": "high",
                    })

        return threats

    def security_eliminate_competition(self, target_player: str, reason: str) -> str:
        """Security eliminates competition from a specific player.

        Args:
            target_player: Player to target
            reason: Why they need to be eliminated
        """
        security_role = self._roles.get("Security")
        if not security_role:
            return ""

        directive_id = self.issue_directive(
            "Security", security_role.bot_id,
            "ai auto",
            f"Security: Eliminate competition from {target_player} - {reason}",
            priority=9,
        )

        logger.info(
            "empire_competition_elimination: target=%s reason=%s",
            target_player, reason,
        )
        return directive_id

    def security_protect_territory(self, map_name: str) -> str:
        """Security protects a claimed territory from intruders.

        Args:
            map_name: Territory to protect
        """
        security_role = self._roles.get("Security")
        if not security_role:
            return ""

        with self._lock:
            claim = self._territories.get(map_name)
            if claim:
                claim.last_defended = time.time()

        directive_id = self.issue_directive(
            "Security", security_role.bot_id,
            f"ai auto",
            f"Security: Protect territory {map_name} - patrol and eliminate intruders",
            priority=8,
        )

        logger.info("empire_territory_protected: map=%s", map_name)
        return directive_id

    # ── Logistics: Resource Management ──────────────────────────────

    def logistics_transfer_item(self, item_name: str, quantity: int, from_bot: str, to_bot: str) -> str:
        """Logistics transfers items between bots.

        Args:
            item_name: Item to transfer
            quantity: How many
            from_bot: Source bot
            to_bot: Destination bot
        """
        directive_id = self.issue_directive(
            "Logistics", from_bot,
            f"ai auto",
            f"Logistics: Transfer {quantity}x {item_name} to {to_bot}",
            priority=7,
        )

        with self._lock:
            self._stats["transfers_executed"] += 1

        logger.info(
            "empire_transfer: %dx %s from %s to %s",
            quantity, item_name, from_bot, to_bot,
        )
        return directive_id

    def logistics_manage_shared_inventory(self) -> dict[str, SharedInventory]:
        """Logistics reviews and manages shared inventory."""
        with self._lock:
            return dict(self._shared_inventory)

    def logistics_add_to_shared(self, item_name: str, quantity: int, source_bot: str) -> None:
        """Add items to shared inventory."""
        with self._lock:
            inv = self._shared_inventory.setdefault(
                item_name,
                SharedInventory(item_name=item_name),
            )
            inv.total_quantity += quantity
            inv.allocated_to[source_bot] = inv.allocated_to.get(source_bot, 0) + quantity
            inv.last_updated = time.time()

    def logistics_remove_from_shared(self, item_name: str, quantity: int, target_bot: str) -> bool:
        """Remove items from shared inventory for a bot."""
        with self._lock:
            inv = self._shared_inventory.get(item_name)
            if not inv or inv.total_quantity < quantity:
                return False
            inv.total_quantity -= quantity
            inv.allocated_to[target_bot] = inv.allocated_to.get(target_bot, 0) + quantity
            inv.last_updated = time.time()
            return True

    # ── Production Pipeline ────────────────────────────────────────

    def setup_production_pipeline(
        self,
        farmer_bot: str,
        crafter_bot: str,
        merchant_bot: str,
        pvp_bot: str = "",
    ) -> None:
        """Set up the farmer -> crafter -> merchant -> PVP production pipeline.

        Args:
            farmer_bot: Bot that farms raw materials
            crafter_bot: Bot that crafts items
            merchant_bot: Bot that sells items
            pvp_bot: Bot that uses items for PVP (optional)
        """
        with self._lock:
            self._pipeline.farmer_bot_id = farmer_bot
            self._pipeline.crafter_bot_id = crafter_bot
            self._pipeline.merchant_bot_id = merchant_bot
            self._pipeline.pvp_bot_id = pvp_bot
            self._pipeline.pipeline_active = True

        # Assign roles
        self.assign_role(farmer_bot, "COO", specialization="farming")
        self.assign_role(crafter_bot, "CTO", specialization="crafting")
        self.assign_role(merchant_bot, "CFO", specialization="merchant")
        if pvp_bot:
            self.assign_role(pvp_bot, "Security", specialization="pvp")

        logger.info(
            "empire_pipeline_setup: farmer=%s crafter=%s merchant=%s pvp=%s",
            farmer_bot, crafter_bot, merchant_bot, pvp_bot or "none",
        )

    def execute_pipeline_transfer(self) -> dict[str, Any]:
        """Execute the next step in the production pipeline.

        Returns:
            Dict describing what was transferred
        """
        with self._lock:
            if not self._pipeline.pipeline_active:
                return {"status": "pipeline_inactive"}

            transfer = {"status": "no_transfer_needed"}

            # Step 1: Farmer -> Crafter (raw materials)
            if self._pipeline.raw_materials:
                item, qty = next(iter(self._pipeline.raw_materials.items()))
                if qty > 0:
                    # Move to crafted goods
                    self._pipeline.raw_materials[item] = 0
                    self._pipeline.crafted_goods[item] = \
                        self._pipeline.crafted_goods.get(item, 0) + qty
                    self._pipeline.last_transfer = time.time()
                    self._stats["pipeline_transfers"] += 1
                    transfer = {
                        "status": "farmer_to_crafter",
                        "item": item,
                        "quantity": qty,
                    }
                    logger.info(
                        "pipeline_transfer: farmer→crafter %dx %s",
                        qty, item,
                    )

            # Step 2: Crafter -> Merchant (crafted goods)
            elif self._pipeline.crafted_goods:
                item, qty = next(iter(self._pipeline.crafted_goods.items()))
                if qty > 0:
                    self._pipeline.crafted_goods[item] = 0
                    self._pipeline.inventory_for_sale[item] = \
                        self._pipeline.inventory_for_sale.get(item, 0) + qty
                    self._pipeline.last_transfer = time.time()
                    self._stats["pipeline_transfers"] += 1
                    transfer = {
                        "status": "crafter_to_merchant",
                        "item": item,
                        "quantity": qty,
                    }
                    logger.info(
                        "pipeline_transfer: crafter→merchant %dx %s",
                        qty, item,
                    )

            return transfer

    def get_pipeline_status(self) -> dict[str, Any]:
        """Get the current status of the production pipeline."""
        with self._lock:
            return {
                "active": self._pipeline.pipeline_active,
                "farmer": self._pipeline.farmer_bot_id,
                "crafter": self._pipeline.crafter_bot_id,
                "merchant": self._pipeline.merchant_bot_id,
                "pvp": self._pipeline.pvp_bot_id,
                "raw_materials": dict(self._pipeline.raw_materials),
                "crafted_goods": dict(self._pipeline.crafted_goods),
                "inventory_for_sale": dict(self._pipeline.inventory_for_sale),
                "zeny_reserve": self._pipeline.zeny_reserve,
                "last_transfer": self._pipeline.last_transfer,
            }

    # ── Empire Reporting ────────────────────────────────────────────

    def generate_report(self) -> EmpireReport:
        """Generate a comprehensive empire status report."""
        with self._lock:
            wealth = self.cfo_track_wealth()
            threats = self.security_detect_threats()
            pipeline_status = self.get_pipeline_status()

            report = EmpireReport(
                timestamp=time.time(),
                total_bots=len(self._roles),
                total_zeny=wealth["total_zeny"],
                total_wealth=wealth["total_wealth"],
                active_directives=len(self.get_pending_directives()),
                pipeline_stage=pipeline_status.get("status", "inactive"),
                territories_held=len(self._territories),
                alliances=len(self._alliances),
                threats_detected=len(threats),
                production_metrics={
                    "raw_materials": len(self._pipeline.raw_materials),
                    "crafted_goods": len(self._pipeline.crafted_goods),
                    "inventory_for_sale": len(self._pipeline.inventory_for_sale),
                },
            )
            self._reports.append(report)
            self._stats["reports_generated"] += 1

            return report

    def get_empire_summary(self) -> str:
        """Get a formatted summary of the empire state."""
        with self._lock:
            lines = [f"── {self._empire_name} Empire ──"]

            # Roles
            if self._roles:
                lines.append(f"  C-Level Roles ({len(self._roles)}):")
                for title, role in sorted(self._roles.items()):
                    lines.append(
                        f"    {title}: {role.bot_id} ({role.class_name} Lv.{role.level}) "
                        f"[{role.status}] perf={role.performance_score:.1%}"
                    )
            else:
                lines.append("  No C-level roles assigned.")

            # Pipeline
            if self._pipeline.pipeline_active:
                lines.append("  Production Pipeline:")
                lines.append(f"    Farmer: {self._pipeline.farmer_bot_id or 'unassigned'}")
                lines.append(f"    Crafter: {self._pipeline.crafter_bot_id or 'unassigned'}")
                lines.append(f"    Merchant: {self._pipeline.merchant_bot_id or 'unassigned'}")
                if self._pipeline.pvp_bot_id:
                    lines.append(f"    PVP: {self._pipeline.pvp_bot_id}")
                lines.append(f"    Zeny Reserve: {self._pipeline.zeny_reserve:,}z")

            # Territories
            if self._territories:
                lines.append(f"  Territories ({len(self._territories)}):")
                for map_name, claim in list(self._territories.items())[:5]:
                    lines.append(
                        f"    {map_name}: claimed by {claim.claimed_by} "
                        f"(competition: {claim.competition_level}/10)"
                    )

            # Alliances
            if self._alliances:
                lines.append(f"  Alliances ({len(self._alliances)}):")
                for name, alliance in list(self._alliances.items())[:3]:
                    lines.append(f"    {name}: {alliance.relationship} (trust: {alliance.trust_level:.0%})")

            # Directives
            pending = self.get_pending_directives()
            if pending:
                lines.append(f"  Active Directives ({len(pending)}):")
                for d in pending[:5]:
                    lines.append(f"    [{d.priority}] {d.issued_by} → {d.target_bot}: {d.action}")

            # Stats
            lines.append(f"  Stats: {self._stats['directives_issued']} directives, "
                        f"{self._stats['territories_claimed']} territories, "
                        f"{self._stats['alliances_formed']} alliances")

            return "\n".join(lines)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    # ── Persistence ──

    def save_state(self) -> int:
        """Save all empire manager state to persistent storage."""
        from ai_sidecar.persistence.strategy_state import StrategyStateDB
        with self._lock:
            data = {
                "roles": {
                    k: {
                        "title": v.title,
                        "bot_id": v.bot_id,
                        "class_name": v.class_name,
                        "level": v.level,
                        "status": v.status,
                        "last_action": v.last_action,
                        "performance_score": v.performance_score,
                        "specialization": v.specialization,
                    }
                    for k, v in self._roles.items()
                },
                "directives": [{
                    "directive_id": d.directive_id,
                    "issued_by": d.issued_by,
                    "target_bot": d.target_bot,
                    "action": d.action,
                    "reason": d.reason,
                    "priority": d.priority,
                    "issued_at": d.issued_at,
                    "deadline": d.deadline,
                    "completed": d.completed,
                    "result": d.result,
                } for d in self._directives],
                "pipeline": {
                    "farmer_bot_id": self._pipeline.farmer_bot_id,
                    "crafter_bot_id": self._pipeline.crafter_bot_id,
                    "merchant_bot_id": self._pipeline.merchant_bot_id,
                    "pvp_bot_id": self._pipeline.pvp_bot_id,
                    "raw_materials": dict(self._pipeline.raw_materials),
                    "crafted_goods": dict(self._pipeline.crafted_goods),
                    "inventory_for_sale": dict(self._pipeline.inventory_for_sale),
                    "zeny_reserve": self._pipeline.zeny_reserve,
                    "pipeline_active": self._pipeline.pipeline_active,
                    "last_transfer": self._pipeline.last_transfer,
                },
                "territories": {
                    k: {
                        "map_name": v.map_name,
                        "claimed_by": v.claimed_by,
                        "claimed_at": v.claimed_at,
                        "priority": v.priority,
                        "competition_level": v.competition_level,
                        "last_defended": v.last_defended,
                        "zeny_per_hour": v.zeny_per_hour,
                    }
                    for k, v in self._territories.items()
                },
                "alliances": {
                    k: {
                        "entity_name": v.entity_name,
                        "entity_type": v.entity_type,
                        "relationship": v.relationship,
                        "formed_at": v.formed_at,
                        "last_contact": v.last_contact,
                        "trust_level": v.trust_level,
                        "benefits": v.benefits,
                        "obligations": v.obligations,
                    }
                    for k, v in self._alliances.items()
                },
                "shared_inventory": {
                    k: {
                        "item_name": v.item_name,
                        "total_quantity": v.total_quantity,
                        "allocated_to": dict(v.allocated_to),
                        "reserved_for": v.reserved_for,
                        "last_updated": v.last_updated,
                    }
                    for k, v in self._shared_inventory.items()
                },
                "stats": dict(self._stats),
            }
            return StrategyStateDB.save_empire_manager(data)

    def load_state(self) -> bool:
        """Load empire manager state from persistent storage."""
        from ai_sidecar.persistence.strategy_state import StrategyStateDB
        data = StrategyStateDB.load_empire_manager()
        if data is None:
            return False
        with self._lock:
            self._roles.clear()
            for title, r_data in data.get("roles", {}).items():
                self._roles[title] = EmpireRole(**r_data)
            self._directives.clear()
            for d_data in data.get("directives", []):
                self._directives.append(EmpireDirective(**d_data))
            pipeline_data = data.get("pipeline", {})
            self._pipeline.farmer_bot_id = pipeline_data.get("farmer_bot_id", "")
            self._pipeline.crafter_bot_id = pipeline_data.get("crafter_bot_id", "")
            self._pipeline.merchant_bot_id = pipeline_data.get("merchant_bot_id", "")
            self._pipeline.pvp_bot_id = pipeline_data.get("pvp_bot_id", "")
            self._pipeline.raw_materials = defaultdict(int, pipeline_data.get("raw_materials", {}))
            self._pipeline.crafted_goods = defaultdict(int, pipeline_data.get("crafted_goods", {}))
            self._pipeline.inventory_for_sale = defaultdict(int, pipeline_data.get("inventory_for_sale", {}))
            self._pipeline.zeny_reserve = pipeline_data.get("zeny_reserve", 0)
            self._pipeline.pipeline_active = pipeline_data.get("pipeline_active", False)
            self._pipeline.last_transfer = pipeline_data.get("last_transfer", 0.0)
            self._territories.clear()
            for name, t_data in data.get("territories", {}).items():
                self._territories[name] = TerritoryClaim(**t_data)
            self._alliances.clear()
            for name, a_data in data.get("alliances", {}).items():
                self._alliances[name] = Alliance(**a_data)
            self._shared_inventory.clear()
            for name, s_data in data.get("shared_inventory", {}).items():
                self._shared_inventory[name] = SharedInventory(**s_data)
            saved_stats = data.get("stats", {})
            for k, v in saved_stats.items():
                if k in self._stats:
                    self._stats[k] = v
            logger.info("empire_manager_state_loaded: %d roles, %d territories, %d alliances",
                        len(self._roles), len(self._territories), len(self._alliances))
            return True


# ── Global instance ──

_empire_manager: EmpireManager | None = None
_empire_lock = RLock()


def get_empire_manager() -> EmpireManager:
    """Get or create the global empire manager."""
    global _empire_manager
    with _empire_lock:
        if _empire_manager is None:
            _empire_manager = EmpireManager()
        return _empire_manager
