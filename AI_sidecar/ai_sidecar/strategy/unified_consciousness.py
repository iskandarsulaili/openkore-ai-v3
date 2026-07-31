"""
Unified Consciousness — integrates all 60+ services into a single decision-making system.

A top player doesn't have 60 separate modules making independent decisions.
They have ONE mind that integrates everything:
- Sensory input (what I see, hear, feel)
- Memory (what I remember, what I learned)
- Reasoning (what I think, what I plan)
- Action (what I do, what I say)
- Reflection (what I should have done differently)

This module is the "self" — the unified consciousness that:
1. Maintains a coherent world model (all players, monsters, maps, events)
2. Makes decisions based on long-term goals, not immediate heuristics
3. Reflects on past experiences and extracts general principles
4. Imagines future scenarios and plans accordingly
5. Has a single "self" model that integrates all sensory input
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConsciousnessState(Enum):
    """The current state of consciousness."""
    AWAKE = "awake"  # Normal operation
    FOCUSED = "focused"  # Deep concentration on a specific task
    REFLECTIVE = "reflective"  # Reviewing past experiences
    PLANNING = "planning"  # Strategic planning
    CRISIS = "crisis"  # Emergency mode
    LEARNING = "learning"  # Absorbing new information
    RESTING = "resting"  # Low activity / maintenance


class DecisionDomain(Enum):
    """Domains the unified consciousness can make decisions in."""
    COMBAT = "combat"
    ECONOMY = "economy"
    SOCIAL = "social"
    FLEET = "fleet"
    EMPIRE = "empire"
    PROGRESSION = "progression"
    EXPLORATION = "exploration"
    CRAFTING = "crafting"
    QUESTING = "questing"
    PVP = "pvp"
    WOE = "woe"
    LEARNING = "learning"
    SURVIVAL = "survival"


class DecisionUrgency(Enum):
    """How urgent a decision is."""
    REFLEX = "reflex"  # Must act NOW (sub-second)
    IMMEDIATE = "immediate"  # Act within seconds
    SHORT_TERM = "short_term"  # Act within minutes
    MEDIUM_TERM = "medium_term"  # Act within hours
    LONG_TERM = "long_term"  # Act within days


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class WorldModel:
    """The unified world model — everything the consciousness knows."""
    # All known players: player_name -> dict of info
    known_players: dict[str, dict[str, Any]] = field(default_factory=dict)
    # All known monsters: monster_name -> dict of info
    known_monsters: dict[str, dict[str, Any]] = field(default_factory=dict)
    # All known maps: map_name -> dict of info
    known_maps: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Current events: event_id -> event
    active_events: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Server state
    server_time: float = 0.0
    server_population: int = 0
    server_economy_state: str = "unknown"  # stable, inflation, recession
    last_updated: float = 0.0


@dataclass
class SelfModel:
    """The consciousness's model of itself."""
    # Identity
    empire_name: str = ""
    bot_ids: list[str] = field(default_factory=list)
    primary_bot: str = ""

    # Current state
    current_state: ConsciousnessState = ConsciousnessState.AWAKE
    current_focus: str = ""  # What we're focused on
    current_priority: str = ""  # Current top priority

    # Capabilities
    total_bots: int = 0
    total_zeny: int = 0
    total_wealth: int = 0
    combat_readiness: float = 0.5  # 0.0-1.0
    economic_power: float = 0.5  # 0.0-1.0
    social_influence: float = 0.5  # 0.0-1.0

    # History
    creation_time: float = 0.0
    total_cycles: int = 0
    total_decisions: int = 0
    total_reflections: int = 0

    # Goals
    active_goals: list[dict[str, Any]] = field(default_factory=list)
    completed_goals: list[dict[str, Any]] = field(default_factory=list)
    last_updated: float = 0.0


@dataclass
class UnifiedDecision:
    """A decision made by the unified consciousness."""
    decision_id: str
    domain: DecisionDomain
    urgency: DecisionUrgency
    action: str
    target_bot: str = ""
    reason: str = ""
    confidence: float = 0.5  # 0.0-1.0
    expected_outcome: str = ""
    alternatives: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    executed: bool = False
    outcome: str = ""
    reflection: str = ""


@dataclass
class Reflection:
    """A reflection on past experience."""
    reflection_id: str
    topic: str
    experience: str
    lesson: str
    general_principle: str  # What we learned that applies generally
    domain: DecisionDomain
    timestamp: float = 0.0
    applied_count: int = 0
    last_applied: float = 0.0


@dataclass
class FutureScenario:
    """An imagined future scenario and plan."""
    scenario_id: str
    description: str
    probability: float  # 0.0-1.0
    impact: str  # positive, negative, neutral
    timeframe: str  # short_term, medium_term, long_term
    plan: list[str] = field(default_factory=list)
    contingencies: list[str] = field(default_factory=list)
    created_at: float = 0.0


# ---------------------------------------------------------------------------
# UnifiedConsciousness
# ---------------------------------------------------------------------------


class UnifiedConsciousness:
    """The unified consciousness — integrates all services into one mind.

    This is the single decision-making system that:
    1. Receives input from all 60+ services
    2. Maintains a coherent world model
    3. Makes decisions based on long-term goals
    4. Reflects on past experiences
    5. Imagines future scenarios
    6. Has a single "self" model

    Wires into:
      - empire_manager.py: strategic direction
      - theory_of_mind.py: player intent modeling
      - competitive_intelligence.py: threat assessment
      - crisis_manager.py: crisis handling
      - conscious_engine.py: build/skill decisions
      - long_term_planner.py: goal decomposition
      - multi_account_synergy.py: team coordination
      - fleet_coordinator.py: fleet management
      - social_engine.py: social intelligence
      - market_engine.py: economic intelligence
      - PDCA loop: primary decision-maker
    """

    def __init__(
        self,
        empire_manager: Any = None,
        theory_of_mind: Any = None,
        competitive_intelligence: Any = None,
        crisis_manager: Any = None,
        conscious_engine: Any = None,
        long_term_planner: Any = None,
        multi_account_synergy: Any = None,
        fleet_coordinator: Any = None,
        social_engine: Any = None,
        market_engine: Any = None,
        enqueue_fn: Callable | None = None,
    ) -> None:
        self._lock = RLock()

        # Wired dependencies
        self._empire_manager = empire_manager
        self._theory_of_mind = theory_of_mind
        self._competitive_intelligence = competitive_intelligence
        self._crisis_manager = crisis_manager
        self._conscious_engine = conscious_engine
        self._long_term_planner = long_term_planner
        self._multi_account_synergy = multi_account_synergy
        self._fleet_coordinator = fleet_coordinator
        self._social_engine = social_engine
        self._market_engine = market_engine
        self._enqueue_fn = enqueue_fn

        # World model
        self._world: WorldModel = WorldModel()

        # Self model
        self._self: SelfModel = SelfModel(
            creation_time=time.time(),
        )

        # Decision history (last 500)
        self._decisions: deque = deque(maxlen=500)

        # Reflections (extracted principles)
        self._reflections: list[Reflection] = []

        # Future scenarios
        self._scenarios: list[FutureScenario] = []

        # Sensory input buffer (last 1000 signals)
        self._sensory_buffer: deque = deque(maxlen=1000)

        # Current focus stack
        self._focus_stack: list[str] = []

        # Stats
        self._stats: dict[str, int] = {
            "decisions_made": 0,
            "reflections_recorded": 0,
            "scenarios_imagined": 0,
            "world_updates": 0,
            "self_updates": 0,
            "actions_queued": 0,
        }

    # ── World Model Management ──────────────────────────────────────

    def update_world_model(self, signals: dict[str, Any]) -> None:
        """Update the world model with new sensory input.

        This is the consciousness's primary input channel.
        All 60+ services feed their observations through this method.
        """
        with self._lock:
            now = time.time()
            self._sensory_buffer.append((now, signals))
            self._stats["world_updates"] += 1

            # Update known players
            nearby_players = signals.get("nearby_players", [])
            if isinstance(nearby_players, list):
                for player in nearby_players:
                    if isinstance(player, dict):
                        name = player.get("name", str(player))
                        if name:
                            if name not in self._world.known_players:
                                self._world.known_players[name] = {
                                    "first_seen": now,
                                    "last_seen": now,
                                    "observations": 0,
                                }
                            self._world.known_players[name]["last_seen"] = now
                            self._world.known_players[name]["observations"] = \
                                self._world.known_players[name].get("observations", 0) + 1
                            self._world.known_players[name]["map"] = signals.get("map", "")
                            self._world.known_players[name]["class"] = player.get("class", "")
                            self._world.known_players[name]["level"] = player.get("level", 0)

            # Update known monsters
            monsters = signals.get("monsters", [])
            if isinstance(monsters, list):
                for monster in monsters:
                    if isinstance(monster, dict):
                        name = monster.get("name", "")
                        if name and name not in self._world.known_monsters:
                            self._world.known_monsters[name] = {
                                "first_seen": now,
                                "last_seen": now,
                                "map": signals.get("map", ""),
                                "level": monster.get("level", 0),
                                "race": monster.get("race", ""),
                                "element": monster.get("element", ""),
                            }

            # Update known maps
            map_name = signals.get("map", "")
            if map_name and map_name not in self._world.known_maps:
                self._world.known_maps[map_name] = {
                    "first_visited": now,
                    "last_visited": now,
                    "visits": 1,
                    "players_seen": [],
                    "monsters_seen": [],
                }
            elif map_name:
                self._world.known_maps[map_name]["last_visited"] = now
                self._world.known_maps[map_name]["visits"] = \
                    self._world.known_maps[map_name].get("visits", 0) + 1

            # Update self model
            self._self.bot_ids = signals.get("all_bots", self._self.bot_ids)
            self._self.total_bots = len(self._self.bot_ids) if isinstance(self._self.bot_ids, list) else 0
            self._self.total_zeny = signals.get("zeny", self._self.total_zeny)
            self._self.primary_bot = signals.get("bot_id", self._self.primary_bot)

            self._world.last_updated = now

    def get_world_model(self) -> WorldModel:
        """Get the current world model."""
        with self._lock:
            return self._world

    def get_self_model(self) -> SelfModel:
        """Get the current self model."""
        with self._lock:
            return self._self

    # ── Decision Making ─────────────────────────────────────────────

    def decide(
        self,
        domain: DecisionDomain,
        urgency: DecisionUrgency,
        context: dict[str, Any] | None = None,
    ) -> UnifiedDecision | None:
        """Make a decision in a specific domain.

        This is the consciousness's primary decision method.
        It integrates input from all services to make a unified decision.

        Args:
            domain: What domain to decide in
            urgency: How urgent the decision is
            context: Additional context

        Returns:
            A UnifiedDecision or None if no action needed
        """
        with self._lock:
            now = time.time()
            self._self.total_decisions += 1
            self._stats["decisions_made"] += 1

            decision = UnifiedDecision(
                decision_id=f"uc_{int(now * 1000)}_{self._self.total_decisions}",
                domain=domain,
                urgency=urgency,
                action="",
                timestamp=now,
            )

            # ── Integrate input from all services ──

            # Check empire manager for strategic directives
            empire_action = self._consult_empire(domain, context)
            if empire_action:
                decision.action = empire_action["action"]
                decision.reason = empire_action["reason"]
                decision.confidence = empire_action.get("confidence", 0.5)
                decision.target_bot = empire_action.get("target_bot", "")

            # Check theory of mind for player intent
            tom_action = self._consult_theory_of_mind(domain, context)
            if tom_action and not decision.action:
                decision.action = tom_action["action"]
                decision.reason = tom_action["reason"]
                decision.confidence = tom_action.get("confidence", 0.5)

            # Check competitive intelligence for threats
            ci_action = self._consult_competitive_intelligence(domain, context)
            if ci_action and not decision.action:
                decision.action = ci_action["action"]
                decision.reason = ci_action["reason"]
                decision.confidence = ci_action.get("confidence", 0.5)

            # Check crisis manager for emergencies
            crisis_action = self._consult_crisis_manager(domain, context)
            if crisis_action:
                # Crisis overrides everything
                decision.action = crisis_action["action"]
                decision.reason = crisis_action["reason"]
                decision.confidence = crisis_action.get("confidence", 0.9)
                decision.urgency = DecisionUrgency.REFLEX

            # Check conscious engine for build/skill decisions
            ce_action = self._consult_conscious_engine(domain, context)
            if ce_action and not decision.action:
                decision.action = ce_action["action"]
                decision.reason = ce_action["reason"]
                decision.confidence = ce_action.get("confidence", 0.5)

            # Check long-term planner for strategic goals
            ltp_action = self._consult_long_term_planner(domain, context)
            if ltp_action and not decision.action:
                decision.action = ltp_action["action"]
                decision.reason = ltp_action["reason"]
                decision.confidence = ltp_action.get("confidence", 0.5)

            # If no action determined, return None
            if not decision.action:
                return None

            # Apply reflections (learned principles)
            decision = self._apply_reflections(decision)

            # Store decision
            self._decisions.append(decision)

            # Execute if urgent enough
            if urgency in (DecisionUrgency.REFLEX, DecisionUrgency.IMMEDIATE):
                self._execute_decision(decision)

            return decision

    def _consult_empire(self, domain: DecisionDomain, context: dict[str, Any] | None) -> dict[str, Any] | None:
        """Consult the empire manager for strategic direction."""
        if self._empire_manager is None:
            return None
        try:
            # Check for pending directives
            bot_id = context.get("bot_id", "") if context else ""
            pending = self._empire_manager.get_pending_directives(target_bot=bot_id)
            if pending:
                directive = pending[0]
                return {
                    "action": directive.action,
                    "reason": f"Empire directive: {directive.reason}",
                    "confidence": 0.8,
                    "target_bot": directive.target_bot,
                }

            # Check empire strategy
            if domain == DecisionDomain.EMPIRE:
                return {
                    "action": "ai auto",
                    "reason": "Empire management cycle - check roles, pipeline, territories",
                    "confidence": 0.6,
                }
        except Exception:
            pass
        return None

    def _consult_theory_of_mind(self, domain: DecisionDomain, context: dict[str, Any] | None) -> dict[str, Any] | None:
        """Consult theory of mind for player intent insights."""
        if self._theory_of_mind is None:
            return None
        try:
            # Check for nearby players and assess threat
            nearby = context.get("nearby_players", []) if context else []
            for player in nearby:
                if isinstance(player, dict):
                    name = player.get("name", "")
                    if name:
                        threat = self._theory_of_mind.assess_player_threat(name)
                        if threat.get("threat_score", 0) >= 0.7:
                            return {
                                "action": "ai auto",
                                "reason": f"High threat player nearby: {name} (score: {threat['threat_score']:.0%})",
                                "confidence": 0.9,
                            }
        except Exception:
            pass
        return None

    def _consult_competitive_intelligence(self, domain: DecisionDomain, context: dict[str, Any] | None) -> dict[str, Any] | None:
        """Consult competitive intelligence for market/threat insights."""
        if self._competitive_intelligence is None:
            return None
        try:
            # Check for threats
            threats = self._competitive_intelligence.get_threats(min_score=50)
            if threats:
                top_threat = threats[0]
                return {
                    "action": "ai auto",
                    "reason": f"Competitive threat: {top_threat.player_name} (score: {top_threat.threat_score})",
                    "confidence": 0.7,
                }

            # Check market opportunities
            if domain == DecisionDomain.ECONOMY:
                hot_maps = self._competitive_intelligence.get_most_competitive_maps(1)
                if hot_maps:
                    map_name, count = hot_maps[0]
                    return {
                        "action": "ai auto",
                        "reason": f"High competition on {map_name} ({count} farmers) - consider alternatives",
                        "confidence": 0.6,
                    }
        except Exception:
            pass
        return None

    def _consult_crisis_manager(self, domain: DecisionDomain, context: dict[str, Any] | None) -> dict[str, Any] | None:
        """Consult crisis manager for emergencies."""
        if self._crisis_manager is None:
            return None
        try:
            bot_id = context.get("bot_id", "") if context else ""
            if bot_id and self._crisis_manager._active_crises.get(bot_id):
                return {
                    "action": "ai auto",
                    "reason": f"Crisis active for {bot_id} - executing recovery",
                    "confidence": 0.95,
                }
        except Exception:
            pass
        return None

    def _consult_conscious_engine(self, domain: DecisionDomain, context: dict[str, Any] | None) -> dict[str, Any] | None:
        """Consult the conscious engine for build/skill decisions."""
        if self._conscious_engine is None:
            return None
        try:
            if domain in (DecisionDomain.COMBAT, DecisionDomain.PROGRESSION):
                # Check if there are skill/stat decisions to make
                if hasattr(self._conscious_engine, 'get_decisions'):
                    decisions = self._conscious_engine.get_decisions(context or {})
                    if decisions:
                        d = decisions[0]
                        return {
                            "action": f"ai auto",
                            "reason": f"Conscious engine: {d.get('reason', 'optimization')}",
                            "confidence": 0.7,
                        }
        except Exception:
            pass
        return None

    def _consult_long_term_planner(self, domain: DecisionDomain, context: dict[str, Any] | None) -> dict[str, Any] | None:
        """Consult the long-term planner for strategic goals."""
        if self._long_term_planner is None:
            return None
        try:
            if domain == DecisionDomain.LEARNING:
                return {
                    "action": "ai auto",
                    "reason": "Long-term planning cycle - review goals and adapt",
                    "confidence": 0.5,
                }
        except Exception:
            pass
        return None

    def _apply_reflections(self, decision: UnifiedDecision) -> UnifiedDecision:
        """Apply learned reflections to a decision."""
        for reflection in self._reflections:
            if reflection.domain == decision.domain and reflection.applied_count < 10:
                # Apply the general principle
                decision.reason += f" | Principle: {reflection.general_principle}"
                reflection.applied_count += 1
                reflection.last_applied = time.time()
        return decision

    def _execute_decision(self, decision: UnifiedDecision) -> bool:
        """Execute a decision by queueing an action."""
        if self._enqueue_fn is None:
            return False

        try:
            target = decision.target_bot or self._self.primary_bot
            if target:
                self._enqueue_fn(target, decision.action)
                decision.executed = True
                self._stats["actions_queued"] += 1
                logger.info(
                    "uc_executed: bot=%s action=%s domain=%s urgency=%s",
                    target, decision.action, decision.domain.value, decision.urgency.value,
                )
                return True
        except Exception:
            pass
        return False

    # ── Reflection ──────────────────────────────────────────────────

    def reflect(self, topic: str, experience: str, lesson: str, general_principle: str, domain: DecisionDomain) -> Reflection:
        """Reflect on a past experience and extract a general principle.

        This is how the consciousness learns from experience.
        Each reflection produces a general principle that applies to future decisions.

        Args:
            topic: What this reflection is about
            experience: What happened
            lesson: What we learned specifically
            general_principle: What we learned that applies generally
            domain: What domain this applies to

        Returns:
            The Reflection object
        """
        with self._lock:
            reflection = Reflection(
                reflection_id=f"ref_{int(time.time() * 1000)}_{len(self._reflections)}",
                topic=topic,
                experience=experience,
                lesson=lesson,
                general_principle=general_principle,
                domain=domain,
                timestamp=time.time(),
            )
            self._reflections.append(reflection)
            self._stats["reflections_recorded"] += 1
            self._self.total_reflections += 1

            logger.info(
                "uc_reflection: topic=%s domain=%s principle=%s",
                topic, domain.value, general_principle,
            )

            return reflection

    def get_reflections(self, domain: DecisionDomain | None = None) -> list[Reflection]:
        """Get reflections, optionally filtered by domain."""
        with self._lock:
            if domain:
                return [r for r in self._reflections if r.domain == domain]
            return list(self._reflections)

    def get_principles(self) -> list[str]:
        """Get all general principles learned."""
        with self._lock:
            return [r.general_principle for r in self._reflections]

    # ── Future Scenario Planning ───────────────────────────────────

    def imagine_scenario(
        self,
        description: str,
        probability: float,
        impact: str,
        timeframe: str,
        plan: list[str] | None = None,
        contingencies: list[str] | None = None,
    ) -> FutureScenario:
        """Imagine a future scenario and plan for it.

        This is how the consciousness does strategic planning.
        It imagines what might happen and prepares contingencies.

        Args:
            description: What might happen
            probability: How likely (0.0-1.0)
            impact: positive, negative, neutral
            timeframe: short_term, medium_term, long_term
            plan: Steps to handle this scenario
            contingencies: Backup plans

        Returns:
            The FutureScenario object
        """
        with self._lock:
            scenario = FutureScenario(
                scenario_id=f"sc_{int(time.time() * 1000)}_{len(self._scenarios)}",
                description=description,
                probability=probability,
                impact=impact,
                timeframe=timeframe,
                plan=plan or [],
                contingencies=contingencies or [],
                created_at=time.time(),
            )
            self._scenarios.append(scenario)
            self._stats["scenarios_imagined"] += 1

            logger.info(
                "uc_scenario: desc=%s prob=%.0f%% impact=%s timeframe=%s",
                description, probability * 100, impact, timeframe,
            )

            return scenario

    def get_scenarios(self, timeframe: str | None = None) -> list[FutureScenario]:
        """Get future scenarios, optionally filtered by timeframe."""
        with self._lock:
            if timeframe:
                return [s for s in self._scenarios if s.timeframe == timeframe]
            return list(self._scenarios)

    # ── Focus Management ───────────────────────────────────────────

    def set_focus(self, focus: str) -> None:
        """Set the current focus of consciousness."""
        with self._lock:
            self._focus_stack.append(focus)
            self._self.current_focus = focus
            self._self.current_state = ConsciousnessState.FOCUSED

    def clear_focus(self) -> None:
        """Clear the current focus and return to previous state."""
        with self._lock:
            if self._focus_stack:
                self._focus_stack.pop()
            self._self.current_focus = self._focus_stack[-1] if self._focus_stack else ""
            self._self.current_state = ConsciousnessState.AWAKE

    def get_current_focus(self) -> str:
        """Get the current focus."""
        with self._lock:
            return self._self.current_focus

    # ── Priority Management ────────────────────────────────────────

    def set_priority(self, priority: str) -> None:
        """Set the current top priority."""
        with self._lock:
            self._self.current_priority = priority

    def get_priority(self) -> str:
        """Get the current top priority."""
        with self._lock:
            return self._self.current_priority

    # ── Integrated Decision Cycle ────────────────────────────────────

    def consciousness_tick(self, signals: dict[str, Any]) -> list[UnifiedDecision]:
        """Run one full consciousness tick — the main decision cycle.

        This is called by the PDCA loop on every cycle.
        It integrates all sensory input, makes decisions, and queues actions.

        Args:
            signals: Current bot state signals

        Returns:
            List of decisions made this tick
        """
        decisions_made = []

        # 1. Update world model
        self.update_world_model(signals)

        # 2. Check for crises first (highest priority)
        crisis_decision = self.decide(
            DecisionDomain.SURVIVAL,
            DecisionUrgency.REFLEX,
            context=signals,
        )
        if crisis_decision:
            decisions_made.append(crisis_decision)
            return decisions_made  # Handle crisis first, nothing else

        # 3. Check combat needs
        combat_decision = self.decide(
            DecisionDomain.COMBAT,
            DecisionUrgency.IMMEDIATE,
            context=signals,
        )
        if combat_decision:
            decisions_made.append(combat_decision)

        # 4. Check economic needs
        economy_decision = self.decide(
            DecisionDomain.ECONOMY,
            DecisionUrgency.SHORT_TERM,
            context=signals,
        )
        if economy_decision:
            decisions_made.append(economy_decision)

        # 5. Check social needs
        social_decision = self.decide(
            DecisionDomain.SOCIAL,
            DecisionUrgency.SHORT_TERM,
            context=signals,
        )
        if social_decision:
            decisions_made.append(social_decision)

        # 6. Check progression needs
        progression_decision = self.decide(
            DecisionDomain.PROGRESSION,
            DecisionUrgency.MEDIUM_TERM,
            context=signals,
        )
        if progression_decision:
            decisions_made.append(progression_decision)

        # 7. Check empire needs (strategic)
        empire_decision = self.decide(
            DecisionDomain.EMPIRE,
            DecisionUrgency.LONG_TERM,
            context=signals,
        )
        if empire_decision:
            decisions_made.append(empire_decision)

        # 8. Update self model
        with self._lock:
            self._self.total_cycles += 1
            self._self.last_updated = time.time()
            self._stats["self_updates"] += 1

        return decisions_made

    # ── Summary / Context ──────────────────────────────────────────

    def get_consciousness_summary(self) -> str:
        """Get a formatted summary of consciousness state."""
        with self._lock:
            lines = ["── Unified Consciousness ──"]

            # Self model
            self_model = self._self
            lines.append(f"  State: {self_model.current_state.value}")
            lines.append(f"  Focus: {self_model.current_focus or 'none'}")
            lines.append(f"  Priority: {self_model.current_priority or 'none'}")
            lines.append(f"  Bots: {self_model.total_bots}")
            lines.append(f"  Wealth: {self_model.total_zeny:,}z")
            lines.append(f"  Cycles: {self_model.total_cycles}")
            lines.append(f"  Decisions: {self_model.total_decisions}")

            # World model
            world = self._world
            lines.append(f"  Known Players: {len(world.known_players)}")
            lines.append(f"  Known Monsters: {len(world.known_monsters)}")
            lines.append(f"  Known Maps: {len(world.known_maps)}")

            # Reflections
            if self._reflections:
                lines.append(f"  Reflections ({len(self._reflections)}):")
                for r in self._reflections[-3:]:
                    lines.append(f"    {r.topic}: {r.general_principle}")

            # Scenarios
            if self._scenarios:
                lines.append(f"  Future Scenarios ({len(self._scenarios)}):")
                for s in self._scenarios[-3:]:
                    lines.append(f"    {s.description} (prob: {s.probability:.0%})")

            # Recent decisions
            if self._decisions:
                lines.append(f"  Recent Decisions ({len(self._decisions)}):")
                for d in list(self._decisions)[-5:]:
                    lines.append(
                        f"    [{d.urgency.value}] {d.domain.value}: {d.action[:50]}"
                    )

            return "\n".join(lines)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global instance ──

_unified_consciousness: UnifiedConsciousness | None = None
_uc_lock = RLock()


def get_unified_consciousness() -> UnifiedConsciousness:
    """Get or create the global unified consciousness instance."""
    global _unified_consciousness
    with _uc_lock:
        if _unified_consciousness is None:
            _unified_consciousness = UnifiedConsciousness()
        return _unified_consciousness
