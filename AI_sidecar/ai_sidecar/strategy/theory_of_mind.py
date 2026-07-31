"""
Theory of Mind — models other players' intentions, predicts behavior, tracks relationships.

A top player doesn't just react to what other players do. They understand WHY.
They model other players' mental states: goals, beliefs, desires, intentions.
They predict what others will do next and plan accordingly.

This module implements:
1. Player intention modeling (what are they trying to do?)
2. Behavior prediction (what will they do next?)
3. Goal inference (why are they doing what they're doing?)
4. Relationship tracking (friend, rival, ally, enemy, neutral)
5. Pattern detection (farming schedule, WOE participation, market activity)
6. Deception detection (is this player lying or manipulating?)
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PlayerIntention(Enum):
    """What a player is likely trying to do."""
    FARMING = "farming"
    LEVELING = "leveling"
    SHOPPING = "shopping"
    PVP = "pvp"
    WOE = "woe"
    SCOUTING = "scouting"
    TRADING = "trading"
    CRAFTING = "crafting"
    QUESTING = "questing"
    SOCIALIZING = "socializing"
    GM_MONITORING = "gm_monitoring"
    BOTTING = "botting"
    UNKNOWN = "unknown"


class PlayerDisposition(Enum):
    """How a player feels about us."""
    FRIENDLY = "friendly"
    NEUTRAL = "neutral"
    HOSTILE = "hostile"
    ALLY = "ally"
    RIVAL = "rival"
    ENEMY = "enemy"
    TRUSTED = "trusted"
    SUSPICIOUS = "suspicious"
    UNKNOWN = "unknown"


class BehaviorPattern(Enum):
    """Types of behavioral patterns we can detect."""
    FARMING_SCHEDULE = "farming_schedule"
    WOE_PARTICIPATION = "woe_participation"
    MARKET_ACTIVITY = "market_activity"
    LEVELING_SPEED = "leveling_speed"
    SOCIAL_HOURS = "social_hours"
    PVP_TIMES = "pvp_times"
    GUILD_ACTIVITY = "guild_activity"
    TRADING_PARTNERS = "trading_partners"
    MAP_ROTATION = "map_rotation"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PlayerObservation:
    """An observation of a player's action."""
    player_name: str
    action: str  # what they did
    map_name: str = ""
    class_name: str = ""
    level: int = 0
    timestamp: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5  # how sure we are about this observation


@dataclass
class IntentionEstimate:
    """Our estimate of a player's current intention."""
    player_name: str
    primary_intention: PlayerIntention = PlayerIntention.UNKNOWN
    secondary_intention: PlayerIntention = PlayerIntention.UNKNOWN
    confidence: float = 0.0  # 0.0-1.0
    evidence: list[str] = field(default_factory=list)
    last_updated: float = 0.0


@dataclass
class BehaviorPrediction:
    """A prediction about what a player will do next."""
    player_name: str
    predicted_action: str
    predicted_map: str = ""
    predicted_timeframe: str = ""  # immediate, short_term, long_term
    probability: float = 0.0  # 0.0-1.0
    reasoning: str = ""
    expires_at: float = 0.0


@dataclass
class PlayerRelationship:
    """Tracked relationship with another player."""
    player_name: str
    disposition: PlayerDisposition = PlayerDisposition.UNKNOWN
    trust_level: float = 0.5  # 0.0 (distrust) - 1.0 (complete trust)
    familiarity: float = 0.0  # 0.0 (stranger) - 1.0 (known well)
    interaction_count: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    last_interaction: float = 0.0
    first_seen: float = 0.0
    notes: str = ""
    tags: list[str] = field(default_factory=list)  # e.g., ["farmer", "pker", "merchant", "guild_member"]


@dataclass
class DetectedPattern:
    """A detected behavioral pattern for a player."""
    player_name: str
    pattern_type: BehaviorPattern
    description: str
    confidence: float = 0.0
    first_detected: float = 0.0
    last_observed: float = 0.0
    data_points: int = 0
    prediction: str = ""  # what we predict based on this pattern


@dataclass
class DeceptionIndicator:
    """An indicator that a player might be deceptive."""
    player_name: str
    indicator_type: str  # inconsistent_story, unusual_behavior, too_helpful, probing_questions
    description: str
    severity: float = 0.5  # 0.0-1.0
    timestamp: float = 0.0
    evidence: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TheoryOfMind
# ---------------------------------------------------------------------------


class TheoryOfMind:
    """Models other players' intentions, predicts behavior, tracks relationships.

    Wires into:
      - competitive_intelligence.py: player tracking and threat assessment
      - social_engine.py: relationship management
      - empire_manager.py: strategic decision-making
      - PDCA loop: informs decisions with player intent data
    """

    def __init__(
        self,
        competitive_intelligence: Any = None,
        social_engine: Any = None,
        empire_manager: Any = None,
    ) -> None:
        self._lock = RLock()

        # Wired dependencies
        self._competitive_intelligence = competitive_intelligence
        self._social_engine = social_engine
        self._empire_manager = empire_manager

        # Player observations: player_name -> deque of PlayerObservation
        self._observations: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))

        # Intention estimates: player_name -> IntentionEstimate
        self._intentions: dict[str, IntentionEstimate] = {}

        # Behavior predictions: player_name -> list of BehaviorPrediction
        self._predictions: dict[str, list[BehaviorPrediction]] = defaultdict(list)

        # Relationships: player_name -> PlayerRelationship
        self._relationships: dict[str, PlayerRelationship] = {}

        # Detected patterns: player_name -> list of DetectedPattern
        self._patterns: dict[str, list[DetectedPattern]] = defaultdict(list)

        # Deception indicators: player_name -> list of DeceptionIndicator
        self._deception_indicators: dict[str, list[DeceptionIndicator]] = defaultdict(list)

        # Action-outcome memory: (player, action) -> outcome
        self._action_outcomes: dict[tuple[str, str], str] = {}

        # Stats
        self._stats: dict[str, int] = {
            "observations_recorded": 0,
            "intentions_inferred": 0,
            "predictions_made": 0,
            "predictions_correct": 0,
            "predictions_wrong": 0,
            "patterns_detected": 0,
            "deception_indicators": 0,
            "relationships_tracked": 0,
        }

    # ── Observation Recording ───────────────────────────────────────

    def observe_player(
        self,
        player_name: str,
        action: str,
        map_name: str = "",
        class_name: str = "",
        level: int = 0,
        context: dict[str, Any] | None = None,
        confidence: float = 0.5,
    ) -> None:
        """Record an observation of a player's action.

        Args:
            player_name: The player observed
            action: What they did (e.g., "farming", "moving", "trading", "pvp", "idle")
            map_name: Where they were
            class_name: Their class
            level: Their level
            context: Additional context
            confidence: How sure we are
        """
        with self._lock:
            now = time.time()
            observation = PlayerObservation(
                player_name=player_name,
                action=action,
                map_name=map_name,
                class_name=class_name,
                level=level,
                timestamp=now,
                context=context or {},
                confidence=confidence,
            )
            self._observations[player_name].append(observation)
            self._stats["observations_recorded"] += 1

            # Update relationship tracking
            self._ensure_relationship(player_name)
            rel = self._relationships[player_name]
            rel.last_interaction = now
            rel.interaction_count += 1

            # Update familiarity (decaying average)
            rel.familiarity = min(1.0, rel.familiarity + 0.05)

            # Infer intention from this observation
            self._infer_intention(player_name, observation)

            # Detect patterns
            self._detect_patterns(player_name)

            # Update competitive intelligence
            if self._competitive_intelligence is not None:
                try:
                    if action == "farming":
                        self._competitive_intelligence.observe_farming(
                            player_name=player_name,
                            map_name=map_name,
                            monster_name=context.get("monster", "") if context else "",
                        )
                    elif action == "trading":
                        self._competitive_intelligence.observe_market(
                            player_name=player_name,
                            item_name=context.get("item", "") if context else "",
                            price=context.get("price", 0) if context else 0,
                            quantity=context.get("quantity", 1) if context else 1,
                            shop_location=map_name,
                        )
                    elif action in ("leveling", "changing_job"):
                        self._competitive_intelligence.track_player_build(
                            player_name=player_name,
                            job_class=class_name,
                            base_level=level,
                        )
                except Exception:
                    pass

    def _ensure_relationship(self, player_name: str) -> PlayerRelationship:
        """Get or create a relationship entry for a player."""
        if player_name not in self._relationships:
            self._relationships[player_name] = PlayerRelationship(
                player_name=player_name,
                first_seen=time.time(),
            )
            self._stats["relationships_tracked"] += 1
        return self._relationships[player_name]

    # ── Intention Inference ────────────────────────────────────────

    def _infer_intention(self, player_name: str, observation: PlayerObservation) -> None:
        """Infer a player's intention from an observation."""
        action = observation.action.lower()
        map_name = observation.map_name.lower()
        context = observation.context or {}

        intention = PlayerIntention.UNKNOWN
        evidence = []

        # Map actions to intentions
        if action in ("farming", "killing", "hunting"):
            intention = PlayerIntention.FARMING
            evidence.append(f"Observed {action} on {map_name}")
        elif action in ("leveling", "grinding", "training"):
            intention = PlayerIntention.LEVELING
            evidence.append(f"Observed {action} on {map_name}")
        elif action in ("trading", "vending", "buying", "selling"):
            intention = PlayerIntention.TRADING
            evidence.append(f"Observed {action} on {map_name}")
        elif action in ("crafting", "producing", "making"):
            intention = PlayerIntention.CRAFTING
            evidence.append(f"Observed {action}")
        elif action in ("pvp", "pking", "attacking_player"):
            intention = PlayerIntention.PVP
            evidence.append(f"Observed {action} on {map_name}")
        elif action in ("woe", "castle", "emperium"):
            intention = PlayerIntention.WOE
            evidence.append(f"Observed {action} on {map_name}")
        elif action in ("scouting", "exploring", "wandering"):
            intention = PlayerIntention.SCOUTING
            evidence.append(f"Observed {action} on {map_name}")
        elif action in ("quest", "questing", "npc_talk"):
            intention = PlayerIntention.QUESTING
            evidence.append(f"Observed {action}")
        elif action in ("chat", "social", "party_invite"):
            intention = PlayerIntention.SOCIALIZING
            evidence.append(f"Observed {action}")
        elif action in ("gm", "warp", "hide", "invisible"):
            intention = PlayerIntention.GM_MONITORING
            evidence.append(f"Observed {action} - possible GM activity")
        elif action in ("botting", "auto", "macro"):
            intention = PlayerIntention.BOTTING
            evidence.append(f"Observed {action} - possible bot")

        # Map-based inference
        if "shop" in map_name or "market" in map_name or "prt_in" in map_name:
            if intention == PlayerIntention.UNKNOWN:
                intention = PlayerIntention.SHOPPING
                evidence.append(f"On shopping map {map_name}")
        elif "pvp" in map_name or "arena" in map_name:
            if intention == PlayerIntention.UNKNOWN:
                intention = PlayerIntention.PVP
                evidence.append(f"On PVP map {map_name}")
        elif "dungeon" in map_name or "dun" in map_name:
            if intention == PlayerIntention.UNKNOWN:
                intention = PlayerIntention.FARMING
                evidence.append(f"On dungeon map {map_name}")

        # Update intention estimate
        with self._lock:
            estimate = self._intentions.get(player_name)
            if estimate is None:
                estimate = IntentionEstimate(
                    player_name=player_name,
                    primary_intention=intention,
                    last_updated=time.time(),
                )
                self._intentions[player_name] = estimate
                self._stats["intentions_inferred"] += 1
            else:
                # Update with decay: recent observations weigh more
                time_factor = max(0.1, 1.0 - (time.time() - estimate.last_updated) / 3600)
                if estimate.primary_intention == intention:
                    estimate.confidence = min(1.0, estimate.confidence + 0.1 * time_factor)
                else:
                    estimate.confidence = max(0.1, estimate.confidence - 0.05 * time_factor)
                    if estimate.confidence < 0.3:
                        estimate.secondary_intention = estimate.primary_intention
                        estimate.primary_intention = intention
                        estimate.confidence = 0.5

                estimate.last_updated = time.time()

            estimate.evidence.extend(evidence)
            if len(estimate.evidence) > 20:
                estimate.evidence = estimate.evidence[-20:]

    def get_intention(self, player_name: str) -> IntentionEstimate | None:
        """Get our estimate of a player's current intention."""
        with self._lock:
            estimate = self._intentions.get(player_name)
            if estimate and time.time() - estimate.last_updated > 3600:
                # Stale estimate - confidence decays
                estimate.confidence = max(0.1, estimate.confidence - 0.2)
            return estimate

    def get_players_by_intention(self, intention: PlayerIntention) -> list[str]:
        """Get all players we believe have a specific intention."""
        with self._lock:
            result = []
            for name, estimate in self._intentions.items():
                if estimate.primary_intention == intention and estimate.confidence > 0.3:
                    result.append(name)
            return result

    # ── Behavior Prediction ─────────────────────────────────────────

    def predict_next_action(self, player_name: str) -> BehaviorPrediction | None:
        """Predict what a player will do next based on observations.

        Uses pattern matching and intention inference to make predictions.
        """
        with self._lock:
            observations = list(self._observations.get(player_name, []))
            if not observations:
                return None

            intention = self._intentions.get(player_name)
            patterns = self._patterns.get(player_name, [])
            now = time.time()

            prediction = BehaviorPrediction(
                player_name=player_name,
                predicted_action="unknown",
                probability=0.0,
                expires_at=now + 300,  # Predictions expire in 5 minutes
            )

            # Predict based on current intention
            if intention:
                if intention.primary_intention == PlayerIntention.FARMING:
                    # Predict they'll stay on the same map farming
                    last_obs = observations[-1]
                    prediction.predicted_action = f"continue farming on {last_obs.map_name}"
                    prediction.predicted_map = last_obs.map_name
                    prediction.predicted_timeframe = "short_term"
                    prediction.probability = 0.7 * intention.confidence
                    prediction.reasoning = f"Player has been farming. {intention.confidence:.0%} confident."

                elif intention.primary_intention == PlayerIntention.TRADING:
                    prediction.predicted_action = "set up shop or browse vendors"
                    prediction.predicted_timeframe = "short_term"
                    prediction.probability = 0.6 * intention.confidence
                    prediction.reasoning = "Player is in trading mode."

                elif intention.primary_intention == PlayerIntention.PVP:
                    prediction.predicted_action = "look for PVP targets"
                    prediction.predicted_timeframe = "immediate"
                    prediction.probability = 0.8 * intention.confidence
                    prediction.reasoning = "Player is in PVP mode - high threat."

                elif intention.primary_intention == PlayerIntention.WOE:
                    prediction.predicted_action = "participate in WOE"
                    prediction.predicted_timeframe = "short_term"
                    prediction.probability = 0.9 * intention.confidence
                    prediction.reasoning = "Player is in WOE mode."

            # Refine with pattern data
            for pattern in patterns:
                if pattern.pattern_type == BehaviorPattern.FARMING_SCHEDULE and pattern.prediction:
                    prediction.predicted_action = pattern.prediction
                    prediction.probability = max(prediction.probability, pattern.confidence * 0.8)
                    prediction.reasoning += f" Pattern: {pattern.description}"
                elif pattern.pattern_type == BehaviorPattern.WOE_PARTICIPATION and pattern.prediction:
                    if "woe" in pattern.prediction.lower():
                        prediction.predicted_action = pattern.prediction
                        prediction.predicted_timeframe = "short_term"
                        prediction.probability = max(prediction.probability, pattern.confidence * 0.9)
                        prediction.reasoning += f" WOE pattern: {pattern.description}"

            # Store prediction
            self._predictions[player_name].append(prediction)
            self._stats["predictions_made"] += 1

            return prediction

    def record_prediction_outcome(self, player_name: str, was_correct: bool) -> None:
        """Record whether a prediction was correct (for learning)."""
        with self._lock:
            if was_correct:
                self._stats["predictions_correct"] += 1
            else:
                self._stats["predictions_wrong"] += 1

    def get_predictions(self, player_name: str) -> list[BehaviorPrediction]:
        """Get active predictions for a player."""
        with self._lock:
            now = time.time()
            return [
                p for p in self._predictions.get(player_name, [])
                if p.expires_at > now
            ]

    # ── Pattern Detection ──────────────────────────────────────────

    def _detect_patterns(self, player_name: str) -> None:
        """Detect behavioral patterns from observations."""
        observations = list(self._observations.get(player_name, []))
        if len(observations) < 5:
            return

        now = time.time()

        # Detect farming schedule
        farming_obs = [o for o in observations if o.action.lower() in ("farming", "hunting", "killing")]
        if len(farming_obs) >= 3:
            # Check if there's a time-based pattern
            hours = [o.timestamp for o in farming_obs]
            if hours:
                # Simple pattern: check if they farm at consistent times
                hour_of_day = [time.localtime(h).tm_hour for h in hours[-10:]]
                if len(set(hour_of_day)) <= 3:  # Farms at consistent hours
                    common_hours = sorted(set(hour_of_day))
                    pattern = DetectedPattern(
                        player_name=player_name,
                        pattern_type=BehaviorPattern.FARMING_SCHEDULE,
                        description=f"Farms at hours: {common_hours}",
                        confidence=min(0.9, 0.3 + len(farming_obs) * 0.05),
                        first_detected=now,
                        last_observed=now,
                        data_points=len(farming_obs),
                        prediction=f"Likely to farm around hour {common_hours[0]}",
                    )
                    self._add_pattern(player_name, pattern)

        # Detect WOE participation
        woe_obs = [o for o in observations if o.action.lower() in ("woe", "castle", "emperium")]
        if len(woe_obs) >= 2:
            pattern = DetectedPattern(
                player_name=player_name,
                pattern_type=BehaviorPattern.WOE_PARTICIPATION,
                description="Participates in WOE",
                confidence=min(0.9, 0.4 + len(woe_obs) * 0.1),
                first_detected=now,
                last_observed=now,
                data_points=len(woe_obs),
                prediction="Will participate in next WOE",
            )
            self._add_pattern(player_name, pattern)

        # Detect market activity
        market_obs = [o for o in observations if o.action.lower() in ("trading", "vending", "selling")]
        if len(market_obs) >= 3:
            pattern = DetectedPattern(
                player_name=player_name,
                pattern_type=BehaviorPattern.MARKET_ACTIVITY,
                description="Active in market/trading",
                confidence=min(0.8, 0.3 + len(market_obs) * 0.05),
                first_detected=now,
                last_observed=now,
                data_points=len(market_obs),
                prediction="Will continue trading activities",
            )
            self._add_pattern(player_name, pattern)

        # Detect leveling speed
        level_obs = [o for o in observations if o.action.lower() in ("leveling", "grinding")]
        if len(level_obs) >= 3:
            levels = [o.level for o in level_obs if o.level > 0]
            if len(levels) >= 2:
                level_gain = levels[-1] - levels[0]
                time_span = level_obs[-1].timestamp - level_obs[0].timestamp
                if time_span > 0:
                    levels_per_hour = level_gain / (time_span / 3600)
                    pattern = DetectedPattern(
                        player_name=player_name,
                        pattern_type=BehaviorPattern.LEVELING_SPEED,
                        description=f"Leveling at {levels_per_hour:.1f} levels/hour",
                        confidence=0.6,
                        first_detected=now,
                        last_observed=now,
                        data_points=len(level_obs),
                        prediction=f"Will gain ~{levels_per_hour:.0f} levels in next hour",
                    )
                    self._add_pattern(player_name, pattern)

    def _add_pattern(self, player_name: str, pattern: DetectedPattern) -> None:
        """Add or update a detected pattern."""
        existing = self._patterns.get(player_name, [])
        for i, p in enumerate(existing):
            if p.pattern_type == pattern.pattern_type:
                # Update existing pattern
                existing[i] = pattern
                self._stats["patterns_detected"] += 1
                return
        existing.append(pattern)
        self._stats["patterns_detected"] += 1

    def get_patterns(self, player_name: str) -> list[DetectedPattern]:
        """Get detected patterns for a player."""
        with self._lock:
            return list(self._patterns.get(player_name, []))

    # ── Relationship Management ─────────────────────────────────────

    def update_relationship(
        self,
        player_name: str,
        disposition: PlayerDisposition | None = None,
        trust_delta: float = 0.0,
        interaction_type: str = "neutral",
    ) -> PlayerRelationship:
        """Update relationship with a player.

        Args:
            player_name: The player
            disposition: New disposition (or None to keep current)
            trust_delta: Change in trust (-1.0 to 1.0)
            interaction_type: positive, negative, neutral

        Returns:
            Updated relationship
        """
        with self._lock:
            rel = self._ensure_relationship(player_name)

            if disposition:
                rel.disposition = disposition

            # Update trust
            rel.trust_level = max(0.0, min(1.0, rel.trust_level + trust_delta))

            # Track interaction type
            if interaction_type == "positive":
                rel.positive_interactions += 1
            elif interaction_type == "negative":
                rel.negative_interactions += 1

            rel.last_interaction = time.time()

            # Auto-tag based on observations
            intention = self._intentions.get(player_name)
            if intention:
                tag = intention.primary_intention.value
                if tag not in rel.tags:
                    rel.tags.append(tag)

            return rel

    def get_relationship(self, player_name: str) -> PlayerRelationship | None:
        """Get relationship with a player."""
        with self._lock:
            return self._relationships.get(player_name)

    def get_players_by_disposition(self, disposition: PlayerDisposition) -> list[str]:
        """Get all players with a specific disposition."""
        with self._lock:
            return [
                name for name, rel in self._relationships.items()
                if rel.disposition == disposition
            ]

    def get_trusted_players(self, min_trust: float = 0.7) -> list[str]:
        """Get players we trust above a threshold."""
        with self._lock:
            return [
                name for name, rel in self._relationships.items()
                if rel.trust_level >= min_trust
            ]

    def get_hostile_players(self) -> list[str]:
        """Get players we consider hostile."""
        with self._lock:
            return [
                name for name, rel in self._relationships.items()
                if rel.disposition in (PlayerDisposition.HOSTILE, PlayerDisposition.ENEMY)
            ]

    # ── Deception Detection ─────────────────────────────────────────

    def record_deception_indicator(
        self,
        player_name: str,
        indicator_type: str,
        description: str,
        severity: float = 0.5,
        evidence: list[str] | None = None,
    ) -> None:
        """Record an indicator that a player might be deceptive.

        Args:
            player_name: The player
            indicator_type: inconsistent_story, unusual_behavior, too_helpful, probing_questions
            description: What happened
            severity: 0.0-1.0
            evidence: Supporting evidence
        """
        with self._lock:
            indicator = DeceptionIndicator(
                player_name=player_name,
                indicator_type=indicator_type,
                description=description,
                severity=severity,
                timestamp=time.time(),
                evidence=evidence or [],
            )
            self._deception_indicators[player_name].append(indicator)
            self._stats["deception_indicators"] += 1

            # Reduce trust
            rel = self._ensure_relationship(player_name)
            rel.trust_level = max(0.0, rel.trust_level - severity * 0.2)

            logger.info(
                "deception_indicator: player=%s type=%s severity=%.2f desc=%s",
                player_name, indicator_type, severity, description,
            )

    def get_deception_score(self, player_name: str) -> float:
        """Get a deception score for a player (0.0 = honest, 1.0 = deceptive)."""
        with self._lock:
            indicators = self._deception_indicators.get(player_name, [])
            if not indicators:
                return 0.0

            # Average severity of recent indicators
            recent = [i for i in indicators if time.time() - i.timestamp < 86400]
            if not recent:
                return 0.0

            return sum(i.severity for i in recent) / len(recent)

    # ── Goal Inference ──────────────────────────────────────────────

    def infer_player_goals(self, player_name: str) -> list[str]:
        """Infer what goals a player might have based on their behavior.

        Returns:
            List of inferred goals with confidence
        """
        goals = []
        with self._lock:
            intention = self._intentions.get(player_name)
            patterns = self._patterns.get(player_name, [])
            observations = list(self._observations.get(player_name, []))

            if not observations:
                return ["Unknown - insufficient data"]

            # Infer from intention
            if intention:
                if intention.primary_intention == PlayerIntention.FARMING:
                    goals.append(f"Likely farming for zeny/items (confidence: {intention.confidence:.0%})")
                elif intention.primary_intention == PlayerIntention.LEVELING:
                    goals.append(f"Likely leveling to {self._estimate_target_level(player_name)} (confidence: {intention.confidence:.0%})")
                elif intention.primary_intention == PlayerIntention.PVP:
                    goals.append("Likely seeking PVP combat or building a PVP character")
                elif intention.primary_intention == PlayerIntention.TRADING:
                    goals.append("Likely building wealth through trading")
                elif intention.primary_intention == PlayerIntention.WOE:
                    goals.append("Likely preparing for or participating in WOE")

            # Infer from patterns
            for pattern in patterns:
                if pattern.pattern_type == BehaviorPattern.LEVELING_SPEED:
                    goals.append(f"Leveling at {pattern.description} - may be rushing to max level")
                elif pattern.pattern_type == BehaviorPattern.MARKET_ACTIVITY:
                    goals.append("Active in economy - may be a merchant alt")

            # Infer from class
            if observations:
                latest_class = observations[-1].class_name
                if latest_class:
                    if "merchant" in latest_class.lower() or "vendor" in latest_class.lower():
                        goals.append("Merchant class - likely focused on economy")
                    elif "knight" in latest_class.lower() or "sword" in latest_class.lower():
                        goals.append("Melee class - likely focused on combat/farming")
                    elif "mage" in latest_class.lower() or "wizard" in latest_class.lower():
                        goals.append("Magic class - likely focused on AoE farming")
                    elif "priest" in latest_class.lower() or "acolyte" in latest_class.lower():
                        goals.append("Support class - likely focused on party play")

            return goals if goals else ["Unknown goals"]

    def _estimate_target_level(self, player_name: str) -> str:
        """Estimate what level a player is aiming for."""
        observations = list(self._observations.get(player_name, []))
        level_obs = [o for o in observations if o.level > 0]
        if not level_obs:
            return "max level"

        current_level = level_obs[-1].level
        if current_level < 50:
            return f"level 50+ (currently {current_level})"
        elif current_level < 70:
            return f"level 70+ (currently {current_level})"
        elif current_level < 90:
            return f"level 90+ (currently {current_level})"
        else:
            return f"max level (currently {current_level})"

    # ── Threat Assessment ──────────────────────────────────────────

    def assess_player_threat(self, player_name: str) -> dict[str, Any]:
        """Assess how much of a threat a player is to our empire.

        Returns:
            Dict with threat level, reasoning, and recommendations
        """
        with self._lock:
            rel = self._relationships.get(player_name)
            intention = self._intentions.get(player_name)
            patterns = self._patterns.get(player_name, [])
            deception_score = self.get_deception_score(player_name)

            threat_score = 0.0
            reasons = []

            # Base threat from disposition
            if rel:
                if rel.disposition in (PlayerDisposition.ENEMY, PlayerDisposition.HOSTILE):
                    threat_score += 0.4
                    reasons.append("Marked as enemy/hostile")
                elif rel.disposition == PlayerDisposition.RIVAL:
                    threat_score += 0.2
                    reasons.append("Marked as rival")

            # Threat from intention
            if intention:
                if intention.primary_intention == PlayerIntention.PVP:
                    threat_score += 0.3
                    reasons.append("In PVP mode")
                elif intention.primary_intention == PlayerIntention.GM_MONITORING:
                    threat_score += 0.5
                    reasons.append("Possible GM activity")
                elif intention.primary_intention == PlayerIntention.SCOUTING:
                    threat_score += 0.2
                    reasons.append("Scouting behavior")

            # Threat from deception
            if deception_score > 0.5:
                threat_score += 0.2
                reasons.append(f"Deception score: {deception_score:.0%}")

            # Threat from patterns
            for pattern in patterns:
                if pattern.pattern_type == BehaviorPattern.WOE_PARTICIPATION:
                    threat_score += 0.1
                    reasons.append("WOE participant")

            # Clamp
            threat_score = min(1.0, threat_score)

            # Recommendations
            recommendations = []
            if threat_score >= 0.7:
                recommendations.append("Avoid this player - high threat")
                recommendations.append("Alert security team")
            elif threat_score >= 0.4:
                recommendations.append("Monitor this player - medium threat")
                recommendations.append("Avoid sharing sensitive information")
            else:
                recommendations.append("Low threat - no special precautions needed")

            return {
                "player": player_name,
                "threat_score": threat_score,
                "threat_level": "high" if threat_score >= 0.7 else "medium" if threat_score >= 0.4 else "low",
                "reasons": reasons,
                "recommendations": recommendations,
                "deception_score": deception_score,
                "disposition": rel.disposition.value if rel else "unknown",
            }

    # ── Summary / Context ──────────────────────────────────────────

    def get_theory_of_mind_summary(self) -> str:
        """Get a formatted summary of theory of mind state."""
        with self._lock:
            lines = ["── Theory of Mind ──"]

            # Intentions
            if self._intentions:
                lines.append(f"  Tracked Players ({len(self._intentions)}):")
                for name, estimate in sorted(self._intentions.items())[:10]:
                    lines.append(
                        f"    {name}: {estimate.primary_intention.value} "
                        f"(conf: {estimate.confidence:.0%})"
                    )
            else:
                lines.append("  No players tracked yet.")

            # Relationships
            hostile = self.get_hostile_players()
            trusted = self.get_trusted_players()
            if hostile:
                lines.append(f"  Hostile: {', '.join(hostile[:5])}")
            if trusted:
                lines.append(f"  Trusted: {', '.join(trusted[:5])}")

            # Predictions
            active_predictions = sum(
                len(ps) for ps in self._predictions.values()
            )
            if active_predictions:
                lines.append(f"  Active Predictions: {active_predictions}")

            # Patterns
            pattern_count = sum(len(ps) for ps in self._patterns.values())
            if pattern_count:
                lines.append(f"  Detected Patterns: {pattern_count}")

            # Stats
            lines.append(
                f"  Stats: {self._stats['observations_recorded']} observations, "
                f"{self._stats['predictions_made']} predictions, "
                f"{self._stats['patterns_detected']} patterns"
            )

            return "\n".join(lines)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global instance ──

_theory_of_mind: TheoryOfMind | None = None
_tom_lock = RLock()


def get_theory_of_mind() -> TheoryOfMind:
    """Get or create the global theory of mind instance."""
    global _theory_of_mind
    with _tom_lock:
        if _theory_of_mind is None:
            _theory_of_mind = TheoryOfMind()
        return _theory_of_mind
