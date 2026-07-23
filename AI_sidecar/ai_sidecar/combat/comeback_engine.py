"""
Comeback Strategy Engine — adapt after failure, don't repeat mistakes.

When things go wrong (death, gear loss, PK, failed quest, lost money),
this engine provides a structured comeback plan instead of repeating
the same failed behavior.

Extended with root cause analysis:
  - FailureAnalyzer: determines WHY a failure occurred across 5 dimensions
  - RootCause: structured root cause with evidence and recommended fix
  - FixVerifier: tracks whether applied fixes actually prevented recurrence
  - KnowledgeBase: builds a persistent map of failure patterns → countermeasures
"""

from __future__ import annotations

import logging
import random
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Root Cause Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class RootCause:
    """A diagnosed root cause for a bot failure.

    Attributes:
        cause_type: One of 'element_mismatch', 'gear_insufficient', 'bad_positioning',
                    'bad_timing', 'overaggro', 'wrong_target', 'heal_starvation',
                    'skill_miss', 'map_hazard', 'pk_ambush', 'unknown'
        confidence: 0.0–1.0 how sure we are this is the real cause
        evidence: List of observed facts that support this cause
        recommended_fix: Specific, actionable fix to apply
        fix_parameters: Dict of parameters to pass to the fix executor
    """
    cause_type: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    recommended_fix: str = ""
    fix_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FailureContext:
    """Normalised failure context extracted from raw context dict."""
    failure_type: str
    map_name: str = "unknown"
    mob_name: str = "unknown"
    mob_element: str = "unknown"
    mob_race: str = "unknown"
    mob_size: str = "unknown"
    player_hp_pct: float = 1.0
    player_element: str = "neutral"
    player_weapon: str = "unknown"
    player_gear_score: float = 0.0
    aggro_count: int = 0
    nearby_allies: int = 0
    time_of_day: str = "day"
    position_x: int = 0
    position_y: int = 0
    skill_used: str = ""
    damage_taken_spike: bool = False
    was_healing: bool = False
    was_moving: bool = False
    duration_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# FailureAnalyzer — Diagnose WHY a failure occurred
# ═══════════════════════════════════════════════════════════════════════════════

class FailureAnalyzer:
    """Analyzes failure context to determine root cause across 5 dimensions.

    Dimensions analyzed:
      1. Element — was the wrong element used against the target?
      2. Gear — was the player's gear insufficient for the threat?
      3. Positioning — was the player in a bad spot (cornered, trapped)?
      4. Timing — did the player engage at the wrong moment (too many mobs, low HP)?
      5. Aggro — did the player pull more than they could handle?
    """

    # Element advantage lookup: attacking_element -> defending_element -> multiplier
    # Simplified RO chart — full chart lives in elemental_matrix.py
    _ELEMENT_ADVANTAGE: dict[str, dict[str, float]] = {
        "fire": {"earth": 1.5, "fire": 0.5, "water": 0.5, "wind": 1.0,
                 "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.75, "undead": 1.0, "neutral": 1.0},
        "water": {"fire": 1.5, "earth": 0.5, "water": 0.5, "wind": 1.0,
                  "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.75, "undead": 1.0, "neutral": 1.0},
        "earth": {"fire": 1.0, "water": 1.5, "earth": 0.5, "wind": 0.5,
                  "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.75, "undead": 1.0, "neutral": 1.0},
        "wind": {"fire": 1.0, "water": 1.0, "earth": 1.5, "wind": 0.5,
                 "poison": 1.0, "holy": 1.0, "dark": 1.0, "ghost": 0.75, "undead": 1.0, "neutral": 1.0},
        "holy": {"dark": 1.5, "undead": 1.5, "ghost": 0.5, "holy": 0.5},
        "dark": {"holy": 0.5, "dark": 0.5, "ghost": 0.5},
        "ghost": {"ghost": 1.5, "holy": 1.0, "dark": 1.0},
        "undead": {"holy": 0.0, "undead": 0.5},
    }

    # Gear score thresholds by map difficulty tier
    _GEAR_TIERS: dict[str, tuple[float, float]] = {
        "easy": (0.0, 0.3),
        "medium": (0.3, 0.5),
        "hard": (0.5, 0.7),
        "mvp": (0.7, 0.9),
        "woe": (0.8, 1.0),
    }

    def __init__(self) -> None:
        self._lock = RLock()
        # Track dimension-specific failure counts for pattern detection
        self._dimension_failures: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def analyze(self, context: FailureContext) -> list[RootCause]:
        """Run all dimension analyzers and return ranked root causes.

        Returns causes sorted by confidence descending. The top result is
        the most likely root cause.
        """
        with self._lock:
            causes: list[RootCause] = []

            # Run each dimension analyzer
            element_cause = self._analyze_element(context)
            if element_cause:
                causes.append(element_cause)

            gear_cause = self._analyze_gear(context)
            if gear_cause:
                causes.append(gear_cause)

            positioning_cause = self._analyze_positioning(context)
            if positioning_cause:
                causes.append(positioning_cause)

            timing_cause = self._analyze_timing(context)
            if timing_cause:
                causes.append(timing_cause)

            aggro_cause = self._analyze_aggro(context)
            if aggro_cause:
                causes.append(aggro_cause)

            # Sort by confidence descending
            causes.sort(key=lambda c: c.confidence, reverse=True)

            # Record dimension failures for pattern tracking
            for cause in causes:
                self._dimension_failures[cause.cause_type].append({
                    "context": context,
                    "timestamp": time.time(),
                })

            return causes

    def get_primary_cause(self, context: FailureContext) -> RootCause:
        """Get the single most likely root cause."""
        causes = self.analyze(context)
        if causes:
            return causes[0]
        return RootCause(
            cause_type="unknown",
            confidence=0.3,
            evidence=["No specific root cause pattern matched"],
            recommended_fix="general caution — monitor for recurring patterns",
        )

    def _analyze_element(self, ctx: FailureContext) -> RootCause | None:
        """Check if element mismatch caused the failure.

        Triggers when:
        - Player's element is weak against the mob's element
        - Player is using neutral vs an element that resists it
        - Mob element hard-counters the player's attack element
        """
        evidence: list[str] = []
        player_el = ctx.player_element.lower()
        mob_el = ctx.mob_element.lower()

        if player_el == "unknown" or mob_el == "unknown":
            return None

        # Check if player element is disadvantaged against mob
        advantage_table = self._ELEMENT_ADVANTAGE.get(player_el, {})
        multiplier = advantage_table.get(mob_el, 1.0)

        if multiplier < 0.75:
            evidence.append(
                f"Player element '{player_el}' deals {multiplier:.1f}x damage "
                f"to mob element '{mob_el}' — significant disadvantage"
            )
        elif multiplier <= 0.0:
            evidence.append(
                f"Player element '{player_el}' deals ZERO damage "
                f"to mob element '{mob_el}' — complete immunity"
            )

        # Check if mob element resists neutral (common bot mistake)
        if player_el == "neutral" and mob_el in ("ghost", "undead"):
            evidence.append(
                f"Neutral attacks are resisted by '{mob_el}' — "
                f"need elemental weapon or skill"
            )

        # Check if player is using wrong weapon element for the map
        if ctx.player_weapon and ctx.player_weapon != "unknown":
            weapon_el = ctx.player_weapon.lower()
            if weapon_el != player_el and weapon_el != "neutral":
                w_adv = self._ELEMENT_ADVANTAGE.get(weapon_el, {}).get(mob_el, 1.0)
                if w_adv < 0.75:
                    evidence.append(
                        f"Weapon element '{weapon_el}' is also disadvantaged "
                        f"({w_adv:.1f}x) against mob element '{mob_el}'"
                    )

        if evidence:
            return RootCause(
                cause_type="element_mismatch",
                confidence=min(0.95, 0.5 + 0.15 * len(evidence)),
                evidence=evidence,
                recommended_fix=(
                    f"Switch to an element that counters '{mob_el}'. "
                    f"Use elemental weapon or endow skill. "
                    f"Check elemental_matrix.py for the full advantage table."
                ),
                fix_parameters={
                    "target_element": mob_el,
                    "recommended_attack_element": self._best_element_against(mob_el),
                    "current_element": player_el,
                },
            )

        return None

    def _analyze_gear(self, ctx: FailureContext) -> RootCause | None:
        """Check if insufficient gear caused the failure.

        Triggers when:
        - Gear score is below expected tier for the map
        - Player took a damage spike that better gear would mitigate
        - Known dangerous mob on this map requires specific gear
        """
        evidence: list[str] = []

        # Map difficulty estimation from context
        map_tier = self._estimate_map_tier(ctx.map_name, ctx.mob_name)

        # Check gear score against map tier
        tier_range = self._GEAR_TIERS.get(map_tier, (0.0, 0.5))
        if ctx.player_gear_score < tier_range[0]:
            evidence.append(
                f"Gear score {ctx.player_gear_score:.2f} is below "
                f"recommended {tier_range[0]:.2f} for '{map_tier}' tier map"
            )

        # Damage spike suggests gear mitigation failure
        if ctx.damage_taken_spike:
            evidence.append(
                "Damage spike detected — gear may lack sufficient "
                "DEF/MDEF or elemental reduction cards"
            )

        # Low HP at time of death suggests insufficient sustain
        if ctx.player_hp_pct < 0.3:
            evidence.append(
                f"HP was critically low ({ctx.player_hp_pct:.0%}) — "
                f"gear may lack HP sustain or VIT investment"
            )

        if evidence:
            return RootCause(
                cause_type="gear_insufficient",
                confidence=min(0.9, 0.4 + 0.2 * len(evidence)),
                evidence=evidence,
                recommended_fix=(
                    f"Upgrade gear for '{map_tier}' tier content. "
                    f"Prioritize: elemental reduction cards > DEF/MDEF > HP sustain. "
                    f"Consider switching to a lower-tier map until geared."
                ),
                fix_parameters={
                    "map_tier": map_tier,
                    "current_gear_score": ctx.player_gear_score,
                    "target_gear_score": tier_range[0],
                    "needs_elemental_reduction": ctx.damage_taken_spike,
                },
            )

        return None

    def _analyze_positioning(self, ctx: FailureContext) -> RootCause | None:
        """Check if bad positioning caused the failure.

        Triggers when:
        - Player was cornered or trapped
        - Player was in a known danger zone
        - Player was surrounded by mobs
        - Player was too close to a dangerous AoE caster
        """
        evidence: list[str] = []

        # High aggro count + low mobility = trapped
        if ctx.aggro_count >= 3 and not ctx.was_moving:
            evidence.append(
                f"Player was stationary with {ctx.aggro_count} mobs aggroed — "
                f"likely cornered or trapped"
            )

        # Low HP + not moving = failed to retreat
        if ctx.player_hp_pct < 0.3 and not ctx.was_moving:
            evidence.append(
                f"HP was {ctx.player_hp_pct:.0%} but player was not retreating — "
                f"positioning prevented escape"
            )

        # Not healing when taking damage
        if ctx.damage_taken_spike and not ctx.was_healing:
            evidence.append(
                "Took damage spike without healing — "
                "may have been out of position for safe heal"
            )

        if evidence:
            return RootCause(
                cause_type="bad_positioning",
                confidence=min(0.85, 0.4 + 0.15 * len(evidence)),
                evidence=evidence,
                recommended_fix=(
                    "Improve positioning: stay near map edges/escape routes, "
                    "keep escape teleport ready, don't stand still with multiple aggro. "
                    "Use spatial_combat.py positioning system for safer spots."
                ),
                fix_parameters={
                    "aggro_at_death": ctx.aggro_count,
                    "was_stationary": not ctx.was_moving,
                    "failed_to_heal": ctx.damage_taken_spike and not ctx.was_healing,
                },
            )

        return None

    def _analyze_timing(self, ctx: FailureContext) -> RootCause | None:
        """Check if bad timing caused the failure.

        Triggers when:
        - Engaged while HP was already low
        - Engaged while already fighting multiple mobs
        - Engaged during dangerous time (night for undead maps)
        - Skill was on cooldown when needed
        """
        evidence: list[str] = []

        # Engaged with low HP
        if ctx.player_hp_pct < 0.5:
            evidence.append(
                f"Engaged combat at {ctx.player_hp_pct:.0%} HP — "
                f"should have healed first"
            )

        # Engaged while overaggroed
        if ctx.aggro_count > 2:
            evidence.append(
                f"Engaged new target with {ctx.aggro_count} existing aggro — "
                f"should have cleared or fled first"
            )

        # Night time on undead map (undead get buffed at night on some servers)
        if ctx.time_of_day == "night" and ctx.mob_element in ("undead", "dark"):
            evidence.append(
                f"Engaged '{ctx.mob_element}' mob at night — "
                f"may be buffed or more aggressive"
            )

        if evidence:
            return RootCause(
                cause_type="bad_timing",
                confidence=min(0.85, 0.4 + 0.15 * len(evidence)),
                evidence=evidence,
                recommended_fix=(
                    "Improve engagement timing: always heal above 70% HP before engaging, "
                    "clear surrounding mobs first, avoid night on undead/dark maps. "
                    "Use predictive_threat.py for engagement safety checks."
                ),
                fix_parameters={
                    "hp_at_engagement": ctx.player_hp_pct,
                    "aggro_at_engagement": ctx.aggro_count,
                    "time_of_day": ctx.time_of_day,
                },
            )

        return None

    def _analyze_aggro(self, ctx: FailureContext) -> RootCause | None:
        """Check if over-aggro caused the failure.

        Triggers when:
        - Too many mobs aggroed at once
        - Aggro count exceeds safe threshold for gear/level
        - Chain-pulled multiple spawns
        """
        evidence: list[str] = []

        # Raw aggro count
        if ctx.aggro_count >= 4:
            evidence.append(
                f"Overwhelmed by {ctx.aggro_count} simultaneous aggro — "
                f"exceeds safe handling capacity"
            )
        elif ctx.aggro_count >= 2:
            evidence.append(
                f"Had {ctx.aggro_count} mobs aggroed — "
                f"may exceed safe threshold for current gear"
            )

        # Aggro + low gear = death sentence
        if ctx.aggro_count >= 2 and ctx.player_gear_score < 0.4:
            evidence.append(
                f"Low gear score ({ctx.player_gear_score:.2f}) + "
                f"{ctx.aggro_count} aggro = high death probability"
            )

        # No nearby allies to share aggro
        if ctx.aggro_count >= 3 and ctx.nearby_allies == 0:
            evidence.append(
                f"No allies nearby to peel {ctx.aggro_count} aggro"
            )

        if evidence:
            return RootCause(
                cause_type="overaggro",
                confidence=min(0.9, 0.4 + 0.15 * len(evidence)),
                evidence=evidence,
                recommended_fix=(
                    "Reduce pull radius, use safe positioning to avoid chain-pulling, "
                    "set max aggro threshold in combat config, "
                    "use escape teleport when aggro exceeds limit."
                ),
                fix_parameters={
                    "aggro_count": ctx.aggro_count,
                    "safe_threshold": max(1, 3 - int(ctx.player_gear_score * 3)),
                    "nearby_allies": ctx.nearby_allies,
                },
            )

        return None

    def _estimate_map_tier(self, map_name: str, mob_name: str) -> str:
        """Estimate the difficulty tier of a map based on name heuristics."""
        map_lower = map_name.lower()
        mob_lower = mob_name.lower()

        if any(kw in map_lower for kw in ("woe", "castle", "empire", "valkyrie")):
            return "woe"
        if any(kw in map_lower for kw in ("mvp", "boss", "thanatos", "turtle", "bio")):
            return "mvp"
        if any(kw in map_lower for kw in ("dungeon", "culvert", "sewer", "cave", "tower")):
            return "hard"
        if any(kw in map_lower for kw in ("field", "plains", "forest", "beach", "desert")):
            return "medium"

        # Fallback: check mob name for MVP indicators
        if any(kw in mob_lower for kw in ("mvp", "boss", "mini", "king", "lord")):
            return "mvp"

        return "easy"

    def _best_element_against(self, mob_element: str) -> str:
        """Find the best attacking element against a given mob element."""
        best_el = "neutral"
        best_mult = 0.0
        for attacker_el, table in self._ELEMENT_ADVANTAGE.items():
            mult = table.get(mob_element, 1.0)
            if mult > best_mult:
                best_mult = mult
                best_el = attacker_el
        return best_el

    def get_dimension_stats(self) -> dict[str, int]:
        """Get failure counts per dimension for pattern analysis."""
        with self._lock:
            return {dim: len(fails) for dim, fails in self._dimension_failures.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# FixVerifier — Track whether applied fixes actually prevented recurrence
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class FixRecord:
    """Record of a fix that was applied and its outcome."""
    cause_type: str
    fix_applied: str
    fix_parameters: dict[str, Any]
    applied_at: float
    context_snapshot: dict[str, Any]
    verified: bool = False
    verified_at: float = 0.0
    recurrence: bool = False
    recurrence_count: int = 0
    notes: str = ""


class FixVerifier:
    """Tracks whether applied fixes actually prevented recurrence.

    After a fix is applied, the verifier monitors subsequent failures
    to determine if the same root cause recurs. If it does, the fix
    was insufficient and needs adjustment.
    """

    def __init__(self, observation_window: float = 3600.0) -> None:
        self._lock = RLock()
        self._observation_window = observation_window  # seconds to monitor
        self._pending_fixes: dict[str, FixRecord] = {}  # key -> pending verification
        self._verified_fixes: list[FixRecord] = []
        self._fix_success_rate: dict[str, list[bool]] = defaultdict(list)

    def register_fix(self, cause_type: str, fix: str,
                     parameters: dict[str, Any],
                     context: dict[str, Any]) -> str:
        """Register a fix that was applied and needs verification.

        Returns a fix_id that can be used to report recurrence.
        """
        with self._lock:
            fix_id = f"{cause_type}_{int(time.time())}_{random.randint(1000, 9999)}"
            record = FixRecord(
                cause_type=cause_type,
                fix_applied=fix,
                fix_parameters=parameters,
                applied_at=time.time(),
                context_snapshot=context,
            )
            self._pending_fixes[fix_id] = record
            return fix_id

    def report_recurrence(self, fix_id: str,
                          new_context: dict[str, Any]) -> bool:
        """Report that a failure recurred after a fix was applied.

        Returns True if the fix is now considered failed.
        """
        with self._lock:
            record = self._pending_fixes.get(fix_id)
            if record is None:
                logger.warning("No pending fix found for id %s", fix_id)
                return False

            record.recurrence = True
            record.recurrence_count += 1
            record.notes = f"Recurred at {time.strftime('%H:%M:%S')}"

            # Mark as failed if recurrence count exceeds threshold
            if record.recurrence_count >= 2:
                self._finalize_fix(fix_id, success=False)
                return True

            return False

    def verify_fix(self, fix_id: str, success: bool,
                   notes: str = "") -> None:
        """Manually verify whether a fix succeeded or failed."""
        with self._lock:
            self._finalize_fix(fix_id, success=success, notes=notes)

    def _finalize_fix(self, fix_id: str, success: bool,
                      notes: str = "") -> None:
        """Move a fix from pending to verified."""
        record = self._pending_fixes.pop(fix_id, None)
        if record is None:
            return

        record.verified = True
        record.verified_at = time.time()
        record.notes = notes or record.notes
        record.recurrence = not success

        self._verified_fixes.append(record)
        self._fix_success_rate[record.cause_type].append(success)

    def get_success_rate(self, cause_type: str | None = None) -> float:
        """Get the success rate for a specific cause type, or overall."""
        with self._lock:
            if cause_type:
                results = self._fix_success_rate.get(cause_type, [])
            else:
                results = [r for rs in self._fix_success_rate.values() for r in rs]

            if not results:
                return 0.0
            return sum(results) / len(results)

    def get_pending_count(self) -> int:
        """Get number of fixes awaiting verification."""
        with self._lock:
            return len(self._pending_fixes)

    def get_fix_summary(self) -> dict[str, Any]:
        """Get a summary of fix performance."""
        with self._lock:
            total = len(self._verified_fixes)
            successful = sum(1 for f in self._verified_fixes if not f.recurrence)
            return {
                "total_fixes": total,
                "successful": successful,
                "failed": total - successful,
                "pending": len(self._pending_fixes),
                "by_cause": {
                    ct: {
                        "attempts": len(results),
                        "successes": sum(results),
                        "rate": sum(results) / len(results) if results else 0.0,
                    }
                    for ct, results in self._fix_success_rate.items()
                },
            }

    def auto_verify_pending(self, recent_failures: list[dict[str, Any]]) -> None:
        """Auto-verify pending fixes by checking if the same failure recurred.

        Called periodically with recent failures to check for recurrence.
        """
        with self._lock:
            now = time.time()
            expired: list[str] = []

            for fix_id, record in list(self._pending_fixes.items()):
                # Check if observation window expired
                if now - record.applied_at > self._observation_window:
                    # No recurrence within window — fix succeeded
                    self._finalize_fix(fix_id, success=True,
                                       notes="Auto-verified: no recurrence within observation window")
                    expired.append(fix_id)
                    continue

                # Check recent failures for same cause type + map
                for failure in recent_failures:
                    if failure.get("cause_type") == record.cause_type:
                        same_map = (
                            failure.get("context", {}).get("map", "")
                            == record.context_snapshot.get("map", "")
                        )
                        if same_map:
                            record.recurrence_count += 1
                            if record.recurrence_count >= 2:
                                self._finalize_fix(fix_id, success=False,
                                                   notes="Auto-verified: recurrence detected")
                                expired.append(fix_id)
                                break


# ═══════════════════════════════════════════════════════════════════════════════
# KnowledgeBase — Build map of failure patterns to countermeasures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class FailurePattern:
    """A learned failure pattern with proven countermeasures."""
    pattern_id: str
    cause_type: str
    map_name: str
    mob_name: str
    conditions: dict[str, Any]  # e.g. {"min_aggro": 3, "time": "night"}
    countermeasures: list[str]  # e.g. ["use fire weapon", "avoid night"]
    effectiveness: float  # 0.0–1.0 how well countermeasures work
    occurrences: int = 1
    last_seen: float = 0.0
    tags: list[str] = field(default_factory=list)


class KnowledgeBase:
    """Builds a persistent map of 'what kills bots' with countermeasures.

    Learns from every failure:
      1. Extract pattern (map + mob + conditions)
      2. Record what countermeasure was applied
      3. Track whether the countermeasure worked (via FixVerifier)
      4. Update effectiveness scores
      5. Surface known dangerous patterns proactively
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._patterns: dict[str, FailurePattern] = {}  # pattern_id -> pattern
        self._map_danger_scores: dict[str, float] = defaultdict(float)  # map -> danger
        self._mob_kill_counts: Counter[str] = Counter()  # mob -> times killed bot
        self._time_danger: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )  # map -> time_of_day -> deaths

    def learn(self, cause: RootCause, context: FailureContext) -> str:
        """Learn from a failure by recording the pattern.

        Returns the pattern_id for cross-referencing with FixVerifier.
        """
        with self._lock:
            pattern_id = self._make_pattern_id(cause, context)
            now = time.time()

            if pattern_id in self._patterns:
                pattern = self._patterns[pattern_id]
                pattern.occurrences += 1
                pattern.last_seen = now
                # Decay effectiveness slightly with each recurrence
                pattern.effectiveness *= 0.95
            else:
                pattern = FailurePattern(
                    pattern_id=pattern_id,
                    cause_type=cause.cause_type,
                    map_name=context.map_name,
                    mob_name=context.mob_name,
                    conditions=self._extract_conditions(cause, context),
                    countermeasures=[cause.recommended_fix],
                    effectiveness=0.5,  # Start neutral
                    occurrences=1,
                    last_seen=now,
                    tags=self._generate_tags(cause, context),
                )
                self._patterns[pattern_id] = pattern

            # Update aggregate stats
            self._map_danger_scores[context.map_name] += 1.0
            self._mob_kill_counts[context.mob_name] += 1
            self._time_danger[context.map_name][context.time_of_day] += 1

            return pattern_id

    def update_countermeasure_effectiveness(
        self, pattern_id: str, fix_succeeded: bool
    ) -> None:
        """Update how well a countermeasure worked for a pattern."""
        with self._lock:
            pattern = self._patterns.get(pattern_id)
            if pattern is None:
                return

            if fix_succeeded:
                pattern.effectiveness = min(1.0, pattern.effectiveness + 0.1)
            else:
                pattern.effectiveness = max(0.0, pattern.effectiveness - 0.15)

    def get_dangerous_patterns(self, map_name: str | None = None,
                               min_occurrences: int = 2) -> list[FailurePattern]:
        """Get the most dangerous known patterns, optionally filtered by map."""
        with self._lock:
            patterns = list(self._patterns.values())
            if map_name:
                patterns = [p for p in patterns if p.map_name == map_name]
            patterns = [p for p in patterns if p.occurrences >= min_occurrences]
            patterns.sort(key=lambda p: p.occurrences, reverse=True)
            return patterns

    def get_map_risk_assessment(self, map_name: str) -> dict[str, Any]:
        """Get a risk assessment for a specific map."""
        with self._lock:
            patterns = self.get_dangerous_patterns(map_name=map_name)
            time_danger = dict(self._time_danger.get(map_name, {}))

            return {
                "map": map_name,
                "danger_score": self._map_danger_scores.get(map_name, 0.0),
                "known_patterns": len(patterns),
                "deadliest_mobs": [
                    {"mob": mob, "deaths": count}
                    for mob, count in self._mob_kill_counts.most_common(5)
                ],
                "time_danger": time_danger,
                "patterns": [
                    {
                        "cause": p.cause_type,
                        "mob": p.mob_name,
                        "occurrences": p.occurrences,
                        "effectiveness": p.effectiveness,
                        "countermeasures": p.countermeasures,
                    }
                    for p in patterns[:5]  # Top 5 most dangerous
                ],
            }

    def get_top_killers(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get the top mobs that have killed the bot."""
        with self._lock:
            return [
                {"mob": mob, "deaths": count}
                for mob, count in self._mob_kill_counts.most_common(limit)
            ]

    def get_proactive_warnings(self, map_name: str) -> list[str]:
        """Get proactive warnings before entering a map."""
        with self._lock:
            warnings: list[str] = []
            assessment = self.get_map_risk_assessment(map_name)

            if assessment["danger_score"] >= 5:
                warnings.append(
                    f"⚠ Map '{map_name}' has high failure rate "
                    f"({assessment['danger_score']:.0f} recorded failures)"
                )

            for pattern in assessment["patterns"]:
                if pattern["occurrences"] >= 3 and pattern["effectiveness"] < 0.4:
                    warnings.append(
                        f"⚠ Known threat: {pattern['cause']} by '{pattern['mob']}' "
                        f"({pattern['occurrences']}x) — current countermeasures "
                        f"only {pattern['effectiveness']:.0%} effective"
                    )

            time_danger = assessment.get("time_danger", {})
            if time_danger.get("night", 0) > time_danger.get("day", 0) * 2:
                warnings.append(
                    f"⚠ Most failures on '{map_name}' occur at night — "
                    f"consider daytime-only farming"
                )

            return warnings

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge base statistics."""
        with self._lock:
            return {
                "total_patterns": len(self._patterns),
                "maps_tracked": len(self._map_danger_scores),
                "mobs_killed_bot": len(self._mob_kill_counts),
                "total_deaths": sum(p.occurrences for p in self._patterns.values()),
                "most_dangerous_map": max(self._map_danger_scores,
                                          key=self._map_danger_scores.get,
                                          default="none"),
                "top_killer": self._mob_kill_counts.most_common(1)[0][0]
                if self._mob_kill_counts else "none",
            }

    def _make_pattern_id(self, cause: RootCause, context: FailureContext) -> str:
        """Create a unique pattern ID from cause + context."""
        return f"{cause.cause_type}|{context.map_name}|{context.mob_name}"

    def _extract_conditions(self, cause: RootCause,
                            context: FailureContext) -> dict[str, Any]:
        """Extract relevant conditions from context for this cause type."""
        conditions: dict[str, Any] = {}

        if cause.cause_type == "element_mismatch":
            conditions["player_element"] = context.player_element
            conditions["mob_element"] = context.mob_element
        elif cause.cause_type == "gear_insufficient":
            conditions["gear_score"] = context.player_gear_score
            conditions["damage_spike"] = context.damage_taken_spike
        elif cause.cause_type == "bad_positioning":
            conditions["aggro_count"] = context.aggro_count
            conditions["was_moving"] = context.was_moving
        elif cause.cause_type == "bad_timing":
            conditions["hp_pct"] = context.player_hp_pct
            conditions["time_of_day"] = context.time_of_day
        elif cause.cause_type == "overaggro":
            conditions["aggro_count"] = context.aggro_count
            conditions["nearby_allies"] = context.nearby_allies

        return conditions

    def _generate_tags(self, cause: RootCause,
                       context: FailureContext) -> list[str]:
        """Generate searchable tags for a pattern."""
        tags = [cause.cause_type, context.map_name, context.mob_name]
        if context.time_of_day:
            tags.append(context.time_of_day)
        if context.aggro_count >= 3:
            tags.append("multi_aggro")
        if context.player_hp_pct < 0.3:
            tags.append("low_hp")
        if context.damage_taken_spike:
            tags.append("damage_spike")
        return tags


# ═══════════════════════════════════════════════════════════════════════════════
# Original ComebackEngine — Extended with root cause analysis
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class ComebackPlan:
    """A structured plan for recovering from failure."""
    failure_type: str  # death, gear_loss, pk, quest_fail, money_loss
    severity: int  # 1-10
    immediate_action: str  # What to do right now
    recovery_strategy: str  # How to recover
    avoid_for_minutes: int  # How long to avoid the failed activity
    alternative_plan: str  # What to do instead
    confidence: float  # 0.0-1.0
    # Root cause analysis extensions (populated by record_failure)
    _root_cause: RootCause | None = None
    _all_causes: list[RootCause] = field(default_factory=list)
    _pattern_id: str = ""


class ComebackEngine:
    """Generates comeback plans after failures.

    Instead of repeating the same failed behavior, this engine:
    1. Analyzes what went wrong (with root cause analysis)
    2. Generates a recovery plan
    3. Avoids the failed activity for a period
    4. Tracks recovery progress
    5. Learns from patterns to prevent future failures
    """

    def __init__(self):
        self._lock = RLock()

        # Failure history
        self._failures: deque[dict[str, Any]] = deque(maxlen=50)

        # Active avoidance
        self._avoid_until: dict[str, float] = {}  # key -> timestamp

        # Recovery tracking
        self._recovery_attempts: dict[str, int] = defaultdict(int)
        self._recovery_successes: dict[str, int] = defaultdict(int)

        # Stats
        self._stats: dict[str, int] = defaultdict(int)

        # ── Root cause analysis extensions ──
        self._analyzer = FailureAnalyzer()
        self._fix_verifier = FixVerifier()
        self._knowledge_base = KnowledgeBase()

        # Track fix_ids for pending fixes
        self._active_fix_ids: dict[str, str] = {}  # failure_key -> fix_id

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def record_failure(self, failure_type: str,
                       context: dict[str, Any]) -> ComebackPlan:
        """Record a failure and generate a comeback plan.

        Extended with root cause analysis:
        1. Normalize context into FailureContext
        2. Run FailureAnalyzer to find root cause
        3. Learn from the failure in KnowledgeBase
        4. Generate comeback plan (original behavior)
        5. Register fix with FixVerifier for tracking
        """
        with self._lock:
            now = time.time()

            # ── Step 1: Normalize context ──
            failure_ctx = self._normalize_context(failure_type, context)

            # ── Step 2: Root cause analysis ──
            primary_cause = self._analyzer.get_primary_cause(failure_ctx)
            all_causes = self._analyzer.analyze(failure_ctx)

            # ── Step 3: Learn from failure ──
            pattern_id = self._knowledge_base.learn(primary_cause, failure_ctx)

            # ── Step 4: Record the failure (original) ──
            self._failures.append({
                "type": failure_type,
                "context": context,
                "timestamp": now,
                "root_cause": primary_cause.cause_type,
                "pattern_id": pattern_id,
            })
            self._stats["failures"] += 1

            # Count recent failures of this type
            recent = [f for f in self._failures
                      if f["type"] == failure_type
                      and now - f["timestamp"] < 3600]
            severity = min(10, len(recent) * 2)

            # ── Step 5: Generate plan (original) ──
            if failure_type == "death":
                plan = self._handle_death(context, severity, now)
            elif failure_type == "gear_loss":
                plan = self._handle_gear_loss(context, severity, now)
            elif failure_type == "pk":
                plan = self._handle_pk(context, severity, now)
            elif failure_type == "quest_fail":
                plan = self._handle_quest_fail(context, severity, now)
            elif failure_type == "money_loss":
                plan = self._handle_money_loss(context, severity, now)
            else:
                plan = ComebackPlan(
                    failure_type=failure_type,
                    severity=severity,
                    immediate_action="stand up and assess",
                    recovery_strategy="return to safe zone",
                    avoid_for_minutes=5,
                    alternative_plan="continue normal activity",
                    confidence=0.5,
                )

            # ── Step 6: Enhance plan with root cause insights ──
            if primary_cause.cause_type != "unknown":
                plan.recovery_strategy = (
                    f"[ROOT CAUSE: {primary_cause.cause_type}] "
                    f"{primary_cause.recommended_fix} | "
                    f"{plan.recovery_strategy}"
                )
                plan.confidence = max(plan.confidence, primary_cause.confidence * 0.8)

            # ── Step 7: Register fix for verification ──
            fix_id = self._fix_verifier.register_fix(
                cause_type=primary_cause.cause_type,
                fix=primary_cause.recommended_fix,
                parameters=primary_cause.fix_parameters,
                context=context,
            )
            failure_key = f"{failure_type}:{context.get('map', 'unknown')}:{now}"
            self._active_fix_ids[failure_key] = fix_id

            # ── Step 8: Set avoidance (original) ──
            if plan.avoid_for_minutes > 0:
                key = f"{failure_type}:{context.get('map', 'unknown')}"
                self._avoid_until[key] = now + (plan.avoid_for_minutes * 60)

            # Attach root cause info to the plan for downstream consumers
            plan._root_cause = primary_cause  # type: ignore[attr-defined]
            plan._all_causes = all_causes  # type: ignore[attr-defined]
            plan._pattern_id = pattern_id  # type: ignore[attr-defined]

            return plan

    def _normalize_context(self, failure_type: str,
                           context: dict[str, Any]) -> FailureContext:
        """Normalize a raw context dict into a FailureContext for analysis."""
        return FailureContext(
            failure_type=failure_type,
            map_name=context.get("map", "unknown"),
            mob_name=context.get("mob", context.get("monster", "unknown")),
            mob_element=context.get("mob_element", "unknown"),
            mob_race=context.get("mob_race", "unknown"),
            mob_size=context.get("mob_size", "unknown"),
            player_hp_pct=context.get("hp_pct", context.get("hp", 1.0)),
            player_element=context.get("player_element", "neutral"),
            player_weapon=context.get("weapon", "unknown"),
            player_gear_score=context.get("gear_score", 0.0),
            aggro_count=context.get("aggro_count", context.get("aggro", 0)),
            nearby_allies=context.get("nearby_allies", 0),
            time_of_day=context.get("time_of_day", "day"),
            position_x=context.get("x", 0),
            position_y=context.get("y", 0),
            skill_used=context.get("skill", ""),
            damage_taken_spike=context.get("damage_spike", False),
            was_healing=context.get("was_healing", False),
            was_moving=context.get("was_moving", True),
            duration_seconds=context.get("duration", 0.0),
            extra={k: v for k, v in context.items()
                   if k not in ("map", "mob", "monster", "mob_element",
                                "mob_race", "mob_size", "hp_pct", "hp",
                                "player_element", "weapon", "gear_score",
                                "aggro_count", "aggro", "nearby_allies",
                                "time_of_day", "x", "y", "skill",
                                "damage_spike", "was_healing", "was_moving",
                                "duration")},
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Root cause analysis public accessors
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_failure(self, context: dict[str, Any]) -> list[RootCause]:
        """Run root cause analysis on a failure context without recording it.

        Useful for pre-engagement safety checks.
        """
        failure_ctx = self._normalize_context("analysis", context)
        return self._analyzer.analyze(failure_ctx)

    def get_risk_warnings(self, map_name: str) -> list[str]:
        """Get proactive warnings about known dangers on a map."""
        return self._knowledge_base.get_proactive_warnings(map_name)

    def get_map_risk(self, map_name: str) -> dict[str, Any]:
        """Get a full risk assessment for a map."""
        return self._knowledge_base.get_map_risk_assessment(map_name)

    def get_top_killers(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get the top mobs that have killed the bot."""
        return self._knowledge_base.get_top_killers(limit=limit)

    def get_fix_success_rate(self, cause_type: str | None = None) -> float:
        """Get how well fixes are working for a cause type."""
        return self._fix_verifier.get_success_rate(cause_type)

    def get_fix_summary(self) -> dict[str, Any]:
        """Get a summary of fix performance."""
        return self._fix_verifier.get_fix_summary()

    def get_knowledge_stats(self) -> dict[str, Any]:
        """Get knowledge base statistics."""
        return self._knowledge_base.get_stats()

    def get_dimension_stats(self) -> dict[str, int]:
        """Get failure counts per root cause dimension."""
        return self._analyzer.get_dimension_stats()

    def report_recurrence(self, failure_type: str, map_name: str,
                          context: dict[str, Any]) -> bool:
        """Report that a failure recurred after a fix was applied.

        This should be called when the same failure happens again
        after a comeback plan was executed.
        """
        with self._lock:
            # Find the most recent fix_id for this failure type + map
            for key, fix_id in list(self._active_fix_ids.items()):
                if key.startswith(f"{failure_type}:{map_name}"):
                    recurred = self._fix_verifier.report_recurrence(fix_id, context)
                    if recurred:
                        # Update knowledge base that countermeasure failed
                        pattern_id = f"{failure_type}|{map_name}|{context.get('mob', 'unknown')}"
                        self._knowledge_base.update_countermeasure_effectiveness(
                            pattern_id, fix_succeeded=False
                        )
                    return recurred
            return False

    def verify_fix(self, failure_type: str, map_name: str,
                   success: bool, notes: str = "") -> None:
        """Manually verify whether a fix for a failure succeeded."""
        with self._lock:
            for key, fix_id in list(self._active_fix_ids.items()):
                if key.startswith(f"{failure_type}:{map_name}"):
                    self._fix_verifier.verify_fix(fix_id, success=success, notes=notes)
                    if success:
                        pattern_id = f"{failure_type}|{map_name}|unknown"
                        self._knowledge_base.update_countermeasure_effectiveness(
                            pattern_id, fix_succeeded=True
                        )
                    break

    def auto_verify_fixes(self) -> None:
        """Auto-verify pending fixes against recent failures."""
        with self._lock:
            recent_failures = [
                {
                    "cause_type": f.get("root_cause", "unknown"),
                    "context": f.get("context", {}),
                }
                for f in self._failures
            ]
            self._fix_verifier.auto_verify_pending(recent_failures)

    # ──────────────────────────────────────────────────────────────────────────
    # Original handlers (unchanged)
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_death(self, context: dict[str, Any],
                      severity: int, now: float) -> ComebackPlan:
        """Generate a comeback plan after death."""
        map_name = context.get("map", "unknown")
        reason = context.get("reason", "unknown")

        if severity >= 6:
            return ComebackPlan(
                failure_type="death",
                severity=severity,
                immediate_action="teleport to safe town",
                recovery_strategy=f"avoid {map_name} for 2 hours, farm easier mobs to recover exp",
                avoid_for_minutes=120,
                alternative_plan=f"farm on a different map with lower-level mobs",
                confidence=0.8,
            )
        elif severity >= 3:
            return ComebackPlan(
                failure_type="death",
                severity=severity,
                immediate_action="check equipment, heal, return carefully",
                recovery_strategy=f"return to {map_name} but stay near escape route",
                avoid_for_minutes=30,
                alternative_plan=f"farm on a safer area of {map_name}",
                confidence=0.7,
            )
        else:
            return ComebackPlan(
                failure_type="death",
                severity=severity,
                immediate_action="heal and continue",
                recovery_strategy=f"be more careful on {map_name}",
                avoid_for_minutes=5,
                alternative_plan="continue current activity",
                confidence=0.6,
            )

    def _handle_gear_loss(self, context: dict[str, Any],
                          severity: int, now: float) -> ComebackPlan:
        """Generate a comeback plan after gear loss."""
        item = context.get("item", "unknown")

        return ComebackPlan(
            failure_type="gear_loss",
            severity=severity,
            immediate_action="switch to backup gear",
            recovery_strategy=f"farm zeny to replace {item}, check player shops for deals",
            avoid_for_minutes=10,
            alternative_plan="use backup weapon and adjust build",
            confidence=0.7,
        )

    def _handle_pk(self, context: dict[str, Any],
                   severity: int, now: float) -> ComebackPlan:
        """Generate a comeback plan after being PK'd."""
        attacker = context.get("attacker", "unknown")
        map_name = context.get("map", "unknown")

        return ComebackPlan(
            failure_type="pk",
            severity=severity,
            immediate_action="teleport to safe town, log alt to scout",
            recovery_strategy=f"avoid {map_name} for 24 hours, scout with alt before returning",
            avoid_for_minutes=1440,  # 24 hours
            alternative_plan=f"farm on a different map, add {attacker} to watchlist",
            confidence=0.9,
        )

    def _handle_quest_fail(self, context: dict[str, Any],
                           severity: int, now: float) -> ComebackPlan:
        """Generate a comeback plan after quest failure."""
        quest = context.get("quest", "unknown")

        return ComebackPlan(
            failure_type="quest_fail",
            severity=severity,
            immediate_action="review what went wrong, prepare better",
            recovery_strategy=f"retry {quest} with different approach, check guide for tips",
            avoid_for_minutes=15,
            alternative_plan=f"level up more before retrying {quest}",
            confidence=0.6,
        )

    def _handle_money_loss(self, context: dict[str, Any],
                           severity: int, now: float) -> ComebackPlan:
        """Generate a comeback plan after losing money."""
        amount = context.get("amount", 0)

        return ComebackPlan(
            failure_type="money_loss",
            severity=severity,
            immediate_action="stop current activity",
            recovery_strategy=f"switch to safe farming until {amount * 2} zeny recovered",
            avoid_for_minutes=30,
            alternative_plan="farm easy mobs for guaranteed drops",
            confidence=0.7,
        )

    def is_avoiding(self, failure_type: str, map_name: str) -> bool:
        """Check if a map/activity is currently being avoided."""
        with self._lock:
            now = time.time()
            key = f"{failure_type}:{map_name}"
            avoid_until = self._avoid_until.get(key, 0)
            if now < avoid_until:
                return True

            # Also check for broader avoidance
            for k, v in self._avoid_until.items():
                if k.startswith(failure_type) and map_name in k and now < v:
                    return True

            return False

    def record_recovery(self, failure_type: str, success: bool) -> None:
        """Record whether a recovery attempt was successful."""
        with self._lock:
            self._recovery_attempts[failure_type] += 1
            if success:
                self._recovery_successes[failure_type] += 1
                self._stats["recoveries"] += 1

    def get_recovery_rate(self, failure_type: str) -> float:
        """Get the success rate for a recovery type."""
        with self._lock:
            attempts = self._recovery_attempts.get(failure_type, 0)
            if attempts == 0:
                return 0.0
            return self._recovery_successes.get(failure_type, 0) / attempts

    def get_stats(self) -> dict[str, int]:
        """Get comeback engine statistics."""
        with self._lock:
            return dict(self._stats)

    def get_extended_stats(self) -> dict[str, Any]:
        """Get comprehensive stats including root cause analysis data."""
        with self._lock:
            return {
                "basic": dict(self._stats),
                "recovery_rates": {
                    ft: self.get_recovery_rate(ft)
                    for ft in list(self._recovery_attempts.keys())
                },
                "dimensions": self._analyzer.get_dimension_stats(),
                "fix_performance": self._fix_verifier.get_fix_summary(),
                "knowledge_base": self._knowledge_base.get_stats(),
                "active_avoidance": len(self._avoid_until),
            }


# ═══════════════════════════════════════════════════════════════════════════════
# Global singleton
# ═══════════════════════════════════════════════════════════════════════════════

_engine: ComebackEngine | None = None


def get_comeback_engine() -> ComebackEngine:
    """Get the global ComebackEngine instance."""
    global _engine
    if _engine is None:
        _engine = ComebackEngine()
    return _engine
