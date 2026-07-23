"""
Spatial Combat Awareness — PositioningSystem for RO tile-based combat.

A pro player doesn't stand still. They:
  1. Stand diagonal to mobs (take 1 hit instead of 2)
  2. Predict caster AoE and move preemptively
  3. Close distance on casters to interrupt, zigzag against archers
  4. Stop attacking when DoT will finish the mob (overkill awareness)
  5. Chain skills by purpose (Fire Wall to block path, Safety Wall to buy time)

This module scores every candidate position by expected damage intake vs output
and integrates with combat_loop.py via setters and tick-phase hooks.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import RLock
from typing import Any

from ai_sidecar.combat.damage_formulas import (
    SkillCooldownTracker,
    calculate_cast_time,
    estimate_hits_to_kill,
    get_skill_range,
)
from ai_sidecar.combat.elemental_matrix import ElementalMatrix

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CAST_TIME_DANGER_THRESHOLD = 3.0  # seconds — cast time above this calls for interrupt
DEFAULT_MOVE_SPEED_CELLS_PER_SECOND = 3.0  # RO default walk speed (~150 ASPD walk)
DEFAULT_ZIGZAG_INTERVAL = 0.8  # seconds between zigzag direction changes
MAX_POSITION_CANDIDATES = 16  # cap candidate positions for performance
OVERKILL_DOT_THRESHOLD = 0.5  # stop attacking if DoT will take >50% remaining HP
SAFETY_BUFFER_CELLS = 3  # cells to keep from edges/chokepoints

# ── Skill purpose tags used by _chain_skill_by_purpose ──
SKILL_PURPOSE_BLOCK_PATH = {"Fire Wall", "Ice Wall", "Quagmire"}
SKILL_PURPOSE_BUY_TIME = {"Safety Wall", "Endure", "Kyrie Eleison", "Pneuma"}
SKILL_PURPOSE_ZONE_DENIAL = {"Fire Wall", "Frost Nova", "Storm Gust", "Meteor Storm"}
SKILL_PURPOSE_DOT_FINISH = {"Fire Wall", "Poison", "Envenom"}


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────


class MovementMode(Enum):
    """Reason for a movement command."""
    APPROACH = auto()
    MAINTAIN_RANGE = auto()
    DIAGONAL_POSITION = auto()
    AOE_EVASION = auto()
    CASTER_CLOSE = auto()
    ZIGZAG = auto()
    RETREAT = auto()


class PositionVerdict(Enum):
    """Recommended action after scoring a position."""
    STAY = auto()
    MOVE = auto()
    INTERRUPT = auto()
    STOP_ATTACKING = auto()
    CHAIN_SKILL = auto()


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PositionScore:
    """Score for a candidate (x, y) position.

    Positive score = good position (low intake, high output).
    Higher is better.
    """
    x: int
    y: int
    damage_intake: float = 0.0  # Expected damage we'll take per second
    damage_output: float = 0.0  # Expected damage we'll deal per second
    intake_output_ratio: float = 1.0  # intake / output (lower = better)
    diagonal_advantage: float = 0.0  # bonus for diagonal positioning (0-1)
    aoe_safety: float = 1.0  # 0 = in an AoE blast, 1 = completely safe
    cover_from_ranged: float = 1.0  # 0 = exposed to archers, 1 = well-covered
    interrupt_opportunity: float = 0.0  # 0-1, how good this position is for stopping casters
    total_score: float = 0.0  # Weighted sum of all factors


@dataclass(slots=True)
class MovementIntent:
    """A planned movement with reasoning."""
    target_x: int
    target_y: int
    mode: MovementMode
    priority: int = 50  # higher = more urgent
    reason: str = ""
    estimated_arrival_s: float = 0.0
    should_skip_if_interrupted: bool = False


@dataclass(slots=True)
class SkillChainIntent:
    """Intent to chain a skill for spatial/tactical purpose rather than DPS."""
    skill_name: str
    purpose: str  # "block_path", "buy_time", "zone_denial", "dot_finish"
    target_x: int
    target_y: int
    priority: int = 50
    reason: str = ""


@dataclass(slots=True)
class OverkillAssessment:
    """Assessment of whether current damage-over-time will finish the target."""
    will_die_to_dot: bool = False
    dot_damage_remaining: float = 0.0
    target_hp_remaining: int = 0
    estimated_time_to_die_s: float = 0.0
    should_stop_attacking: bool = False


@dataclass
class SpatialSnapshot:
    """Spatial snapshot of the current combat situation.

    Extracted from the bridge snapshot by the positioning system.
    """
    my_x: int = 0
    my_y: int = 0
    my_hp_pct: float = 1.0
    my_job_class: str = "novice"
    my_available_skills: list[str] = field(default_factory=list)
    map_name: str = ""
    aggro_count: int = 0

    # Current target
    target_id: int = 0
    target_name: str = ""
    target_x: int = 0
    target_y: int = 0
    target_distance: float = 0.0
    target_hp_pct: float = 1.0
    target_max_hp: int = 0
    target_is_casting: bool = False
    target_casting_skill: str = ""
    target_element: str = "neutral"
    target_race: str = "formless"
    target_size: str = "medium"
    target_is_boss: bool = False
    target_attack_type: str = "melee"  # "melee", "ranged", "magic"

    # All monsters in range
    monsters: list[dict] = field(default_factory=list)  # each has x, y, id, name, distance, hp_pct, is_casting, etc.

    # Active DoTs on target
    active_dots: list[dict] = field(default_factory=list)  # {skill_name, damage_per_tick, ticks_remaining}


# ──────────────────────────────────────────────────────────────────────────────
# PositioningSystem
# ──────────────────────────────────────────────────────────────────────────────


class PositioningSystem:
    """Evaluates spatial combat situations and recommends positions + actions.

    Thread-safe. Designed to be called once per combat tick from combat_loop.py.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._cd_tracker = SkillCooldownTracker()
        self._elemental_matrix = ElementalMatrix()

        # Movement state
        self._zigzag_toggle: bool = False
        self._last_zigzag_switch: float = 0.0
        self._last_move_time: float = 0.0
        self._move_cooldown: float = 0.5  # don't re-path faster than this

        # Last spatial snapshot
        self._snap: SpatialSnapshot = SpatialSnapshot()

        # Grid of walkable cells (populated lazily from map data).
        # For now we assume all tiles are walkable — the aggro_pathfinder
        # already handles walkability. We work relative to monster positions.
        self._map_width: int = 500
        self._map_height: int = 500

        # Stats
        self._stats: dict[str, int] = {
            "evaluations": 0,
            "diagonal_positions": 0,
            "aoe_evasions": 0,
            "caster_interrupts": 0,
            "overkill_stops": 0,
            "skill_chains": 0,
        }

    # ── Public API ──────────────────────────────────────────────────────────

    def update_snapshot(self, snapshot: dict) -> None:
        """Extract spatial data from the bridge snapshot."""
        with self._lock:
            actors = snapshot.get("actors", [])
            position = snapshot.get("position", {})
            combat = snapshot.get("combat", {})
            vitals = snapshot.get("vitals", {}) or snapshot.get("stats", {})
            status = snapshot.get("status", {})

            snap = SpatialSnapshot()
            snap.my_x = int(position.get("x", position.get("pos_x", 0)))
            snap.my_y = int(position.get("y", position.get("pos_y", 0)))
            snap.my_hp_pct = float(vitals.get("hp_ratio", 1.0))
            snap.my_job_class = str(vitals.get("job_name", vitals.get("class", "novice"))).lower()
            snap.map_name = str(position.get("map", ""))
            snap.aggro_count = int(combat.get("aggro_count", 0))

            skills_data = snapshot.get("skills", [])
            if isinstance(skills_data, list):
                snap.my_available_skills = [
                    s.get("name", "") if isinstance(s, dict) else str(s)
                    for s in skills_data
                ]

            # Current target
            target_id = int(combat.get("target_id", 0))
            snap.target_id = target_id
            if target_id > 0:
                for actor in actors:
                    aid = int(actor.get("actor_id", actor.get("id", 0)))
                    if aid == target_id:
                        snap.target_name = str(actor.get("name", ""))
                        snap.target_x = int(actor.get("x", actor.get("pos_x", 0)))
                        snap.target_y = int(actor.get("y", actor.get("pos_y", 0)))
                        snap.target_distance = float(actor.get("distance", 0))
                        snap.target_hp_pct = float(actor.get("hp_pct", actor.get("hp_ratio", 1.0)))
                        snap.target_max_hp = int(actor.get("max_hp", 1))
                        snap.target_is_casting = bool(actor.get("is_casting", False))
                        snap.target_casting_skill = str(actor.get("casting_skill", ""))
                        snap.target_element = str(actor.get("element", "neutral")).lower()
                        snap.target_race = str(actor.get("race", "formless")).lower()
                        snap.target_size = str(actor.get("size", "medium")).lower()
                        snap.target_is_boss = bool(actor.get("is_boss", False))
                        # Infer attack type from monster name heuristics / class
                        snap.target_attack_type = self._infer_attack_type(actor)
                        break

            # All monsters
            snap.monsters = [
                a for a in actors
                if a.get("type", "") == "monster" and int(a.get("hp", 1)) > 0
            ]

            # Active DoTs (from snapshot status effects)
            snap.active_dots = self._extract_dots(snapshot)

            self._snap = snap

    def evaluate(self) -> PositionVerdict:
        """Main evaluation: score current position and decide action.

        Returns a Verdict that the combat loop should act on.
        """
        with self._lock:
            snap = self._snap
            self._stats["evaluations"] += 1

            # Phase 1: Overkill check — can we stop attacking?
            overkill = self._assess_overkill()
            if overkill.should_stop_attacking:
                self._stats["overkill_stops"] += 1
                return PositionVerdict.STOP_ATTACKING

            # Phase 2: AoE evasion — are we in a blast zone?
            aoe_threat = self._assess_aoe_threat()
            if aoe_threat is not None:
                self._stats["aoe_evasions"] += 1
                return PositionVerdict.MOVE

            # Phase 3: Cast time manipulation — should we interrupt?
            if snap.target_is_casting and snap.target_distance < 10:
                cast_time = calculate_cast_time(snap.target_casting_skill)
                if cast_time > DEFAULT_CAST_TIME_DANGER_THRESHOLD:
                    self._stats["caster_interrupts"] += 1
                    return PositionVerdict.INTERRUPT

            # Phase 4: Score current position
            score = self.score_position(snap.my_x, snap.my_y)

            # Phase 5: Score best alternative
            best = self._find_best_position()
            if best and best.total_score > score.total_score * 1.15:  # 15% improvement threshold
                return PositionVerdict.MOVE

            return PositionVerdict.STAY

    def get_movement_intent(self) -> MovementIntent | None:
        """Compute the best movement intent based on current spatial situation.

        Returns a MovementIntent or None if no movement is needed.
        """
        with self._lock:
            snap = self._snap

            # Rate-limit re-pathing
            now = time.time()
            if now - self._last_move_time < self._move_cooldown:
                return None

            snap = self._snap

            # 1. AoE evasion takes highest priority
            aoe_threat = self._assess_aoe_threat()
            if aoe_threat is not None:
                self._last_move_time = now
                return MovementIntent(
                    target_x=aoe_threat[0],
                    target_y=aoe_threat[1],
                    mode=MovementMode.AOE_EVASION,
                    priority=90,
                    reason=f"AoE evasion from {aoe_threat[2]}",
                    estimated_arrival_s=self._estimate_travel_time(snap.my_x, snap.my_y, aoe_threat[0], aoe_threat[1]),
                )

            # 2. Caster interrupt approach
            if snap.target_is_casting and snap.target_distance > 1:
                cast_time = calculate_cast_time(snap.target_casting_skill)
                if cast_time > DEFAULT_CAST_TIME_DANGER_THRESHOLD:
                    travel_time = self._estimate_travel_time(
                        snap.my_x, snap.my_y, snap.target_x, snap.target_y
                    )
                    if travel_time < cast_time * 0.8:  # We can reach before cast finishes
                        self._last_move_time = now
                        return MovementIntent(
                            target_x=snap.target_x + 1,  # Close enough to melee interrupt
                            target_y=snap.target_y + 1,
                            mode=MovementMode.CASTER_CLOSE,
                            priority=85,
                            reason=f"Close to interrupt {snap.target_casting_skill} (cast={cast_time:.1f}s, travel={travel_time:.1f}s)",
                            estimated_arrival_s=travel_time,
                        )

            # 3. Zigzag against ranged attackers
            if snap.target_attack_type == "ranged" and snap.target_distance > 3:
                intent = self._get_zigzag_intent()
                if intent is not None:
                    self._last_move_time = now
                    return intent

            # 4. Diagonal positioning against melee
            if snap.target_attack_type == "melee" and snap.target_distance <= 3:
                intent = self._get_diagonal_intent()
                if intent is not None:
                    self._last_move_time = now
                    return intent

            # 5. Position to best scoring spot
            best = self._find_best_position()
            if best is not None:
                current_score = self.score_position(snap.my_x, snap.my_y)
                if best.total_score > current_score.total_score * 1.15:
                    self._last_move_time = now
                    return MovementIntent(
                        target_x=best.x,
                        target_y=best.y,
                        mode=MovementMode.APPROACH,
                        priority=60,
                        reason=f"Position scoring: intake={best.damage_intake:.0f}, output={best.damage_output:.0f}",
                        estimated_arrival_s=self._estimate_travel_time(snap.my_x, snap.my_y, best.x, best.y),
                    )

            return None

    def get_skill_chain_intent(self) -> SkillChainIntent | None:
        """Check if we should chain a skill for spatial/tactical purposes.

        Called after get_movement_intent(). Returns a SkillChainIntent
        or None if no skill chain is needed.
        """
        with self._lock:
            snap = self._snap

            # 1. Block path with Fire Wall / Ice Wall when being chased
            if snap.aggro_count >= 3:
                # Find a position between us and the closest monster cluster
                block_pos = self._find_block_position()
                if block_pos and "Fire Wall" in snap.my_available_skills:
                    self._stats["skill_chains"] += 1
                    return SkillChainIntent(
                        skill_name="Fire Wall",
                        purpose="block_path",
                        target_x=block_pos[0],
                        target_y=block_pos[1],
                        priority=80,
                        reason=f"Block {snap.aggro_count} chasing monsters with Fire Wall",
                    )

            # 2. Buy time with Safety Wall when low HP + surrounded
            if snap.my_hp_pct < 0.5 and snap.aggro_count >= 2:
                if "Safety Wall" in snap.my_available_skills:
                    self._stats["skill_chains"] += 1
                    return SkillChainIntent(
                        skill_name="Safety Wall",
                        purpose="buy_time",
                        target_x=snap.my_x,
                        target_y=snap.my_y,
                        priority=75,
                        reason=f"Safety Wall at low HP ({snap.my_hp_pct:.0%}) with {snap.aggro_count} aggro",
                    )

            # 3. Zone denial with AoE when group is clustered
            if snap.aggro_count >= 3:
                cluster_center = self._find_monster_cluster_center()
                if cluster_center:
                    for skill_name in SKILL_PURPOSE_ZONE_DENIAL:
                        if skill_name in snap.my_available_skills:
                            self._stats["skill_chains"] += 1
                            return SkillChainIntent(
                                skill_name=skill_name,
                                purpose="zone_denial",
                                target_x=cluster_center[0],
                                target_y=cluster_center[1],
                                priority=70,
                                reason=f"Zone denial on {snap.aggro_count} clustered monsters",
                            )

            return None

    def score_position(self, x: int, y: int) -> PositionScore:
        """Score a candidate position by expected damage intake vs output.

        Returns a PositionScore with all factors computed.
        Positive total_score = good position. Higher is better.
        """
        with self._lock:
            snap = self._snap
            score = PositionScore(x=x, y=y)

            if not snap.monsters:
                score.total_score = 100.0  # No enemies = perfect position
                return score

            # ── Damage intake estimation ──
            intake = 0.0
            diagonal_bonus = 0.0
            aoe_exposure = 0.0
            ranged_exposure = 0.0

            for mob in snap.monsters:
                mx = int(mob.get("x", mob.get("pos_x", 0)))
                my = int(mob.get("y", mob.get("pos_y", 0)))
                dist = self._tile_distance(x, y, mx, my)

                if dist > 15:
                    continue  # Too far to matter

                # Base damage estimation from monster threat level
                mob_hp_pct = float(mob.get("hp_pct", mob.get("hp_ratio", 1.0)))
                is_casting = bool(mob.get("is_casting", False))
                mob_attack_type = self._infer_attack_type(mob)

                # Monsters deal more damage when healthy
                damage_factor = (1.0 - mob_hp_pct * 0.5)  # 0.5-1.0, lower HP = less threat

                # Distance falloff for melee monsters
                if mob_attack_type == "melee":
                    if dist <= 1:
                        melee_danger = 10.0 * damage_factor
                    elif dist <= 2:
                        melee_danger = 5.0 * damage_factor
                    elif dist <= 5:
                        melee_danger = 2.0 * damage_factor
                    else:
                        melee_danger = 1.0 * damage_factor

                    # Diagonal advantage: standing diagonal reduces melee hits
                    if self._is_diagonal(x, y, mx, my):
                        # Monsters at diagonal positions hit less because
                        # in RO, diagonal approach takes longer for melee mobs
                        diag_factor = 0.6
                        diagonal_bonus += 10.0
                    else:
                        diag_factor = 1.0

                    intake += melee_danger * diag_factor

                elif mob_attack_type == "ranged":
                    if dist <= 10:
                        ranged_danger = 5.0 * damage_factor * max(0.2, 1.0 - dist / 15.0)
                        intake += ranged_danger
                        if dist < 5:
                            ranged_exposure += 3.0 * damage_factor
                    else:
                        ranged_danger = 0.5 * damage_factor
                        intake += ranged_danger

                elif mob_attack_type == "magic":
                    if is_casting:
                        # Danger from the casting monster
                        cast_skill = str(mob.get("casting_skill", ""))
                        cast_time = calculate_cast_time(cast_skill) if cast_skill else 2.0
                        aoe_danger = 8.0 * damage_factor
                        intake += aoe_danger
                        if self._is_in_aoe_blast(x, y, mx, my, cast_skill):
                            aoe_exposure += 15.0 * damage_factor

                # Casting monsters add interrupt value
                if is_casting:
                    cast_skill = str(mob.get("casting_skill", ""))
                    ct = calculate_cast_time(cast_skill) if cast_skill else 2.0
                    if dist <= 5 and ct > 1.0:
                        score.interrupt_opportunity = max(
                            score.interrupt_opportunity,
                            min(1.0, ct / 5.0),
                        )

            score.damage_intake = intake
            score.diagonal_advantage = min(1.0, diagonal_bonus / 20.0)
            score.aoe_safety = max(0.0, 1.0 - aoe_exposure / 30.0)
            score.cover_from_ranged = max(0.0, 1.0 - ranged_exposure / 15.0)

            # ── Damage output estimation ──
            output = 0.0
            if snap.target_id > 0 and snap.target_distance > 0:
                # Better output at appropriate range for class
                if snap.my_job_class in ("archer", "hunter", "sniper", "ranger"):
                    if 5 <= snap.target_distance <= 9:
                        output = 10.0  # Sweet spot for ranged
                    elif snap.target_distance <= 4:
                        output = 5.0  # Too close for comfort
                    else:
                        output = 3.0  # Too far
                elif snap.my_job_class in ("mage", "wizard", "high_wizard", "warlock"):
                    if 5 <= snap.target_distance <= 9:
                        output = 10.0  # Safe casting range
                    elif snap.target_distance <= 2:
                        output = 2.0  # Needs to back off
                    else:
                        output = 6.0
                elif snap.my_job_class in ("acolyte", "priest", "high_priest", "archbishop"):
                    if 3 <= snap.target_distance <= 7:
                        output = 8.0
                    else:
                        output = 5.0
                else:
                    # Melee classes want to be close
                    if snap.target_distance <= 2:
                        output = 10.0
                    elif snap.target_distance <= 4:
                        output = 7.0
                    else:
                        output = 3.0

                # Elemental advantage bonus
                if snap.target_element:
                    mult = self._elemental_matrix.get_elemental_multiplier(
                        "Neutral", snap.target_element
                    )
                    if mult > 1.0:
                        output *= mult

            score.damage_output = output

            # ── Combined score ──
            # Base: output - intake. Lower intake = higher score.
            score.intake_output_ratio = (
                score.damage_intake / max(0.01, score.damage_output)
            )

            score.total_score = (
                score.damage_output * 1.0
                - score.damage_intake * 1.5  # Penalize damage intake more
                + score.diagonal_advantage * 15.0
                + score.aoe_safety * 20.0
                + score.cover_from_ranged * 10.0
                + score.interrupt_opportunity * 12.0
            )

            # Clamp to reasonable range
            score.total_score = max(-100.0, min(200.0, score.total_score))

            return score

    def assess_overkill(self) -> OverkillAssessment:
        """Public wrapper for _assess_overkill, thread-safe."""
        with self._lock:
            return self._assess_overkill()

    def should_zigzag(self) -> bool:
        """Check if we should be zigzagging against current target."""
        with self._lock:
            snap = self._snap
            return (
                snap.target_attack_type == "ranged"
                and snap.target_distance > 3
            )

    def get_stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def reset_stats(self) -> None:
        with self._lock:
            self._stats = {k: 0 for k in self._stats}

    # ── Internal: Overkill Assessment ──────────────────────────────────────

    def _assess_overkill(self) -> OverkillAssessment:
        """Check if active DoTs will finish the target.

        If yes, we can stop attacking and save SP/time.
        """
        snap = self._snap
        assessment = OverkillAssessment()

        if not snap.active_dots or snap.target_max_hp <= 0:
            return assessment

        target_hp_current = int(snap.target_max_hp * snap.target_hp_pct)
        assessment.target_hp_remaining = target_hp_current

        total_dot_damage = 0.0
        longest_dot_time = 0.0

        for dot in snap.active_dots:
            dmg_per_tick = float(dot.get("damage_per_tick", 0))
            ticks = int(dot.get("ticks_remaining", 0))
            tick_interval = float(dot.get("tick_interval_s", 1.0))

            total_dot_damage += dmg_per_tick * ticks
            dot_time = ticks * tick_interval
            if dot_time > longest_dot_time:
                longest_dot_time = dot_time

        assessment.dot_damage_remaining = total_dot_damage
        assessment.estimated_time_to_die_s = longest_dot_time

        # If DoT will do more than OVERKILL_DOT_THRESHOLD of remaining HP
        if total_dot_damage >= target_hp_current * OVERKILL_DOT_THRESHOLD:
            assessment.will_die_to_dot = True
            assessment.should_stop_attacking = True

        return assessment

    # ── Internal: AoE Threat Assessment ────────────────────────────────────

    def _assess_aoe_threat(self) -> tuple[int, int, str] | None:
        """Check if we're in an AoE blast zone.

        Returns (safe_x, safe_y, skill_name) if we need to move,
        or None if we're safe.
        """
        snap = self._snap

        for mob in snap.monsters:
            if not mob.get("is_casting", False):
                continue

            cast_skill = str(mob.get("casting_skill", ""))
            if not cast_skill:
                continue

            # Known dangerous AoE skills
            aoe_skills = {
                "Storm Gust": 7,
                "Meteor Storm": 7,
                "Lord of Vermilion": 7,
                "Heaven's Drive": 5,
                "Fire Ball": 3,
                "Frost Nova": 4,
                "Fire Storm": 5,
                "Fire Wall": 2,
                "Ice Wall": 2,
                "Quagmire": 4,
            }

            radius = aoe_skills.get(cast_skill, 0)
            if radius <= 0:
                continue

            mx = int(mob.get("x", mob.get("pos_x", 0)))
            my = int(mob.get("y", mob.get("pos_y", 0)))
            dist_to_center = self._tile_distance(snap.my_x, snap.my_y, mx, my)

            if dist_to_center <= radius + 2:
                # We're in the blast zone — find the fastest escape vector
                safe_x, safe_y = self._find_escape_vector(mx, my, radius + 3)
                return (safe_x, safe_y, cast_skill)

        return None

    # ── Internal: Diagonal Positioning ─────────────────────────────────────

    def _get_diagonal_intent(self) -> MovementIntent | None:
        """Compute intent to move to a diagonal position relative to target.

        In RO, standing diagonally to a mob means only 1 mob can melee you
        at a time (2x2 hitbox interaction). Standing axis-aligned means
        2+ mobs can reach you.
        """
        snap = self._snap

        # Compute 4 diagonal positions around the target
        diagonals = [
            (snap.target_x + 2, snap.target_y + 2),
            (snap.target_x + 2, snap.target_y - 2),
            (snap.target_x - 2, snap.target_y + 2),
            (snap.target_x - 2, snap.target_y - 2),
        ]

        best_score = -999.0
        best_pos = None

        for dx, dy in diagonals:
            # Bounds check
            if not (0 <= dx < self._map_width and 0 <= dy < self._map_height):
                continue

            score = self.score_position(dx, dy)

            # Bonus for actually being diagonal
            if self._is_diagonal(snap.my_x, snap.my_y, dx, dy):
                score.total_score += 10.0
                self._stats["diagonal_positions"] += 1

            if score.total_score > best_score:
                best_score = score.total_score
                best_pos = (dx, dy)

        if best_pos:
            return MovementIntent(
                target_x=best_pos[0],
                target_y=best_pos[1],
                mode=MovementMode.DIAGONAL_POSITION,
                priority=65,
                reason="Move to diagonal position for 1-hit melee reduction",
                estimated_arrival_s=self._estimate_travel_time(
                    snap.my_x, snap.my_y, best_pos[0], best_pos[1]
                ),
            )
        return None

    # ── Internal: Zigzag Movement ──────────────────────────────────────────

    def _get_zigzag_intent(self) -> MovementIntent | None:
        """Compute zigzag movement against ranged attackers.

        Zigzagging makes archers miss more often and reduces incoming damage
        by varying the approach vector.
        """
        snap = self._snap
        now = time.time()

        # Toggle zigzag direction periodically
        if now - self._last_zigzag_switch > DEFAULT_ZIGZAG_INTERVAL:
            self._zigzag_toggle = not self._zigzag_toggle
            self._last_zigzag_switch = now

        # Zigzag perpendicular to target direction
        dx = snap.target_x - snap.my_x
        dy = snap.target_y - snap.my_y

        # Perpendicular vector (swap and negate)
        if self._zigzag_toggle:
            perp_x = -dy
            perp_y = dx
        else:
            perp_x = dy
            perp_y = -dx

        # Normalize to 2-cell steps
        length = max(1, int(math.sqrt(perp_x * perp_x + perp_y * perp_y)))
        step_x = perp_x // length * 2 if abs(perp_x) > 0 else 0
        step_y = perp_y // length * 2 if abs(perp_y) > 0 else 0

        target_x = snap.my_x + step_x
        target_y = snap.my_y + step_y

        # Bounds check
        target_x = max(2, min(self._map_width - 2, target_x))
        target_y = max(2, min(self._map_height - 2, target_y))

        return MovementIntent(
            target_x=target_x,
            target_y=target_y,
            mode=MovementMode.ZIGZAG,
            priority=70,
            reason=f"Zigzag against ranged target ({'right' if self._zigzag_toggle else 'left'})",
            estimated_arrival_s=self._estimate_travel_time(snap.my_x, snap.my_y, target_x, target_y),
            should_skip_if_interrupted=True,
        )

    # ── Internal: Find Best Position ───────────────────────────────────────

    def _find_best_position(self) -> PositionScore | None:
        """Search for the highest-scored position near the player."""
        snap = self._snap
        candidates: list[PositionScore] = []

        # Search in expanding rings around current position
        for radius in range(1, min(8, self._map_width // 4)):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1, max(1, radius)):
                    if abs(dx) == radius or abs(dy) == radius:
                        cx = snap.my_x + dx
                        cy = snap.my_y + dy

                        # Bounds
                        if not (2 <= cx < self._map_width - 2 and 2 <= cy < self._map_height - 2):
                            continue

                        score = self.score_position(cx, cy)
                        candidates.append(score)

            if len(candidates) >= MAX_POSITION_CANDIDATES:
                break

        if not candidates:
            return None

        # Sort by total_score descending
        candidates.sort(key=lambda s: s.total_score, reverse=True)
        return candidates[0]

    # ── Internal: Block / Escape Vector ────────────────────────────────────

    def _find_block_position(self) -> tuple[int, int] | None:
        """Find a position between us and the closest monster cluster to place a blocking skill."""
        snap = self._snap
        if not snap.monsters:
            return None

        # Find average position of monsters within 10 cells
        nearby = [
            m for m in snap.monsters
            if self._tile_distance(
                snap.my_x, snap.my_y,
                int(m.get("x", m.get("pos_x", 0))),
                int(m.get("y", m.get("pos_y", 0))),
            ) <= 10
        ]
        if not nearby:
            return None

        avg_monster_x = int(sum(
            int(m.get("x", m.get("pos_x", 0))) for m in nearby
        ) / len(nearby))
        avg_monster_y = int(sum(
            int(m.get("y", m.get("pos_y", 0))) for m in nearby
        ) / len(nearby))

        # Place wall halfway between us and the monster cluster
        wall_x = (snap.my_x + avg_monster_x) // 2
        wall_y = (snap.my_y + avg_monster_y) // 2

        return (wall_x, wall_y)

    def _find_escape_vector(
        self, from_x: int, from_y: int, min_distance: int
    ) -> tuple[int, int]:
        """Find the safest direction to flee from a danger point."""
        snap = self._snap

        # Use map dimensions to find the farthest point
        dx = snap.my_x - from_x
        dy = snap.my_y - from_y

        if dx == 0 and dy == 0:
            dx, dy = 1, 1

        length = math.sqrt(dx * dx + dy * dy)
        norm_x = dx / length
        norm_y = dy / length

        # Project outward
        safe_x = int(snap.my_x + norm_x * min_distance * 2)
        safe_y = int(snap.my_y + norm_y * min_distance * 2)

        # Bounds
        safe_x = max(2, min(self._map_width - 2, safe_x))
        safe_y = max(2, min(self._map_height - 2, safe_y))

        return (safe_x, safe_y)

    def _find_monster_cluster_center(self) -> tuple[int, int] | None:
        """Find the center of the monster cluster for AoE targeting."""
        snap = self._snap
        monsters_nearby = [
            m for m in snap.monsters
            if self._tile_distance(
                snap.my_x, snap.my_y,
                int(m.get("x", m.get("pos_x", 0))),
                int(m.get("y", m.get("pos_y", 0))),
            ) <= 12
        ]

        if len(monsters_nearby) < 3:
            return None

        avg_x = int(sum(
            int(m.get("x", m.get("pos_x", 0))) for m in monsters_nearby
        ) / len(monsters_nearby))
        avg_y = int(sum(
            int(m.get("y", m.get("pos_y", 0))) for m in monsters_nearby
        ) / len(monsters_nearby))

        return (avg_x, avg_y)

    # ── Internal: Helpers ──────────────────────────────────────────────────

    def _infer_attack_type(self, monster: dict) -> str:
        """Infer monster attack type from available data."""
        name = str(monster.get("name", "")).lower()
        is_casting = bool(monster.get("is_casting", False))

        # Ranged monster heuristics
        ranged_keywords = ("archer", "hunter", "sniper", "ranger", "bow", "arrow")
        magic_keywords = ("mage", "wizard", "warlock", "sorc", "sage", "professor")

        if any(kw in name for kw in ranged_keywords):
            return "ranged"
        if any(kw in name for kw in magic_keywords) or is_casting:
            return "magic"
        return "melee"

    def _extract_dots(self, snapshot: dict) -> list[dict]:
        """Extract active damage-over-time effects from snapshot status."""
        dots: list[dict] = []
        status = snapshot.get("status", {})
        statuses = status.get("statuses", []) if isinstance(status, dict) else []

        for effect in statuses:
            if isinstance(effect, dict):
                name = str(effect.get("name", "")).lower()
                if any(kw in name for kw in ("fire wall", "poison", "venom", "burn", "bleed", "dot")):
                    dots.append({
                        "skill_name": effect.get("name", "unknown"),
                        "damage_per_tick": int(effect.get("damage_per_tick", effect.get("value", 10))),
                        "ticks_remaining": int(effect.get("ticks_remaining", effect.get("duration", 5))),
                        "tick_interval_s": float(effect.get("tick_interval", 1.0)),
                    })
        return dots

    @staticmethod
    def _tile_distance(x1: int, y1: int, x2: int, y2: int) -> float:
        """Euclidean distance in tiles."""
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    @staticmethod
    def _is_diagonal(x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if two positions are diagonal to each other.

        Diagonal = both x AND y differ (not axis-aligned).
        """
        return x1 != x2 and y1 != y2

    @staticmethod
    def _is_in_aoe_blast(
        px: int, py: int, center_x: int, center_y: int, skill_name: str
    ) -> bool:
        """Check if a point is within the blast radius of a known AoE skill."""
        aoe_radii = {
            "Storm Gust": 7,
            "Meteor Storm": 7,
            "Lord of Vermilion": 7,
            "Heaven's Drive": 5,
            "Fire Ball": 3,
            "Frost Nova": 4,
            "Fire Storm": 5,
            "Fire Wall": 2,
            "Quagmire": 4,
        }
        radius = aoe_radii.get(skill_name, 0)
        if radius <= 0:
            return False
        dist = math.sqrt((px - center_x) ** 2 + (py - center_y) ** 2)
        return dist <= radius

    def _estimate_travel_time(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Estimate travel time in seconds between two tiles."""
        dist = self._tile_distance(x1, y1, x2, y2)
        return dist / DEFAULT_MOVE_SPEED_CELLS_PER_SECOND


# ── Global Singleton ──────────────────────────────────────────────────────────

_positioning_system: PositioningSystem | None = None
_positioning_system_lock = RLock()


def get_positioning_system() -> PositioningSystem:
    """Get or create the global PositioningSystem singleton."""
    global _positioning_system
    with _positioning_system_lock:
        if _positioning_system is None:
            _positioning_system = PositioningSystem()
        return _positioning_system
