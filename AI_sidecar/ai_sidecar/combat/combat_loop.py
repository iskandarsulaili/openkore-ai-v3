"""
Real Combat Loop — continuous combat execution independent of the LLM.

Target → approach → skill rotation → cooldown management → reposition → next target.
The LLM only gets involved for strategic decisions. Combat is a reflex loop.

This module wires together:
  - threat_targeting.py  → target selection
  - skill_rotation.py   → skill execution based on build
  - elemental_matrix.py  → elemental advantage
  - buff_maintenance.py  → buff upkeep
  - gear_swapper.py      → gear swapping for situation
  - resource_manager.py  → potion/consumable usage
  - build_manager.py     → build-aware decisions
  - reflex_combat.py     → emergency reflexes
  - action_executor.py   → enqueue actions to bridge
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)

COMBAT_TICK_INTERVAL = 0.2  # 200ms per combat tick

# Potion/consumable thresholds — can be overridden by death analysis
HP_EMERGENCY_THRESHOLD = 0.35
HP_CRITICAL_THRESHOLD = 0.20
HP_DANGER_THRESHOLD = 0.10
SP_LOW_THRESHOLD = 0.20
SP_EMPTY_THRESHOLD = 0.10
MAX_AGGRO_DEFAULT = 5

# Buff recast window
BUFF_RECAST_WINDOW = 15  # seconds before expiry to recast

# Gear swap cooldown
GEAR_SWAP_COOLDOWN = 5.0  # seconds between gear swaps


@dataclass
class CombatState:
    """Current state of the combat loop."""
    is_active: bool = False
    current_target_id: int = 0
    current_target_name: str = ""
    current_target_hp_pct: float = 1.0
    current_target_distance: float = 0.0
    current_target_element: str = "neutral"
    current_target_size: str = "medium"
    current_target_race: str = "formless"
    current_target_is_boss: bool = False
    current_target_is_casting: bool = False
    current_skill: str = ""
    current_skill_index: int = 0
    current_rotation_name: str = ""
    last_skill_time: float = 0.0
    skill_cooldowns: dict[str, float] = field(default_factory=dict)
    aggro_count: int = 0
    my_hp_pct: float = 1.0
    my_sp_pct: float = 1.0
    my_sp: int = 0
    my_hp: int = 0
    my_max_hp: int = 0
    my_job_class: str = "novice"
    my_build_name: str = ""
    my_weapon_element: str = "neutral"
    my_weapon_type: str = "sword"
    my_buffs: list[str] = field(default_factory=list)
    my_available_skills: list[str] = field(default_factory=list)
    is_in_combat: bool = False
    combat_started_at: float = 0.0
    kills_this_session: int = 0
    ticks_this_session: int = 0
    last_gear_swap_time: float = 0.0
    current_gear_set: str = "farming_set"
    last_buff_check_time: float = 0.0
    last_potion_time: float = 0.0
    last_target_acquire_time: float = 0.0
    is_sitting: bool = False
    map_name: str = ""
    enemies_nearby: int = 0
    party_members_nearby: int = 0
    was_dead: bool = False
    # Dynamic thresholds (overridden by death analysis)
    max_aggro: int = 5
    heal_threshold: float = 0.6
    flee_hp_pct: float = 0.3
    min_potion_stock: int = 10


class CombatLoop:
    """Continuous combat loop that drives all combat subsystems."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._state: CombatState = CombatState()
        self._enqueue_fn: Callable | None = None
        self._get_snapshot_fn: Callable | None = None
        self._last_tick: float = 0.0

        # Subsystem references (set via setters)
        self._threat_targeting: Any = None
        self._skill_rotation: Any = None
        self._elemental_matrix: Any = None
        self._buff_maintenance: Any = None
        self._gear_swapper: Any = None
        self._resource_manager: Any = None
        self._build_manager: Any = None
        self._reflex_combat: Any = None
        self._action_executor: Any = None
        self._action_queue: Any = None
        self._gather_and_kill: Any = None

    # ── Public API ──

    def start(self) -> None:
        with self._lock:
            self._state.is_active = True
            logger.info("combat_loop_started")

    def stop(self) -> None:
        with self._lock:
            self._state.is_active = False
            logger.info("combat_loop_stopped")

    def tick(self) -> str | None:
        """Execute one combat tick. Returns the action performed, or None."""
        with self._lock:
            if not self._state.is_active:
                return None

            now = time.time()
            if now - self._last_tick < COMBAT_TICK_INTERVAL:
                return None
            self._last_tick = now
            self._state.ticks_this_session += 1

            # ── PHASE 0: Emergency reflexes (highest priority) ──
            action = self._check_emergency_reflexes(now)
            if action:
                return action

            # ── PHASE 1: Buff maintenance ──
            action = self._check_buffs(now)
            if action:
                return action

            # ── PHASE 2: Resource management (potions, SP) ──
            action = self._check_resources(now)
            if action:
                return action

            # ── PHASE 3: Gear swap if needed ──
            action = self._check_gear_swap(now)
            if action:
                return action

            # ── PHASE 4: Target acquisition ──
            if self._state.current_target_id == 0 or self._state.current_target_hp_pct <= 0:
                action = self._acquire_target(now)
                if action:
                    return action
                # No target found — sit to regen
                if self._state.my_sp_pct < SP_EMPTY_THRESHOLD and not self._state.is_sitting:
                    self._enqueue_action("sit", "ai_manual")
                    self._state.is_sitting = True
                    return "sit_to_regen"
                return None

            # ── PHASE 5: Movement during combat ──
            action = self._handle_combat_movement(now)
            if action:
                return action

            # ── PHASE 6: Skill execution ──
            action = self._execute_skill_rotation(now)
            if action:
                return action

            # ── PHASE 7: Basic attack fallback ──
            return self._basic_attack()

    def update_state(self, snapshot: dict) -> None:
        """Update combat state from a bridge snapshot."""
        with self._lock:
            vitals = snapshot.get("vitals", {}) or snapshot.get("stats", {})
            combat = snapshot.get("combat", {})
            actors = snapshot.get("actors", [])
            position = snapshot.get("position", {})
            inventory = snapshot.get("inventory", {})
            skills_data = snapshot.get("skills", [])
            status = snapshot.get("status", {})

            # ── Vitals ──
            self._state.my_hp = int(vitals.get("hp", vitals.get("hp", 1)))
            self._state.my_max_hp = int(vitals.get("hp_max", vitals.get("max_hp", 1)))
            self._state.my_hp_pct = float(vitals.get("hp_ratio", 1.0))
            if self._state.my_max_hp > 0:
                self._state.my_hp_pct = self._state.my_hp / self._state.my_max_hp
            self._state.my_sp = int(vitals.get("sp", 0))
            my_max_sp = int(vitals.get("sp_max", vitals.get("max_sp", 1)))
            self._state.my_sp_pct = float(vitals.get("sp_ratio", 1.0))
            if my_max_sp > 0:
                self._state.my_sp_pct = self._state.my_sp / my_max_sp

            # ── Position ──
            self._state.map_name = str(position.get("map", ""))

            # ── Status ──
            self._state.is_sitting = bool(status.get("sitting", False))

            # ── Combat info ──
            self._state.aggro_count = int(combat.get("aggro_count", 0))
            self._state.enemies_nearby = len([a for a in actors if a.get("type", "") == "monster" and a.get("hp", 0) > 0])

            # ── Job class ──
            self._state.my_job_class = str(vitals.get("job_name", vitals.get("class", "novice"))).lower()

            # ── Available skills ──
            if isinstance(skills_data, list):
                self._state.my_available_skills = [
                    s.get("name", "") if isinstance(s, dict) else str(s)
                    for s in skills_data
                ]

            # ── Buffs ──
            buffs_raw = snapshot.get("buffs", [])
            if isinstance(buffs_raw, list):
                self._state.my_buffs = [
                    b.get("name", "") if isinstance(b, dict) else str(b)
                    for b in buffs_raw
                ]

            # ── Find current target ──
            target_id = int(combat.get("target_id", 0))
            if target_id > 0:
                for actor in actors:
                    if int(actor.get("actor_id", actor.get("id", 0))) == target_id:
                        self._state.current_target_id = target_id
                        self._state.current_target_name = str(actor.get("name", ""))
                        self._state.current_target_hp_pct = float(actor.get("hp_pct", actor.get("hp_ratio", 1.0)))
                        self._state.current_target_distance = float(actor.get("distance", 0))
                        self._state.current_target_element = str(actor.get("element", "neutral")).lower()
                        self._state.current_target_size = str(actor.get("size", "medium")).lower()
                        self._state.current_target_race = str(actor.get("race", "formless")).lower()
                        self._state.current_target_is_boss = bool(actor.get("is_boss", False))
                        self._state.current_target_is_casting = bool(actor.get("is_casting", False))
                        self._state.is_in_combat = True
                        if self._state.combat_started_at == 0:
                            self._state.combat_started_at = time.time()
                        break

            # ── Check if target is dead ──
            if self._state.current_target_hp_pct <= 0 and self._state.is_in_combat:
                self._state.kills_this_session += 1
                self._state.current_target_id = 0
                self._state.is_in_combat = False
                self._state.combat_started_at = 0
                self._state.current_skill_index = 0

            # ── Update resource manager with inventory ──
            if self._resource_manager and isinstance(inventory, dict):
                potions = {}
                for item_name, item_data in inventory.items():
                    if isinstance(item_data, dict):
                        name = item_data.get("name", item_name) if isinstance(item_data, dict) else item_name
                        if "potion" in str(name).lower() or "white" in str(name).lower() or "blue" in str(name).lower():
                            potions[name] = item_data.get("amount", item_data.get("quantity", 1)) if isinstance(item_data, dict) else 1
                if potions:
                    self._resource_manager.update_potion_stock(potions)

            # ── Track economic data ──
            try:
                _ee = getattr(self._runtime, "economic_engine", None) if hasattr(self, '_runtime') else None
                if _ee is None:
                    from ai_sidecar.economy.economic_engine import get_economic_engine
                    _ee = get_economic_engine()
                if _ee is not None and self._state.map_name:
                    # Record a lightweight economic snapshot
                    from ai_sidecar.economy.economic_engine import EconomicSnapshot
                    _ee.record_snapshot(EconomicSnapshot(
                        map_name=self._state.map_name,
                        timestamp=time.time(),
                        zeny_earned=0,
                        zeny_spent=0,
                        items_dropped=[],
                        exp_earned=0,
                        time_elapsed_seconds=5.0,
                        monsters_killed=0,
                        deaths=0,
                    ))
            except Exception:
                pass

            # ── Track death events ──
            try:
                _da = getattr(self._runtime, "death_analysis", None) if hasattr(self, '_runtime') else None
                if _da is None:
                    from ai_sidecar.learning.death_analysis import get_death_analyzer
                    _da = get_death_analyzer()
                if _da is not None and status.get("dead", False) and not self._state.was_dead:
                    self._state.was_dead = True
                    from ai_sidecar.learning.death_analysis import DeathRecord
                    _da.record_death(DeathRecord(
                        timestamp=time.time(),
                        map_name=self._state.map_name,
                        position=(0, 0),
                        monster_name=self._state.current_target_name or "unknown",
                        monster_id=self._state.current_target_id,
                        hp_before_death=self._state.my_hp,
                        max_hp=self._state.my_max_hp,
                        aggro_count=self._state.aggro_count,
                        had_potions=True,
                        was_casting=False,
                        buffs_active=list(self._state.my_buffs),
                        seconds_since_last_heal=0,
                        cause_of_death="unknown",
                        lesson_learned="",
                    ))
                elif not status.get("dead", False):
                    self._state.was_dead = False
            except Exception:
                pass

            # ── Apply death analysis adjustments ──
            try:
                _da = getattr(self._runtime, "death_analysis", None) if hasattr(self, '_runtime') else None
                if _da is None:
                    from ai_sidecar.learning.death_analysis import get_death_analyzer
                    _da = get_death_analyzer()
                if _da is not None:
                    suggestions = _da.get_suggested_adjustments()
                    for adj in suggestions:
                        param = adj.parameter
                        new_val = adj.new_value
                        if param == "max_aggro":
                            self._state.max_aggro = max(1, int(new_val))
                        elif param == "heal_threshold":
                            self._state.heal_threshold = max(0.1, min(0.9, new_val))
                        elif param == "flee_hp_pct":
                            self._state.flee_hp_pct = max(0.05, min(0.5, new_val))
                        elif param == "min_potion_stock":
                            self._state.min_potion_stock = max(0, int(new_val))
            except Exception:
                pass

            # ── Update threat targeting ──
            if self._threat_targeting and actors:
                active_ids = set()
                for actor in actors:
                    aid = int(actor.get("actor_id", actor.get("id", 0)))
                    if aid > 0 and actor.get("type", "") == "monster":
                        active_ids.add(aid)
                        self._threat_targeting.update_monster(
                            aid,
                            name=actor.get("name", ""),
                            hp=int(actor.get("hp", 1)),
                            max_hp=int(actor.get("max_hp", 1)),
                            distance=int(actor.get("distance", 0)),
                            element=actor.get("element", "neutral"),
                            size=actor.get("size", "medium"),
                            race=actor.get("race", "formless"),
                            is_boss=bool(actor.get("is_boss", False)),
                            is_aggressive=bool(actor.get("is_aggressive", True)),
                            is_casting=bool(actor.get("is_casting", False)),
                            casting_skill=actor.get("casting_skill", ""),
                        )
                self._threat_targeting.cleanup_monsters(active_ids)

    # ── Internal Phases ──

    def _check_emergency_reflexes(self, now: float) -> str | None:
        """Phase 0: Check emergency reflexes — bypass everything else."""
        s = self._state

        # 1. HP emergency — use potion
        if s.my_hp_pct < HP_EMERGENCY_THRESHOLD:
            if now - s.last_potion_time > 0.5:  # potion cooldown
                s.last_potion_time = now
                self._enqueue_action("use_potion_or_heal", "use_item")
                logger.info("combat_emergency: hp=%.0f%% < %.0f%% — using potion",
                            s.my_hp_pct * 100, HP_EMERGENCY_THRESHOLD * 100)
                return "use_potion_or_heal"

        # 2. Critical HP + overwhelmed — flee
        if s.my_hp_pct < HP_CRITICAL_THRESHOLD and s.aggro_count > 3:
            self._enqueue_action("flee_to_safe_spot", "ai_manual")
            logger.warning("combat_flee: hp=%.0f%% aggro=%d", s.my_hp_pct * 100, s.aggro_count)
            return "flee_to_safe_spot"

        # 3. Danger HP — teleport
        if s.my_hp_pct < HP_DANGER_THRESHOLD:
            self._enqueue_action("teleport_away", "use_item")
            logger.warning("combat_teleport: hp=%.0f%%", s.my_hp_pct * 100)
            return "teleport_away"

        # 4. Interrupt casting target
        if s.current_target_is_casting and s.current_target_distance < 10:
            self._enqueue_action("interrupt_caster", "attack_skill")
            return "interrupt_caster"

        # 5. Too many aggro — AoE clear
        if s.aggro_count > 5:
            self._enqueue_action("use_strongest_aoe", "attack_skill")
            return "use_strongest_aoe"
        if s.aggro_count > 3:
            self._enqueue_action("use_aoe_skill", "attack_skill")
            return "use_aoe_skill"

        return None

    def _check_buffs(self, now: float) -> str | None:
        """Phase 1: Check and maintain buffs."""
        s = self._state
        if not self._buff_maintenance:
            return None

        # Only check buffs every 5 seconds
        if now - s.last_buff_check_time < 5.0:
            return None
        s.last_buff_check_time = now

        # Get buffs that need recasting
        active_buffs = {name: {"expires_at": 0} for name in s.my_buffs}
        buffs_to_cast = self._buff_maintenance.get_buffs_to_cast(
            active_buffs=active_buffs,
            current_sp=s.my_sp,
            available_skills=set(s.my_available_skills),
        )

        if buffs_to_cast:
            buff = buffs_to_cast[0]
            self._enqueue_action(f"cast_{buff.skill_name.lower().replace(' ', '_')}", "attack_skill")
            logger.info("combat_buff: casting %s (priority=%d)", buff.name, buff.priority)
            return f"cast_{buff.name}"

        return None

    def _check_resources(self, now: float) -> str | None:
        """Phase 2: Check resources — potions, SP management."""
        s = self._state

        # Low SP — sit to regen
        if s.my_sp_pct < SP_EMPTY_THRESHOLD and s.aggro_count == 0 and not s.is_sitting:
            self._enqueue_action("rest_and_regen_sp", "ai_manual")
            s.is_sitting = True
            logger.info("combat_regen: sp=%.0f%% — sitting to regen", s.my_sp_pct * 100)
            return "rest_and_regen_sp"

        # Low SP in combat — use basic attack
        if s.my_sp_pct < SP_LOW_THRESHOLD and s.is_in_combat:
            return "use_basic_attack_only"

        # Check resource manager for restock needs
        if self._resource_manager:
            resource_state = self._resource_manager.get_resource_state()
            if resource_state.needs_restock and resource_state.potion_count == 0:
                logger.warning("combat_out_of_potions: restocking needed")
                return "restock_potions"

        return None

    def _check_gear_swap(self, now: float) -> str | None:
        """Phase 3: Check if gear needs swapping for elemental/race advantage."""
        s = self._state
        if not self._gear_swapper or not s.current_target_id:
            return None

        # Only swap every GEAR_SWAP_COOLDOWN seconds
        if now - s.last_gear_swap_time < GEAR_SWAP_COOLDOWN:
            return None

        # Get best gear for current target
        best_gear = self._gear_swapper.get_best_gear_for_target(
            target_element=s.current_target_element,
            target_size=s.current_target_size,
            target_race=s.current_target_race,
            is_boss=s.current_target_is_boss,
            job_class=s.my_job_class,
        )

        if best_gear and best_gear.name != s.current_gear_set:
            s.last_gear_swap_time = now
            s.current_gear_set = best_gear.name
            s.my_weapon_element = best_gear.weapon_element
            s.my_weapon_type = best_gear.weapon_type

            commands = self._gear_swapper.get_gear_swap_commands(None, best_gear)
            for cmd in commands:
                self._enqueue_action_raw(cmd, "use_item")
            logger.info("combat_gear_swap: %s → %s (element=%s, weapon=%s)",
                        s.current_gear_set, best_gear.name,
                        best_gear.weapon_element, best_gear.weapon_type)
            return f"swap_gear_{best_gear.name}"

        return None

    def _acquire_target(self, now: float) -> str | None:
        """Phase 4: Acquire a new target using threat targeting."""
        s = self._state
        if not self._threat_targeting:
            return None

        # Rate-limit target acquisition
        if now - s.last_target_acquire_time < 1.0:
            return None
        s.last_target_acquire_time = now

        # Get best target from threat targeting
        has_aoe = any("aoe" in sk.lower() for sk in s.my_available_skills)
        best_target = self._threat_targeting.get_best_target(
            player_class=s.my_job_class,
            party_size=1,
            has_aoe=has_aoe,
        )

        if best_target:
            target_id = best_target.get("monster_id", 0)
            if target_id > 0:
                s.current_target_id = target_id
                s.current_target_name = best_target.get("name", "")
                s.current_target_hp_pct = 1.0
                s.current_target_distance = best_target.get("distance", 0)
                s.is_in_combat = True
                s.combat_started_at = now
                s.current_skill_index = 0

                # Select rotation based on build and target
                self._select_rotation(best_target)

                # Enqueue attack command
                self._enqueue_action_raw(f"attack {target_id}", "attack_skill")
                logger.info("combat_acquire_target: %s (id=%d, dist=%d, score=%.1f)",
                            s.current_target_name, target_id,
                            best_target.get("distance", 0),
                            best_target.get("threat_score", 0))
                return f"attack_{s.current_target_name}"

        return None

    def _select_rotation(self, target_info: dict) -> None:
        """Select the best skill rotation based on build and target."""
        s = self._state
        if not self._skill_rotation:
            return

        # Get build info
        build_name = s.my_build_name
        if self._build_manager:
            build = self._build_manager._active_builds.get("default", {})
            build_name = build.get("name", "")

        # Get recommended rotation
        target_data = {
            "element": s.current_target_element,
            "race": s.current_target_race,
            "size": s.current_target_size,
            "aggro": s.aggro_count,
            "is_boss": s.current_target_is_boss,
            "hp_pct": s.current_target_hp_pct,
        }

        rotation_skills = self._skill_rotation.get_rotation_for_target(
            target_info=target_data,
            available_skills=s.my_available_skills,
            current_sp=s.my_sp,
        )

        if rotation_skills:
            s.current_skill = rotation_skills[0].name if rotation_skills else ""
            s.current_skill_index = 0
            # Store rotation for subsequent ticks
            s._cached_rotation = rotation_skills  # type: ignore
            logger.info("combat_rotation_selected: %s → %s (build=%s)",
                        s.current_target_name, s.current_skill, build_name)
        else:
            s.current_skill = ""
            s.current_skill_index = 0
            s._cached_rotation = []  # type: ignore

    def _handle_combat_movement(self, now: float) -> str | None:
        """Phase 5: Handle movement during combat."""
        s = self._state

        # Approach target if too far
        if s.current_target_distance > 9 and s.my_job_class in ("archer", "hunter", "sniper", "ranger"):
            # Ranged class — maintain distance
            return None
        elif s.current_target_distance > 3:
            # Melee or close-range — approach
            self._enqueue_action_raw(f"move {s.current_target_id}", "map_move")
            return "approach_target"

        # Kite for ranged classes
        if s.current_target_distance < 3 and s.my_job_class in ("mage", "wizard", "high_wizard", "warlock", "archer", "hunter", "sniper", "ranger"):
            self._enqueue_action("maintain_distance", "map_move")
            return "maintain_distance"

        return None

    def _execute_skill_rotation(self, now: float) -> str | None:
        """Phase 6: Execute the current skill rotation."""
        s = self._state
        if not self._skill_rotation:
            return None

        # Get cached rotation
        rotation = getattr(s, "_cached_rotation", [])
        if not rotation:
            return None

        # Check cooldowns
        current_skill = rotation[s.current_skill_index % len(rotation)]
        skill_name = current_skill.name

        # Check if skill is on cooldown
        if skill_name in s.skill_cooldowns:
            if now - s.skill_cooldowns[skill_name] < (current_skill.cooldown_ms / 1000.0 if current_skill.cooldown_ms > 0 else 1.0):
                # Try next skill in rotation
                s.current_skill_index = (s.current_skill_index + 1) % len(rotation)
                return self._execute_skill_rotation(now)

        # Check SP cost
        if current_skill.sp_cost > 0 and current_skill.sp_cost > s.my_sp:
            # Not enough SP — try next skill or basic attack
            s.current_skill_index = (s.current_skill_index + 1) % len(rotation)
            if s.current_skill_index == 0:
                return "use_basic_attack_only"
            return self._execute_skill_rotation(now)

        # Check elemental advantage
        if self._elemental_matrix:
            multiplier = self._elemental_matrix.get_elemental_multiplier(
                current_skill.element,
                s.current_target_element,
            )
            if multiplier < 0.5:
                # Bad element — skip this skill, try next
                s.current_skill_index = (s.current_skill_index + 1) % len(rotation)
                return self._execute_skill_rotation(now)

        # Execute the skill
        s.current_skill = skill_name
        s.last_skill_time = now
        s.skill_cooldowns[skill_name] = now
        s.current_skill_index = (s.current_skill_index + 1) % len(rotation)

        # Enqueue the skill action
        target_id = s.current_target_id
        self._enqueue_action_raw(f"skill {skill_name} {target_id}", "attack_skill")
        logger.debug("combat_skill: %s → %s (idx=%d/%d, sp=%d, mult=%.1f)",
                     skill_name, s.current_target_name,
                     s.current_skill_index, len(rotation),
                     current_skill.sp_cost,
                     self._elemental_matrix.get_elemental_multiplier(
                         current_skill.element, s.current_target_element
                     ) if self._elemental_matrix else 1.0)

        return f"use_skill_{skill_name}"

    def _basic_attack(self) -> str:
        """Phase 7: Basic attack fallback."""
        s = self._state
        target_id = s.current_target_id
        if target_id > 0:
            self._enqueue_action_raw(f"attack {target_id}", "attack_skill")
        return "use_basic_attack_only"

    # ── Action Enqueue Helpers ──

    def _enqueue_action(self, reflex_action: str, kind: str) -> bool:
        """Enqueue a reflex action via the action executor."""
        if self._action_executor and self._action_queue:
            return self._action_executor.execute(reflex_action, "default", self._action_queue)
        if self._enqueue_fn:
            try:
                self._enqueue_fn(reflex_action)
                return True
            except Exception:
                pass
        return False

    def _enqueue_action_raw(self, command: str, kind: str) -> bool:
        """Enqueue a raw command string."""
        if self._action_executor and self._action_queue:
            return self._action_executor.execute_command(command, "default", self._action_queue)
        if self._enqueue_fn:
            try:
                self._enqueue_fn(command)
                return True
            except Exception:
                pass
        return False

    # ── State Management ──

    def set_target(self, target_id: int, target_name: str = "") -> None:
        with self._lock:
            self._state.current_target_id = target_id
            self._state.current_target_name = target_name
            self._state.is_in_combat = True
            self._state.combat_started_at = time.time()

    def set_skill(self, skill_name: str, cooldown_ms: int = 0) -> None:
        with self._lock:
            self._state.current_skill = skill_name
            self._state.skill_cooldowns[skill_name] = time.time()

    def get_state(self) -> CombatState:
        with self._lock:
            return self._state

    def get_combat_summary(self) -> str:
        with self._lock:
            s = self._state
            rotation_name = getattr(s, "_cached_rotation", [])
            rot_str = rotation_name[0].name if rotation_name else "none"
            return (
                f"── Combat Loop ──\n"
                f"Active: {s.is_active}\n"
                f"Target: {s.current_target_name} (ID={s.current_target_id}, "
                f"HP={s.current_target_hp_pct:.0%}, elem={s.current_target_element})\n"
                f"Distance: {s.current_target_distance:.1f}\n"
                f"Aggro: {s.aggro_count} | HP: {s.my_hp_pct:.0%} | SP: {s.my_sp_pct:.0%}\n"
                f"Build: {s.my_build_name} | Class: {s.my_job_class}\n"
                f"Rotation: {rot_str} | Skill: {s.current_skill}\n"
                f"Gear: {s.current_gear_set} | Weapon: {s.my_weapon_element}/{s.my_weapon_type}\n"
                f"Buffs: {len(s.my_buffs)} active\n"
                f"Kills: {s.kills_this_session} | Ticks: {s.ticks_this_session}\n"
                f"Map: {s.map_name}"
            )

    def reset(self) -> None:
        with self._lock:
            self._state = CombatState()

    # ── Subsystem Wiring ──

    def set_threat_targeting(self, obj: Any) -> None:
        with self._lock:
            self._threat_targeting = obj

    def set_skill_rotation(self, obj: Any) -> None:
        with self._lock:
            self._skill_rotation = obj

    def set_elemental_matrix(self, obj: Any) -> None:
        with self._lock:
            self._elemental_matrix = obj

    def set_buff_maintenance(self, obj: Any) -> None:
        with self._lock:
            self._buff_maintenance = obj

    def set_gear_swapper(self, obj: Any) -> None:
        with self._lock:
            self._gear_swapper = obj

    def set_resource_manager(self, obj: Any) -> None:
        with self._lock:
            self._resource_manager = obj

    def set_build_manager(self, obj: Any) -> None:
        with self._lock:
            self._build_manager = obj

    def set_reflex_combat(self, obj: Any) -> None:
        with self._lock:
            self._reflex_combat = obj

    def set_action_executor(self, obj: Any) -> None:
        with self._lock:
            self._action_executor = obj

    def set_action_queue(self, obj: Any) -> None:
        with self._lock:
            self._action_queue = obj

    def set_gather_and_kill(self, obj: Any) -> None:
        with self._lock:
            self._gather_and_kill = obj

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def set_get_snapshot_fn(self, fn: Callable) -> None:
        with self._lock:
            self._get_snapshot_fn = fn

    def set_build(self, build_name: str) -> None:
        """Set the bot's build for build-aware combat decisions."""
        with self._lock:
            self._state.my_build_name = build_name
            logger.info("combat_build_set: %s", build_name)


# ── Global Singleton ──

_combat_loop: CombatLoop | None = None
_combat_loop_lock = RLock()


def get_combat_loop() -> CombatLoop:
    global _combat_loop
    with _combat_loop_lock:
        if _combat_loop is None:
            _combat_loop = CombatLoop()
        return _combat_loop
