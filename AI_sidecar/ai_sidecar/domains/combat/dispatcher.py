"""Tactics dispatcher — selects tactics based on job name and routes combat decisions.

The dispatcher is the entry point for the combat domain. It:
  1. Resolves the character's job to a tactics module.
  2. Delegates target selection, skill selection, and positioning.
  3. Emits HeuristicAction objects for the action queue.
"""

from __future__ import annotations

import logging
from typing import Any

from ai_sidecar.actions import HeuristicAction
from ai_sidecar.domains.combat.tactics.base import BaseTactics, TacticsContext, TargetInfo
from ai_sidecar.domains.combat.jobs.registry import get_job_registry
from ai_sidecar.domains.combat.skills import get_skill_registry
from ai_sidecar.domains.combat.targeting import TargetScorer, enrich_monster_list

logger = logging.getLogger(__name__)


class TacticsDispatcher:
    """Dispatches combat decisions to the appropriate tactics module.

    Stateless — all context lives in TacticsContext.
    """

    def __init__(self) -> None:
        self._job_registry = get_job_registry()
        self._skill_registry = get_skill_registry()
        self._default_tactics: BaseTactics | None = None

    def build_context(self, signals: dict[str, Any]) -> TacticsContext:
        """Build a TacticsContext from raw signals dict.

        This is the main entry point for the bridge integration: it converts
        the raw state signals into a structured TacticsContext that tactics
        modules can consume.
        """
        vitals = signals.get("vitals", {}) or signals.get("stats", {})
        combat = signals.get("combat", {})
        actors = signals.get("actors", [])
        cooldowns_raw = signals.get("cooldowns", {})
        buffs_raw = signals.get("buffs", [])
        skills_data = signals.get("skills", [])
        status = signals.get("status", {})
        position = signals.get("position", {})

        # Monsters from actors
        monsters = [a for a in actors if a.get("type", "") == "monster" and a.get("hp", 0) > 0]

        # Party members from actors
        party_members = [a for a in actors if a.get("type", "") == "player"
                         and a.get("is_party", False)]

        # Parse cooldowns
        cooldowns: dict[str, float] = {}
        if isinstance(cooldowns_raw, dict):
            cooldowns = {k.lower(): float(v) for k, v in cooldowns_raw.items()}

        # Parse buffs
        buffs: list[str] = []
        if isinstance(buffs_raw, list):
            buffs = [b.get("name", "") if isinstance(b, dict) else str(b) for b in buffs_raw]

        # Parse available skills
        available_skills: list[str] = []
        if isinstance(skills_data, list):
            available_skills = [
                s.get("name", "") if isinstance(s, dict) else str(s)
                for s in skills_data
            ]

        my_hp = int(vitals.get("hp", vitals.get("hp", 1)))
        my_max_hp = int(vitals.get("hp_max", vitals.get("max_hp", 1)))
        my_sp = int(vitals.get("sp", 0))
        my_max_sp = int(vitals.get("sp_max", vitals.get("max_sp", 1)))

        return TacticsContext(
            my_hp_pct=my_hp / max(1, my_max_hp),
            my_sp_pct=my_sp / max(1, my_max_sp),
            my_hp=my_hp,
            my_max_hp=my_max_hp,
            my_sp=my_sp,
            my_max_sp=my_max_sp,
            my_job_class=str(vitals.get("job_name", vitals.get("class", "novice"))).lower(),
            my_base_level=int(vitals.get("base_level", vitals.get("level", 1))),
            my_weapon_type=str(vitals.get("weapon_type", "dagger")).lower(),
            my_weapon_element=str(vitals.get("weapon_element", "neutral")).lower(),
            current_target_id=int(combat.get("target_id", 0)),
            aggro_count=int(combat.get("aggro_count", 0)),
            enemies_nearby=len(monsters),
            party_members_nearby=len(party_members),
            has_party=len(party_members) > 0,
            is_sitting=bool(status.get("sitting", False)),
            is_in_combat=bool(combat.get("in_combat", False)),
            map_name=str(position.get("map", "")),
            monsters=monsters,
            party_members=party_members,
            cooldowns=cooldowns,
            active_buffs=buffs,
            available_skills=available_skills,
            config=signals.get("config", {}),
        )

    def get_tactics(self, job_name: str) -> BaseTactics:
        """Get the tactics module for a job name.

        Falls back to hybrid tactics for unknown jobs.
        """
        tactics = self._job_registry.get_tactics_for_job(job_name)
        if isinstance(tactics, BaseTactics):
            return tactics
        # Fallback: import hybrid
        from ai_sidecar.domains.combat.tactics.hybrid import HybridTactics
        return HybridTactics()

    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction],
               bot_id: str) -> None:
        """Main assess method — called by DomainRegistry for each combat tick.

        Builds context from signals, selects tactics, then runs all decision
        phases and appends actions.

        Args:
            signals: Raw state signals from the bridge snapshot.
            actions: List to append HeuristicAction objects to.
            bot_id: Bot identifier string.
        """
        try:
            ctx = self.build_context(signals)
            tactics = self.get_tactics(ctx.my_job_class)

            # Phase 1: Emergency check
            emergency = tactics.assess_emergency(ctx)
            if emergency:
                actions.append(emergency)
                return  # Emergency overrides everything else

            # Phase 2: Buff maintenance
            needed_buffs = tactics.assess_buffs(ctx)
            for buff_name in needed_buffs:
                actions.append(self._make_buffer_action(buff_name, ctx, tactics))

            # Phase 3: Target selection
            target = self._select_target(ctx, tactics)

            # Phase 4: Positioning
            position = tactics.evaluate_positioning(ctx, target)
            if position and position.get("urgency", 0) > 0.5:
                actions.append(self._make_move_action(position, ctx, tactics, target))

            # Phase 5: Skill selection
            skill = tactics.select_skill(ctx, target)
            if skill:
                skill_name, skill_level = skill
                actions.append(self._make_skill_action(
                    skill_name, skill_level, ctx, tactics, target
                ))
            else:
                # Basic attack fallback
                if target:
                    actions.append(self._make_attack_action(target, ctx, tactics))

        except Exception as e:
            logger.error("tactics_dispatcher.assess() failed: %s", e, exc_info=True)

    # ── Internal helpers ──

    def _select_target(self, ctx: TacticsContext, tactics: BaseTactics) -> TargetInfo | None:
        """Select target using the tactics module, enriched with scoring."""
        if not ctx.monsters:
            return None
        target = tactics.select_target(ctx)
        return target

    def _make_buffer_action(self, buff_name: str, ctx: TacticsContext,
                            tactics: BaseTactics) -> HeuristicAction:
        return HeuristicAction(
            kind="command",
            command=f"use_skill {buff_name}",
            confidence=0.9,
            domain="combat_tactics",
            reason=f"{tactics.name}_buff_{buff_name}",
            metadata={
                "tactics": tactics.name,
                "buff_name": buff_name,
                "sp_cost": self._skill_registry.get_sp_cost(buff_name),
            },
        )

    def _make_move_action(self, position: dict[str, Any], ctx: TacticsContext,
                          tactics: BaseTactics, target: TargetInfo | None) -> HeuristicAction:
        _mx = position.get('move_x', 0)
        _my = position.get('move_y', 0)
        _reason = position.get("reason", "tactical_positioning")
        if (not _mx and not _my):
            # The kiting/positioning modules signal an INTENT (retreat, back_up,
            # approach, reposition_los) with tactic labels, but do not compute an
            # absolute coordinate vector (TacticsContext has no bot/target x,y).
            # Emitting `move 0 0` would path the bot to the map origin (a no-op /
            # teleport hazard). Instead honour the intent as an observability
            # record and let OpenKore's native AI (attackAuto routing + emergency
            # flee) execute the actual reposition — a safe, honest fallback rather
            # than a bogus coordinate write.
            return HeuristicAction(
                kind="log",
                command=f"tactics_reposition:{position.get('tactic', 'stand')}",
                confidence=0.7,
                domain="combat_tactics",
                reason=_reason,
                metadata={
                    "tactics": tactics.name,
                    "urgency": position.get("urgency", 0.5),
                    "tactic": position.get("tactic", "") or "",
                    "target_id": target.actor_id if target else 0,
                },
            )
        return HeuristicAction(
            kind="command",
            command=f"move {_mx} {_my}",
            confidence=0.7,
            domain="combat_tactics",
            reason=_reason,
            metadata={
                "tactics": tactics.name,
                "urgency": position.get("urgency", 0.5),
                "target_id": target.actor_id if target else 0,
            },
        )

    def _make_skill_action(self, skill_name: str, skill_level: int,
                           ctx: TacticsContext, tactics: BaseTactics,
                           target: TargetInfo | None) -> HeuristicAction:
        return HeuristicAction(
            kind="command",
            command=f"use_skill {skill_name}",
            confidence=0.85,
            domain="combat_tactics",
            reason=f"{tactics.name}_skill_{skill_name}",
            metadata={
                "tactics": tactics.name,
                "skill_name": skill_name,
                "skill_level": skill_level,
                "target_id": target.actor_id if target else 0,
                "target_name": target.name if target else "none",
            },
        )

    def _make_attack_action(self, target: TargetInfo, ctx: TacticsContext,
                            tactics: BaseTactics) -> HeuristicAction:
        return HeuristicAction(
            kind="command",
            command=f"attack {target.actor_id}",
            confidence=0.7,
            domain="combat_tactics",
            reason=f"{tactics.name}_basic_attack",
            metadata={
                "tactics": tactics.name,
                "target_id": target.actor_id,
                "target_name": target.name,
            },
        )


# ── Global Singleton ──

_dispatcher: TacticsDispatcher | None = None
_dispatcher_lock = __import__("threading").RLock()


def get_tactics_dispatcher() -> TacticsDispatcher:
    global _dispatcher
    with _dispatcher_lock:
        if _dispatcher is None:
            _dispatcher = TacticsDispatcher()
        return _dispatcher


def assess_combat_tactics(signals: dict[str, Any], actions: list[HeuristicAction],
                          bot_id: str) -> None:
    """Convenience function: assess combat with the dispatcher.

    Designed to be called from the DomainRegistry or bridge directly.
    """
    dispatcher = get_tactics_dispatcher()
    dispatcher.assess(signals, actions, bot_id)
