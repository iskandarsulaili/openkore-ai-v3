"""Job change automation — walk to NPC → talk → confirm → equip new weapon.

Provides:
  - JobChangeStep enum: phases of the job-change interaction
  - JobChangePlan: NPC coordinates, talk sequence, and gear requirements
  - AdvancementDomain: auto-detects when a job change is due and executes it
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ai_sidecar.actions import HeuristicAction

logger = __import__("logging").getLogger(__name__)


# ── Job change data ──────────────────────────────────────────────────────

class JobChangeStep(str, Enum):
    """Step of the job change interaction sequence."""
    IDLE = "idle"
    WALK_TO_MAP = "walk_to_map"            # Move to the NPC's town/map
    APPROACH_NPC = "approach_npc"          # Walk to NPC coordinates
    TALK_NPC = "talk_npc"                  # Initiate dialogue
    SELECT_JOB = "select_job"              # Choose the job option
    CONFIRM = "confirm"                    # Confirm job change
    EQUIP_WEAPON = "equip_weapon"          # Equip new class weapon
    LEAVE = "leave"                        # Walk away / done


@dataclass
class JobChangePlan:
    """Full plan for changing from one job to another."""
    from_job: str
    to_job: str
    npc_map: str                      # Map where the job NPC lives
    npc_x: int
    npc_y: int
    talk_sequence: list[str] = field(default_factory=lambda: [
        "talk continue",
        "talk resp 1",
        "talk resp 2",
        "talk resp 1",
    ])
    weapon_to_equip: str = ""         # Item ID of the new class weapon
    required_base_level: int = 10
    required_job_level: int = 10


# ── Job change NPC database ──────────────────────────────────────────────

# Novice → first job (from Prontera guild building)
NOVICE_JOB_NPCS: dict[str, tuple[str, int, int]] = {
    "swordman":  ("prontera", 160, 191),
    "mage":      ("prontera", 185, 135),
    "archer":    ("prontera", 215, 135),
    "acolyte":   ("prontera", 190, 166),
    "thief":     ("morocc", 216, 48),
    "merchant":  ("alberta", 45, 151),
}

# First job → second job (2-1 class) - major town NPC locations
FIRST_TO_SECOND_NPCS: dict[str, tuple[str, int, int, list[str]]] = {
    "swordman":  ("prt_in", 30, 93, [
        "talk continue", "talk resp 1",
        "talk resp 2", "talk resp 1",
        "talk resp 1",
    ]),
    "thief":     ("in_moc_16", 80, 131, [
        "talk continue", "talk resp 1",
        "talk resp 2",
    ]),
    "archer":    ("payon", 194, 138, [
        "talk continue", "talk resp 1",
        "talk resp 2",
    ]),
    "mage":      ("geffen", 123, 60, [
        "talk continue", "talk resp 1",
        "talk resp 2",
    ]),
    "acolyte":   ("prt_church", 153, 152, [
        "talk continue", "talk resp 1",
        "talk resp 1",
    ]),
    "merchant":  ("alberta", 62, 47, [
        "talk continue", "talk resp 1",
    ]),
}

# First job → 2-2 class alternate locations
FIRST_TO_ALTERNATE_NPCS: dict[str, tuple[str, int, int, list[str]]] = {
    "swordman":  ("prontera", 265, 188, [
        "talk continue", "talk resp 2", "talk resp 1",
    ]),
    "thief":     ("morocc", 123, 210, [
        "talk continue", "talk resp 2",
    ]),
    "archer":    ("payon", 218, 224, [
        "talk continue", "talk resp 2",
    ]),
    "mage":      ("geffen", 249, 69, [
        "talk continue", "talk resp 2",
    ]),
    "acolyte":   ("prt_church", 55, 62, [
        "talk continue", "talk resp 2",
    ]),
    "merchant":  ("alberta", 94, 162, [
        "talk continue", "talk resp 2",
    ]),
}

# Weapon items per first job class (for equip after changing)
CLASS_WEAPONS: dict[str, str] = {
    "swordman":  "1201",   # Knife → will become Sword
    "knight":    "1901",   # Katana
    "thief":     "1301",   # Dagger
    "assassin":  "1301",
    "archer":    "1701",   # Bow
    "hunter":    "1701",
    "mage":      "1601",   # Rod
    "wizard":    "1601",
    "acolyte":   "1501",   # Mace
    "priest":    "1501",
    "merchant":  "1201",
    "blacksmith": "1901",
}

# Alternate class names for 2-2 jobs
ALTERNATE_CLASS_MAP: dict[str, str] = {
    "rogue": "thief",
    "bard": "archer",
    "dancer": "archer",
    "sage": "mage",
    "crusader": "swordman",
    "monk": "acolyte",
    "alchemist": "merchant",
}


# ── Job change detection ─────────────────────────────────────────────────

def get_job_change_plan(
    current_job: str,
    target_job: str,
    base_level: int,
    job_level: int,
) -> tuple[JobChangePlan | None, str]:
    """Determine the job change plan and a human-readable reason.

    Returns:
        (JobChangePlan | None, reason_str)
        - None means no change needed or prerequisites not met.
    """
    cj = current_job.lower().strip()
    tj = target_job.lower().strip()

    # Normalise some common job names
    _norm: dict[str, str] = {
        "novice": "novice",
        "first": "novice",
    }

    # ── No change needed ──
    if cj == tj:
        return None, f"Already a {target_job}"

    # ── Novice → First job ──
    if cj == "novice":
        if base_level < 10:
            return None, f"Need base level 10 for {target_job} (currently {base_level})"
        npc = NOVICE_JOB_NPCS.get(tj)
        if not npc:
            return None, f"No NPC data for {target_job}"
        npc_map, npc_x, npc_y = npc
        plan = JobChangePlan(
            from_job="novice",
            to_job=tj,
            npc_map=npc_map,
            npc_x=npc_x,
            npc_y=npc_y,
            talk_sequence=["talk continue", "talk resp 1", "talk resp 2", "talk resp 1"],
            weapon_to_equip=CLASS_WEAPONS.get(tj, ""),
            required_base_level=10,
        )
        return plan, f"Novice → {target_job} at {npc_map} ({npc_x},{npc_y})"

    # ── First job → Second job (2-1) ──
    _2_1_data = FIRST_TO_SECOND_NPCS.get(cj)
    if _2_1_data:
        npc_map, npc_x, npc_y, talk_seq = _2_1_data
        plan = JobChangePlan(
            from_job=cj,
            to_job=tj,
            npc_map=npc_map,
            npc_x=npc_x,
            npc_y=npc_y,
            talk_sequence=talk_seq,
            weapon_to_equip=CLASS_WEAPONS.get(tj, ""),
            required_base_level=40,
            required_job_level=40,
        )
        return plan, f"{current_job} → {target_job} (2-1) at {npc_map}"

    # ── First job → Alternate (2-2) ──
    _base = ALTERNATE_CLASS_MAP.get(cj)
    if _base:
        _a_data = FIRST_TO_ALTERNATE_NPCS.get(_base)
        if _a_data:
            npc_map, npc_x, npc_y, talk_seq = _a_data
            plan = JobChangePlan(
                from_job=cj,
                to_job=tj,
                npc_map=npc_map,
                npc_x=npc_x,
                npc_y=npc_y,
                talk_sequence=talk_seq,
                weapon_to_equip=CLASS_WEAPONS.get(tj, ""),
                required_base_level=40,
                required_job_level=40,
            )
            return plan, f"{current_job} → {target_job} (2-2) at {npc_map}"

    return None, f"No job plan: {current_job} → {target_job}"


# ── Advancement domain ───────────────────────────────────────────────────

class AdvancementDomain:
    """Detects when a job change is due and emits the NPC interaction sequence.

    Tracks per-bot state so the multi-step process survives multiple
    assess() cycles.
    """

    def __init__(self) -> None:
        self._steps: dict[str, JobChangeStep] = {}       # bot_id → current step
        self._plans: dict[str, JobChangePlan] = {}        # bot_id → active plan
        self._completed: dict[str, list[str]] = {}        # bot_id → [job names changed to]

    # ── Public API ────────────────────────────────────────────────────

    def get_step(self, bot_id: str) -> JobChangeStep:
        return self._steps.get(bot_id, JobChangeStep.IDLE)

    def has_completed(self, bot_id: str) -> list[str]:
        return self._completed.get(bot_id, [])

    def assess(
        self,
        signals: dict[str, Any],
        actions: list[HeuristicAction],
        bot_id: str,
    ) -> None:
        """Check if a job change is needed and execute it."""
        current_job = str(signals.get("job_name", "novice") or "novice").lower()
        base_level = int(signals.get("base_level", 1) or 1)
        job_level = int(signals.get("job_level", 1) or 1)
        map_name = str(signals.get("map", "") or "").lower().replace(".gat", "")

        step = self._steps.get(bot_id, JobChangeStep.IDLE)

        # ── Determine if a new job change is needed ──
        if step == JobChangeStep.IDLE:
            target_job = self._detect_needed_change(current_job, base_level, job_level)
            if target_job is None:
                return  # No change needed
            plan, reason = get_job_change_plan(current_job, target_job, base_level, job_level)
            if plan is None:
                return  # Prerequisites not met
            self._plans[bot_id] = plan
            self._steps[bot_id] = JobChangeStep.WALK_TO_MAP
            logger.info("[advancement] %s: %s", bot_id, reason)
            return

        # ── Execute the active plan ──
        plan = self._plans.get(bot_id)
        if plan is None:
            self._steps[bot_id] = JobChangeStep.IDLE
            return

        if step == JobChangeStep.WALK_TO_MAP:
            self._walk_to_map(actions, bot_id, plan, map_name)
        elif step == JobChangeStep.APPROACH_NPC:
            self._approach_npc(actions, bot_id, plan, map_name)
        elif step == JobChangeStep.TALK_NPC:
            self._talk_npc(actions, bot_id, plan, map_name)
        elif step == JobChangeStep.SELECT_JOB:
            self._do_talk_sequence(actions, bot_id, plan, map_name)
        elif step == JobChangeStep.CONFIRM:
            self._confirm_job(actions, bot_id, plan, map_name)
        elif step == JobChangeStep.EQUIP_WEAPON:
            self._equip_weapon(actions, bot_id, plan)
        elif step == JobChangeStep.LEAVE:
            self._leave_npc(actions, bot_id, plan)
            # Mark complete
            self._completed.setdefault(bot_id, []).append(plan.to_job)
            del self._plans[bot_id]
            self._steps[bot_id] = JobChangeStep.IDLE

    # ── Change detection ──────────────────────────────────────────────

    @staticmethod
    def _detect_needed_change(
        current_job: str,
        base_level: int,
        job_level: int,
    ) -> str | None:
        """Detect if a job change is needed, return target job or None."""
        if current_job == "novice" and base_level >= 10:
            return "acolyte"  # Default for autobots — subagent can override
        if current_job in ("swordman", "thief", "archer", "mage",
                            "acolyte", "merchant") and job_level >= 40:
            # Return the obvious 2-1 upgrade
            _upgrade = {
                "swordman": "knight",
                "thief": "assassin",
                "archer": "hunter",
                "mage": "wizard",
                "acolyte": "priest",
                "merchant": "blacksmith",
            }
            return _upgrade.get(current_job)
        return None  # No change needed

    # ── Step executors ────────────────────────────────────────────────

    def _walk_to_map(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        plan: JobChangePlan,
        current_map: str,
    ) -> None:
        """Move to the NPC's map."""
        if current_map != plan.npc_map:
            actions.append(HeuristicAction(
                kind="command",
                command=f"move {plan.npc_map}",
                confidence=0.99,
                domain="progression",
                reason=f"Job change: walking to {plan.npc_map} for {plan.to_job}",
                metadata={
                    "step": "walk_to_map",
                    "from_job": plan.from_job,
                    "to_job": plan.to_job,
                },
            ))
        else:
            self._steps[bot_id] = JobChangeStep.APPROACH_NPC

    def _approach_npc(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        plan: JobChangePlan,
        current_map: str,
    ) -> None:
        """Walk to the NPC coordinates."""
        if current_map != plan.npc_map:
            self._steps[bot_id] = JobChangeStep.WALK_TO_MAP
            return
        actions.append(HeuristicAction(
            kind="command",
            command=f"move {plan.npc_x} {plan.npc_y}",
            confidence=0.99,
            domain="progression",
            reason=f"Job change: walking to NPC at ({plan.npc_x},{plan.npc_y}) "
                   f"for {plan.to_job}",
            metadata={
                "step": "approach_npc",
                "npc_x": plan.npc_x,
                "npc_y": plan.npc_y,
                "to_job": plan.to_job,
            },
        ))
        self._steps[bot_id] = JobChangeStep.TALK_NPC

    def _talk_npc(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        plan: JobChangePlan,
        current_map: str,
    ) -> None:
        """Initiate dialogue with the job change NPC."""
        if current_map != plan.npc_map:
            self._steps[bot_id] = JobChangeStep.WALK_TO_MAP
            return
        actions.append(HeuristicAction(
            kind="command",
            command=f"talknpc {plan.npc_x} {plan.npc_y}",
            confidence=0.95,
            domain="progression",
            reason=f"Job change: talking to NPC for {plan.to_job}",
            metadata={
                "step": "talk_npc",
                "npc_x": plan.npc_x,
                "npc_y": plan.npc_y,
                "to_job": plan.to_job,
            },
        ))
        self._steps[bot_id] = JobChangeStep.SELECT_JOB

    def _do_talk_sequence(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        plan: JobChangePlan,
        current_map: str,
    ) -> None:
        """Send the talk response sequence to navigate the NPC dialogue."""
        if current_map != plan.npc_map:
            self._steps[bot_id] = JobChangeStep.WALK_TO_MAP
            return
        for idx, talk_cmd in enumerate(plan.talk_sequence):
            confidence = max(0.70, 0.95 - (idx * 0.04))
            actions.append(HeuristicAction(
                kind="command",
                command=talk_cmd,
                confidence=confidence,
                domain="progression",
                reason=f"Job change dialogue: step {idx+1}/{len(plan.talk_sequence)} "
                       f"for {plan.to_job}",
                metadata={
                    "step": f"dialogue_{idx}",
                    "talk_cmd": talk_cmd,
                    "to_job": plan.to_job,
                },
            ))
        self._steps[bot_id] = JobChangeStep.CONFIRM

    def _confirm_job(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        plan: JobChangePlan,
        current_map: str,
    ) -> None:
        """Confirm the job change in the final NPC dialogue."""
        if current_map != plan.npc_map:
            self._steps[bot_id] = JobChangeStep.WALK_TO_MAP
            return
        actions.append(HeuristicAction(
            kind="command",
            command="talk resp 1",  # Assume "Yes" is option 1
            confidence=0.90,
            domain="progression",
            reason=f"Job change: confirming {plan.to_job}",
            metadata={
                "step": "confirm_job",
                "to_job": plan.to_job,
            },
        ))
        self._steps[bot_id] = JobChangeStep.EQUIP_WEAPON

    def _equip_weapon(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        plan: JobChangePlan,
    ) -> None:
        """Equip the new class weapon after successful job change."""
        if not plan.weapon_to_equip:
            self._steps[bot_id] = JobChangeStep.LEAVE
            return
        actions.append(HeuristicAction(
            kind="command",
            command=f"equip {plan.weapon_to_equip}",
            confidence=0.90,
            domain="progression",
            reason=f"Job change: equipping weapon {plan.weapon_to_equip} for {plan.to_job}",
            metadata={
                "step": "equip_weapon",
                "weapon_id": plan.weapon_to_equip,
                "to_job": plan.to_job,
            },
        ))
        self._steps[bot_id] = JobChangeStep.LEAVE

    def _leave_npc(
        self,
        actions: list[HeuristicAction],
        bot_id: str,
        plan: JobChangePlan,
    ) -> None:
        """Finalise: walk away from the NPC."""
        actions.append(HeuristicAction(
            kind="command",
            command="talk continue",
            confidence=0.85,
            domain="progression",
            reason=f"Job change complete: leaving NPC for {plan.to_job}",
            metadata={
                "step": "leave_npc",
                "to_job": plan.to_job,
            },
        ))
        actions.append(HeuristicAction(
            kind="command",
            command="stand",
            confidence=0.90,
            domain="progression",
            reason=f"Job change complete: stand up as {plan.to_job}",
        ))


# ── Convenience ──────────────────────────────────────────────────────────

def create_advancement_domain() -> AdvancementDomain:
    return AdvancementDomain()
