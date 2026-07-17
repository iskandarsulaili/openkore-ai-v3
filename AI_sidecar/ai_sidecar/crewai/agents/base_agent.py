"""Enhanced BehaviorProfile with role-specific logic, performance tracking & role change requests."""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)
# ── Skill creation helper for agents ──


def maybe_create_skill(
    name: str,
    content: str,
    category: str = None,
    provenance: str = "background_review",
) -> dict:
    """Create a skill from an agent's discovery.
    
    All CrewAI agents can call this when they discover something
    worth persisting as a reusable skill.
    
    Args:
        name: Skill name (lowercase, hyphen-separated)
        content: SKILL.md format content with YAML frontmatter
        category: Domain category (healing, navigation, economy, etc.)
        provenance: 'foreground' (user) or 'background_review' (agent)
    
    Returns:
        Dict with success status
    """
    from ai_sidecar.skills_manager import create_skill as _create
    from ai_sidecar.skills_usage import bump as _bump
    
    result = _create(name=name, content=content, category=category, provenance=provenance)
    if result.get("success"):
        _bump(name, event="use")
        import logging
        logging.getLogger(__name__).info(
            "Agent created skill: %s (category=%s)", name, category or "general"
        )
    return result



# ── RO roles that profiles can fulfill ──────────────────────────────────────

ROLE_AGENT_MAP: dict[str, list[str]] = {
    "tank": ["combat", "tactical_commander"],
    "healer": ["safety", "social_coordinator"],
    "dps_melee": ["combat", "tactical_commander"],
    "dps_ranged": ["combat", "tactical_commander"],
    "dps_magic": ["combat", "tactical_commander"],
    "support": ["safety", "social_coordinator"],
    "buffer": ["social_coordinator"],
    "debuff": ["social_coordinator", "tactical_commander"],
    "crafter": ["economy", "opportunistic_trader"],
    "merchant": ["economy", "opportunistic_trader"],
    "refiner": ["economy"],
    "farmer": ["economy", "navigation"],
    "looter": ["economy", "navigation"],
    "quester": ["questing", "navigation"],
    "mvp_hunter": ["combat", "tactical_commander"],
    "pvp_attacker": ["combat", "tactical_commander"],
    "pvp_defender": ["combat", "safety"],
    "gvg_frontline": ["combat", "tactical_commander"],
    "gvg_siege": ["combat", "navigation"],
    "gvg_support": ["safety", "social_coordinator"],
    "scout": ["navigation", "state_assessor"],
    "idle": ["manager"],
}


class BehaviorProfile:
    """Base class for heuristic behavior profiles with role awareness.

    Each profile can be assigned a specific RO role that influences
    can_handle() scoring and get_action() behavior.
    """

    agent_id: str
    role: str  # default descriptive role
    goal: str
    backstory: str

    def __init__(self, role_override: str | None = None) -> None:
        self._role = role_override or self.role
        self._outcomes: dict[str, dict[str, int]] = {}
        self._role_outcomes: dict[str, dict[str, dict[str, int]]] = {}
        self._role_performance: dict[str, float] = {}
        self._role_assignment_count: dict[str, int] = {}
        self._role_change_requested: bool = False

    # ── Role management ────────────────────────────────────────────────

    def set_role(self, role: str) -> None:
        """Assign a role to this profile instance."""
        self._role = role
        if role not in self._role_outcomes:
            self._role_outcomes[role] = {}
        if role not in self._role_assignment_count:
            self._role_assignment_count[role] = 0
        self._role_assignment_count[role] = self._role_assignment_count[role] + 1

    def get_assigned_role(self) -> str:
        return self._role

    def can_handle_role(self, role: str) -> bool:
        """Check if this profile is compatible with a given RO role."""
        compatible_ids = ROLE_AGENT_MAP.get(role, [])
        return self.agent_id in compatible_ids

    def request_role_change(self, current_role: str) -> dict[str, Any] | None:
        """Request a role change if current role underperforms for this profile."""
        if current_role not in self._role_outcomes:
            return None
        outcomes = self._role_outcomes[current_role]
        total = sum(s["attempts"] for s in outcomes.values())
        if total < 10:
            return None  # Not enough data
        successes = sum(s["successes"] for s in outcomes.values())
        rate = successes / max(1, total)
        self._role_performance[current_role] = rate

        # Find best alternative
        best_role, best_rate = current_role, rate
        for role, outcomes_dict in self._role_outcomes.items():
            if role == current_role:
                continue
            role_total = sum(s["attempts"] for s in outcomes_dict.values())
            if role_total < 5:
                continue
            role_successes = sum(s["successes"] for s in outcomes_dict.values())
            role_rate = role_successes / max(1, role_total)
            if role_rate > best_rate + 0.15:
                best_rate = role_rate
                best_role = role

        if best_role != current_role:
            self._role_change_requested = True
            return {
                "requested_change": True,
                "from_role": current_role,
                "to_role": best_role,
                "current_rate": rate,
                "target_rate": best_rate,
                "improvement": best_rate - rate,
                "reason": f"Performance in {current_role} ({rate:.0%}) < {best_role} ({best_rate:.0%})",
            }
        return {"requested_change": False, "reason": "current_role_optimal"}

    # ── Enhanced outcome tracking ──────────────────────────────────────

    def record_outcome(self, command: str, success: bool, role: str | None = None) -> None:
        """Record whether a heuristic action succeeded or failed, per role."""
        actual_role = role or self._role
        # Global outcomes
        if command not in self._outcomes:
            self._outcomes[command] = {"attempts": 0, "successes": 0, "failures": 0}
        self._outcomes[command]["attempts"] += 1
        if success:
            self._outcomes[command]["successes"] += 1
        else:
            self._outcomes[command]["failures"] += 1

        # Role-specific outcomes
        role_outcomes = self._role_outcomes.setdefault(actual_role, {})
        if command not in role_outcomes:
            role_outcomes[command] = {"attempts": 0, "successes": 0, "failures": 0}
        role_outcomes[command]["attempts"] += 1
        if success:
            role_outcomes[command]["successes"] += 1
        else:
            role_outcomes[command]["failures"] += 1

    # ── Role-adjusted can_handle ────────────────────────────────────────

    def can_handle(self, signals: dict[str, Any]) -> float:
        """Return a relevance score [0.0, 1.0] for the current game state.

        Subclasses override this. The base implementation considers the
        assigned role as a signal-weighting factor.
        """
        return 0.0

    def can_handle_with_role(self, signals: dict[str, Any], role: str | None = None) -> float:
        """Score can_handle adjusted for the given role.

        If the profile is not compatible with the requested role, returns 0.0.
        """
        target_role = role or self._role
        if not self.can_handle_role(target_role):
            return 0.0
        base_score = self.can_handle(signals)
        if base_score <= 0.0:
            return 0.0
        # Boost confidence if we have good performance in this role
        perf = self._role_performance.get(target_role, 0.5)
        role_boost = (perf - 0.5) * 0.2  # -0.1 to +0.1 boost
        return max(0.0, min(1.0, base_score + role_boost))

    # ── Role-adjusted get_action ────────────────────────────────────────

    def get_action(self, signals: dict[str, Any]) -> dict[str, Any] | None:
        """Return a heuristic action dict or None.

        Subclasses override this.
        """
        return None

    def get_action_with_role(self, signals: dict[str, Any], role: str | None = None) -> dict[str, Any] | None:
        """Get an action adjusted for a specific role. This base implementation
        delegates to get_action() but can be overridden in subclasses for
        role-specific behavior."""
        target_role = role or self._role
        if not self.can_handle_role(target_role):
            return None

        action = self.get_action(signals)
        if action is None:
            return None

        action["role"] = target_role
        action["confidence"] *= self._role_performance.get(target_role, 0.5) * 2
        action["confidence"] = max(0.0, min(1.0, action.get("confidence", 1.0)))
        return action

    # ── Success penalty (from original) ────────────────────────────────

    def _success_penalty(self, command: str) -> float:
        cmd_stats = self._outcomes.get(command)
        if not cmd_stats or cmd_stats["attempts"] < 3:
            return 1.0
        failure_rate = cmd_stats["failures"] / cmd_stats["attempts"]
        return max(0.5, 1.0 - failure_rate)

    # ── Stats ───────────────────────────────────────────────────────────

    def outcome_stats(self) -> dict[str, dict[str, int]]:
        return dict(self._outcomes)

    def role_outcome_stats(self, role: str | None = None) -> dict[str, dict[str, int]]:
        target = role or self._role
        return dict(self._role_outcomes.get(target, {}))

    def role_performance_summary(self) -> dict[str, Any]:
        return {
            "current_role": self._role,
            "role_performance": dict(self._role_performance),
            "role_assignment_counts": dict(self._role_assignment_count),
            "supported_roles": [r for r in ROLE_AGENT_MAP if self.agent_id in ROLE_AGENT_MAP[r]],
            "change_requested": self._role_change_requested,
        }
