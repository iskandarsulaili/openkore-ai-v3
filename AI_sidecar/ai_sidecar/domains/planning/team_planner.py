"""Team Composition Planner — role assignment, synergy scoring, and efficiency calculation.

Analyzes a group of bots and recommends how to specialise them into a
complementary farming team.  Produces HeuristicAction commands that the
PDCA loop can execute.

Roles (mutually exclusive, assigned by best-fitting stat):
  - farmer    Highest INT/MATK → Wizard/Sage       — AoE map clearing
  - buffer    Highest buff/heal → Priest/Monk      — follow, buff, heal, loot
  - tank      Highest VIT/DEF    → Knight/Paladin  — party dungeons
  - vender    Highest STR        → Merchant/BS     — town vending
  - looter    High AGI           → Rogue/Assassin  — vacuum loot, move fast
  - scout     Highest AGI        → Hunter/Archer   — MVP hunting, kiting

Team composition rules used in scoring:
  - Wizard + Priest farms 3× faster than solo at same level
  - Merchant vender turns items → zeny without leaving town
  - Rogue with max AGI can vacuum loot in ~2 s
  - Hunter/kiting class best for MVP hunting
  - Knight/Paladin tank essential for party dungeons
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

ALL_STATS = ("STR", "AGI", "VIT", "INT", "DEX", "LUK")

# Stat → role mapping priority (first match wins when multiple bots qualify)
ROLE_STAT_MAP: dict[str, str] = {
    "INT": "farmer",
    "MATK": "farmer",
    "DEX": "farmer",   # cast time reduction also favours farmer
    "HEAL": "buffer",
    "STR": "vender",
    "AGI": "looter",
    "VIT": "tank",
}

# Job → role affinities (used to break ties)
JOB_ROLE_AFFINITY: dict[str, str] = {
    "Wizard": "farmer",
    "Sage": "farmer",
    "High Wizard": "farmer",
    "Professor": "farmer",
    "Priest": "buffer",
    "Monk": "buffer",
    "High Priest": "buffer",
    "Champion": "buffer",
    "Merchant": "vender",
    "Blacksmith": "vender",
    "Whitesmith": "vender",
    "Alchemist": "vender",
    "Creator": "vender",
    "Rogue": "looter",
    "Stalker": "looter",
    "Assassin": "looter",
    "Assassin Cross": "looter",
    "Hunter": "scout",
    "Sniper": "scout",
    "Bard": "scout",
    "Dancer": "scout",
    "Knight": "tank",
    "Paladin": "tank",
    "Crusader": "tank",
    "Lord Knight": "tank",
}

# Job-progression paths (first-job → second-job → transcendent)
JOB_PROGRESSION: dict[str, list[str]] = {
    "Novice": ["Mage", "Wizard", "High Wizard"],
    "Mage": ["Wizard", "High Wizard"],
    "Wizard": ["High Wizard"],
    "Acolyte": ["Priest", "High Priest"],
    "Priest": ["High Priest"],
    "Monk": ["Champion"],
    "Merchant": ["Blacksmith", "Whitesmith"],
    "Blacksmith": ["Whitesmith"],
    "Alchemist": ["Creator"],
    "Thief": ["Rogue", "Stalker"],
    "Rogue": ["Stalker"],
    "Assassin": ["Assassin Cross"],
    "Archer": ["Hunter", "Sniper"],
    "Hunter": ["Sniper"],
    "Bard": ["Clown"],
    "Dancer": ["Gypsy"],
    "Swordsman": ["Knight", "Lord Knight"],
    "Knight": ["Lord Knight"],
    "Crusader": ["Paladin"],
}

# Synergy weights for pair combinations
# (role_a, role_b) → multiplier on base efficiency
SYNERGY_MATRIX: dict[tuple[str, str], float] = {
    ("farmer", "buffer"): 3.0,  # Wizard + Priest = 3× farming
    ("farmer", "looter"): 1.5,  # Wizard + Rogue = fast clear + vacuum
    ("buffer", "tank"): 2.0,   # Priest + Knight = unkillable duo
    ("tank", "buffer"): 2.0,
    ("farmer", "tank"): 1.3,
    ("tank", "farmer"): 1.3,
    ("farmer", "vender"): 1.4,  # Farmer + Merchant = efficient economy
    ("vender", "farmer"): 1.4,
    ("buffer", "vender"): 1.2,
    ("vender", "buffer"): 1.2,
    ("scout", "farmer"): 1.2,
    ("farmer", "scout"): 1.2,
    ("scout", "buffer"): 1.3,
    ("buffer", "scout"): 1.3,
    ("looter", "vender"): 1.6,  # Rogue vacuums + Merchant sells
    ("vender", "looter"): 1.6,
}

# Synergy penalties for redundant roles
REDUNDANCY_PENALTY = 0.5  # per duplicate role beyond first


class Role(Enum):
    """Bot roles within a farming team."""
    FARMER = "farmer"
    BUFFER = "buffer"
    TANK = "tank"
    VENDER = "vender"
    LOOTER = "looter"
    SCOUT = "scout"
    UNASSIGNED = "unassigned"


@dataclass
class TeamComposition:
    """Describes one optimal team archetype loaded from YAML."""
    name: str
    description: str
    roles: dict[str, str]          # role → recommended job
    expected_efficiency: int       # zeny per hour
    min_level: int = 1
    recommended_level: int = 40
    requirements: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════


def _load_compositions(yaml_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load team composition definitions from YAML."""
    if yaml_path is None:
        yaml_path = (
            Path(__file__).resolve().parents[3] / "data" / "team_compositions.yaml"
        )
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data.get("compositions", [])


# ═══════════════════════════════════════════════════════════════════════
# TeamPlanner
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class TeamPlanner:
    """Analyse and recommend bot team compositions.

    Usage::

        planner = TeamPlanner()
        bots = [
            {"id": "bot1", "job": "Novice", "level": 12,
             "stats": {"STR": 10, "AGI": 9, "VIT": 5, "INT": 25, "DEX": 15, "LUK": 1}},
            ...
        ]
        roles = planner.assign_roles(bots)
        synergy = planner.get_synergy_score(bots)
        efficiency = planner.get_team_efficiency(bots)
        recommendation = planner.get_optimal_team(bots)
    """

    yaml_path: str | Path | None = None
    _compositions: list[TeamComposition] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        raw = _load_compositions(self.yaml_path)
        for c in raw:
            self._compositions.append(TeamComposition(
                name=c["name"],
                description=c.get("description", ""),
                roles=c.get("roles", {}),
                expected_efficiency=c.get("expected_efficiency", 0),
                min_level=c.get("min_level", 1),
                recommended_level=c.get("recommended_level", 40),
                requirements=c.get("requirements", []),
            ))

    # ── Public API ────────────────────────────────────────────────────

    def assign_roles(self, bots: list[dict[str, Any]]) -> dict[str, str]:
        """Assign each bot a team role based on its stats and job.

        Returns ``{bot_id: role_name}``.
        """
        if not bots:
            return {}

        # Score each bot for each role
        scored: list[tuple[str, str, float]] = []  # (bot_id, role, score)
        for bot in bots:
            bot_id = bot["id"]
            stats = bot.get("stats", {})
            job = bot.get("job", "Novice")
            level = bot.get("level", 1)

            for role in Role:
                if role == Role.UNASSIGNED:
                    continue
                score = self._score_bot_for_role(bot_id, stats, job, level, role.value)
                scored.append((bot_id, role.value, score))

        # Greedy assignment: highest score picks role first
        scored.sort(key=lambda x: -x[2])  # descending score
        assigned: dict[str, str] = {}
        taken_roles: set[str] = set()

        for bot_id, role, _score in scored:
            if bot_id in assigned:
                continue
            if role in taken_roles:
                continue
            assigned[bot_id] = role
            taken_roles.add(role)

        # Any remaining bots get UNASSIGNED
        for bot in bots:
            if bot["id"] not in assigned:
                assigned[bot["id"]] = Role.UNASSIGNED.value

        return assigned

    def get_optimal_team(self, current_bots: list[dict[str, Any]]) -> dict[str, Any]:
        """Return a full recommendation dict with assigned roles, best
        composition match, job targets, level order, and HeuristicActions.

        Returns::

            {
                "assigned_roles": {bot_id: role, ...},
                "best_composition": TeamComposition | None,
                "synergy_score": float,
                "efficiency_estimate": int,
                "job_targets": {bot_id: target_job, ...},
                "level_order": [bot_id, ...],
                "actions": [HeuristicAction, ...],
            }
        """
        if not current_bots:
            return {
                "assigned_roles": {},
                "best_composition": None,
                "synergy_score": 0.0,
                "efficiency_estimate": 0,
                "job_targets": {},
                "level_order": [],
                "actions": [],
            }

        # 1. Assign roles
        roles = self.assign_roles(current_bots)

        # 2. Find best-matching composition
        best_comp: TeamComposition | None = None
        best_overlap = -1
        for comp in self._compositions:
            overlap = self._composition_overlap(
                roles, comp.roles, current_bots
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_comp = comp

        # 3. Synergy & efficiency
        synergy = self.get_synergy_score(current_bots, roles)
        efficiency = self.get_team_efficiency(current_bots, roles)

        # 4. Job targets
        job_targets = self._recommend_job_targets(current_bots, roles)

        # 5. Level order
        level_order = self._recommend_level_order(current_bots, roles)

        # 6. Build actions
        actions: list[HeuristicAction] = []
        for bot_id, role in roles.items():
            if role != Role.UNASSIGNED.value:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"team_assign_role {bot_id} {role}",
                    confidence=0.9,
                    reason=f"{bot_id} assigned as {role}",
                    domain="planning",
                    metadata={"bot_id": bot_id, "role": role},
                ))

        # Level-order action
        level_order_str = ", ".join(level_order)
        actions.append(HeuristicAction(
            kind="command",
            command=f"team_level_order [{level_order_str}]",
            confidence=0.85,
            reason=f"Optimal leveling order: farmer first, then buffer, then support",
            domain="planning",
            metadata={"level_order": level_order},
        ))

        # Job-target actions
        for bot_id, target_job in job_targets.items():
            current_job = self._bot_job(current_bots, bot_id)
            if target_job and target_job != current_job:
                actions.append(HeuristicAction(
                    kind="command",
                    command=f"team_job_target {bot_id} {target_job}",
                    confidence=0.85,
                    reason=f"{bot_id} should aim for {target_job} for {roles.get(bot_id, '?')} role",
                    domain="planning",
                    metadata={"bot_id": bot_id, "target_job": target_job},
                ))

        return {
            "assigned_roles": roles,
            "best_composition": best_comp,
            "synergy_score": synergy,
            "efficiency_estimate": efficiency.get("expected_zeny_per_hour", 0),
            "job_targets": job_targets,
            "level_order": level_order,
            "actions": actions,
        }

    def get_synergy_score(
        self,
        bots: list[dict[str, Any]],
        roles: dict[str, str] | None = None,
    ) -> float:
        """Calculate team synergy as a float in [0.0, 1.0].

        Uses the synergy matrix to score how well the assigned roles
        complement each other.
        """
        if not bots:
            return 0.0
        if roles is None:
            roles = self.assign_roles(bots)

        role_list = list(roles.values())
        unique_roles = set(role_list)

        # Base: each unique role beyond the first adds 0.15
        diversity_score = min((len(unique_roles) - 1) * 0.15, 0.6)

        # Pairwise synergy: sum from matrix
        pair_score = 0.0
        pair_count = 0
        for i in range(len(role_list)):
            for j in range(i + 1, len(role_list)):
                r1, r2 = role_list[i], role_list[j]
                key = (r1, r2)
                if key in SYNERGY_MATRIX:
                    pair_score += SYNERGY_MATRIX[key]
                elif (r2, r1) in SYNERGY_MATRIX:
                    pair_score += SYNERGY_MATRIX[(r2, r1)]
                pair_count += 1

        avg_pair_synergy = pair_score / max(pair_count, 1)
        # Normalise so max realistic (e.g. 3.0) maps to ~0.35
        synergy_from_pairs = min(avg_pair_synergy * 0.12, 0.35)

        # Penalty for redundant roles
        redundancy_penalty = 0.0
        for role in unique_roles:
            count = role_list.count(role)
            if count > 1:
                redundancy_penalty += (count - 1) * REDUNDANCY_PENALTY * 0.1

        raw = diversity_score + synergy_from_pairs - redundancy_penalty
        return max(0.0, min(raw, 1.0))

    def get_team_efficiency(
        self,
        bots: list[dict[str, Any]],
        roles: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Estimate team efficiency in zeny/hr based on roles and level.

        Returns a dict with keys:
          - expected_zeny_per_hour
          - breakdown: per-role contribution
          - confidence: model confidence (0.0–1.0)
        """
        if not bots:
            return {
                "expected_zeny_per_hour": 0,
                "breakdown": {},
                "confidence": 0.0,
            }

        if roles is None:
            roles = self.assign_roles(bots)

        avg_level = sum(b.get("level", 1) for b in bots) / max(len(bots), 1)
        role_bonus = {
            "farmer": 300,
            "buffer": 100,
            "tank": 60,
            "looter": 80,
            "vender": 120,
            "scout": 150,
            "unassigned": 10,
        }

        total = 0
        breakdown: dict[str, int] = {}
        for bot in bots:
            role = roles.get(bot["id"], "unassigned")
            base = role_bonus.get(role, 10)
            # Scale by level (diminishing returns after 99)
            level = bot.get("level", 1)
            level_factor = min(level / 50.0, 2.0)
            contribution = int(base * level_factor)
            breakdown[bot["id"]] = contribution
            total += contribution

        # Apply synergy multiplier
        synergy = self.get_synergy_score(bots, roles)
        synergy_mult = 1.0 + synergy * 1.5  # up to 2.5× with max synergy
        total = int(total * synergy_mult)

        # Round to nearest 100
        total = round(total / 100) * 100

        # Confidence based on data completeness
        complete = all(
            bot.get("stats") is not None and bot.get("job") is not None
            for bot in bots
        )
        confidence = 0.8 if complete else 0.4

        return {
            "expected_zeny_per_hour": total,
            "breakdown": breakdown,
            "confidence": confidence,
        }

    def list_compositions(self) -> list[TeamComposition]:
        """Return all known team compositions from data."""
        return list(self._compositions)

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _score_bot_for_role(
        bot_id: str,
        stats: dict[str, int],
        job: str,
        level: int,
        role: str,
    ) -> float:
        """Score a single bot's suitability for a given role (0–100)."""
        score = 0.0

        # 1) Job affinity (+30 if the bot's job naturally fits the role)
        # Normalise for Transcendent classes
        for affinity_job, affinity_role in JOB_ROLE_AFFINITY.items():
            if affinity_role == role and affinity_job.lower() in job.lower():
                score += 30.0
                break

        # 2) Stat-based scoring
        if role == "farmer":
            score += stats.get("INT", 0) * 1.5
            score += stats.get("DEX", 0) * 0.8
            score -= stats.get("STR", 0) * 0.3  # STR doesn't help farming
        elif role == "buffer":
            score += stats.get("INT", 0) * 1.2
            score += stats.get("DEX", 0) * 0.5
        elif role == "vender":
            score += stats.get("STR", 0) * 1.5
            score -= stats.get("INT", 0) * 0.2
        elif role == "looter":
            score += stats.get("AGI", 0) * 1.5
            score += stats.get("DEX", 0) * 0.5
        elif role == "tank":
            score += stats.get("VIT", 0) * 1.5
            score += stats.get("STR", 0) * 0.3
        elif role == "scout":
            score += stats.get("AGI", 0) * 1.0
            score += stats.get("DEX", 0) * 1.0
            score += stats.get("INT", 0) * 0.2

        # 3) Level bonus (higher level = more effective in role)
        score += level * 0.5

        return score

    @staticmethod
    def _bot_job(bots: list[dict[str, Any]], bot_id: str) -> str:
        """Get a bot's current job."""
        for bot in bots:
            if bot["id"] == bot_id:
                return bot.get("job", "Novice")
        return "Novice"

    def _composition_overlap(
        self,
        assigned: dict[str, str],
        comp_roles: dict[str, str],
        bots: list[dict[str, Any]],
    ) -> int:
        """Count how many roles in *assigned* match the composition's
        role→job expectations."""
        match_count = 0
        for bot in bots:
            bot_id = bot["id"]
            role = assigned.get(bot_id, Role.UNASSIGNED.value)
            expected_job = comp_roles.get(role)
            if expected_job is None:
                continue
            bot_job = bot.get("job", "")
            if expected_job.lower() in bot_job.lower():
                match_count += 1
        return match_count

    def _recommend_job_targets(
        self,
        bots: list[dict[str, Any]],
        roles: dict[str, str],
    ) -> dict[str, str]:
        """For each bot, recommend the best job progression target."""
        targets: dict[str, str] = {}
        for bot in bots:
            bot_id = bot["id"]
            role = roles.get(bot_id, Role.UNASSIGNED.value)
            current_job = bot.get("job", "Novice")
            job_target = self._find_job_target(current_job, role)
            if job_target:
                targets[bot_id] = job_target
        return targets

    def _find_job_target(self, current_job: str, role: str) -> str | None:
        """Find the next job advancement target for a bot given its role."""
        # Check if we already have a progression path
        if current_job in JOB_PROGRESSION:
            path = JOB_PROGRESSION[current_job]
            if path:
                # Return the last (highest) job in the path
                ultimate = path[-1]
                # But check if there's a role-specific preference
                for affinity_job, affinity_role in JOB_ROLE_AFFINITY.items():
                    if affinity_role == role and affinity_job in path:
                        # Prefer the role-aligned job, but aim for highest
                        idx = path.index(affinity_job)
                        # Return the top of the path that aligns with role
                        return path[min(idx + 1, len(path) - 1)]
                return ultimate

        # No path found; try Novice as fallback
        if current_job == "Novice":
            role_to_first_job = {
                "farmer": "Mage",
                "buffer": "Acolyte",
                "vender": "Merchant",
                "looter": "Thief",
                "tank": "Swordsman",
                "scout": "Archer",
            }
            first = role_to_first_job.get(role)
            if first:
                return first
        return current_job  # stay as-is

    def _recommend_level_order(
        self,
        bots: list[dict[str, Any]],
        roles: dict[str, str],
    ) -> list[str]:
        """Recommend which bot to level first (farmer → buffer → rest)."""
        priority = {
            "farmer": 0,
            "scout": 1,
            "buffer": 2,
            "tank": 3,
            "looter": 4,
            "vender": 5,
            "unassigned": 6,
        }

        # Sort by role priority, then by current level (lower level first
        # = needs more leveling)
        def _sort_key(bot: dict[str, Any]) -> tuple[int, int]:
            role = roles.get(bot["id"], "unassigned")
            return (priority.get(role, 6), bot.get("level", 1))

        sorted_bots = sorted(bots, key=_sort_key)
        return [b["id"] for b in sorted_bots]

    def __repr__(self) -> str:
        return f"<TeamPlanner: {len(self._compositions)} compositions>"
