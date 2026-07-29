"""Stat Breakpoint Planner — real RO stat thresholds and recommendations.

RO stats have meaningful breakpoints that affect gameplay:
  - DEX: 95 = instant cast, 150 = minimum cast time
  - AGI: 150 = 99% dodge for most PvE, 99 = max Flee per point
  - INT: 75 = max SP regen, 99 = max MATK
  - STR: 99 = max weight bonus
  - VIT: 100 = stun immunity
  - LUK: 100 = max crit rate benefit

This planner loads the same YAML data as BuildPlanner and adds
breakpoint-aware recommendations.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _load_data(yaml_path: str | Path | None = None) -> dict[str, Any]:
    """Load data from build_plans.yaml."""
    if yaml_path is None:
        yaml_path = (
            Path(__file__).resolve().parents[3] / "data" / "build_plans.yaml"
        )
    with open(yaml_path) as f:
        return yaml.safe_load(f)


# Standard RO stat order
ALL_STATS = ["STR", "AGI", "VIT", "INT", "DEX", "LUK"]

# Hard-coded reference breakpoints (also in YAML; kept here as fallback)
KNOWN_BREAKPOINTS: dict[str, dict[int, dict[str, Any]]] = {
    "DEX": {
        95: {"description": "Instant cast — no cast time for most skills", "effect": "cast_time_zero"},
        150: {"description": "Minimum cast time (fixed cast time floor)", "effect": "cast_time_minimum"},
    },
    "AGI": {
        99: {"description": "Max Flee per point (diminishing returns after)", "effect": "flee_efficiency_peak"},
        150: {"description": "99% dodge rate for most PvE monsters", "effect": "near_perfect_dodge_pve"},
    },
    "INT": {
        75: {"description": "Maximum SP regeneration tick", "effect": "max_sp_regen"},
        99: {"description": "Maximum MATK from INT", "effect": "max_matk"},
    },
    "STR": {
        99: {"description": "Maximum weight bonus (800 -> 1300 additional weight)", "effect": "max_weight_bonus"},
    },
    "VIT": {
        100: {"description": "Stun immunity", "effect": "stun_immunity"},
    },
    "LUK": {
        100: {"description": "Maximum crit rate benefit from LUK", "effect": "max_crit_rate"},
    },
}


class StatBreakpointPlanner:
    """Plans stat allocation with awareness of RO breakpoints.

    This is a companion to BuildPlanner — it adds breakpoint-aware
    recommendations on top of target stat distributions.
    """

    def __init__(self, yaml_path: str | Path | None = None) -> None:
        data = _load_data(yaml_path)
        self._builds: list[dict[str, Any]] = data.get("builds", [])
        yaml_bps: dict[str, dict[str, Any]] = data.get("breakpoints", {})

        # Merge YAML breakpoints with known fallback
        self._breakpoints: dict[str, dict[int, dict[str, Any]]] = {}
        for stat in ALL_STATS:
            merged: dict[int, dict[str, Any]] = dict(KNOWN_BREAKPOINTS.get(stat, {}))
            yaml_stat = yaml_bps.get(stat, {})
            for k, v in yaml_stat.items():
                merged[int(k)] = dict(v)
            self._breakpoints[stat] = merged

        # Index builds by id
        self._by_id: dict[str, dict[str, Any]] = {}
        for b in self._builds:
            self._by_id[b["id"]] = b

        # Index builds by job
        self._by_job: dict[str, list[dict[str, Any]]] = {}
        for b in self._builds:
            self._by_job.setdefault(b["job"], []).append(b)

    # ── Target stats ─────────────────────────────────────────────────

    def get_target_stats(
        self,
        job: str,
        build_name: str | None = None,
    ) -> dict[str, int] | None:
        """Get target stats for a job's first (or named) meta build."""
        builds = self._by_job.get(job, [])
        if not builds:
            return None
        if build_name:
            name_lower = build_name.lower()
            for b in builds:
                if name_lower in b["name"].lower():
                    return dict(b["target_stats"])
        return dict(builds[0]["target_stats"])

    # ── Breakpoint info ──────────────────────────────────────────────

    def get_all_breakpoints(self) -> dict[str, dict[int, dict[str, Any]]]:
        """Return all known stat breakpoints."""
        return dict(self._breakpoints)

    def get_breakpoint_info(self, stat: str, value: int) -> dict[str, Any] | None:
        """Get info about what's unlocked at a specific stat value.

        Returns the closest breakpoint at or below the given value.
        """
        stat_upper = stat.upper()
        if stat_upper not in self._breakpoints:
            return None
        bps = self._breakpoints[stat_upper]
        best: dict[str, Any] | None = None
        best_value = 0
        for bp_value in sorted(bps.keys(), reverse=True):
            if bp_value <= value:
                info = dict(bps[bp_value])
                info["breakpoint_value"] = bp_value
                info["current_value"] = value
                info["remaining"] = bp_value - value if value < bp_value else 0
                best = info
                break
        return best

    def is_stat_breakpoint(self, stat: str, value: int) -> bool:
        """Check if a specific value is an exact stat breakpoint."""
        stat_upper = stat.upper()
        if stat_upper not in self._breakpoints:
            return False
        return value in self._breakpoints[stat_upper]

    def get_next_breakpoint(self, stat: str, current_value: int) -> dict[str, Any] | None:
        """Get the next breakpoint for a stat above the current value."""
        stat_upper = stat.upper()
        if stat_upper not in self._breakpoints:
            return None
        bps = self._breakpoints[stat_upper]
        for bp_value in sorted(bps.keys()):
            if bp_value > current_value:
                info = dict(bps[bp_value])
                info["breakpoint_value"] = bp_value
                info["current_value"] = current_value
                info["remaining"] = bp_value - current_value
                return info
        return None

    def get_breakpoints_for_stat(self, stat: str) -> list[dict[str, Any]]:
        """Get all breakpoints for a given stat as a sorted list."""
        stat_upper = stat.upper()
        if stat_upper not in self._breakpoints:
            return []
        bps = self._breakpoints[stat_upper]
        result = []
        for bp_value in sorted(bps.keys()):
            info = dict(bps[bp_value])
            info["breakpoint_value"] = bp_value
            result.append(info)
        return result

    # ── Recommendation ───────────────────────────────────────────────

    def recommend_next_stat(
        self,
        current_stats: dict[str, int],
        target_stats: dict[str, int],
    ) -> str:
        """Recommend which stat to raise next level-up.

        Strategy:
          1. First check if any stat is *approaching* a breakpoint (within 5 points)
             and not yet at target — prioritise hitting the breakpoint.
          2. Otherwise, pick the stat furthest from its target.
        """
        # Priority order
        priority_order = ["STR", "AGI", "VIT", "INT", "DEX", "LUK"]

        # Phase 1: check for imminent breakpoints (within 5 levels, not at target)
        candidates: list[tuple[str, int]] = []
        for stat in priority_order:
            current = current_stats.get(stat, 0)
            target = target_stats.get(stat, 0)
            if current >= target:
                continue
            next_bp = self.get_next_breakpoint(stat, current)
            if next_bp:
                bp_value = next_bp["breakpoint_value"]
                if bp_value <= target and (bp_value - current) <= 5:
                    candidates.append((stat, bp_value - current))

        # Prioritise closest imminent breakpoint
        if candidates:
            candidates.sort(key=lambda x: (x[1], priority_order.index(x[0])))
            return candidates[0][0]

        # Phase 2: furthest from target
        best_stat = priority_order[0]
        best_gap = -1
        for stat in priority_order:
            current = current_stats.get(stat, 0)
            target = target_stats.get(stat, 0)
            if current < target:
                gap = target - current
                if gap > best_gap:
                    best_gap = gap
                    best_stat = stat
        return best_stat

    # ── Skill builds ─────────────────────────────────────────────────

    def get_skill_build(
        self,
        job: str,
        build_name: str | None = None,
    ) -> list[tuple[str, int]]:
        """Get the skill allocation for a build as (skill_id, level) pairs."""
        builds = self._by_job.get(job, [])
        if not builds:
            return []
        if build_name:
            name_lower = build_name.lower()
            for b in builds:
                if name_lower in b["name"].lower():
                    return [
                        (s["id"], s["level"])
                        for s in b.get("skills", [])
                        if s.get("level", 0) > 0
                    ]
        return [
            (s["id"], s["level"])
            for s in builds[0].get("skills", [])
            if s.get("level", 0) > 0
        ]

    def get_trap_skills(self, job: str) -> list[str]:
        """Aggregate all trap skills across all builds for a job.

        These are skill IDs the bot should never auto-pick.
        """
        seen: set[str] = set()
        for build in self._by_job.get(job, []):
            seen.update(build.get("trap_skills", []))
        return sorted(seen)

    # ── Convenience ──────────────────────────────────────────────────

    def get_stat_summary(self, current_stats: dict[str, int]) -> list[dict[str, Any]]:
        """Produce a summary of what breakpoints are met for each stat.

        Returns a list of dicts with keys: stat, value, breakpoints_hit,
        next_breakpoint, remaining.
        """
        summary = []
        for stat in ALL_STATS:
            value = current_stats.get(stat, 0)
            bp_info = self.get_breakpoint_info(stat, value)
            next_bp = self.get_next_breakpoint(stat, value)
            info: dict[str, Any] = {
                "stat": stat,
                "current_value": value,
                "breakpoint_active": bp_info["description"] if bp_info else None,
                "next_breakpoint": next_bp["breakpoint_value"] if next_bp else None,
                "remaining_to_next": next_bp["remaining"] if next_bp else None,
            }
            summary.append(info)
        return summary

    def __repr__(self) -> str:
        return f"<StatBreakpointPlanner: {len(self._builds)} builds across {len(self._by_job)} jobs>"
