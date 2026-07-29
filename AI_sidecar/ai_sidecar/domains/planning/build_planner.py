"""Build Planner — knows meta builds for each RO job/class.

Loads build data from build_plans.yaml and provides:
  - Looking up meta builds by job or build name
  - Recommending stat allocation per level up
  - Recommending skills per job advancement
  - Identifying trap skills to avoid
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Data helpers ────────────────────────────────────────────────────


def _load_builds(yaml_path: str | Path | None = None) -> dict[str, Any]:
    """Load build plans from YAML.

    Defaults to ``AI_sidecar/data/build_plans.yaml`` relative to this file.
    """
    if yaml_path is None:
        yaml_path = (
            Path(__file__).resolve().parents[3] / "data" / "build_plans.yaml"
        )
    with open(yaml_path) as f:
        return yaml.safe_load(f)


# ── BuildPlanner ────────────────────────────────────────────────────


class BuildPlanner:
    """RO meta-build knowledge base.

    For each job/class, knows the recommended meta builds, stat targets,
    skill allocations, and trap skills to avoid.
    """

    def __init__(self, yaml_path: str | Path | None = None) -> None:
        data = _load_builds(yaml_path)
        self._builds: list[dict[str, Any]] = data.get("builds", [])
        self._breakpoints: dict[str, dict[str, Any]] = data.get("breakpoints", {})

        # Index builds by job and by id for fast lookup
        self._by_job: dict[str, list[dict[str, Any]]] = {}
        self._by_id: dict[str, dict[str, Any]] = {}
        for b in self._builds:
            job = b["job"]
            self._by_job.setdefault(job, []).append(b)
            self._by_id[b["id"]] = b

    # ── Queries ──────────────────────────────────────────────────────

    def list_builds(self, job: str | None = None) -> list[dict[str, Any]]:
        """List all known builds, optionally filtered by job name."""
        if job:
            return [
                b for b in self._builds
                if b["job"].lower() == job.lower()
            ]
        return list(self._builds)

    def get_build_by_id(self, build_id: str) -> dict[str, Any] | None:
        """Look up a specific build by its unique id."""
        return self._by_id.get(build_id)

    def get_builds_for_job(self, job: str) -> list[dict[str, Any]]:
        """Get all meta builds for a given job class."""
        return self._by_job.get(job, [])

    def get_jobs(self) -> list[str]:
        """List all jobs that have meta builds defined."""
        return sorted(self._by_job.keys())

    # ── Stat recommendations ─────────────────────────────────────────

    def get_target_stats(
        self,
        build_id: str | None = None,
        job: str | None = None,
        build_name: str | None = None,
    ) -> dict[str, int] | None:
        """Return the target stat distribution for a build.

        Accepts either ``build_id``, ``job`` (uses first build for that
        job), or ``build_name`` (fuzzy match).
        """
        build = self._resolve_build(build_id, job, build_name)
        if build is None:
            return None
        return dict(build["target_stats"])

    def recommend_next_stat(
        self,
        current_stats: dict[str, int],
        target_stats: dict[str, int],
    ) -> str:
        """Recommend which stat to raise next based on priority.

        Prioritises stats furthest from their target, weighted by the
        build's stat_priority order.
        """
        best_stat = "STR"  # fallback
        best_gap = -1

        # Determine stat priority — fall back to alphabetical
        priority_order = ["STR", "AGI", "VIT", "INT", "DEX", "LUK"]

        for stat in priority_order:
            current = current_stats.get(stat, 0)
            target = target_stats.get(stat, 0)
            if current < target:
                gap = target - current
                if gap > best_gap:
                    best_gap = gap
                    best_stat = stat

        return best_stat

    def get_stat_priority(self, build_id: str) -> list[str]:
        """Get the stat raising priority order for a build."""
        build = self._by_id.get(build_id)
        if build is None:
            return []
        return list(build.get("stat_priority", []))

    # ── Skill recommendations ────────────────────────────────────────

    def get_skill_build(
        self,
        build_id: str | None = None,
        job: str | None = None,
        build_name: str | None = None,
    ) -> list[tuple[str, int]]:
        """Get the skill allocation for a build as (skill_id, level) pairs.

        Skills with level 0 are explicitly excluded — they are traps
        or intentionally skipped.
        """
        build = self._resolve_build(build_id, job, build_name)
        if build is None:
            return []
        return [
            (s["id"], s["level"])
            for s in build.get("skills", [])
            if s.get("level", 0) > 0
        ]

    def get_trap_skills(
        self,
        build_id: str | None = None,
        job: str | None = None,
        build_name: str | None = None,
    ) -> list[str]:
        """Get skill IDs to never pick for a build."""
        build = self._resolve_build(build_id, job, build_name)
        if build is None:
            return []
        return list(build.get("trap_skills", []))

    def get_trap_skills_for_job(self, job: str) -> list[str]:
        """Aggregate all trap skills across all builds for a job."""
        seen: set[str] = set()
        for build in self._by_job.get(job, []):
            seen.update(build.get("trap_skills", []))
        return sorted(seen)

    # ── Breakpoints ──────────────────────────────────────────────────

    def get_all_breakpoints(self) -> dict[str, dict[str, Any]]:
        """Return the full breakpoints table (stat -> value -> info)."""
        return dict(self._breakpoints)

    def get_breakpoint_info(self, stat: str, value: int) -> dict[str, Any] | None:
        """Get what's unlocked at a specific stat value."""
        stat_bps = self._breakpoints.get(stat.upper())
        if stat_bps is None:
            return None
        # Return the *closest breakpoint* at or below the given value
        best: dict[str, Any] | None = None
        best_value = 0
        for bp_value_str, info in stat_bps.items():
            bp_value = int(bp_value_str)
            if bp_value <= value and bp_value > best_value:
                best = info
                best_value = bp_value
        return best

    def is_stat_breakpoint(self, stat: str, value: int) -> bool:
        """Check if a specific value is an exact breakpoint for a stat."""
        stat_bps = self._breakpoints.get(stat.upper())
        if stat_bps is None:
            return False
        return value in stat_bps

    # ── Helpers ──────────────────────────────────────────────────────

    def _resolve_build(
        self,
        build_id: str | None = None,
        job: str | None = None,
        build_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Resolve a build from one of the three lookup strategies."""
        if build_id:
            return self._by_id.get(build_id)
        if job:
            builds = self._by_job.get(job, [])
            if builds:
                return builds[0]
        if build_name:
            name_lower = build_name.lower()
            for b in self._builds:
                if name_lower in b["name"].lower():
                    return b
        return None

    def __repr__(self) -> str:
        return f"<BuildPlanner: {len(self._builds)} builds, {len(self._by_job)} jobs>"
