"""Efficiency tracker — tracks exp/hr, zeny/hr, kills/hr, deaths/hr from snapshot diffs.

The Pro RO Player uses these metrics to evaluate whether current hunting grounds
are effective, and to compare alternative strategies quantitatively.

Usage:
    tracker = EfficiencyTracker()
    tracker.update(snapshot)  # call every cycle
    metrics = tracker.get_metrics(bot_id)
    # Returns: {exp_hour, zeny_hour, kills_hour, deaths_hour, ...}
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


class EfficiencyTracker:
    """Tracks per-bot efficiency metrics across snapshots.
    
    Stores a sliding window of snapshot state diffs and computes
    exp/hour, zeny/hour, kills/hour, deaths/hour, and uptime.
    All metrics are server-agnostic — they just track deltas.
    """

    def __init__(self, window_minutes: int = 10):
        self._lock = RLock()
        self._window_minutes = window_minutes
        # bot_id -> {timestamp: state_snapshot}
        self._history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        # bot_id -> running totals
        self._totals: dict[str, dict[str, float]] = defaultdict(lambda: {
            "exp_total": 0.0, "zeny_total": 0.0, "kills_total": 0,
            "deaths_total": 0, "sessions_total": 0, "uptime_seconds": 0.0,
            "last_exp": 0.0, "last_zeny": 0, "last_base_level": 0,
            "last_job_level": 0, "session_start": time.time(),
        })

    def update(self, snapshot: Any, bot_id: str | None = None) -> None:
        """Process a new snapshot and update efficiency metrics."""
        if snapshot is None:
            return
        
        resolved_bot_id = bot_id or "default"
        now = time.time()
        
        with self._lock:
            totals = self._totals[resolved_bot_id]
            history = self._history[resolved_bot_id]
            
            # Extract progression data
            prog = getattr(snapshot, "progression", None) if not isinstance(snapshot, dict) else snapshot.get("progression")
            status = getattr(snapshot, "status", None) if not isinstance(snapshot, dict) else snapshot.get("status")
            inventory = getattr(snapshot, "inventory", None) if not isinstance(snapshot, dict) else snapshot.get("inventory")
            
            current_exp = 0.0
            current_zeny = 0
            current_base_level = 0
            current_job_level = 0
            
            if isinstance(snapshot, dict):
                if prog:
                    current_exp = float(prog.get("exp", 0) or 0)
                    current_zeny = int(inventory.get("zeny", 0) or 0) if inventory else 0
                    current_base_level = int(prog.get("base_level", 0) or 0)
                    current_job_level = int(prog.get("job_level", 0) or 0)
            else:
                if prog:
                    current_exp = float(getattr(prog, "exp", 0) or 0)
                    current_zeny = int(getattr(inventory, "zeny", 0) or 0) if inventory else 0
                    current_base_level = int(getattr(prog, "base_level", 0) or 0)
                    current_job_level = int(getattr(prog, "job_level", 0) or 0)
            
            # First snapshot — just record
            if totals["last_base_level"] == 0:
                totals["last_exp"] = current_exp
                totals["last_zeny"] = current_zeny
                totals["last_base_level"] = current_base_level
                totals["last_job_level"] = current_job_level
                totals["session_start"] = now
                history.append({
                    "ts": now, "exp": current_exp, "zeny": current_zeny,
                    "base_level": current_base_level, "job_level": current_job_level,
                })
                return
            
            # Track level-ups
            if current_base_level > totals["last_base_level"]:
                totals["last_base_level"] = current_base_level
            if current_job_level > totals["last_job_level"]:
                totals["last_job_level"] = current_job_level
            
            # Track exp/zeny gains (handle level-up resets)
            if current_exp > 0 and current_exp >= totals["last_exp"]:
                exp_gain = current_exp - totals["last_exp"]
                if exp_gain > 0 and exp_gain < 100_000_000:  # sanity cap
                    totals["exp_total"] += exp_gain
                totals["last_exp"] = current_exp
            elif current_exp == 0:
                # Level up reset — store current but don't lose the delta
                pass
            
            zeny_diff = current_zeny - totals["last_zeny"]
            if abs(zeny_diff) < 10_000_000:  # sanity cap
                if zeny_diff > 0:
                    totals["zeny_total"] += zeny_diff
                elif zeny_diff < 0:
                    # Spent money — track as expense
                    pass
            totals["last_zeny"] = current_zeny
            
            # Prune history to time window
            cutoff = now - (self._window_minutes * 60)
            history[:] = [h for h in history if h["ts"] >= cutoff]
            history.append({
                "ts": now, "exp": current_exp, "zeny": current_zeny,
                "base_level": current_base_level, "job_level": current_job_level,
            })

    def record_death(self, bot_id: str) -> None:
        """Record a death event for efficiency tracking."""
        with self._lock:
            self._totals[bot_id]["deaths_total"] += 1

    def record_kill(self, bot_id: str) -> None:
        """Record a monster kill."""
        with self._lock:
            self._totals[bot_id]["kills_total"] += 1

    def get_metrics(self, bot_id: str) -> dict[str, Any]:
        """Get efficiency metrics for a bot.
        
        Returns dict with exp_hour, zeny_hour, kills_hour, deaths_hour,
        uptime, level, and estimated leveling time.
        """
        with self._lock:
            totals = self._totals.get(bot_id)
            if totals is None:
                return {}
            
            now = time.time()
            elapsed = max(now - totals["session_start"], 1)
            elapsed_hours = elapsed / 3600.0
            
            metrics = {
                "exp_hour": round(totals["exp_total"] / elapsed_hours, 0) if elapsed_hours > 0 else 0.0,
                "zeny_hour": round(totals["zeny_total"] / elapsed_hours, 0) if elapsed_hours > 0 else 0.0,
                "kills_hour": round(totals["kills_total"] / elapsed_hours, 1) if elapsed_hours > 0 else 0.0,
                "deaths_hour": round(totals["deaths_total"] / elapsed_hours, 2) if elapsed_hours > 0 else 0.0,
                "uptime_minutes": round(elapsed / 60, 1),
                "exp_total": round(totals["exp_total"], 0),
                "zeny_total": round(totals["zeny_total"], 0),
                "kills_total": totals["kills_total"],
                "deaths_total": totals["deaths_total"],
                "base_level": int(totals["last_base_level"]),
                "job_level": int(totals["last_job_level"]),
            }
            
            # Estimate time to next level (if we have exp rate)
            if metrics["exp_hour"] > 0:
                # Rough estimate: level N needs ~N^2 * 1000 exp at low levels
                level = max(metrics["base_level"], 1)
                exp_to_next = level * level * 1000
                hours_to_next = exp_to_next / metrics["exp_hour"]
                metrics["estimated_hours_to_next_level"] = round(hours_to_next, 1)
            else:
                metrics["estimated_hours_to_next_level"] = 999.0
            
            return metrics

    def get_summary(self, bot_id: str) -> str:
        """Get a human-readable efficiency summary."""
        m = self.get_metrics(bot_id)
        if not m:
            return "No efficiency data yet"
        
        parts = [
            f"Level {m.get('base_level', '?')}/{m.get('job_level', '?')}",
            f"EXP: {m.get('exp_hour', 0):.0f}/hr",
            f"Zeny: {m.get('zeny_hour', 0):.0f}/hr",
        ]
        if m.get('deaths_hour', 0) > 0:
            parts.append(f"Deaths: {m.get('deaths_hour', 0):.1f}/hr ⚠️")
        if m.get('kills_hour', 0) > 0:
            parts.append(f"Kills: {m.get('kills_hour', 0):.0f}/hr")
        parts.append(f"Uptime: {m.get('uptime_minutes', 0):.0f}min")
        
        est = m.get('estimated_hours_to_next_level', 0)
        if est < 999:
            parts.append(f"~{est:.1f}h to next level")
        
        return " | ".join(parts)

    def get_all_bot_ids(self) -> list[str]:
        return list(self._totals.keys())
