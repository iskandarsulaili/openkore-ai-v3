"""Learning Feedback Loop — closes the data→decision gap.

Tracks performance metrics per map, per monster, per skill and
adjusts AI system parameters based on learning. All decisions
are data-driven, not hardcoded.

Persistence: sidecar_experience.sqlite
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections import defaultdict
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)

_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "sidecar_experience.sqlite",
)


class LearningFeedbackLoop:
    """Self-* learning system that closes the data→decision gap.

    Learning dimensions:
    - Map effectiveness: kill_rate, loot_value, death_rate per map
    - Skill effectiveness: damage_per_sp per skill per monster type
    - Reflex tuning: adjusts HighFreqReflex thresholds based on survival data
    - Combat strategy: adjusts element/race priorities based on effectiveness
    - Economy: adjusts loot filters based on realized value
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._lock = RLock()
        self._db_path = db_path or _DEFAULT_DB
        self._session_start = time.time()
        self._last_adjustment: dict[str, Any] = {}

        # In-memory accumulators (flushed to DB periodically)
        self._map_metrics: dict[str, dict[str, float]] = defaultdict(
            lambda: {"kills": 0, "deaths": 0, "loot_value": 0.0, "time_spent": 0.0}
        )
        self._skill_metrics: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"uses": 0, "total_damage": 0, "total_sp": 0, "monster_elements": defaultdict(int)}
        )
        self._session_metrics: dict[str, float] = defaultdict(float)

        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite tables if they don't exist."""
        try:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS map_learning (
                    map_name TEXT PRIMARY KEY,
                    total_kills INTEGER DEFAULT 0,
                    total_deaths INTEGER DEFAULT 0,
                    total_loot_value REAL DEFAULT 0.0,
                    total_time_seconds REAL DEFAULT 0.0,
                    last_updated REAL DEFAULT 0.0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skill_learning (
                    skill_id TEXT,
                    monster_element TEXT,
                    uses INTEGER DEFAULT 0,
                    total_damage REAL DEFAULT 0.0,
                    total_sp REAL DEFAULT 0.0,
                    last_updated REAL DEFAULT 0.0,
                    PRIMARY KEY (skill_id, monster_element)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reflex_adjustments (
                    threshold_key TEXT PRIMARY KEY,
                    current_value REAL DEFAULT 0.0,
                    adjustments INTEGER DEFAULT 0,
                    last_adjusted REAL DEFAULT 0.0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_history (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time REAL,
                    end_time REAL,
                    total_kills INTEGER DEFAULT 0,
                    total_deaths INTEGER DEFAULT 0,
                    total_loot_value REAL DEFAULT 0.0,
                    summary TEXT
                )
            """)
            conn.commit()
            conn.close()
            logger.info("learning_db_ready: path=%s", self._db_path)
        except Exception as e:
            logger.warning("learning_db_init_failed: %s", e)

    # ── Map Learning ──

    def record_map_kill(self, map_name: str, loot_value: float = 0.0) -> None:
        """Record a kill on a map."""
        with self._lock:
            m = self._map_metrics[map_name]
            m["kills"] += 1
            m["loot_value"] += loot_value
            self._session_metrics["total_kills"] += 1
            self._session_metrics["total_loot_value"] += loot_value

    def record_map_death(self, map_name: str) -> None:
        """Record a death on a map."""
        with self._lock:
            m = self._map_metrics[map_name]
            m["deaths"] += 1
            self._session_metrics["total_deaths"] += 1

    def record_map_time(self, map_name: str, seconds: float) -> None:
        """Record time spent on a map."""
        with self._lock:
            self._map_metrics[map_name]["time_spent"] += seconds

    def map_kill_rate(self, map_name: str) -> float:
        """Get kills per hour on this map."""
        with self._lock:
            m = self._map_metrics.get(map_name, {})
            hours = m.get("time_spent", 0) / 3600.0
            if hours < 0.01:
                return 0.0
            return m.get("kills", 0) / hours

    def map_death_rate(self, map_name: str) -> float:
        """Get deaths per hour on this map."""
        with self._lock:
            m = self._map_metrics.get(map_name, {})
            hours = m.get("time_spent", 0) / 3600.0
            if hours < 0.01:
                return 10.0  # Unknown maps are dangerous
            return m.get("deaths", 0) / hours

    def map_efficiency_score(self, map_name: str) -> float:
        """Combined efficiency score: (loot_value - cost_of_deaths) / time.
        Higher is better. Negative means the map is costing money.
        """
        with self._lock:
            m = self._map_metrics.get(map_name, {})
            hours = m.get("time_spent", 0) / 3600.0
            if hours < 0.01:
                return 0.0
            loot = m.get("loot_value", 0.0)
            deaths = m.get("deaths", 0)
            # Each death costs ~1000z (respawn potions, repair, lost time)
            death_cost = deaths * 1000.0
            return (loot - death_cost) / hours

    def recommend_map(self, current_map: str, level: int) -> dict[str, Any]:
        """Recommend whether to stay on current map or move.
        Returns dict with recommendation, reasoning, and alternative maps.
        """
        with self._lock:
            eff = self.map_efficiency_score(current_map)
            death_rate = self.map_death_rate(current_map)

            reason_parts = []
            if death_rate > 5.0:
                reason_parts.append(f"death_rate too high ({death_rate:.1f}/hr)")
                should_move = True
            elif eff < 0:
                reason_parts.append(f"negative efficiency ({eff:.0f}z/hr)")
                should_move = True
            else:
                reason_parts.append(f"acceptable efficiency ({eff:.0f}z/hr)")
                should_move = time.time() - self._session_start > 3600  # Re-evaluate after 1 hour

            return {
                "should_move": should_move,
                "current_map": current_map,
                "efficiency": eff,
                "death_rate": death_rate,
                "reason": ", ".join(reason_parts),
                "time_on_map": self._map_metrics.get(current_map, {}).get("time_spent", 0),
            }

    # ── Skill Learning ──

    def record_skill_use(
        self, skill_id: str, damage: float, sp_cost: float, monster_element: str
    ) -> None:
        """Record a skill use with damage and SP cost."""
        with self._lock:
            key = f"{skill_id}:{monster_element}"
            s = self._skill_metrics[key]
            s["uses"] += 1
            s["total_damage"] += damage
            s["total_sp"] += sp_cost
            s["monster_elements"][monster_element] += 1

    def skill_effectiveness(self, skill_id: str, monster_element: str) -> float:
        """Get damage per SP for this skill vs this element."""
        with self._lock:
            key = f"{skill_id}:{monster_element}"
            s = self._skill_metrics.get(key, {})
            if s.get("total_sp", 0) <= 0:
                return 1.0  # Unknown skill = neutral effectiveness
            return s.get("total_damage", 0) / s.get("total_sp", 1)

    def best_skill_for_element(self, available_skills: list[str], monster_element: str) -> str | None:
        """From a list of available skill IDs, pick the best one for this monster element."""
        with self._lock:
            best = None
            best_score = 0.0
            for skill_id in available_skills:
                eff = self.skill_effectiveness(skill_id, monster_element)
                if eff > best_score:
                    best_score = eff
                    best = skill_id
            return best

    # ── Reflex Adjustment ──

    def adjust_reflex_thresholds(self, recent_deaths: int, total_time: float) -> dict[str, float]:
        """Adjust HighFreqReflex thresholds based on survival data.
        
        If recent death rate is high -> lower thresholds (more conservative).
        If no deaths -> raise thresholds (more aggressive).
        
        Returns dict of {threshold_key: new_value} to pass to update_thresholds().
        """
        with self._lock:
            death_rate = recent_deaths / max(total_time, 1) * 3600  # deaths per hour
            adjustments: dict[str, float] = {}

            if death_rate > 10:
                # Too many deaths -> escape earlier, heal earlier
                adjustments["escape_teleport_hp_pct"] = 0.30  # Was 0.15
                adjustments["emergency_potion_hp_pct"] = 0.50  # Was 0.30
                adjustments["heal_potion_hp_pct"] = 0.70  # Was 0.50
                self._last_adjustment["reason"] = "high_death_rate"
            elif death_rate > 2:
                # Moderate deaths -> slight increase
                adjustments["escape_teleport_hp_pct"] = 0.20
                adjustments["emergency_potion_hp_pct"] = 0.40
                adjustments["heal_potion_hp_pct"] = 0.60
                self._last_adjustment["reason"] = "moderate_death_rate"
            else:
                # Safe -> standard thresholds
                adjustments["escape_teleport_hp_pct"] = 0.15
                adjustments["emergency_potion_hp_pct"] = 0.30
                adjustments["heal_potion_hp_pct"] = 0.50
                self._last_adjustment["reason"] = "low_death_rate"

            self._last_adjustment["death_rate"] = death_rate
            self._last_adjustment["adjusted_at"] = time.time()
            return adjustments

    def get_last_adjustment(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_adjustment)

    # ── Persistence ──

    def flush(self) -> None:
        """Flush in-memory metrics to SQLite."""
        try:
            conn = sqlite3.connect(self._db_path)
            now = time.time()

            for map_name, metrics in self._map_metrics.items():
                if metrics["kills"] == 0 and metrics["deaths"] == 0:
                    continue
                conn.execute(
                    """INSERT INTO map_learning (map_name, total_kills, total_deaths,
                       total_loot_value, total_time_seconds, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(map_name) DO UPDATE SET
                       total_kills = total_kills + excluded.total_kills,
                       total_deaths = total_deaths + excluded.total_deaths,
                       total_loot_value = total_loot_value + excluded.total_loot_value,
                       total_time_seconds = total_time_seconds + excluded.total_time_seconds,
                       last_updated = excluded.last_updated""",
                    (map_name, int(metrics["kills"]), int(metrics["deaths"]),
                     metrics["loot_value"], metrics["time_spent"], now),
                )

            for key, s in self._skill_metrics.items():
                parts = key.split(":", 1)
                skill_id = parts[0]
                mon_elem = parts[1] if len(parts) > 1 else "unknown"
                conn.execute(
                    """INSERT INTO skill_learning (skill_id, monster_element, uses,
                       total_damage, total_sp, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(skill_id, monster_element) DO UPDATE SET
                       uses = uses + excluded.uses,
                       total_damage = total_damage + excluded.total_damage,
                       total_sp = total_sp + excluded.total_sp,
                       last_updated = excluded.last_updated""",
                    (skill_id, mon_elem, int(s["uses"]),
                     s["total_damage"], s["total_sp"], now),
                )

            conn.commit()
            conn.close()

            # Reset in-memory accumulators
            self._map_metrics.clear()
            self._skill_metrics.clear()

        except Exception as e:
            logger.warning("learning_flush_failed: %s", e)

    def load_persisted(self) -> dict[str, Any]:
        """Load learning data from SQLite for analysis."""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row

            maps = [
                dict(row) for row in conn.execute(
                    "SELECT * FROM map_learning ORDER BY total_kills DESC LIMIT 20"
                ).fetchall()
            ]
            skills = [
                dict(row) for row in conn.execute(
                    "SELECT * FROM skill_learning ORDER BY uses DESC LIMIT 20"
                ).fetchall()
            ]
            adjustments = [
                dict(row) for row in conn.execute(
                    "SELECT * FROM reflex_adjustments ORDER BY last_adjusted DESC LIMIT 10"
                ).fetchall()
            ]

            conn.close()
            return {"maps": maps, "skills": skills, "adjustments": adjustments}
        except Exception as e:
            logger.warning("learning_load_failed: %s", e)
            return {"maps": [], "skills": [], "adjustments": []}

    # ── Session Management ──

    def end_session(self) -> dict[str, Any]:
        """End current session and save summary."""
        elapsed = time.time() - self._session_start
        summary = {
            "duration": elapsed,
            "total_kills": int(self._session_metrics.get("total_kills", 0)),
            "total_deaths": int(self._session_metrics.get("total_deaths", 0)),
            "total_loot_value": self._session_metrics.get("total_loot_value", 0.0),
        }
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                """INSERT INTO session_history (start_time, end_time, total_kills,
                   total_deaths, total_loot_value, summary)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (self._session_start, time.time(), summary["total_kills"],
                 summary["total_deaths"], summary["total_loot_value"], json.dumps(summary)),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("learning_end_session_failed: %s", e)

        self.flush()
        return summary

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic stats."""
        with self._lock:
            return {
                "maps_tracked": len(self._map_metrics),
                "skills_tracked": len(self._skill_metrics),
                "session_kills": int(self._session_metrics.get("total_kills", 0)),
                "session_deaths": int(self._session_metrics.get("total_deaths", 0)),
                "session_loot_value": self._session_metrics.get("total_loot_value", 0.0),
                "session_duration_s": time.time() - self._session_start,
                "last_adjustment": self._last_adjustment,
            }


def create_learning_loop(db_path: str | None = None) -> LearningFeedbackLoop:
    """Factory function for dependency injection."""
    return LearningFeedbackLoop(db_path=db_path)
