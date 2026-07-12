"""Fleet self-learning system — zone ELO scoring, party composition learning, mined-out detection.

Tracks experience rates per zone, computes ELO scores for zone difficulty/reward,
recommends optimal zones based on bot level, detects when zones are "mined out",
and learns optimal party compositions from historical success rates.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ZoneOutcome:
    """Result of a bot spending time in a zone."""
    bot_id: str
    map_name: str
    bot_level: int  # base level at the time
    duration_minutes: float
    base_exp_gained: float
    job_exp_gained: float
    zeny_gained: float
    death_count: int = 0
    party_size: int = 1
    party_composition: dict[str, int] = field(default_factory=dict)
    # e.g. {"Knight": 1, "Priest": 1, "Wizard": 1}
    success: bool = True
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)


class FleetLearningSystem:
    """Fleet-level learning: zone optimization, party composition, mined-out detection.

    Tracks experience rates per zone, computes ELO-like scores,
    recommends zones based on bot level, detects mined-out zones,
    and learns optimal party compositions.

    Thread-safe. Persists to SQLite for cross-session durability.
    """

    # ELO constants
    ELO_K = 32         # Sensitivity of ELO adjustments
    ELO_INITIAL = 1500  # Starting ELO for a new zone

    # Mined-out detection
    MINED_OUT_WINDOW = 10   # Check last N outcomes for a zone
    MINED_OUT_THRESHOLD = 0.5  # If exp rate dropped by 50%+, consider mined out

    def __init__(self, db_path: str | Path | None = None, max_outcomes_per_zone: int = 1000):
        self._lock = threading.RLock()
        self._db_path = str(db_path) if db_path else None
        self._max_outcomes_per_zone = max_outcomes_per_zone

        # In-memory caches (synced from DB on init)
        self._outcomes: list[ZoneOutcome] = []
        self._zone_stats: dict[str, dict[str, Any]] = {}  # map_name -> stats dict
        self._party_compositions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        # map_name -> list of composition records

        if self._db_path:
            self._init_db()
            self._load_from_db()

    # ── SQLite persistence ──────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            db = sqlite3.connect(self._db_path, timeout=5.0)
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS zone_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id TEXT NOT NULL,
                map_name TEXT NOT NULL,
                bot_level INTEGER NOT NULL,
                duration_minutes REAL NOT NULL,
                base_exp_gained REAL NOT NULL DEFAULT 0,
                job_exp_gained REAL NOT NULL DEFAULT 0,
                zeny_gained REAL NOT NULL DEFAULT 0,
                death_count INTEGER NOT NULL DEFAULT 0,
                party_size INTEGER NOT NULL DEFAULT 1,
                party_composition TEXT NOT NULL DEFAULT '{}',
                success INTEGER NOT NULL DEFAULT 1,
                timestamp REAL NOT NULL,
                details TEXT NOT NULL DEFAULT '{}'
            )""")
            db.execute("""CREATE INDEX IF NOT EXISTS idx_zo_map
                ON zone_outcomes(map_name, timestamp DESC)""")
            db.execute("""CREATE INDEX IF NOT EXISTS idx_zo_bot
                ON zone_outcomes(bot_id, timestamp DESC)""")
            db.execute("""CREATE TABLE IF NOT EXISTS zone_elo (
                map_name TEXT PRIMARY KEY,
                elo_score REAL NOT NULL DEFAULT 1500,
                total_outcomes INTEGER NOT NULL DEFAULT 0,
                avg_base_exp_rate REAL NOT NULL DEFAULT 0,
                avg_job_exp_rate REAL NOT NULL DEFAULT 0,
                avg_zeny_rate REAL NOT NULL DEFAULT 0,
                avg_party_size REAL NOT NULL DEFAULT 1,
                success_rate REAL NOT NULL DEFAULT 0.5,
                mined_out INTEGER NOT NULL DEFAULT 0,
                last_updated REAL NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS party_compositions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                map_name TEXT NOT NULL,
                composition_json TEXT NOT NULL,
                party_size INTEGER NOT NULL,
                total_runs INTEGER NOT NULL DEFAULT 0,
                successful_runs INTEGER NOT NULL DEFAULT 0,
                avg_base_exp_rate REAL NOT NULL DEFAULT 0,
                avg_job_exp_rate REAL NOT NULL DEFAULT 0,
                avg_zeny_rate REAL NOT NULL DEFAULT 0,
                success_rate REAL NOT NULL DEFAULT 0.5,
                score REAL NOT NULL DEFAULT 0,
                last_used REAL NOT NULL
            )""")
            db.execute("""CREATE INDEX IF NOT EXISTS idx_pc_map_score
                ON party_compositions(map_name, score DESC)""")
            db.commit()
            db.close()
        except Exception as e:
            logger.warning("FleetLearningSystem: DB init failed: %s", e)

    def _save_outcome(self, outcome: ZoneOutcome) -> None:
        if not self._db_path:
            return
        try:
            db = sqlite3.connect(self._db_path, timeout=3.0)
            db.execute(
                """INSERT INTO zone_outcomes
                   (bot_id, map_name, bot_level, duration_minutes,
                    base_exp_gained, job_exp_gained, zeny_gained,
                    death_count, party_size, party_composition,
                    success, timestamp, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (outcome.bot_id, outcome.map_name, outcome.bot_level,
                 outcome.duration_minutes,
                 outcome.base_exp_gained, outcome.job_exp_gained, outcome.zeny_gained,
                 outcome.death_count, outcome.party_size,
                 json.dumps(outcome.party_composition),
                 int(outcome.success), outcome.timestamp,
                 json.dumps(outcome.details)),
            )
            db.commit()
            db.close()
        except Exception as e:
            logger.debug("FleetLearningSystem: save_outcome failed: %s", e)

    def _load_from_db(self) -> None:
        if not self._db_path:
            return
        try:
            db = sqlite3.connect(self._db_path, timeout=5.0)
            # Load recent outcomes
            cursor = db.execute(
                "SELECT * FROM zone_outcomes ORDER BY timestamp DESC LIMIT 5000"
            )
            for row in cursor.fetchall():
                outcome = ZoneOutcome(
                    bot_id=row[1], map_name=row[2], bot_level=row[3],
                    duration_minutes=row[4],
                    base_exp_gained=row[5], job_exp_gained=row[6],
                    zeny_gained=row[7], death_count=row[8],
                    party_size=row[9],
                    party_composition=json.loads(row[10]) if row[10] else {},
                    success=bool(row[11]),
                    timestamp=row[12],
                    details=json.loads(row[13]) if row[13] else {},
                )
                self._outcomes.append(outcome)

            # Load zone ELO scores
            cursor = db.execute("SELECT * FROM zone_elo")
            for row in cursor.fetchall():
                self._zone_stats[row[0]] = {
                    "elo_score": row[1],
                    "total_outcomes": row[2],
                    "avg_base_exp_rate": row[3],
                    "avg_job_exp_rate": row[4],
                    "avg_zeny_rate": row[5],
                    "avg_party_size": row[6],
                    "success_rate": row[7],
                    "mined_out": bool(row[8]),
                    "last_updated": row[9],
                }

            # Load party compositions
            cursor = db.execute(
                "SELECT * FROM party_compositions ORDER BY score DESC"
            )
            for row in cursor.fetchall():
                self._party_compositions[row[0]].append({
                    "map_name": row[0],
                    "composition": json.loads(row[1]),
                    "party_size": row[2],
                    "total_runs": row[3],
                    "successful_runs": row[4],
                    "avg_base_exp_rate": row[5],
                    "avg_job_exp_rate": row[6],
                    "avg_zeny_rate": row[7],
                    "success_rate": row[8],
                    "score": row[9],
                    "last_used": row[10],
                })

            db.close()
            logger.info(
                "FleetLearningSystem: loaded %d outcomes, %d zones, %d compositions",
                len(self._outcomes), len(self._zone_stats),
                sum(len(v) for v in self._party_compositions.values()),
            )
        except Exception as e:
            logger.warning("FleetLearningSystem: load failed: %s", e)

    # ── Public API ──────────────────────────────────────────────────────

    def record_outcome(self, outcome: ZoneOutcome) -> None:
        """Record a zone outcome and update all derived metrics."""
        with self._lock:
            self._outcomes.append(outcome)
            map_name = outcome.map_name

            # Prune in-memory list
            zone_outcomes = [o for o in self._outcomes if o.map_name == map_name]
            if len(zone_outcomes) > self._max_outcomes_per_zone:
                excess = len(zone_outcomes) - self._max_outcomes_per_zone
                self._outcomes = [o for o in self._outcomes
                                  if o.map_name != map_name or
                                  o not in zone_outcomes[:excess]]

            # Update zone ELO
            self._update_zone_elo(outcome)

            # Update party composition knowledge
            if outcome.party_composition:
                self._update_party_composition(outcome)

        # Persist outside lock
        self._save_outcome(outcome)
        self._persist_zone_stats(map_name)
        if outcome.party_composition:
            self._persist_party_composition(outcome.map_name)

    def get_zone_score(self, map_name: str) -> dict[str, Any]:
        """Get comprehensive scoring for a zone."""
        with self._lock:
            stats = self._zone_stats.get(map_name)
            if stats is None:
                return {
                    "map_name": map_name,
                    "elo_score": self.ELO_INITIAL,
                    "confidence": 0.0,
                    "total_outcomes": 0,
                    "avg_base_exp_rate": 0.0,
                    "avg_job_exp_rate": 0.0,
                    "avg_zeny_rate": 0.0,
                    "avg_party_size": 1.0,
                    "success_rate": 0.5,
                    "mined_out": False,
                    "mined_out_confidence": 0.0,
                }

            return {
                "map_name": map_name,
                "elo_score": stats["elo_score"],
                "confidence": min(1.0, stats["total_outcomes"] / 50.0),
                "total_outcomes": stats["total_outcomes"],
                "avg_base_exp_rate": stats["avg_base_exp_rate"],
                "avg_job_exp_rate": stats["avg_job_exp_rate"],
                "avg_zeny_rate": stats["avg_zeny_rate"],
                "avg_party_size": stats["avg_party_size"],
                "success_rate": stats["success_rate"],
                "mined_out": stats.get("mined_out", False),
                "mined_out_confidence": stats.get("mined_out_confidence", 0.0),
            }

    def get_best_zone(
        self, bot_level: int, min_samples: int = 3,
        exclude_mined_out: bool = True, top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Recommend the best zone(s) based on ELO + bot level.

        Zones are scored by a weighted formula:
        - ELO score (zone difficulty/reward)
        - Base exp rate (higher = better)
        - Level appropriateness (not too high/low)
        - Not mined out

        Returns top_n zones sorted by composite score.
        """
        with self._lock:
            candidates: list[dict[str, Any]] = []
            for map_name, stats in self._zone_stats.items():
                if stats["total_outcomes"] < min_samples:
                    continue
                if exclude_mined_out and stats.get("mined_out", False):
                    continue

                # Level appropriateness: zone avg level vs bot level
                elo = stats["elo_score"]
                # Normalize ELO to a 0-1 range (typical range 1000-2000)
                elo_score_norm = max(0.0, min(1.0, (elo - 1000.0) / 1000.0))

                # Exp rate normalized (cap at 10M/hour as 1.0)
                base_rate_norm = min(1.0, stats["avg_base_exp_rate"] / 10_000_000.0)

                # Level match: zones with ELO ~ bot_level*10 are appropriate
                # (elite = level 150 -> ELO ~ 1500, so bot_level*10 roughly maps)
                target_elo = bot_level * 10
                level_match = max(0.0, 1.0 - abs(elo - target_elo) / 500.0)

                # Success rate
                sr = stats["success_rate"]

                # Composite score
                composite = (
                    elo_score_norm * 0.25 +
                    base_rate_norm * 0.30 +
                    level_match * 0.25 +
                    sr * 0.20
                )

                candidates.append({
                    "map_name": map_name,
                    "composite_score": composite,
                    "elo_score": elo,
                    "avg_base_exp_rate": stats["avg_base_exp_rate"],
                    "avg_job_exp_rate": stats["avg_job_exp_rate"],
                    "avg_zeny_rate": stats["avg_zeny_rate"],
                    "success_rate": sr,
                    "confidence": min(1.0, stats["total_outcomes"] / 50.0),
                    "mined_out": stats.get("mined_out", False),
                })

            candidates.sort(key=lambda x: x["composite_score"], reverse=True)
            return candidates[:top_n]

    def get_party_composition(self, map_name: str) -> dict[str, Any]:
        """Get the best known party composition for a zone.

        Returns the composition with the highest score, or a default recommendation.
        """
        with self._lock:
            comps = self._party_compositions.get(map_name, [])
            if not comps:
                return {
                    "map_name": map_name,
                    "composition": {},
                    "party_size": 1,
                    "success_rate": 0.5,
                    "score": 0.0,
                    "total_runs": 0,
                    "confidence": 0.0,
                }

            best = max(comps, key=lambda c: c["score"])
            return {
                "map_name": map_name,
                "composition": best["composition"],
                "party_size": best["party_size"],
                "success_rate": best["success_rate"],
                "score": best["score"],
                "total_runs": best["total_runs"],
                "confidence": min(1.0, best["total_runs"] / 20.0),
            }

    # ── Internal: Zone ELO ──────────────────────────────────────────────

    def _update_zone_elo(self, outcome: ZoneOutcome) -> None:
        """Update ELO score for a zone based on outcome."""
        map_name = outcome.map_name
        if map_name not in self._zone_stats:
            self._zone_stats[map_name] = {
                "elo_score": self.ELO_INITIAL,
                "total_outcomes": 0,
                "avg_base_exp_rate": 0.0,
                "avg_job_exp_rate": 0.0,
                "avg_zeny_rate": 0.0,
                "avg_party_size": 1.0,
                "success_rate": 0.5,
                "mined_out": False,
                "mined_out_confidence": 0.0,
                "last_updated": outcome.timestamp,
                # Track recent rates for mined-out detection
                "_recent_base_rates": [],
            }

        stats = self._zone_stats[map_name]
        n = stats["total_outcomes"]

        # Update running averages
        stats["avg_base_exp_rate"] = (
            (stats["avg_base_exp_rate"] * n + outcome.base_exp_gained / max(0.001, outcome.duration_minutes / 60.0))
            / (n + 1)
        ) if outcome.duration_minutes > 0 else 0.0
        stats["avg_job_exp_rate"] = (
            (stats["avg_job_exp_rate"] * n + outcome.job_exp_gained / max(0.001, outcome.duration_minutes / 60.0))
            / (n + 1)
        ) if outcome.duration_minutes > 0 else 0.0
        stats["avg_zeny_rate"] = (
            (stats["avg_zeny_rate"] * n + outcome.zeny_gained / max(0.001, outcome.duration_minutes / 60.0))
            / (n + 1)
        )
        stats["avg_party_size"] = (
            (stats["avg_party_size"] * n + outcome.party_size) / (n + 1)
        )
        stats["total_outcomes"] = n + 1

        # Update success rate
        successes = sum(1 for o in self._outcomes if o.map_name == map_name and o.success)
        total = sum(1 for o in self._outcomes if o.map_name == map_name)
        stats["success_rate"] = successes / max(1, total)

        # ELO update: treat zone as "player" and outcome as win/loss
        # Expected score based on current ELO
        expected = 1.0 / (1.0 + math.pow(10, (1500 - stats["elo_score"]) / 400.0))
        actual = 1.0 if outcome.success else 0.0

        # Death penalty: adjust actual score downward
        if outcome.death_count > 0:
            death_penalty = min(0.5, outcome.death_count * 0.1)
            actual = max(0.0, actual - death_penalty)

        # Bonus for high exp gain
        if outcome.base_exp_gained > 0 and outcome.duration_minutes > 0:
            exp_per_hour = outcome.base_exp_gained / max(0.001, outcome.duration_minutes / 60.0)
            if exp_per_hour > 1_000_000:  # >1M exp/hr = bonus
                actual = min(1.0, actual + 0.1)

        delta = self.ELO_K * (actual - expected)
        stats["elo_score"] = max(500.0, min(2500.0, stats["elo_score"] + delta))
        stats["last_updated"] = outcome.timestamp

        # Detect mined out
        self._check_mined_out(map_name, outcome)

    def _check_mined_out(self, map_name: str, outcome: ZoneOutcome) -> None:
        """Detect if a zone is 'mined out' (exp rate dropping significantly)."""
        stats = self._zone_stats[map_name]

        # Track recent base EXP rates
        exp_rate = 0.0
        if outcome.duration_minutes > 0:
            exp_rate = outcome.base_exp_gained / max(0.001, outcome.duration_minutes / 60.0)

        recent_rates = stats.setdefault("_recent_base_rates", [])
        recent_rates.append(exp_rate)
        # Keep only the most recent N
        if len(recent_rates) > self.MINED_OUT_WINDOW:
            recent_rates.pop(0)

        # Need enough data points
        if len(recent_rates) < self.MINED_OUT_WINDOW:
            return

        # Compare first half vs second half
        half = self.MINED_OUT_WINDOW // 2
        early_avg = sum(recent_rates[:half]) / half
        late_avg = sum(recent_rates[half:]) / half

        if early_avg > 0 and late_avg > 0:
            drop_ratio = late_avg / early_avg
            # If the latest rate dropped significantly
            if drop_ratio < (1.0 - self.MINED_OUT_THRESHOLD):
                stats["mined_out"] = True
                # Confidence increases with more confirming data
                confirmations = 0
                for i in range(half, len(recent_rates)):
                    if recent_rates[i] < early_avg * (1.0 - self.MINED_OUT_THRESHOLD):
                        confirmations += 1
                stats["mined_out_confidence"] = confirmations / half
            else:
                # Gradually reduce mined_out flag if rates recover
                if stats["mined_out"] and drop_ratio > 0.8:
                    stats["mined_out_confidence"] = max(
                        0.0, stats.get("mined_out_confidence", 0.0) - 0.1
                    )
                    if stats["mined_out_confidence"] <= 0.0:
                        stats["mined_out"] = False

    # ── Internal: Party composition ─────────────────────────────────────

    def _update_party_composition(self, outcome: ZoneOutcome) -> None:
        """Update party composition knowledge for a zone."""
        map_name = outcome.map_name
        comps = self._party_compositions[map_name]
        comp_json = json.dumps(outcome.party_composition, sort_keys=True)

        # Find existing entry for this composition
        existing = None
        for c in comps:
            if json.dumps(c["composition"], sort_keys=True) == comp_json:
                existing = c
                break

        exp_rate = 0.0
        if outcome.duration_minutes > 0:
            exp_rate = outcome.base_exp_gained / max(0.001, outcome.duration_minutes / 60.0)

        if existing:
            n = existing["total_runs"]
            existing["total_runs"] = n + 1
            existing["successful_runs"] += 1 if outcome.success else 0
            existing["avg_base_exp_rate"] = (
                existing["avg_base_exp_rate"] * n + exp_rate
            ) / (n + 1)
            existing["last_used"] = outcome.timestamp
            sr = existing["successful_runs"] / existing["total_runs"]
            existing["success_rate"] = sr
            # Score: success_rate * 0.5 + normalized_exp_rate * 0.5
            norm_rate = min(1.0, existing["avg_base_exp_rate"] / 10_000_000.0)
            existing["score"] = sr * 0.5 + norm_rate * 0.5
        else:
            new_entry = {
                "map_name": map_name,
                "composition": outcome.party_composition,
                "party_size": outcome.party_size,
                "total_runs": 1,
                "successful_runs": 1 if outcome.success else 0,
                "avg_base_exp_rate": exp_rate,
                "avg_job_exp_rate": 0.0,
                "avg_zeny_rate": outcome.zeny_gained / max(0.001, outcome.duration_minutes / 60.0) if outcome.duration_minutes > 0 else 0.0,
                "success_rate": 1.0 if outcome.success else 0.0,
                "score": 0.5 + (min(1.0, exp_rate / 10_000_000.0)) * 0.5 if outcome.success else 0.0,
                "last_used": outcome.timestamp,
            }
            comps.append(new_entry)

        # Sort compositions by score descending
        comps.sort(key=lambda c: c["score"], reverse=True)

    # ── Persistence helpers ─────────────────────────────────────────────

    def _persist_zone_stats(self, map_name: str) -> None:
        if not self._db_path:
            return
        with self._lock:
            stats = self._zone_stats.get(map_name)
            if not stats:
                return
        try:
            db = sqlite3.connect(self._db_path, timeout=3.0)
            db.execute(
                """INSERT OR REPLACE INTO zone_elo
                   (map_name, elo_score, total_outcomes,
                    avg_base_exp_rate, avg_job_exp_rate, avg_zeny_rate,
                    avg_party_size, success_rate, mined_out, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (map_name, stats["elo_score"], stats["total_outcomes"],
                 stats["avg_base_exp_rate"], stats["avg_job_exp_rate"],
                 stats["avg_zeny_rate"], stats["avg_party_size"],
                 stats["success_rate"], int(stats.get("mined_out", False)),
                 stats["last_updated"]),
            )
            db.commit()
            db.close()
        except Exception as e:
            logger.debug("FleetLearningSystem: persist_zone_stats failed: %s", e)

    def _persist_party_composition(self, map_name: str) -> None:
        if not self._db_path:
            return
        with self._lock:
            comps = list(self._party_compositions.get(map_name, []))
        try:
            db = sqlite3.connect(self._db_path, timeout=3.0)
            # Delete old entries for this map, insert current ones
            db.execute(
                "DELETE FROM party_compositions WHERE map_name = ?", (map_name,)
            )
            for c in comps:
                db.execute(
                    """INSERT INTO party_compositions
                       (map_name, composition_json, party_size,
                        total_runs, successful_runs,
                        avg_base_exp_rate, avg_job_exp_rate, avg_zeny_rate,
                        success_rate, score, last_used)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (map_name, json.dumps(c["composition"]), c["party_size"],
                     c["total_runs"], c["successful_runs"],
                     c["avg_base_exp_rate"], c.get("avg_job_exp_rate", 0.0),
                     c.get("avg_zeny_rate", 0.0),
                     c["success_rate"], c["score"], c["last_used"]),
                )
            db.commit()
            db.close()
        except Exception as e:
            logger.debug("FleetLearningSystem: persist_party_comp failed: %s", e)

    # ── Utility ─────────────────────────────────────────────────────────

    def get_all_zones(self) -> list[dict[str, Any]]:
        """Return all known zones with their scores."""
        with self._lock:
            return [
                self.get_zone_score(map_name)
                for map_name in self._zone_stats
            ]

    def get_mined_out_zones(self) -> list[dict[str, Any]]:
        """Return zones currently flagged as mined out."""
        with self._lock:
            return [
                {
                    "map_name": map_name,
                    "confidence": stats.get("mined_out_confidence", 0.0),
                    "avg_base_exp_rate": stats["avg_base_exp_rate"],
                    "elo_score": stats["elo_score"],
                }
                for map_name, stats in self._zone_stats.items()
                if stats.get("mined_out", False)
            ]

    def reset_zone(self, map_name: str) -> None:
        """Reset zone stats (e.g., after a game update repopulates the zone)."""
        with self._lock:
            if map_name in self._zone_stats:
                del self._zone_stats[map_name]
            self._outcomes = [o for o in self._outcomes if o.map_name != map_name]
            self._party_compositions.pop(map_name, None)

        if self._db_path:
            try:
                db = sqlite3.connect(self._db_path, timeout=3.0)
                db.execute("DELETE FROM zone_elo WHERE map_name = ?", (map_name,))
                db.execute("DELETE FROM zone_outcomes WHERE map_name = ?", (map_name,))
                db.execute(
                    "DELETE FROM party_compositions WHERE map_name = ?", (map_name,)
                )
                db.commit()
                db.close()
            except Exception:
                pass

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics for the fleet learning system."""
        with self._lock:
            mined_out_count = sum(
                1 for s in self._zone_stats.values() if s.get("mined_out", False)
            )
            return {
                "total_outcomes": len(self._outcomes),
                "known_zones": len(self._zone_stats),
                "known_compositions": sum(len(v) for v in self._party_compositions.values()),
                "mined_out_zones": mined_out_count,
                "unique_bots": len(set(o.bot_id for o in self._outcomes)),
                "unique_maps": len(set(o.map_name for o in self._outcomes)),
            }


# ── Backward-compatible alias ────────────────────────────────────────
SelfLearningSystem = FleetLearningSystem

# ── Backward-compatible aliases ────────────────────────────────────────
PerformanceRecord = ZoneOutcome
SkillEffectiveness = ZoneOutcome  # Legacy: generalized outcome type
ItemEffectiveness = ZoneOutcome   # Legacy: generalized outcome type
