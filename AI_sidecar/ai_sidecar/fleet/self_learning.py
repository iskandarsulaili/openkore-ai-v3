"""Self-learning system — cross-bot experience, strategy adaptation, skill & item optimization."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PerformanceRecord:
    bot_id: str
    context_type: str
    map_name: str
    monster_name: str
    role: str
    action_taken: str
    success: bool
    reward: float
    duration_ms: float
    hp_consumed: float
    sp_consumed: float
    items_used: list[str]
    skills_used: list[str]
    timestamp: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillEffectiveness:
    skill_name: str
    context_type: str
    bot_class: str
    map_name: str
    monster_name: str
    role: str
    attempts: int = 0
    successes: int = 0
    avg_damage: float = 0.0
    avg_healing: float = 0.0
    avg_sp_cost: float = 0.0
    dps: float = 0.0
    score: float = 0.0


@dataclass
class ItemEffectiveness:
    item_name: str
    context_type: str
    map_name: str
    role: str
    attempts: int = 0
    successes: int = 0
    avg_hp_restored: float = 0.0
    avg_sp_restored: float = 0.0
    score: float = 0.0


class SelfLearningSystem:
    """Cross-bot experience & strategy optimization system.

    Tracks performance per role/strategy/map/monster across all bots.
    Learns optimal skill rotations and item usage patterns.
    Persists to SQLite for cross-session durability.
    """

    def __init__(self, db_path: str | Path | None = None, max_records: int = 100000):
        self._lock = threading.RLock()
        self._max_records = max_records
        self._records: list[PerformanceRecord] = []
        self._skill_effectiveness: dict[str, SkillEffectiveness] = {}
        self._item_effectiveness: dict[str, ItemEffectiveness] = {}
        self._strategy_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._db_path = str(db_path) if db_path else None
        if self._db_path:
            self._init_db()

    # ── SQLite persistence ──────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            db = sqlite3.connect(self._db_path, timeout=5.0)
            db.execute("""CREATE TABLE IF NOT EXISTS self_learning_records (
                bot_id TEXT, context_type TEXT, map_name TEXT, monster_name TEXT,
                role TEXT, action_taken TEXT, success INTEGER, reward REAL,
                duration_ms REAL, hp_consumed REAL, sp_consumed REAL,
                items_used TEXT, skills_used TEXT, timestamp REAL, details TEXT
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS self_learning_strategies (
                context_type TEXT, map_name TEXT, role TEXT, action_taken TEXT,
                attempts INTEGER, successes INTEGER, avg_reward REAL, score REAL,
                PRIMARY KEY (context_type, map_name, role, action_taken)
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS self_learning_skills (
                skill_name TEXT, context_type TEXT, bot_class TEXT, map_name TEXT,
                monster_name TEXT, role TEXT, attempts INTEGER, successes INTEGER,
                avg_damage REAL, avg_healing REAL, avg_sp_cost REAL, dps REAL, score REAL,
                PRIMARY KEY (skill_name, context_type, bot_class, map_name, monster_name)
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS self_learning_items (
                item_name TEXT, context_type TEXT, map_name TEXT, role TEXT,
                attempts INTEGER, successes INTEGER, avg_hp_restored REAL,
                avg_sp_restored REAL, score REAL,
                PRIMARY KEY (item_name, context_type, map_name, role)
            )""")
            db.commit()
            db.close()
        except Exception as e:
            logger.warning("SelfLearningSystem: DB init failed: %s", e)

    def _save_record(self, record: PerformanceRecord) -> None:
        if not self._db_path:
            return
        try:
            db = sqlite3.connect(self._db_path, timeout=3.0)
            db.execute(
                "INSERT INTO self_learning_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record.bot_id, record.context_type, record.map_name, record.monster_name,
                 record.role, record.action_taken, int(record.success), record.reward,
                 record.duration_ms, record.hp_consumed, record.sp_consumed,
                 json.dumps(record.items_used), json.dumps(record.skills_used),
                 record.timestamp, json.dumps(record.details)),
            )
            db.commit()
            db.close()
        except Exception as e:
            logger.debug("SelfLearningSystem: save failed: %s", e)

    def _load_records(self, limit: int = 5000) -> list[dict[str, Any]]:
        if not self._db_path:
            return []
        try:
            db = sqlite3.connect(self._db_path, timeout=3.0)
            cursor = db.execute(
                "SELECT * FROM self_learning_records ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = []
            for row in cursor.fetchall():
                rows.append({
                    "bot_id": row[0], "context_type": row[1], "map_name": row[2],
                    "monster_name": row[3], "role": row[4], "action_taken": row[5],
                    "success": bool(row[6]), "reward": row[7], "duration_ms": row[8],
                    "hp_consumed": row[9], "sp_consumed": row[10],
                    "items_used": json.loads(row[11]) if row[11] else [],
                    "skills_used": json.loads(row[12]) if row[12] else [],
                    "timestamp": row[13], "details": json.loads(row[14]) if row[14] else {},
                })
            db.close()
            return rows
        except Exception as e:
            logger.debug("SelfLearningSystem: load failed: %s", e)
            return []

    # ── Record experience ───────────────────────────────────────────────

    def record(self, record: PerformanceRecord) -> None:
        """Record a performance datapoint."""
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]

            # Update strategy score
            key = (record.context_type, record.map_name, record.role)
            strat_key = f"{record.context_type}:{record.map_name}:{record.role}:{record.action_taken}"
            stats = self._strategy_scores[strat_key]
            stats["attempts"] = stats.get("attempts", 0) + 1
            stats["successes"] = stats.get("successes", 0) + (1 if record.success else 0)
            avg = stats.get("avg_reward", 0.0)
            n = stats["attempts"]
            stats["avg_reward"] = (avg * (n - 1) + record.reward) / n
            sr = stats["successes"] / stats["attempts"]
            stats["score"] = sr * 0.6 + (stats["avg_reward"] / max(1.0, stats["avg_reward"] + 100)) * 0.4

            # Update skill effectiveness
            for skill in record.skills_used:
                skey = f"{skill}:{record.context_type}:{record.bot_id}:{record.map_name}:{record.monster_name}"
                se = self._skill_effectiveness.setdefault(
                    skey, SkillEffectiveness(
                        skill_name=skill, context_type=record.context_type,
                        bot_class=record.bot_id, map_name=record.map_name,
                        monster_name=record.monster_name, role=record.role))
                se.attempts += 1
                if record.success:
                    se.successes += 1
                avg_dmg = record.details.get("damage_dealt", 0.0)
                avg_heal = record.details.get("healing_done", 0.0)
                sp_cost = record.details.get("sp_cost", 0.0)
                if se.attempts > 0:
                    se.avg_damage = (se.avg_damage * (se.attempts - 1) + avg_dmg) / se.attempts
                    se.avg_healing = (se.avg_healing * (se.attempts - 1) + avg_heal) / se.attempts
                    se.avg_sp_cost = (se.avg_sp_cost * (se.attempts - 1) + sp_cost) / se.attempts
                dur_s = max(0.001, record.duration_ms / 1000.0)
                se.dps = (se.avg_damage + se.avg_healing) / dur_s
                sr = se.successes / max(1, se.attempts)
                se.score = sr * 0.3 + (se.dps / max(1.0, se.dps + 1000)) * 0.4 + (1.0 - min(1.0, se.avg_sp_cost / 100.0)) * 0.3

            # Update item effectiveness
            for item in record.items_used:
                ikey = f"{item}:{record.context_type}:{record.map_name}:{record.role}"
                ie = self._item_effectiveness.setdefault(
                    ikey, ItemEffectiveness(
                        item_name=item, context_type=record.context_type,
                        map_name=record.map_name, role=record.role))
                ie.attempts += 1
                if record.success:
                    ie.successes += 1
                hp_r = record.details.get("hp_restored", 0.0)
                sp_r = record.details.get("sp_restored", 0.0)
                if ie.attempts > 0:
                    ie.avg_hp_restored = (ie.avg_hp_restored * (ie.attempts - 1) + hp_r) / ie.attempts
                    ie.avg_sp_restored = (ie.avg_sp_restored * (ie.attempts - 1) + sp_r) / ie.attempts
                sr = ie.successes / max(1, ie.attempts)
                ie.score = sr * 0.5 + (ie.avg_hp_restored / max(1.0, ie.avg_hp_restored + 500)) * 0.3 + (
                    ie.avg_sp_restored / max(1.0, ie.avg_sp_restored + 200)) * 0.2

        self._save_record(record)

    # ── Query learned knowledge ─────────────────────────────────────────

    def best_strategy(self, context_type: str, map_name: str = "", role: str = "",
                      min_samples: int = 3) -> tuple[str, float]:
        """Return the best (action, score) for a given context."""
        with self._lock:
            best_action, best_score = "", 0.0
            prefix = f"{context_type}:{map_name}:{role}:"
            for key, stats in self._strategy_scores.items():
                if key.startswith(prefix):
                    score = stats.get("score", 0.0)
                    attempts = stats.get("attempts", 0)
                    if attempts >= min_samples and score > best_score:
                        best_score = score
                        best_action = key.split(":")[-1]
            return best_action, best_score

    def best_skill_rotation(self, context_type: str, bot_class: str = "",
                            map_name: str = "", monster_name: str = "",
                            role: str = "", limit: int = 6) -> list[dict[str, Any]]:
        """Return the best skills ranked by effectiveness."""
        with self._lock:
            candidates: list[SkillEffectiveness] = []
            for se in self._skill_effectiveness.values():
                if se.context_type != context_type:
                    continue
                if bot_class and se.bot_class != bot_class:
                    continue
                if monster_name and se.monster_name != monster_name:
                    continue
                if role and se.role != role:
                    continue
                candidates.append(se)
            candidates.sort(key=lambda x: x.score, reverse=True)
            return [
                {"skill": c.skill_name, "score": c.score, "avg_damage": c.avg_damage,
                 "avg_healing": c.avg_healing, "avg_sp_cost": c.avg_sp_cost,
                 "dps": c.dps, "attempts": c.attempts, "success_rate": c.successes / max(1, c.attempts)}
                for c in candidates[:limit]
            ]

    def best_items(self, context_type: str, map_name: str = "",
                   role: str = "", limit: int = 10) -> list[dict[str, Any]]:
        """Return the best items ranked by effectiveness."""
        with self._lock:
            candidates: list[ItemEffectiveness] = []
            for ie in self._item_effectiveness.values():
                if ie.context_type != context_type:
                    continue
                if map_name and ie.map_name != map_name:
                    continue
                if role and ie.role != role:
                    continue
                candidates.append(ie)
            candidates.sort(key=lambda x: x.score, reverse=True)
            return [
                {"item": c.item_name, "score": c.score, "avg_hp_restored": c.avg_hp_restored,
                 "avg_sp_restored": c.avg_sp_restored, "attempts": c.attempts,
                 "success_rate": c.successes / max(1, c.attempts)}
                for c in candidates[:limit]
            ]

    def strategy_adaptation(self, context_type: str, map_name: str = "",
                            role: str = "") -> dict[str, Any]:
        """Produce a strategy adaptation recommendation based on past performance."""
        best_action, best_score = self.best_strategy(context_type, map_name, role)
        current_avg_score = 0.0
        with self._lock:
            scores = [
                s.get("score", 0.0)
                for k, s in self._strategy_scores.items()
                if k.startswith(f"{context_type}:{map_name}:{role}:")
            ]
            current_avg_score = sum(scores) / max(1, len(scores))
        return {
            "context_type": context_type,
            "map_name": map_name,
            "role": role,
            "best_action": best_action,
            "best_score": best_score,
            "current_avg_score": current_avg_score,
            "improvement_potential": best_score - current_avg_score,
            "should_adapt": best_score > current_avg_score + 0.1,
        }

    def cross_bot_recommendation(self, bot_id: str, context_type: str) -> dict[str, Any]:
        """Get recommendations from other bots' experience for a given context."""
        with self._lock:
            # Load other bots' records from DB
            other_records = [r for r in self._records if r.bot_id != bot_id and r.context_type == context_type]
            if not other_records:
                return {
                    "has_cross_bot_data": False,
                    "best_action": "",
                    "confidence": 0.0,
                    "peer_count": 0,
                }

            action_stats: dict[str, list[bool]] = {}
            for r in other_records:
                if r.action_taken:
                    action_stats.setdefault(r.action_taken, []).append(r.success)

            best_action, best_rate = "", 0.0
            for action, outcomes in action_stats.items():
                rate = sum(1 for s in outcomes if s) / len(outcomes)
                if rate > best_rate and len(outcomes) >= 3:
                    best_rate = rate
                    best_action = action

            return {
                "has_cross_bot_data": True,
                "best_action": best_action,
                "success_rate": best_rate,
                "peer_count": len(other_records),
                "unique_bots": len(set(r.bot_id for r in other_records)),
            }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_records": len(self._records),
                "unique_strategies": len(self._strategy_scores),
                "unique_skills": len(self._skill_effectiveness),
                "unique_items": len(self._item_effectiveness),
                "by_context": dict(sorted(
                    (k, len([r for r in self._records if r.context_type == k]))
                    for k in set(r.context_type for r in self._records)
                )),
            }
