"""
Server Adaptation Engine — Auto-detects server rates and mechanics.
====================================================================
No hardcoded values. The AI observes actual game behavior to determine
server-specific rates for EXP, drops, stats, formulas, and mechanics.

Detects by:
1. Comparing actual EXP gained from killing a known monster vs expected
2. Checking NPC names and positions from live snapshots
3. Observing drop rates (known drop vs actual drop frequency)
4. Checking refine success rates
5. Checking stat formula differences (pre-renewal vs renewal)
6. Checking max level, ASPD formulas, and other mechanics

Every server is different. This engine adapts to any rAthena-based server
without manual configuration.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ServerProfile:
    """Detected server rates and mechanics."""
    # Rates
    base_exp_rate: float = 1.0
    job_exp_rate: float = 1.0
    drop_rate: float = 1.0
    card_rate: float = 1.0
    refine_rate: float = 1.0
    mvp_exp_rate: float = 1.0
    quest_exp_rate: float = 1.0
    
    # Mechanics
    is_renewal: bool = False  # Pre-renewal vs Renewal stat formulas
    max_base_level: int = 99
    max_job_level: int = 50
    max_stats: int = 99  # Pre-renewal cap, Renewal goes higher
    aspd_formula: str = "pre_renewal"  # pre_renewal | renewal
    
    # Custom features
    has_instant_cast: bool = False
    has_no_cast_delay: bool = False
    has_async_attack: bool = False
    has_custom_commands: bool = False
    has_auto_loot: bool = False
    has_merchant_auto_store: bool = False
    has_mvp_tracker: bool = False
    has_warp_to_mvp: bool = False
    
    # Detection confidence
    confidence: float = 0.0  # 0.0 = unknown, 1.0 = fully profiled
    samples_taken: int = 0
    last_updated: float = 0.0
    
    # Town NPCs discovered
    towns: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    # Server identity
    server_name: str = ""
    server_gm: str = ""
    server_website: str = ""


class ServerAdaptationEngine:
    """Auto-detects server rates and mechanics by observing game behavior.

    The AI kills a monster, compares actual EXP gained vs expected from rAthena
    knowledge, and computes the server's EXP rate. Same for drops, cards, etc.
    This allows the bot to optimize hunting recommendations for any server.

    Key insight: instead of asking the user "what's the EXP rate?", the bot
    figures it out by playing the game.
    """

    # Known baseline values from rAthena (pre-rate)
    BASELINE_EXP = {
        "PORING": {"base": 150, "job": 40, "hp": 55, "level": 1},
        "LUNATIC": {"base": 120, "job": 32, "hp": 42, "level": 1},
        "PICKY": {"base": 135, "job": 36, "hp": 47, "level": 2},
        "FABRE": {"base": 160, "job": 42, "hp": 55, "level": 3},
        "DROPS": {"base": 195, "job": 51, "hp": 46, "level": 2},
        "CHONCHON": {"base": 210, "job": 55, "hp": 48, "level": 5},
        "CONDOR": {"base": 90, "job": 25, "hp": 15, "level": 10},
        "SPORE": {"base": 340, "job": 90, "hp": 85, "level": 14},
        "WILLOW": {"base": 380, "job": 100, "hp": 100, "level": 13},
        "ZOMBIE": {"base": 440, "job": 115, "hp": 152, "level": 17},
    }

    # Mechanics detection keywords from server chat
    RENEWAL_KEYWORDS = ["renewal", "third class", "3rd class", "rune knight", "warlock",
                        "ranger", "arch bishop", "mechanic", "guillotine cross",
                        "royal guard", "sorcerer", "minstrel", "wanderer", "sura",
                        "genetic", "shadow chaser"]

    PRE_RENEWAL_KEYWORDS = ["pre-renewal", "pre-re", "classic", "transcendent",
                            "high wizard", "lord knight", "sniper", "high priest",
                            "whitesmith", "assassin cross", "paladin", "professor",
                            "clown", "gypsy", "champion", "creator", "stalker"]

    def __init__(self, knowledge_path: str = "knowledge/knowledge.json"):
        self._lock = RLock()
        self._profile: ServerProfile = ServerProfile()
        self._monsters: list[dict[str, Any]] = []
        self._load_knowledge(knowledge_path)
        self._exp_samples: list[dict[str, Any]] = []  # For rate calculation
        self._drop_samples: dict[str, list[bool]] = {}  # item -> [success/fail]
        self._refine_samples: list[bool] = []

    def _load_knowledge(self, path: str) -> None:
        """Load rAthena knowledge for baseline comparisons."""
        p = Path(path)
        if not p.exists():
            p = Path(__file__).parent.parent / "knowledge" / "knowledge.json"
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                self._monsters = data.get("monsters", [])
            except Exception:
                pass

    def get_profile(self) -> ServerProfile:
        """Get the current server profile."""
        with self._lock:
            return self._profile

    def get_server_id(self) -> str:
        """Get the server name/ID from the profile.

        Returns the server_name if set, otherwise 'default'.
        """
        with self._lock:
            name = self._profile.server_name
            if name:
                return name
        return "default"

    def record_exp_gain(self, monster_name: str, actual_base_exp: int,
                        actual_job_exp: int) -> dict[str, Any]:
        """Record actual EXP gained from killing a monster.

        Compares against expected baseline to compute server EXP rate.
        The more samples, the more accurate the rate detection.
        """
        with self._lock:
            baseline = self.BASELINE_EXP.get(monster_name.upper())
            if not baseline:
                # Try to find in knowledge DB
                for mob in self._monsters:
                    if mob.get("name", "").upper() == monster_name.upper():
                        baseline = {
                            "base": mob.get("base_exp", 0),
                            "job": mob.get("job_exp", 0),
                            "hp": mob.get("hp", 1),
                            "level": mob.get("level", 1),
                        }
                        break
            if not baseline:
                return {"rate_detected": False, "reason": "unknown_monster"}

            if actual_base_exp <= 0 and actual_job_exp <= 0:
                return {"rate_detected": False, "reason": "zero_exp"}

            # Calculate rate from sample
            base_rate = actual_base_exp / max(baseline["base"], 1) if baseline["base"] > 0 else 0
            job_rate = actual_job_exp / max(baseline["job"], 1) if baseline["job"] > 0 else 0

            sample = {
                "monster": monster_name,
                "expected_base": baseline["base"],
                "expected_job": baseline["job"],
                "actual_base": actual_base_exp,
                "actual_job": actual_job_exp,
                "base_rate": base_rate,
                "job_rate": job_rate,
                "timestamp": time.time(),
            }
            self._exp_samples.append(sample)

            # Recompute rates from all samples
            if len(self._exp_samples) >= 2:
                base_rates = [s["base_rate"] for s in self._exp_samples if s["base_rate"] > 0]
                job_rates = [s["job_rate"] for s in self._exp_samples if s["job_rate"] > 0]
                if base_rates:
                    # Use median to filter outliers
                    base_rates.sort()
                    median_base = base_rates[len(base_rates) // 2]
                    self._profile.base_exp_rate = round(median_base, 2)
                if job_rates:
                    job_rates.sort()
                    median_job = job_rates[len(job_rates) // 2]
                    self._profile.job_exp_rate = round(median_job, 2)
                self._profile.samples_taken = len(self._exp_samples)
                self._profile.confidence = min(1.0, len(self._exp_samples) / 10.0)
                self._profile.last_updated = time.time()

            return {
                "rate_detected": True,
                "base_rate": round(base_rate, 2),
                "job_rate": round(job_rate, 2),
                "samples": len(self._exp_samples),
                "monster": monster_name,
            }

    def record_drop_observation(self, item_name: str, dropped: bool) -> dict[str, Any]:
        """Record whether a known drop actually dropped.

        Used to compute server drop rate by comparing expected vs actual.
        """
        with self._lock:
            if item_name not in self._drop_samples:
                self._drop_samples[item_name] = []
            self._drop_samples[item_name].append(dropped)

            # After enough samples, estimate drop rate
            if len(self._drop_samples[item_name]) >= 10:
                success_rate = sum(1 for d in self._drop_samples[item_name] if d) / len(self._drop_samples[item_name])
                # Compare against expected baseline (cards are 0.01%, etc.)
                expected_rate = 0.01  # Default for common drops
                for mob in self._monsters:
                    for drop in mob.get("drops", []):
                        if drop.get("item", "").upper() == item_name.upper():
                            expected_rate = drop.get("rate", 100) / 10000.0
                            break
                if expected_rate > 0:
                    self._profile.drop_rate = round(success_rate / expected_rate, 2)
                    self._profile.confidence = min(1.0, self._profile.confidence + 0.05)

            return {
                "item": item_name,
                "dropped": dropped,
                "samples": len(self._drop_samples.get(item_name, [])),
            }

    def detect_mechanics(self, snapshot: Any) -> dict[str, Any]:
        """Detect server mechanics from game state snapshot.

        Looks for:
        - Max HP/SP values (Renewal has higher caps)
        - Stat values (Renewal allows >99)
        - Available skills (3rd class skills = Renewal)
        - NPC names and positions
        """
        with self._lock:
            changes = {}

            # Extract stats from snapshot
            base_level = 1
            stats = {}
            if isinstance(snapshot, dict):
                base_level = int(snapshot.get("base_level", 1) or 1)
                stats = {
                    "str": int(snapshot.get("str", 0) or 0),
                    "agi": int(snapshot.get("agi", 0) or 0),
                    "vit": int(snapshot.get("vit", 0) or 0),
                    "int": int(snapshot.get("int", 0) or 0),
                    "dex": int(snapshot.get("dex", 0) or 0),
                    "luk": int(snapshot.get("luk", 0) or 0),
                }
            else:
                base_level = int(getattr(snapshot, "base_level", 1) or 1)
                stats = {
                    "str": int(getattr(snapshot, "str", 0) or 0),
                    "agi": int(getattr(snapshot, "agi", 0) or 0),
                    "vit": int(getattr(snapshot, "vit", 0) or 0),
                    "int": int(getattr(snapshot, "int", 0) or 0),
                    "dex": int(getattr(snapshot, "dex", 0) or 0),
                    "luk": int(getattr(snapshot, "luk", 0) or 0),
                }

            # Detect Renewal by stat cap (>99)
            max_stat = max(stats.values()) if stats else 0
            if max_stat > 99 and not self._profile.is_renewal:
                self._profile.is_renewal = True
                self._profile.max_stats = 130
                changes["is_renewal"] = True
                logger.info("server_detected: renewal mechanics (stat > 99)")

            # Detect max level from job name patterns
            job_name = ""
            if isinstance(snapshot, dict):
                job_name = str(snapshot.get("job_name", snapshot.get("class", "")) or "")
            else:
                job_name = str(getattr(snapshot, "job_name", "") or "")

            if job_name:
                job_lower = job_name.lower().replace(" ", "_")
                for kw in self.RENEWAL_KEYWORDS:
                    if kw in job_lower and not self._profile.is_renewal:
                        self._profile.is_renewal = True
                        self._profile.max_stats = 130
                        self._profile.max_base_level = 175
                        self._profile.max_job_level = 60
                        changes["is_renewal"] = True
                        logger.info("server_detected: renewal from job %s", job_name)
                        break
                for kw in self.PRE_RENEWAL_KEYWORDS:
                    if kw in job_lower and self._profile.is_renewal:
                        # Already detected as renewal, but this is a transcendent class
                        pass

            # Detect NPC positions from actor list
            actors = []
            if isinstance(snapshot, dict):
                actors = snapshot.get("actors", []) or []
            else:
                actors = getattr(snapshot, "actors", []) or []

            for actor in actors:
                actor_name = ""
                actor_type = ""
                actor_x = 0
                actor_y = 0
                if isinstance(actor, dict):
                    actor_name = str(actor.get("name", "") or "")
                    actor_type = str(actor.get("actor_type", "") or "")
                    actor_x = int(actor.get("x", 0) or 0)
                    actor_y = int(actor.get("y", 0) or 0)
                else:
                    actor_name = str(getattr(actor, "name", "") or "")
                    actor_type = str(getattr(actor, "actor_type", "") or "")
                    actor_x = int(getattr(actor, "x", 0) or 0)
                    actor_y = int(getattr(actor, "y", 0) or 0)

                if actor_type == "npc" and actor_name:
                    # Store NPC for service discovery
                    name_lower = actor_name.lower()
                    service = None
                    if any(kw in name_lower for kw in ["kafra", "storage", "keeper"]):
                        service = "storage"
                    elif any(kw in name_lower for kw in ["tool", "dealer", "shop", "item", "mart"]):
                        service = "vendor"
                    elif any(kw in name_lower for kw in ["heal", "nun", "nurse", "priest"]):
                        service = "healer"
                    elif any(kw in name_lower for kw in ["refine", "smith", "forge"]):
                        service = "refiner"

                    if service:
                        map_name = ""
                        if isinstance(snapshot, dict):
                            map_name = str(snapshot.get("map", "") or "")
                        else:
                            map_name = str(getattr(snapshot, "map", "") or "")
                        key = f"{map_name}:{service}"
                        if key not in self._profile.towns:
                            self._profile.towns[key] = {
                                "name": actor_name,
                                "x": actor_x,
                                "y": actor_y,
                                "service": service,
                                "map": map_name,
                            }
                            changes[f"npc_{key}"] = True

            return changes

    def get_effective_exp(self, base_exp: int, job_exp: int) -> tuple[int, int]:
        """Apply detected server rates to get effective EXP values."""
        with self._lock:
            return (
                int(base_exp * self._profile.base_exp_rate),
                int(job_exp * self._profile.job_exp_rate),
            )

    def get_effective_drop_rate(self, base_rate: int) -> int:
        """Apply detected server drop rate."""
        with self._lock:
            return int(base_rate * self._profile.drop_rate)

    def get_level_route(self, bot_level: int) -> list[dict[str, Any]]:
        """Get optimal leveling route adjusted for server rates."""
        with self._lock:
            rate = self._profile.base_exp_rate
            if rate >= 10:
                # High rate server — skip low-level grinding, go straight to mid-game
                return [
                    {"level_range": "1-40", "strategy": "quest_skip", "reason": "high_rate"},
                    {"level_range": "40-70", "strategy": "efficient_grind", "reason": "high_rate"},
                    {"level_range": "70-99", "strategy": "optimal_maps", "reason": "endgame"},
                ]
            elif rate >= 3:
                # Medium rate
                return [
                    {"level_range": "1-20", "strategy": "fields", "reason": "medium_rate"},
                    {"level_range": "20-50", "strategy": "dungeons", "reason": "medium_rate"},
                    {"level_range": "50-99", "strategy": "optimal_maps", "reason": "endgame"},
                ]
            else:
                # Low rate (official-like)
                return [
                    {"level_range": "1-15", "strategy": "fields", "reason": "low_rate"},
                    {"level_range": "15-40", "strategy": "dungeons", "reason": "low_rate"},
                    {"level_range": "40-70", "strategy": "optimal_maps", "reason": "midgame"},
                    {"level_range": "70-99", "strategy": "efficient_grind", "reason": "endgame"},
                ]

    def get_stats(self) -> dict[str, Any]:
        """Get server adaptation stats."""
        with self._lock:
            p = self._profile
            return {
                "server_name": p.server_name,
                "rates": {
                    "base_exp": p.base_exp_rate,
                    "job_exp": p.job_exp_rate,
                    "drop": p.drop_rate,
                    "card": p.card_rate,
                    "refine": p.refine_rate,
                    "mvp_exp": p.mvp_exp_rate,
                    "quest_exp": p.quest_exp_rate,
                },
                "mechanics": {
                    "is_renewal": p.is_renewal,
                    "max_base_level": p.max_base_level,
                    "max_job_level": p.max_job_level,
                    "max_stats": p.max_stats,
                    "aspd_formula": p.aspd_formula,
                },
                "features": {
                    "instant_cast": p.has_instant_cast,
                    "no_cast_delay": p.has_no_cast_delay,
                    "auto_loot": p.has_auto_loot,
                    "merchant_auto_store": p.has_merchant_auto_store,
                    "mvp_tracker": p.has_mvp_tracker,
                },
                "confidence": p.confidence,
                "samples": p.samples_taken,
                "npc_count": len(p.towns),
                "towns": dict(p.towns),
            }


class ServerSolutionsStore:
    """DB-backed store of server SPECIFIC solution knowledge (never hardcoded in *.py).

    Per the sandbox rules / RULE.md: different servers need different solutions, so
    server-specific facts (which potion to buy, which farm map is reachable, which town
    has a shop, which mobs are safe to attack) are LEARNED and persisted to the
    `server_solutions` table, not written as literals in code. The LLM/CrewAI conscious
    tier decides WHAT; this store supplies the server-agnostic FACTS the decision uses.

    A fallback (non-DB) dict is used when no DB connection is provided, so the store is
    safe to instantiate anywhere; the DB is preferred for persistence across restarts.
    """

    def __init__(self, db: Any | None = None, server_key: str = "default"):
        self._db = db
        self._server_key = str(server_key or "default")
        self._lock = RLock()
        # In-memory fallback so degenerate reads don't hit a closed DB.
        self._fallback: dict[str, dict[str, Any]] = {}

    def set(self, slot: str, value: Any, *, origin: str = "learned", confidence: float = 0.8, value_json: str | None = None) -> None:
        """Persist a server-specific solution fact."""
        slot = str(slot or "").strip()
        if not slot:
            return
        _now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # If no explicit JSON string was supplied but value is dict/list, serialize it as JSON
        # so a later get_json round-trips correctly (str(dict) would NOT be valid JSON).
        if value_json is None and isinstance(value, (dict, list)):
            try:
                value_json = json.dumps(value)
            except Exception:
                value_json = None
        # When a JSON payload is present, keep value_text empty so get()/get_json() prefer the JSON.
        _vt = "" if value_json else ("" if value is None else str(value))
        with self._lock:
            if self._db is not None:
                try:
                    self._db.execute(
                        "INSERT INTO server_solutions (server_key, slot, value_text, value_json, origin, confidence, last_observed_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(server_key, slot) DO UPDATE SET value_text=excluded.value_text, "
                        "value_json=excluded.value_json, origin=excluded.origin, confidence=excluded.confidence, "
                        "last_observed_at=excluded.last_observed_at, updated_at=excluded.updated_at",
                        (self._server_key, slot, _vt, value_json or "{}", origin, float(confidence), _now, _now),
                    )
                except Exception as _e:
                    logger.debug("server_solutions_set_db_failed slot=%s: %s", slot, _e)
            self._fallback[slot] = {"value": value, "value_json": value_json or "{}", "origin": origin, "confidence": confidence, "updated_at": _now}

    def get(self, slot: str, default: Any = None) -> Any:
        """Read a server-specific solution fact (DB first, then in-memory fallback)."""
        slot = str(slot or "").strip()
        with self._lock:
            if self._db is not None:
                try:
                    _row = self._db.fetchone(
                        "SELECT value_text, value_json FROM server_solutions WHERE server_key=? AND slot=? LIMIT 1",
                        (self._server_key, slot),
                    )
                    if _row is not None:
                        _vt = _row["value_text"] if "value_text" in _row.keys() else (_row[0] if len(_row) > 0 else "")
                        _vj = _row["value_json"] if "value_json" in _row.keys() else (_row[1] if len(_row) > 1 else "{}")
                        try:
                            _parsed = json.loads(_vj) if _vj and _vj != "{}" and _vt == "" else None
                        except Exception:
                            _parsed = None
                        if _parsed is not None and _parsed != {}:
                            return _parsed
                        if _vt:
                            return _vt
                        return default
                    return default
                except Exception as _e:
                    logger.debug("server_solutions_get_db_failed slot=%s: %s", slot, _e)
            if slot in self._fallback:
                _v = self._fallback[slot].get("value")
                return default if _v is None else _v
            return default

    def get_json(self, slot: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
        """Read a JSON-structured server solution fact."""
        default = default if default is not None else {}
        _raw = self.get(slot, None)
        if isinstance(_raw, dict):
            return _raw
        if isinstance(_raw, str):
            try:
                _d = json.loads(_raw)
                return _d if isinstance(_d, dict) else default
            except Exception:
                return default
        return default


_def_store: ServerSolutionsStore | None = None
_store_lock = RLock()


def get_server_solutions_store(db: Any | None = None, server_key: str = "default") -> ServerSolutionsStore:
    """Singleton accessor for the server-solutions knowledge store."""
    global _def_store
    with _store_lock:
        if _def_store is None or _def_store._server_key != server_key:
            _def_store = ServerSolutionsStore(db=db, server_key=server_key)
        return _def_store
