"""Target Engine — resolves current target monster from bridge snapshot.

Architecture:
  - Reads current target from snapshot's actor data
  - Looks up monster in mob_db.yml by name
  - Returns MonsterData (element, size, race, level, HP, DEF, MDEF)
  - Cache results per monster name for 30s
  - All data from rAthena DB — zero hardcoded values

RULE.md compliance: Monsters from mob_db.yml (2,675 entries). Elements from attr_fix.yml.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Data Structures ────────────────────────────────────────────────


@dataclass
class MonsterData:
    """Resolved monster data from mob_db.yml."""
    id: int
    aegis_name: str
    name: str
    level: int
    hp: int
    base_exp: int
    job_exp: int
    attack: int
    attack2: int
    defense: int
    magic_defense: int
    size: str            # Small | Medium | Large
    race: str            # Angel, Brute, Demihuman, Demon, Dragon, etc.
    element: str         # Neutral, Water, Earth, Fire, Wind, Poison, Holy, Dark, Ghost, Undead
    element_level: int   # 1-4
    walk_speed: int = 200
    attack_delay: int = 1000
    modes: list[str] = field(default_factory=list)


# ── Monster DB Loader ─────────────────────────────────────────────


class MonsterDB:
    """Loads monster data from mob_db.yml (rAthena DB)."""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or self._default_path()
        self._monsters_by_name: dict[str, MonsterData] = {}
        self._monsters_by_id: dict[int, MonsterData] = {}
        self._loaded = False
    
    def _default_path(self) -> str:
        """Find mob_db.yml — RAW runs RENEWAL, so db/re is authoritative for
        the live server's mob stats (pre-re HP/EXP mis-score every target:
        Thief Bug Egg = HP 290 real vs 48 pre-re). Prefer re over pre-re."""
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        base = os.path.join(base, "knowledge", "rathena_db", "db")
        # Try re (renewal — the live RAW server) first, then pre-re
        for variant in ["re", "pre-re"]:
            path = os.path.join(base, variant, "mob_db.yml")
            if os.path.exists(path):
                return path
        return os.path.join(base, "re", "mob_db.yml")
    
    def _find_rathena_path(self) -> str:
        """Try to find the rAthena repo path.

        RAW server runs RENEWAL (db/re is authoritative for the live server's
        mob stats — pre-re HP/EXP mis-score every target). Prefer the RUNNING
        server's db/re over the vendored pre-re copy.
        """
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Check common locations — NO hardcoded repo path (agnostic): the server
        # may live anywhere. re (renewal) wins over pre-re at whatever root.
        candidates = []
        for root in (
            os.path.expanduser("~"),
            base,
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ):
            candidates.append(os.path.join(root, "rathena-AI-world", "db", "re"))
            candidates.append(os.path.join(root, "rathena", "db", "re"))
            candidates.append(os.path.join(base, "knowledge", "rathena_db", "db", "re"))
            candidates.append(os.path.join(root, "rathena-AI-world", "db"))
            candidates.append(os.path.join(root, "rathena", "db"))
        # Dedup while preserving order
        seen = set()
        unique = [p for p in candidates if not (p in seen or seen.add(p))]
        for path in unique:
            if os.path.exists(os.path.join(path, "mob_db.yml")):
                return path
        return candidates[0]
    
    def load(self) -> bool:
        """Load all monsters from mob_db.yml. Returns True on success."""
        if self._loaded:
            return True
        
        path = self._db_path
        if not os.path.exists(path):
            logger.warning("monster_db: file not found at %s", path)
            return False
        
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.warning("monster_db: failed to load YAML from %s: %s", path, e)
            return False
        
        if not data or "Body" not in data:
            logger.warning("monster_db: invalid format in %s", path)
            return False
        
        count = 0
        modes_map = {
            0x1: "can_move", 0x2: "can_attack", 0x4: "aggro",
            0x8: "assist", 0x10: "cast_sensor", 0x20: "boss",
            0x40: "plant", 0x80: "can_talk", 0x100: "statue",
            0x200: "looter", 0x400: "can_escape", 0x800: "change_target",
        }
        
        for entry in data["Body"]:
            try:
                m_id = entry.get("Id", 0)
                name = str(entry.get("Name", entry.get("AegisName", "")))
                lookup_names = set()
                lookup_names.add(name.lower())
                lookup_names.add(str(entry.get("AegisName", "")).lower())
                
                # Parse modes
                mode_val = 0
                if "Modes" in entry and entry["Modes"] is not None:
                    if isinstance(entry["Modes"], dict):
                        for k, v in entry["Modes"].items():
                            if v:
                                mode_val |= int(k) if k.isdigit() else 0
                    else:
                        try:
                            mode_val = int(entry["Modes"])
                        except (ValueError, TypeError):
                            mode_val = 0
                
                resolved_modes = []
                for bit, label in modes_map.items():
                    if mode_val & bit:
                        resolved_modes.append(label)
                
                monster = MonsterData(
                    id=m_id,
                    aegis_name=str(entry.get("AegisName", "")),
                    name=name,
                    level=int(entry.get("Level", 0) or 0),
                    hp=int(entry.get("Hp", 0) or 0),
                    base_exp=int(entry.get("BaseExp", 0) or 0),
                    job_exp=int(entry.get("JobExp", 0) or 0),
                    attack=int(entry.get("Attack", 0) or 0),
                    attack2=int(entry.get("Attack2", 0) or 0),
                    defense=int(entry.get("Defense", 0) or 0),
                    magic_defense=int(entry.get("MagicDefense", 0) or 0),
                    size=str(entry.get("Size", "Medium")),
                    race=str(entry.get("Race", "Formless")),
                    element=str(entry.get("Element", "Neutral")),
                    element_level=int(entry.get("ElementLevel", 1) or 1),
                    walk_speed=int(entry.get("WalkSpeed", 200) or 200),
                    attack_delay=int(entry.get("AttackDelay", 1000) or 1000),
                    modes=resolved_modes,
                )
                
                for lookup in lookup_names:
                    if lookup:
                        self._monsters_by_name[lookup] = monster
                self._monsters_by_id[m_id] = monster
                count += 1
            except Exception as e:
                logger.debug("monster_db: skipped entry %s: %s", entry.get("Name", "?"), e)
        
        self._loaded = True
        logger.info("monster_db: loaded %d monsters from %s", count, path)
        return True
    
    def lookup(self, name: str) -> MonsterData | None:
        """Look up a monster by name (case-insensitive)."""
        if not self._loaded:
            self.load()
        return self._monsters_by_name.get(name.lower().strip())
    
    def lookup_by_id(self, monster_id: int) -> MonsterData | None:
        """Look up a monster by numeric ID."""
        if not self._loaded:
            self.load()
        return self._monsters_by_id.get(monster_id)
    
    def is_boss(self, monster: MonsterData) -> bool:
        """Check if monster has boss mode flags."""
        return "boss" in monster.modes


# ── Target Engine ─────────────────────────────────────────────────


class TargetEngine:
    """Resolves the current target monster from bridge snapshot data."""

    def __init__(self):
        self._monster_db = MonsterDB()
        self._cache: dict[str, tuple[MonsterData, float]] = {}  # name → (data, expiry)
        self._cache_ttl = 30.0  # seconds
        self._last_target: str = ""
        self._target_changed = False
    
    def resolve(self, snapshot, bot_id: str = "") -> MonsterData | None:
        """Resolve current target monster from snapshot.
        
        Steps:
        1. Read target info from snapshot actor data
        2. Look up monster name in mob_db.yml
        3. Cache result for _cache_ttl seconds
        4. Detect target changes
        
        Returns MonsterData or None if no target.
        """
        target_name = self._extract_target_name(snapshot)
        if not target_name:
            self._target_changed = self._last_target != ""
            self._last_target = ""
            return None
        
        # Check cache
        now = time.time()
        cached = self._cache.get(target_name.lower())
        if cached and cached[1] > now:
            self._target_changed = (target_name.lower() != self._last_target.lower())
            self._last_target = target_name
            return cached[0]
        
        # Try exact match first, then partial
        monster = self._monster_db.lookup(target_name)
        if monster is None:
            # Try partial match (remove trailing qualifiers like (0), (1))
            import re
            clean = re.sub(r'\s*\(\d+\)$', '', target_name).strip()
            if clean != target_name:
                monster = self._monster_db.lookup(clean)
        
        if monster is None:
            # Try AegisName (the server-side name which might differ from display name)
            pass
        
        if monster:
            self._cache[target_name.lower()] = (monster, now + self._cache_ttl)
            self._target_changed = (target_name.lower() != self._last_target.lower())
            self._last_target = target_name
            return monster
        
        logger.debug("target_engine: unknown monster '%s' — not in mob_db.yml", target_name)
        self._target_changed = (target_name.lower() != self._last_target.lower())
        self._last_target = target_name
        return None
    
    def _extract_target_name(self, snapshot) -> str:
        """Extract current target monster name from snapshot actors."""
        if snapshot is None:
            return ""
        
        try:
            if isinstance(snapshot, dict):
                actors = snapshot.get("actors", {}) or {}
                # Check for current target in actor list
                for actor in actors.get("list", []):
                    if actor.get("is_target") or actor.get("isCurrentTarget"):
                        return str(actor.get("name", actor.get("display", "")))
                # Fallback: first attacking monster
                for actor in actors.get("list", []):
                    atype = actor.get("type", "")
                    if atype == "monster" and actor.get("hp", 0) > 0:
                        return str(actor.get("name", ""))
            else:
                actors = getattr(snapshot, "actors", None)
                if actors:
                    for actor in getattr(actors, "list", []):
                        if getattr(actor, "is_target", False):
                            return str(getattr(actor, "name", ""))
                    for actor in getattr(actors, "list", []):
                        if getattr(actor, "type", "") == "monster" and getattr(actor, "hp", 0) > 0:
                            return str(getattr(actor, "name", ""))
        except Exception as e:
            logger.debug("target_engine: failed to extract target: %s", e)
        
        return ""
    
    def target_changed(self) -> bool:
        """Check if target changed since last resolve call."""
        return self._target_changed


# ── Singleton ──────────────────────────────────────────────────────

_engine: TargetEngine | None = None


def get_target_engine() -> TargetEngine:
    """Get the global TargetEngine instance."""
    global _engine
    if _engine is None:
        _engine = TargetEngine()
    return _engine


def resolve_target(snapshot, bot_id: str = "") -> MonsterData | None:
    """Convenience function to resolve current target."""
    return get_target_engine().resolve(snapshot, bot_id)
