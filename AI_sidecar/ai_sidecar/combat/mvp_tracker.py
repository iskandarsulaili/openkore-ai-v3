"""
MVP Respawn Tracker — records kill times, predicts respawns,
routes bots to due MVPs, and coordinates multi-bot MVP hunting.

Features:
  - Spawn detection from snapshot data (monster name, map, HP)
  - Respawn timer calculation with random window (rAthena: base ± random%)
  - Party coordination for MVP hunting
  - Gear swapping for MVP-specific mechanics
  - Loot distribution system
  - Spawn camping logic
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class MVPRecord:
    """A record of an MVP kill or sighting."""
    monster_id: int
    monster_name: str
    map_name: str
    kill_time: float = 0.0
    sighting_time: float = 0.0
    respawn_time: float = 0.0
    respawn_window_minutes: int = 120  # Default respawn window
    respawn_variance_pct: float = 0.1  # rAthena: ±10% random variance
    is_due: bool = False
    is_up: bool = False
    killed_by_us: bool = False
    strategy_used: str = ""
    last_hp_pct: float = 1.0
    element: str = "Neutral"
    element_level: int = 1
    race: str = "Formless"
    size: str = "Large"
    level: int = 50
    hp: int = 0
    def_: int = 0
    mdef: int = 0


@dataclass
class MVPHuntTarget:
    """An MVP that's worth hunting right now."""
    monster_id: int
    monster_name: str
    map_name: str
    time_until_respawn_min: float = 0.0
    priority: int = 50
    estimated_value: int = 0
    difficulty: str = "medium"
    is_worth_hunting: bool = True
    recommended_element: str = "Neutral"
    recommended_gear_set: str = ""


@dataclass
class MVPLootShare:
    """Loot distribution for an MVP kill."""
    monster_name: str
    kill_time: float
    participants: list[str]  # bot IDs
    loot_items: list[dict] = field(default_factory=list)
    distributed: bool = False


class MVPTracker:
    """Tracks MVP respawns and coordinates hunting."""

    # Known MVPs with their respawn windows (minutes) and estimated card value
    KNOWN_MVPS: dict[int, dict] = {}

    # rAthena MVP respawn rules:
    # - Base respawn time is defined in mob_db (respawn_time field)
    # - Random variance: ±10% of base time (rAthena default)
    # - Some MVPs have fixed respawn (e.g. 2 hours for Baphomet)
    # - MVP respawns at a random location within its spawn area
    MVP_RESPAWN_DATA: dict[str, dict] = {
        "baphomet": {"base_minutes": 120, "variance": 0.1, "map": "iz_dun04", "element": "Dark", "element_level": 3, "race": "Demon", "size": "Large", "level": 81, "hp": 780000, "def": 40, "mdef": 30},
        "orc_hero": {"base_minutes": 60, "variance": 0.1, "map": "orcsdun02", "element": "Earth", "element_level": 2, "race": "DemiHuman", "size": "Large", "level": 70, "hp": 320000, "def": 40, "mdef": 20},
        "moonlight": {"base_minutes": 60, "variance": 0.1, "map": "um_dun01", "element": "Fire", "element_level": 3, "race": "Demon", "size": "Large", "level": 65, "hp": 280000, "def": 15, "mdef": 30},
        "osiris": {"base_minutes": 60, "variance": 0.1, "map": "moc_pryd04", "element": "Undead", "element_level": 4, "race": "Undead", "size": "Large", "level": 75, "hp": 450000, "def": 30, "mdef": 40},
        "eddga": {"base_minutes": 60, "variance": 0.1, "map": "moc_pryd05", "element": "Fire", "element_level": 2, "race": "Brute", "size": "Large", "level": 62, "hp": 250000, "def": 30, "mdef": 20},
        "doppelganger": {"base_minutes": 60, "variance": 0.1, "map": "gef_dun02", "element": "Dark", "element_level": 3, "race": "Demon", "size": "Large", "level": 77, "hp": 500000, "def": 35, "mdef": 25},
        "phreeoni": {"base_minutes": 60, "variance": 0.1, "map": "mi_dun01", "element": "Neutral", "element_level": 3, "race": "Brute", "size": "Large", "level": 68, "hp": 300000, "def": 25, "mdef": 20},
        "garm": {"base_minutes": 60, "variance": 0.1, "map": "xmas_dun02", "element": "Water", "element_level": 3, "race": "Brute", "size": "Large", "level": 72, "hp": 380000, "def": 30, "mdef": 25},
        "mistress": {"base_minutes": 60, "variance": 0.1, "map": "moc_fild12", "element": "Wind", "element_level": 3, "race": "Insect", "size": "Medium", "level": 66, "hp": 290000, "def": 20, "mdef": 40},
        "drake": {"base_minutes": 60, "variance": 0.1, "map": "treasure02", "element": "Undead", "element_level": 2, "race": "Undead", "size": "Large", "level": 70, "hp": 350000, "def": 30, "mdef": 25},
        "atroce": {"base_minutes": 120, "variance": 0.1, "map": "um_boss", "element": "Brute", "element_level": 3, "race": "Brute", "size": "Large", "level": 85, "hp": 850000, "def": 40, "mdef": 25},
        "kiel": {"base_minutes": 120, "variance": 0.1, "map": "kiel_dun01", "element": "DemiHuman", "element_level": 2, "race": "DemiHuman", "size": "Medium", "level": 80, "hp": 600000, "def": 30, "mdef": 35},
        "turtle_general": {"base_minutes": 120, "variance": 0.1, "map": "tur_dun04", "element": "Water", "element_level": 3, "race": "Brute", "size": "Large", "level": 82, "hp": 720000, "def": 35, "mdef": 30},
        "gloom_under_night": {"base_minutes": 120, "variance": 0.1, "map": "ra_fild01", "element": "Dark", "element_level": 3, "race": "Demon", "size": "Large", "level": 88, "hp": 950000, "def": 35, "mdef": 40},
        "detardeuras": {"base_minutes": 120, "variance": 0.1, "map": "boss_rash", "element": "Holy", "element_level": 3, "race": "Demon", "size": "Large", "level": 90, "hp": 1200000, "def": 40, "mdef": 45},
        "ifrit": {"base_minutes": 120, "variance": 0.1, "map": "mocboss", "element": "Fire", "element_level": 4, "race": "Formless", "size": "Large", "level": 95, "hp": 1500000, "def": 50, "mdef": 40},
        "thanatos": {"base_minutes": 120, "variance": 0.1, "map": "than_d", "element": "Ghost", "element_level": 3, "race": "Demon", "size": "Large", "level": 92, "hp": 1300000, "def": 45, "mdef": 50},
        "beelzebub": {"base_minutes": 120, "variance": 0.1, "map": "beach_dun", "element": "Dark", "element_level": 3, "race": "Demon", "size": "Large", "level": 96, "hp": 1800000, "def": 55, "mdef": 50},
    }

    # Element recommendations for each MVP (best element to attack with)
    MVP_ELEMENT_ADVICE: dict[str, str] = {
        "baphomet": "Holy",        # Dark → Holy deals 2x-3x
        "orc_hero": "Water",       # Earth → Water deals 1.5x-1.75x
        "moonlight": "Water",      # Fire → Water deals 1.5x-2x
        "osiris": "Fire",          # Undead → Fire deals 1.25x-2x
        "eddga": "Water",          # Fire → Water deals 1.5x
        "doppelganger": "Holy",    # Dark → Holy deals 2x-3x
        "phreeoni": "Holy",        # Neutral → Holy is neutral, but Ghost works
        "garm": "Wind",            # Water → Wind deals 1.5x
        "mistress": "Fire",        # Wind → Fire deals 1.5x
        "drake": "Fire",           # Undead → Fire deals 1.25x-2x
        "atroce": "Holy",          # Brute → Holy is neutral, use Fire
        "kiel": "Holy",            # DemiHuman → Holy is neutral
        "turtle_general": "Wind",  # Water → Wind deals 1.5x
        "gloom_under_night": "Holy",  # Dark → Holy deals 2x-3x
        "detardeuras": "Dark",     # Holy → Dark deals 1.25x
        "ifrit": "Water",          # Fire → Water deals 1.5x-2x
        "thanatos": "Dark",        # Ghost → Dark deals 1.0x (Ghost resists most)
        "beelzebub": "Holy",       # Dark → Holy deals 2x-4x
    }

    @classmethod
    def _load_mvps_from_db(cls) -> dict[int, dict]:
        """Load MVPs from the knowledge database."""
        mvps: dict[int, dict] = {}
        try:
            from ai_sidecar.knowledge_loader import get_mvps
            db_mvps = get_mvps()
            for m in db_mvps:
                mid = m.get("Id", 0)
                if mid:
                    level = m.get("Level", 50)
                    respawn = 60 if level < 60 else 120
                    hp = m.get("Hp", 0)
                    value = max(10000000, hp * 10)
                    difficulty = "medium" if level < 70 else "hard"
                    mvps[mid] = {
                        "name": m.get("Name", f"MVP_{mid}"),
                        "respawn": respawn,
                        "value": value,
                        "difficulty": difficulty,
                    }
            logger.info("mvps_loaded_from_db: %d MVPs", len(mvps))
        except Exception as e:
            logger.warning("mvps_db_load_failed: %s (no hardcoded fallback — DB is the source of truth)", e)
        return mvps

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: dict[int, MVPRecord] = {}
        self._hunt_targets: list[MVPHuntTarget] = []
        self._loot_shares: list[MVPLootShare] = []
        self._party_members: list[str] = []
        self._enqueue_fn: Callable | None = None
        self._load_known_mvps()

    def _load_known_mvps(self) -> None:
        """Initialize records for all known MVPs."""
        for mid, data in self.KNOWN_MVPS.items():
            self._records[mid] = MVPRecord(
                monster_id=mid,
                monster_name=data["name"],
                map_name="unknown",
                respawn_window_minutes=data["respawn"],
            )

    # ── Spawn Detection ──

    def detect_spawn_from_snapshot(self, snapshot: dict) -> list[int]:
        """Detect MVP spawns from a snapshot dict.

        Scans for monsters with MVP-level HP and known MVP names.

        Args:
            snapshot: Bot state snapshot with 'monsters' or 'entities' list

        Returns:
            List of monster_ids that are MVPs
        """
        detected: list[int] = []
        monsters = snapshot.get("monsters", snapshot.get("entities", []))
        if not monsters:
            return detected

        for mob in monsters:
            mob_name = mob.get("name", "")
            mob_id = mob.get("id", 0)
            mob_hp = mob.get("hp", 0)
            mob_hp_max = mob.get("hp_max", mob.get("max_hp", 0))
            mob_map = mob.get("map", snapshot.get("map", "unknown"))

            # Check if this monster is a known MVP
            mvp_info = self._match_mvp_by_name(mob_name)
            if mvp_info:
                mid = mvp_info.get("id", mob_id)
                self.record_sighting(mid, mob_map)
                self._update_record_stats(mid, mob_hp, mob_hp_max, mob_name)
                detected.append(mid)
                logger.info("mvp_spawn_detected: %s on %s (HP: %d/%d)", mob_name, mob_map, mob_hp, mob_hp_max)
            elif mob_hp_max > 100000 and mob_id > 0:
                # High-HP monster that might be an MVP — check against known data
                for mvp_name, mvp_data in self.MVP_RESPAWN_DATA.items():
                    if mvp_data.get("hp", 0) > 0 and abs(mob_hp_max - mvp_data["hp"]) / mvp_data["hp"] < 0.2:
                        self.record_sighting(mob_id, mob_map)
                        detected.append(mob_id)
                        logger.info("mvp_spawn_suspected: %s (HP: %d) on %s", mob_name, mob_hp_max, mob_map)
                        break

        return detected

    def _match_mvp_by_name(self, name: str) -> dict | None:
        """Match a monster name to a known MVP."""
        name_lower = name.lower().replace("_", " ").replace("-", " ")
        for mvp_name, mvp_data in self.MVP_RESPAWN_DATA.items():
            if mvp_name in name_lower or name_lower in mvp_name:
                return {"id": hash(mvp_name) % 10000, "name": mvp_name, **mvp_data}
        return None

    def _update_record_stats(self, monster_id: int, hp: int, hp_max: int, name: str) -> None:
        """Update MVP record with current stats."""
        with self._lock:
            record = self._records.get(monster_id)
            if record:
                record.last_hp_pct = hp / max(hp_max, 1)
                record.hp = hp_max

    # ── Respawn Timer ──

    def calculate_respawn_time(self, base_minutes: int, variance_pct: float = 0.1) -> float:
        """Calculate actual respawn time with rAthena random variance.

        rAthena formula: respawn = base_time + random(-variance, +variance) * base_time

        Args:
            base_minutes: Base respawn time in minutes
            variance_pct: Random variance as fraction (0.1 = ±10%)

        Returns:
            Respawn timestamp (time.time() + actual_minutes * 60)
        """
        variance = base_minutes * variance_pct * random.uniform(-1.0, 1.0)
        actual_minutes = base_minutes + variance
        return time.time() + actual_minutes * 60

    def get_respawn_window(self, monster_id: int) -> tuple[float, float]:
        """Get the respawn window for an MVP.

        Returns:
            (earliest_respawn, latest_respawn) as timestamps
        """
        with self._lock:
            record = self._records.get(monster_id)
            if not record or record.kill_time == 0:
                return (0, 0)

            base_minutes = record.respawn_window_minutes
            variance = base_minutes * record.respawn_variance_pct
            earliest = record.kill_time + (base_minutes - variance) * 60
            latest = record.kill_time + (base_minutes + variance) * 60
            return (earliest, latest)

    # ── Party Coordination ──

    def set_party_members(self, members: list[str]) -> None:
        """Set the list of party member bot IDs for MVP hunting."""
        with self._lock:
            self._party_members = members
            logger.info("mvp_party_set: %d members", len(members))

    def get_party_members(self) -> list[str]:
        """Get the list of party member bot IDs."""
        with self._lock:
            return list(self._party_members)

    def coordinate_mvp_hunt(self, target: MVPHuntTarget) -> dict:
        """Generate coordination commands for party MVP hunting.

        Returns:
            Dict with assignments per bot_id
        """
        with self._lock:
            if not self._party_members:
                return {}

            assignments: dict = {}
            mvp_name = target.monster_name.lower().replace(" ", "_")

            # Get MVP data for element advice
            mvp_data = self.MVP_RESPAWN_DATA.get(mvp_name, {})
            recommended_element = self.MVP_ELEMENT_ADVICE.get(mvp_name, "Neutral")

            # Assign roles
            for i, bot_id in enumerate(self._party_members):
                if i == 0:
                    # Lead: primary attacker, tank
                    role = "tank_attacker"
                elif i == 1:
                    # Support: healer/buffer
                    role = "healer_support"
                else:
                    # DPS: damage dealer
                    role = "dps"

                assignments[bot_id] = {
                    "role": role,
                    "target_map": target.map_name,
                    "target_monster": target.monster_name,
                    "recommended_element": recommended_element,
                    "recommended_gear": f"mvp_{mvp_name}",
                    "strategy": self._get_mvp_strategy(mvp_name, role),
                }

            return assignments

    def _get_mvp_strategy(self, mvp_name: str, role: str) -> dict:
        """Get MVP-specific combat strategy."""
        mvp_data = self.MVP_RESPAWN_DATA.get(mvp_name, {})
        element = mvp_data.get("element", "Neutral")
        element_level = mvp_data.get("element_level", 1)

        strategies = {
            "tank_attacker": {
                "primary_action": "engage_mvp",
                "element_to_use": self.MVP_ELEMENT_ADVICE.get(mvp_name, "Neutral"),
                "element_level": element_level,
                "defender_element": element,
                "defender_level": element_level,
                "use_provoke": True,
                "use_endure": True,
                "hp_threshold": 0.5,  # Teleport at 50% HP
            },
            "healer_support": {
                "primary_action": "support_party",
                "heal_threshold": 0.7,
                "buff_skills": ["blessing", "increase_agility", "kyrie_eleison"],
                "stay_at_range": True,
            },
            "dps": {
                "primary_action": "damage_mvp",
                "element_to_use": self.MVP_ELEMENT_ADVICE.get(mvp_name, "Neutral"),
                "element_level": element_level,
                "defender_element": element,
                "defender_level": element_level,
                "stay_at_range": True,
                "use_teleport_on_burst": True,
            },
        }
        return strategies.get(role, {})

    # ── Gear Swapping ──

    def get_recommended_gear(self, monster_name: str) -> dict:
        """Get recommended gear set for an MVP.

        Returns dict with weapon, armor, shield, garment, shoes, accessory recommendations.
        """
        mvp_name = monster_name.lower().replace(" ", "_")
        mvp_data = self.MVP_RESPAWN_DATA.get(mvp_name, {})
        element = mvp_data.get("element", "Neutral")
        recommended_element = self.MVP_ELEMENT_ADVICE.get(mvp_name, "Neutral")

        return {
            "weapon_element": recommended_element,
            "defender_element": element,
            "defender_element_level": mvp_data.get("element_level", 1),
            "mvp_name": monster_name,
            "mvp_level": mvp_data.get("level", 50),
            "mvp_hp": mvp_data.get("hp", 0),
            "mvp_def": mvp_data.get("def", 0),
            "mvp_mdef": mvp_data.get("mdef", 0),
            "mvp_race": mvp_data.get("race", "Formless"),
            "mvp_size": mvp_data.get("size", "Large"),
        }

    # ── Loot Distribution ──

    def record_loot(self, monster_name: str, participants: list[str], loot_items: list[dict]) -> None:
        """Record loot from an MVP kill for distribution."""
        with self._lock:
            share = MVPLootShare(
                monster_name=monster_name,
                kill_time=time.time(),
                participants=participants,
                loot_items=loot_items,
            )
            self._loot_shares.append(share)
            logger.info("mvp_loot_recorded: %s, %d items, %d participants",
                        monster_name, len(loot_items), len(participants))

    def get_pending_loot(self) -> list[MVPLootShare]:
        """Get loot shares that haven't been distributed yet."""
        with self._lock:
            return [s for s in self._loot_shares if not s.distributed]

    def mark_loot_distributed(self, monster_name: str, kill_time: float) -> None:
        """Mark loot as distributed."""
        with self._lock:
            for share in self._loot_shares:
                if share.monster_name == monster_name and share.kill_time == kill_time:
                    share.distributed = True
                    break

    # ── Spawn Camping ──

    def should_camp_spawn(self, monster_id: int) -> bool:
        """Check if we should camp an MVP spawn point.

        Returns True if the MVP is due to respawn within the next 10 minutes.
        """
        with self._lock:
            record = self._records.get(monster_id)
            if not record or record.kill_time == 0:
                return False

            now = time.time()
            earliest, latest = self.get_respawn_window(monster_id)
            # Camp if within 10 minutes of earliest respawn
            return earliest > 0 and (earliest - now) <= 600 and (latest - now) > 0

    def get_camp_targets(self) -> list[MVPHuntTarget]:
        """Get MVPs worth camping right now."""
        targets = self.update_hunt_targets()
        return [t for t in targets if self.should_camp_spawn(t.monster_id)]

    # ── Public API ──

    def record_kill(self, monster_id: int, map_name: str, killed_by_us: bool = False, strategy: str = "") -> None:
        """Record an MVP kill with rAthena-accurate respawn calculation."""
        with self._lock:
            data = self.KNOWN_MVPS.get(monster_id)
            if not data:
                return

            # Find respawn data
            mvp_name = data["name"].lower().replace(" ", "_")
            mvp_data = self.MVP_RESPAWN_DATA.get(mvp_name, {})
            base_minutes = mvp_data.get("base_minutes", data.get("respawn", 120))
            variance = mvp_data.get("variance", 0.1)

            now = time.time()
            respawn_time = self.calculate_respawn_time(base_minutes, variance)

            self._records[monster_id] = MVPRecord(
                monster_id=monster_id,
                monster_name=data["name"],
                map_name=map_name,
                kill_time=now,
                respawn_time=respawn_time,
                respawn_window_minutes=base_minutes,
                respawn_variance_pct=variance,
                is_due=False,
                is_up=False,
                killed_by_us=killed_by_us,
                strategy_used=strategy,
                element=mvp_data.get("element", "Neutral"),
                element_level=mvp_data.get("element_level", 1),
                race=mvp_data.get("race", "Formless"),
                size=mvp_data.get("size", "Large"),
                level=mvp_data.get("level", 50),
                hp=mvp_data.get("hp", 0),
                def_=mvp_data.get("def", 0),
                mdef=mvp_data.get("mdef", 0),
            )

            window_start = respawn_time - base_minutes * 60 * variance
            window_end = respawn_time + base_minutes * 60 * variance
            logger.info("mvp_kill_recorded: %s on %s (respawn window: %s-%s)",
                        data["name"], map_name,
                        time.strftime("%H:%M", time.localtime(window_start)),
                        time.strftime("%H:%M", time.localtime(window_end)))

    def record_sighting(self, monster_id: int, map_name: str) -> None:
        """Record an MVP sighting (it's alive and on this map)."""
        with self._lock:
            data = self.KNOWN_MVPS.get(monster_id)
            if not data:
                return
            now = time.time()
            if monster_id in self._records:
                self._records[monster_id].sighting_time = now
                self._records[monster_id].map_name = map_name
                self._records[monster_id].is_up = True
                self._records[monster_id].is_due = False
            else:
                self._records[monster_id] = MVPRecord(
                    monster_id=monster_id,
                    monster_name=data["name"],
                    map_name=map_name,
                    sighting_time=now,
                    respawn_window_minutes=data["respawn"],
                    is_up=True,
                )

    def update_hunt_targets(self) -> list[MVPHuntTarget]:
        """Update and return the list of MVPs worth hunting right now."""
        with self._lock:
            now = time.time()
            targets: list[MVPHuntTarget] = []

            for mid, record in self._records.items():
                data = self.KNOWN_MVPS.get(mid)
                if not data:
                    continue

                # Check if MVP is due for respawn
                if record.respawn_time > 0 and now >= record.respawn_time:
                    record.is_due = True
                    record.is_up = False

                # Check if MVP is currently up (sighted recently)
                if record.sighting_time > 0 and now - record.sighting_time < 300:
                    record.is_up = True

                if record.is_due or record.is_up:
                    time_until = max(0, record.respawn_time - now) / 60.0 if record.respawn_time > 0 else 0
                    priority = 100 if record.is_up else max(10, 100 - int(time_until * 2))

                    # Get element advice
                    mvp_name = data["name"].lower().replace(" ", "_")
                    recommended_element = self.MVP_ELEMENT_ADVICE.get(mvp_name, "Neutral")

                    targets.append(MVPHuntTarget(
                        monster_id=mid,
                        monster_name=data["name"],
                        map_name=record.map_name,
                        time_until_respawn_min=time_until,
                        priority=priority,
                        estimated_value=data["value"],
                        difficulty=data["difficulty"],
                        is_worth_hunting=priority >= 30,
                        recommended_element=recommended_element,
                        recommended_gear_set=f"mvp_{mvp_name}",
                    ))

            targets.sort(key=lambda t: -t.priority)
            self._hunt_targets = targets
            return targets

    def get_best_hunt_target(self) -> MVPHuntTarget | None:
        """Get the best MVP to hunt right now."""
        targets = self.update_hunt_targets()
        return targets[0] if targets else None

    def get_mvp_status(self, monster_id: int) -> str:
        with self._lock:
            record = self._records.get(monster_id)
            if not record:
                return "Unknown"
            if record.is_up:
                return f"UP on {record.map_name}"
            if record.is_due:
                return f"DUE (respawned on {record.map_name})"
            if record.respawn_time > 0:
                remaining = max(0, record.respawn_time - time.time())
                return f"Respawning in {int(remaining/60)}m"
            return "Unknown"

    def get_mvp_summary(self) -> str:
        with self._lock:
            lines = [f"── MVP Tracker ──"]
            targets = self.update_hunt_targets()
            up = [t for t in targets if t.is_worth_hunting and any(
                r.is_up for r in self._records.values() if r.monster_id == t.monster_id
            )]
            due = [t for t in targets if t.is_worth_hunting and any(
                r.is_due for r in self._records.values() if r.monster_id == t.monster_id
            )]
            if up:
                lines.append(f"Currently UP: {', '.join(f'{t.monster_name}({t.map_name})' for t in up[:5])}")
            if due:
                lines.append(f"Due to respawn: {', '.join(f'{t.monster_name}({t.map_name})' for t in due[:5])}")
            if not up and not due:
                lines.append("No MVPs currently up or due")
            best = self.get_best_hunt_target()
            if best:
                lines.append(f"Best target: {best.monster_name} on {best.map_name} (value={best.estimated_value:,}z)")
                lines.append(f"  Recommended element: {best.recommended_element}")
            # Camp targets
            camp = self.get_camp_targets()
            if camp:
                lines.append(f"Camping: {', '.join(f'{t.monster_name}({t.map_name})' for t in camp[:3])}")
            # Pending loot
            pending = self.get_pending_loot()
            if pending:
                lines.append(f"Pending loot: {len(pending)} shares")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._hunt_targets.clear()
            self._loot_shares.clear()
            self._load_known_mvps()


# ── Global Singleton ──

_mvp_tracker: MVPTracker | None = None
_mvp_tracker_lock = RLock()


def get_mvp_tracker() -> MVPTracker:
    global _mvp_tracker
    with _mvp_tracker_lock:
        if _mvp_tracker is None:
            _mvp_tracker = MVPTracker()
        return _mvp_tracker
