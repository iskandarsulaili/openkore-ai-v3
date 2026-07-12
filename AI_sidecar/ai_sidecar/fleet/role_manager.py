"""Role manager for the fleet — auto-assigns, tracks, and expires bot roles.

Auto-assigns roles based on Ragnarok Online job class, supports manual
overrides, role expiry via TTL, thread-safe registry access, and
re-evaluation when class or level changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Any

# ---------------------------------------------------------------------------
# RO job-class → role definitions
# ---------------------------------------------------------------------------

# Priority order: the first matching role in this list becomes the primary role.
# When a class maps to multiple roles (e.g. soul_linker → Healer + Buffer),
# the earlier entry wins so it gets the most semantically appropriate role.
_ROLE_DEFINITIONS: list[tuple[str, tuple[str, ...]]] = [
    ("tank",       ("swordman", "knight", "paladin", "crusader",
                     "lord_knight", "rune_knight", "dragon_knight", "imperial_guard")),
    ("healer",     ("acolyte", "priest", "high_priest", "arch_bishop",
                     "cardinal", "soul_linker")),
    ("dps_melee",  ("thief", "assassin", "assassin_cross", "guillotine_cross",
                     "shadow_cross", "monk", "champion", "shura",
                     "star_gladiator", "taekwon")),
    ("dps_ranged", ("archer", "hunter", "sniper", "ranger", "windhawk",
                     "bard", "dancer", "minstrel", "gypsy", "gunslinger")),
    ("dps_magic",  ("mage", "wizard", "high_wizard", "arch_mage",
                     "sage", "professor", "sorcerer", "elemental_master", "warlock")),
    ("support",    ("merchant", "alchemist", "creator", "genetic", "biolo",
                     "blacksmith", "whitesmith", "meister")),
    ("debuffer",   ("rogue", "stalker", "shadow_chaser",
                     "ninja", "kagerou", "oboro")),
    ("buffer",     ("soul_linker", "bard", "dancer", "minstrel",
                     "gypsy", "star_gladiator")),
]

# Build reverse lookup: class_name → primary role (first match wins)
_CLASS_TO_ROLE: dict[str, str] = {}
for role, classes in _ROLE_DEFINITIONS:
    for cls in classes:
        if cls not in _CLASS_TO_ROLE:  # first definition wins
            _CLASS_TO_ROLE[cls] = role

# Build secondary-role lookup for classes that appear in multiple definitions
_SECONDARY_ROLES: dict[str, list[str]] = {}
for role, classes in _ROLE_DEFINITIONS:
    for cls in classes:
        if cls in _CLASS_TO_ROLE and _CLASS_TO_ROLE[cls] != role:
            _SECONDARY_ROLES.setdefault(cls, []).append(role)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RoleInfo:
    """Runtime state for a single bot's role assignment."""
    bot_id: str
    role: str | None = None
    confidence: float = 0.0
    source: str = "auto"
    job_class: str = "novice"
    level: int = 1
    expires_at: datetime | None = None
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class LegacyRoleManager:
    """Legacy single-bot role manager — preserved for lifecycle fleet coordinator.
    
    Old API: RoleManager(bot_id=bot_id) with update() and current() methods.
    """
    bot_id: str
    role: str | None = None
    confidence: float = 0.0
    expires_at: datetime | None = None
    source: str = "local"

    def update(self, *, role: str | None, confidence: float, ttl_seconds: int, source: str) -> None:
        self.role = role
        self.confidence = float(confidence)
        self.expires_at = datetime.now(UTC) + timedelta(seconds=max(5, int(ttl_seconds)))
        self.source = source

    def current(self) -> dict[str, object]:
        now = datetime.now(UTC)
        expired = self.expires_at is not None and self.expires_at <= now
        if expired:
            return {"role": None, "confidence": 0.0, "expires_at": self.expires_at, "source": self.source}
        return {"role": self.role, "confidence": self.confidence, "expires_at": self.expires_at, "source": self.source}


# ---------------------------------------------------------------------------
# Role manager
# ---------------------------------------------------------------------------


class RoleManager:
    """Thread-safe registry that assigns, tracks, and expires bot roles.

    Usage
    -----
    >>> mgr = RoleManager()
    >>> mgr.register_bot("bot_01", "knight")
    >>> mgr.get_role("bot_01")
    {"role": "tank", "confidence": 0.8, "source": "auto",
     "class": "knight", "level": 1, "secondary_roles": []}
    >>> mgr.update_role("bot_01", "offtank", confidence=0.9, ttl_seconds=120)
    >>> mgr.re_evaluate("bot_01")  # reset to class-based role
    """

    def __init__(self) -> None:
        self._registry: dict[str, RoleInfo] = {}
        self._lock = RLock()

    # -- public API ---------------------------------------------------------

    def register_bot(
        self,
        bot_id: str,
        class_name: str,
        level: int = 1,
        ttl_seconds: int = 300,
    ) -> dict[str, Any]:
        """Register (or re-register) a bot and auto-assign its role.

        Parameters
        ----------
        bot_id : str
            Unique bot identifier.
        class_name : str
            RO job class name (case-insensitive, underscore-separated).
        level : int
            Current bot level (default 1).
        ttl_seconds : int
            Seconds until the auto-assigned role expires (default 300).

        Returns
        -------
        dict
            The bot's current role snapshot (see ``get_role``).
        """
        normalised_class = class_name.strip().lower().replace(" ", "_")
        role, confidence = self._class_to_role(normalised_class)

        info = RoleInfo(
            bot_id=bot_id,
            role=role,
            confidence=confidence,
            source="auto",
            job_class=normalised_class,
            level=level,
            expires_at=(
                datetime.now(UTC) + timedelta(seconds=max(5, int(ttl_seconds)))
                if role is not None
                else None
            ),
        )

        with self._lock:
            self._registry[bot_id] = info

        return self._snapshot(info)

    def update_role(
        self,
        bot_id: str,
        role: str | None,
        confidence: float = 0.9,
        ttl_seconds: int = 300,
        source: str = "manual",
    ) -> dict[str, Any] | None:
        """Manually override the role for a registered bot.

        Returns the updated snapshot, or ``None`` if the bot is unknown.
        """
        with self._lock:
            info = self._registry.get(bot_id)
            if info is None:
                return None

            info.role = role
            info.confidence = float(confidence)
            info.source = source
            info.expires_at = (
                datetime.now(UTC) + timedelta(seconds=max(5, int(ttl_seconds)))
                if role is not None
                else None
            )
            return self._snapshot(info)

    def get_role(self, bot_id: str) -> dict[str, Any] | None:
        """Return the current role snapshot for a bot, respecting expiry.

        If the role has expired, returns the snapshot with ``role = None``,
        ``confidence = 0.0``, and ``expired = True``.
        """
        with self._lock:
            info = self._registry.get(bot_id)
            if info is None:
                return None
            return self._snapshot(info)

    def list_roles(self) -> dict[str, dict[str, Any]]:
        """Return a snapshot of every registered bot and its (non-expired) role."""
        with self._lock:
            return {
                bid: self._snapshot(info)
                for bid, info in self._registry.items()
            }

    def re_evaluate(self, bot_id: str) -> dict[str, Any] | None:
        """Re-assign the role based on the bot's stored job class.

        Discards any manual override and re-applies auto-assignment.
        Returns the updated snapshot, or ``None`` if the bot is unknown.
        """
        with self._lock:
            info = self._registry.get(bot_id)
            if info is None:
                return None

            role, confidence = self._class_to_role(info.job_class)
            info.role = role
            info.confidence = confidence
            info.source = "auto"
            info.expires_at = (
                datetime.now(UTC) + timedelta(seconds=300)
                if role is not None
                else None
            )
            return self._snapshot(info)

    def remove_bot(self, bot_id: str) -> bool:
        """Remove a bot from the registry.  Returns ``True`` if it existed."""
        with self._lock:
            if bot_id in self._registry:
                del self._registry[bot_id]
                return True
            return False

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _class_to_role(class_name: str) -> tuple[str | None, float]:
        """Map a normalised RO job class to (role, confidence).

        Known classes get confidence 0.8; unknown classes return
        ``(None, 0.0)``.
        """
        role = _CLASS_TO_ROLE.get(class_name)
        if role is not None:
            return role, 0.8
        return None, 0.0

    def _snapshot(self, info: RoleInfo) -> dict[str, Any]:
        """Build a dict snapshot of a RoleInfo, applying expiry logic."""
        now = datetime.now(UTC)
        expired = info.expires_at is not None and info.expires_at <= now

        secondary = _SECONDARY_ROLES.get(info.job_class, [])

        if expired:
            return {
                "bot_id": info.bot_id,
                "role": None,
                "confidence": 0.0,
                "source": info.source,
                "class": info.job_class,
                "level": info.level,
                "secondary_roles": secondary,
                "expired": True,
                "expires_at": info.expires_at.isoformat() if info.expires_at else None,
            }

        return {
            "bot_id": info.bot_id,
            "role": info.role,
            "confidence": info.confidence,
            "source": info.source,
            "class": info.job_class,
            "level": info.level,
            "secondary_roles": secondary,
            "expired": False,
            "expires_at": info.expires_at.isoformat() if info.expires_at else None,
        }
