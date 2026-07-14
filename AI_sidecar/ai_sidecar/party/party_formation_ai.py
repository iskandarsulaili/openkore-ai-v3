"""
Party Formation AI — scans for available players, evaluates their classes
and levels, forms optimal parties, and manages party composition dynamically.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class PartyCandidate:
    """A player who could join a party."""
    name: str
    job_class: str = ""
    base_level: int = 0
    is_available: bool = False
    is_in_party: bool = False
    distance: float = 0.0
    last_seen: float = 0.0
    trust_score: int = 50


@dataclass
class PartyComposition:
    """An optimal party composition."""
    role: str  # tank, healer, dps, support
    ideal_class: str = ""
    ideal_level_range: str = ""
    min_level: int = 0
    max_level: int = 99
    priority: int = 50


class PartyFormationAI:
    """Forms optimal parties based on available players."""

    # Optimal party compositions for different scenarios
    COMPOSITIONS: dict[str, list[PartyComposition]] = {
        "farming": [
            PartyComposition("tank", "Swordman/Knight/Crusader", "40-99", 40, 99, 90),
            PartyComposition("healer", "Acolyte/Priest", "40-99", 40, 99, 100),
            PartyComposition("dps", "Mage/Wizard", "40-99", 40, 99, 80),
            PartyComposition("dps", "Archer/Hunter", "40-99", 40, 99, 70),
            PartyComposition("support", "Acolyte/Priest", "40-99", 40, 99, 60),
        ],
        "mvp_hunting": [
            PartyComposition("tank", "Crusader/Paladin", "70-99", 70, 99, 100),
            PartyComposition("healer", "Priest/High Priest", "70-99", 70, 99, 100),
            PartyComposition("dps", "Wizard/High Wizard", "70-99", 70, 99, 90),
            PartyComposition("dps", "Assassin/Assassin Cross", "70-99", 70, 99, 80),
            PartyComposition("support", "Bard/Dancer", "70-99", 70, 99, 70),
        ],
        "leveling": [
            PartyComposition("dps", "Any", "1-99", 1, 99, 100),
            PartyComposition("healer", "Acolyte/Priest", "1-99", 1, 99, 80),
            PartyComposition("leecher", "Any", "1-99", 1, 99, 50),
        ],
    }

    def __init__(self) -> None:
        self._lock = RLock()
        self._candidates: dict[str, PartyCandidate] = {}
        self._current_party: list[str] = []
        self._enqueue_fn: Callable | None = None

    # ── Public API ──

    def observe_player(self, name: str, job_class: str = "", base_level: int = 0,
                       distance: float = 0.0) -> None:
        """Observe a potential party candidate."""
        with self._lock:
            if name not in self._candidates:
                self._candidates[name] = PartyCandidate(
                    name=name, job_class=job_class, base_level=base_level,
                    distance=distance, last_seen=time.time(),
                )
            else:
                c = self._candidates[name]
                c.job_class = job_class or c.job_class
                c.base_level = base_level or c.base_level
                c.distance = distance
                c.last_seen = time.time()

    def find_optimal_party(self, my_class: str, my_level: int,
                           scenario: str = "farming") -> list[str]:
        """Find the optimal party composition for a scenario."""
        with self._lock:
            composition = self.COMPOSITIONS.get(scenario, [])
            if not composition:
                return []

            # Score each candidate
            scored: list[tuple[float, str]] = []
            for name, candidate in self._candidates.items():
                if candidate.is_in_party or name in self._current_party:
                    continue
                if time.time() - candidate.last_seen > 300:
                    continue  # Stale data

                score = 0.0
                for role in composition:
                    if self._matches_role(candidate, role):
                        score += role.priority
                        # Bonus for level proximity
                        level_diff = abs(candidate.base_level - my_level)
                        if level_diff <= 10:
                            score += 20
                        elif level_diff <= 20:
                            score += 10
                        # Bonus for trust
                        score += candidate.trust_score / 10.0
                        break

                if score > 0:
                    scored.append((score, name))

            # Sort by score and pick best candidates
            scored.sort(key=lambda x: -x[0])
            party = [name for _, name in scored[:5]]
            return party

    def _matches_role(self, candidate: PartyCandidate, role: PartyComposition) -> bool:
        """Check if a candidate matches a role."""
        if role.ideal_class != "Any":
            # Check if candidate's class matches the role's ideal class family
            class_families = {
                "Swordman": ["Swordman", "Knight", "Crusader", "Lord Knight", "Paladin"],
                "Mage": ["Mage", "Wizard", "Sage", "High Wizard", "Professor"],
                "Archer": ["Archer", "Hunter", "Bard", "Dancer", "Sniper", "Clown", "Gypsy"],
                "Acolyte": ["Acolyte", "Priest", "Monk", "High Priest", "Champion"],
                "Thief": ["Thief", "Assassin", "Rogue", "Assassin Cross", "Stalker"],
                "Merchant": ["Merchant", "Blacksmith", "Alchemist", "Whitesmith", "Creator"],
            }
            for family, classes in class_families.items():
                if role.ideal_class.startswith(family) and candidate.job_class in classes:
                    return True
            return False

        # Level check
        if candidate.base_level < role.min_level or candidate.base_level > role.max_level:
            return False

        return True

    def form_party(self, members: list[str]) -> bool:
        """Form a party with the given members."""
        with self._lock:
            if not members:
                return False
            self._current_party = members
            if self._enqueue_fn:
                for member in members:
                    self._enqueue_fn("self", f"p Invite {member}")
            logger.info("party_formed: %s", ", ".join(members))
            return True

    def leave_party(self) -> None:
        with self._lock:
            self._current_party.clear()
            if self._enqueue_fn:
                self._enqueue_fn("self", "leave")

    def get_party_summary(self) -> str:
        with self._lock:
            lines = [f"── Party Formation AI ──"]
            lines.append(f"Current party: {', '.join(self._current_party) if self._current_party else 'none'}")
            lines.append(f"Candidates tracked: {len(self._candidates)}")
            available = [c for c in self._candidates.values() if not c.is_in_party and time.time() - c.last_seen < 300]
            if available:
                lines.append(f"Available: {', '.join(f'{c.name}(L{c.base_level} {c.job_class})' for c in available[:5])}")
            return "\n".join(lines)

    def set_enqueue_fn(self, fn: Callable) -> None:
        with self._lock:
            self._enqueue_fn = fn

    def reset(self) -> None:
        with self._lock:
            self._candidates.clear()
            self._current_party.clear()


# ── Global Singleton ──

_party_ai: PartyFormationAI | None = None
_party_ai_lock = RLock()


def get_party_formation_ai() -> PartyFormationAI:
    global _party_ai
    with _party_ai_lock:
        if _party_ai is None:
            _party_ai = PartyFormationAI()
        return _party_ai
