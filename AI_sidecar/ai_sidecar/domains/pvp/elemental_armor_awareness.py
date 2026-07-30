"""Elemental Armor Awareness — learns player elements from damage observations.

Tracks which elements players use (from their armor, weapon, or attacks),
learns the most common element per player/class, and provides counters.

When a player uses Fire-element attacks, equipping Water armor gives
significant damage reduction. This module learns from observed damage types
and recommends optimal elemental armor.

Self-* properties:
  - Self-learning: builds element profiles from observed damage types
  - Self-optimizing: finds the best counter-element for each target
  - Self-adapting: adjusts when players swap elements mid-fight
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

ELEMENT_DECAY_ALPHA: float = 0.25
ELEMENT_WINDOW_SIZE: int = 20
MIN_OBSERVATIONS_FOR_ELEMENT: int = 2

# Element counter system: attacker_element -> defender_element that resists
ELEMENT_COUNTER: dict[str, str] = {
    "neutral": "neutral",
    "fire": "water",      # Water armor resists Fire
    "water": "wind",       # Wind armor resists Water
    "wind": "earth",       # Earth armor resists Wind
    "earth": "fire",       # Fire armor resists Earth
    "holy": "shadow",      # Shadow armor resists Holy
    "shadow": "holy",      # Holy armor resists Shadow
    "undead": "holy",      # Holy armor resists Undead
    "ghost": "neutral",    # Neutral armor partially resists Ghost
    "poison": "neutral",   # No hard counter, neutral is best
    "dark": "holy",        # Holy armor resists Dark
}

# Element damage bonus: attacker_element -> defender_element -> bonus (1.0 = normal)
# 0.25 = heavily resisted, 0.5 = resisted, 0.75 = slightly resisted
# 1.25 = slight weakness, 1.5 = weakness, 2.0 = heavy weakness
EFFECTIVENESS_MATRIX: dict[str, dict[str, float]] = {
    "neutral": {"neutral": 1.0, "fire": 1.0, "water": 1.0, "wind": 1.0, "earth": 1.0,
                "holy": 1.0, "shadow": 1.0, "undead": 1.0, "ghost": 0.75, "poison": 0.75},
    "fire":    {"neutral": 1.0, "fire": 0.25, "water": 0.5, "wind": 0.75, "earth": 1.5,
                "holy": 1.0, "shadow": 1.0, "undead": 1.0, "ghost": 0.75, "poison": 0.75},
    "water":   {"neutral": 1.0, "fire": 1.5, "water": 0.25, "wind": 0.5, "earth": 0.75,
                "holy": 1.0, "shadow": 1.0, "undead": 1.0, "ghost": 0.75, "poison": 0.75},
    "wind":    {"neutral": 1.0, "fire": 0.75, "water": 1.5, "wind": 0.25, "earth": 0.5,
                "holy": 1.0, "shadow": 1.0, "undead": 1.0, "ghost": 0.75, "poison": 0.75},
    "earth":   {"neutral": 1.0, "fire": 0.5, "water": 0.75, "wind": 1.5, "earth": 0.25,
                "holy": 1.0, "shadow": 1.0, "undead": 1.0, "ghost": 0.75, "poison": 0.75},
    "holy":    {"neutral": 1.0, "fire": 1.0, "water": 1.0, "wind": 1.0, "earth": 1.0,
                "holy": 1.0, "shadow": 2.0, "undead": 2.0, "ghost": 1.0, "poison": 1.0,
                "dark": 2.0},
    "shadow":  {"neutral": 1.0, "fire": 1.0, "water": 1.0, "wind": 1.0, "earth": 1.0,
                "holy": 0.5, "shadow": 0.25, "undead": 1.0, "ghost": 1.0, "poison": 1.0},
    "undead":  {"neutral": 1.0, "fire": 1.0, "water": 1.0, "wind": 1.0, "earth": 1.0,
                "holy": 2.0, "shadow": 0.5, "undead": 0.25, "ghost": 1.0, "poison": 1.0},
    "ghost":   {"neutral": 0.75, "fire": 1.0, "water": 1.0, "wind": 1.0, "earth": 1.0,
                "holy": 1.0, "shadow": 1.0, "undead": 1.0, "ghost": 1.0, "poison": 1.0},
    "poison":  {"neutral": 0.75, "fire": 1.0, "water": 1.0, "wind": 1.0, "earth": 1.0,
                "holy": 1.0, "shadow": 1.0, "undead": 1.0, "ghost": 1.0, "poison": 0.25},
    "dark":    {"neutral": 1.0, "fire": 1.0, "water": 1.0, "wind": 1.0, "earth": 1.0,
                "holy": 2.0, "shadow": 0.5, "undead": 1.0, "ghost": 1.0, "poison": 1.0},
}


@dataclass
class ElementObservation:
    """Observation of an element being used by a player."""
    player_name: str
    element: str
    damage_amount: int
    skill_name: str
    timestamp: float
    is_armor_element: bool  # Whether this is armor element (resisted damage)
    is_attack_element: bool  # Whether this is attack element (damage dealt)


@dataclass
class PlayerElementProfile:
    """Learned element profile for a player."""
    player_name: str
    player_class: str | None = None

    # Element observations with recency weighting
    # {element: decay_weighted_count}
    attack_elements: dict[str, float] = field(default_factory=dict)
    armor_elements: dict[str, float] = field(default_factory=dict)

    # Primary learned elements
    primary_attack_element: str = "neutral"
    primary_armor_element: str = "neutral"
    primary_attack_confidence: float = 0.0
    primary_armor_confidence: float = 0.0

    # Raw observations
    recent_observations: deque[ElementObservation] = field(
        default_factory=lambda: deque(maxlen=ELEMENT_WINDOW_SIZE)
    )

    # Total observations
    total_observations: int = 0
    last_seen: float = 0.0

    def record_attack_element(
        self,
        element: str,
        damage: int = 0,
        skill_name: str = "",
    ) -> None:
        """Record an attack element observation."""
        elem = element.lower().strip()
        current = self.attack_elements.get(elem, 0.0)
        # Weight by damage (bigger hits = more confidence)
        weight = 1.0 + min(1.0, damage / 10000.0)
        self.attack_elements[elem] = current + weight
        self._update_primary_attack()
        self.last_seen = time.time()

    def record_armor_element(
        self,
        element: str,
        damage_reduced: int = 0,
        skill_name: str = "",
    ) -> None:
        """Record an armor element observation (from damage reduction)."""
        elem = element.lower().strip()
        current = self.armor_elements.get(elem, 0.0)
        weight = 1.0 + min(1.0, damage_reduced / 10000.0)
        self.armor_elements[elem] = current + weight
        self._update_primary_armor()
        self.last_seen = time.time()

    def _update_primary_attack(self) -> None:
        """Update the primary attack element and confidence."""
        if not self.attack_elements:
            self.primary_attack_element = "neutral"
            self.primary_attack_confidence = 0.0
            return
        total = sum(self.attack_elements.values())
        best_elem = max(
            self.attack_elements.keys(),
            key=lambda k: self.attack_elements[k],
        )
        best_val = self.attack_elements[best_elem]
        self.primary_attack_element = best_elem
        self.primary_attack_confidence = min(1.0, best_val / total) if total > 0 else 0.0

    def _update_primary_armor(self) -> None:
        """Update the primary armor element and confidence."""
        if not self.armor_elements:
            self.primary_armor_element = "neutral"
            self.primary_armor_confidence = 0.0
            return
        total = sum(self.armor_elements.values())
        best_elem = max(
            self.armor_elements.keys(),
            key=lambda k: self.armor_elements[k],
        )
        best_val = self.armor_elements[best_elem]
        self.primary_armor_element = best_elem
        self.primary_armor_confidence = min(1.0, best_val / total) if total > 0 else 0.0

    def get_counter_armor(self) -> str:
        """Get the best armor element to counter this player's attack element.

        Returns:
            Element name to equip for best defense
        """
        return ELEMENT_COUNTER.get(self.primary_attack_element, "neutral")

    def get_counter_attack(self) -> str:
        """Get the best attack element to use against this player's armor.

        Returns:
            Element name that deals most damage to this player's armor
        """
        # Find element that is most effective against their armor
        armor = self.primary_armor_element
        best_elem = "neutral"
        best_mult = 0.0
        for attack_elem, multipliers in EFFECTIVENESS_MATRIX.items():
            mult = multipliers.get(armor, 1.0)
            if mult > best_mult:
                best_mult = mult
                best_elem = attack_elem
        return best_elem

    def get_damage_multiplier(self, attack_element: str) -> float:
        """Get expected damage multiplier when attacking this player.

        Args:
            attack_element: The element of our attack

        Returns:
            Damage multiplier (0.25 = heavily resisted, 2.0 = double damage)
        """
        return EFFECTIVENESS_MATRIX.get(
            attack_element.lower().strip(), {}
        ).get(self.primary_armor_element, 1.0)

    def get_stats(self) -> dict[str, Any]:
        return {
            "player": self.player_name,
            "class": self.player_class,
            "attack_element": self.primary_attack_element,
            "attack_confidence": round(self.primary_attack_confidence, 3),
            "armor_element": self.primary_armor_element,
            "armor_confidence": round(self.primary_armor_confidence, 3),
            "counter_armor": self.get_counter_armor(),
            "counter_attack": self.get_counter_attack(),
            "observations": self.total_observations,
        }


@dataclass
class ElementRecommendation:
    """Recommended element strategy against a specific target."""
    target_name: str
    recommend_attack_element: str
    recommend_armor_element: str
    attack_damage_mult: float
    armor_damage_reduction_mult: float
    confidence: float
    is_reliable: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target_name,
            "attack_element": self.recommend_attack_element,
            "armor_element": self.recommend_armor_element,
            "attack_mult": round(self.attack_damage_mult, 2),
            "armor_mult": round(self.armor_damage_reduction_mult, 2),
            "confidence": round(self.confidence, 3),
            "reliable": self.is_reliable,
        }


class ElementalArmorAwareness:
    """Tracks player elements and recommends counter-elements.

    Usage:
        elem = ElementalArmorAwareness()

        # When taking damage of a known element:
        elem.record_attack_observed("PlayerA", "fire", damage=2500)

        # When your damage seems reduced (target has armor):
        elem.record_armor_observed("PlayerA", "fire")

        # Get counter recommendation:
        rec = elem.get_counter_recommendation("PlayerA")
        # rec.recommend_armor_element == "water" (for fire attack)
        # rec.recommend_attack_element == "wind" (if PlayerA has water armor)
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # Per-player element profiles: {player_name: PlayerElementProfile}
        self._player_profiles: dict[str, PlayerElementProfile] = {}

        # Class-to-element priors (soft, overwhelmed by data)
        # Known class attack element tendencies
        self._class_element_priors: dict[str, dict[str, float]] = {
            "wizard": {"fire": 0.3, "water": 0.3, "wind": 0.2, "earth": 0.2},
            "high wizard": {"fire": 0.3, "water": 0.3, "wind": 0.2, "earth": 0.2},
            "warlock": {"fire": 0.3, "water": 0.3, "wind": 0.2, "earth": 0.2},
            "hunter": {"neutral": 0.6, "fire": 0.1, "water": 0.1, "wind": 0.1, "earth": 0.1},
            "sniper": {"neutral": 0.6, "fire": 0.1, "water": 0.1, "wind": 0.1, "earth": 0.1},
            "ranger": {"neutral": 0.6, "fire": 0.1, "water": 0.1, "wind": 0.1, "earth": 0.1},
            "assassin": {"neutral": 0.5, "poison": 0.3, "shadow": 0.2},
            "rogue": {"neutral": 0.6, "poison": 0.2, "shadow": 0.2},
            "stalker": {"neutral": 0.6, "poison": 0.2, "shadow": 0.2},
            "paladin": {"holy": 0.5, "neutral": 0.5},
            "crusader": {"holy": 0.4, "neutral": 0.6},
            "royal guard": {"holy": 0.5, "neutral": 0.5},
            "priest": {"holy": 0.7, "undead": 0.3},
            "high priest": {"holy": 0.7, "undead": 0.3},
            "arch bishop": {"holy": 0.7, "undead": 0.3},
            "monk": {"neutral": 0.5, "holy": 0.3, "shadow": 0.2},
            "champion": {"neutral": 0.5, "holy": 0.3, "shadow": 0.2},
            "sura": {"neutral": 0.4, "holy": 0.3, "shadow": 0.3},
        }

        # Stats
        self._total_observations: int = 0
        self._start_time: float = time.time()

    # ── Recording ───────────────────────────────────────────────────────

    def record_attack_observed(
        self,
        player_name: str,
        element: str,
        damage: int = 0,
        skill_name: str = "",
        player_class: str | None = None,
    ) -> None:
        """Record that a player used an attack of a specific element.

        Args:
            player_name: Name of the player
            element: Element seen (fire, water, wind, earth, holy, shadow, etc.)
            damage: Damage dealt (for confidence weighting)
            skill_name: Skill used (for tracking)
            player_class: Player's class (for prior info)
        """
        elem = element.lower().strip()
        if elem not in ELEMENT_COUNTER:
            logger.debug("Unknown element: %s (from %s)", elem, player_name)
            return

        with self._lock:
            profile = self._get_or_create_profile(player_name, player_class)
            profile.record_attack_element(elem, damage, skill_name)
            profile.total_observations += 1
            self._total_observations += 1

    def record_armor_observed(
        self,
        player_name: str,
        element: str,
        damage_reduced: int = 0,
        skill_name: str = "",
        player_class: str | None = None,
    ) -> None:
        """Record that a player's armor seems to be a certain element.

        This is inferred when your attacks deal reduced damage or
        when you see visual element effects on the player.

        Args:
            player_name: Name of the player
            element: Element observed on their armor
            damage_reduced: Estimated damage reduction amount
            skill_name: Skill used (for tracking)
            player_class: Player's class
        """
        elem = element.lower().strip()
        with self._lock:
            profile = self._get_or_create_profile(player_name, player_class)
            profile.record_armor_element(elem, damage_reduced, skill_name)
            profile.total_observations += 1
            self._total_observations += 1

    def _get_or_create_profile(
        self,
        player_name: str,
        player_class: str | None = None,
    ) -> PlayerElementProfile:
        """Get or create a profile for a player, applying class priors."""
        key = player_name.lower().strip()
        if key not in self._player_profiles:
            profile = PlayerElementProfile(
                player_name=player_name,
                player_class=player_class,
            )
            # Apply class priors
            if player_class:
                cls_key = player_class.lower().strip()
                priors = self._class_element_priors.get(cls_key, {})
                for elem, weight in priors.items():
                    if weight > 0:
                        profile.attack_elements[elem] = weight * 0.3  # Weak prior
            self._player_profiles[key] = profile
        else:
            # Update class if provided
            if player_class and self._player_profiles[key].player_class is None:
                self._player_profiles[key].player_class = player_class
        return self._player_profiles[key]

    # ── Query ───────────────────────────────────────────────────────────

    def get_counter_recommendation(
        self,
        player_name: str,
    ) -> ElementRecommendation:
        """Get element counter recommendation for a player.

        Args:
            player_name: Name of the player to counter

        Returns:
            ElementRecommendation with recommended attack and armor elements
        """
        key = player_name.lower().strip()

        with self._lock:
            profile = self._player_profiles.get(key)
            if profile is None:
                return ElementRecommendation(
                    target_name=player_name,
                    recommend_attack_element="neutral",
                    recommend_armor_element="neutral",
                    attack_damage_mult=1.0,
                    armor_damage_reduction_mult=1.0,
                    confidence=0.0,
                    is_reliable=False,
                    reason=f"No element data for {player_name}",
                )

            # Get counter elements
            counter_armor = profile.get_counter_armor()
            counter_attack = profile.get_counter_attack()

            # Calculate multipliers
            # If we equip counter_armor, how much does their attack damage us?
            attack_mult = EFFECTIVENESS_MATRIX.get(
                profile.primary_attack_element, {}
            ).get(counter_armor, 1.0)

            # If we use counter_attack, how much damage to them?
            defense_mult = EFFECTIVENESS_MATRIX.get(
                counter_attack, {}
            ).get(profile.primary_armor_element, 1.0)

            # Confidence
            confidence = max(
                profile.primary_attack_confidence,
                profile.primary_armor_confidence,
            )
            is_reliable = profile.total_observations >= MIN_OBSERVATIONS_FOR_ELEMENT

            # Build reason
            parts = []
            if profile.primary_attack_element != "neutral":
                parts.append(
                    f"{player_name} attacks with {profile.primary_attack_element}"
                )
            if profile.primary_armor_element != "neutral":
                parts.append(
                    f"{player_name} armor is {profile.primary_armor_element}"
                )
            if counter_armor != "neutral":
                parts.append(f"equip {counter_armor} armor to resist")
            if counter_attack != "neutral":
                parts.append(f"use {counter_attack} attacks for bonus damage")

            return ElementRecommendation(
                target_name=player_name,
                recommend_attack_element=counter_attack,
                recommend_armor_element=counter_armor,
                attack_damage_mult=round(attack_mult, 2),
                armor_damage_reduction_mult=round(defense_mult, 2),
                confidence=round(confidence, 3),
                is_reliable=is_reliable,
                reason="; ".join(parts) if parts else "no element data",
            )

    def get_best_element_against(self, player_name: str) -> str:
        """Quick helper: best element to attack with against this player.

        Args:
            player_name: Target player name

        Returns:
            Element name that deals most damage (e.g. 'water', 'wind')
        """
        rec = self.get_counter_recommendation(player_name)
        return rec.recommend_attack_element

    def get_best_armor_against(self, player_name: str) -> str:
        """Quick helper: best armor element against this player.

        Args:
            player_name: Target player name

        Returns:
            Element name to equip for defense
        """
        rec = self.get_counter_recommendation(player_name)
        return rec.recommend_armor_element

    # ── Introspection ───────────────────────────────────────────────────

    def get_player_profile(self, player_name: str) -> dict[str, Any] | None:
        """Get detailed element profile for a player."""
        key = player_name.lower().strip()
        with self._lock:
            profile = self._player_profiles.get(key)
            if profile is None:
                return None
            return profile.get_stats()

    def get_all_profiles(self) -> list[dict[str, Any]]:
        """Get element profiles for all tracked players."""
        with self._lock:
            return [
                p.get_stats()
                for p in sorted(
                    self._player_profiles.values(),
                    key=lambda x: x.total_observations,
                    reverse=True,
                )
            ]

    def get_stats(self) -> dict[str, Any]:
        """Summary statistics."""
        with self._lock:
            return {
                "players_tracked": len(self._player_profiles),
                "total_observations": self._total_observations,
                "known_attack_elements": list(
                    set(
                        p.primary_attack_element
                        for p in self._player_profiles.values()
                    )
                ),
                "known_armor_elements": list(
                    set(
                        p.primary_armor_element
                        for p in self._player_profiles.values()
                    )
                ),
            }
