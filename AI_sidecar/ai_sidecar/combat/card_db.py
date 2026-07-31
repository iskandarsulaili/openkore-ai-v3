"""
Card Database — Ragnarok Online weapon card definitions and bonus calculations.

Provides a CardDatabase singleton that stores known RO cards with their
race/element/size damage bonuses and computes stacked card multipliers
following official RO mechanics (diminishing returns for same-type cards).

RO Card Bonus Rules:
  - Same-type cards stack with diminishing returns:
    total_bonus = 1 + (c1 + c2*0.8 + c3*0.6 + c4*0.4)
    where c1-c4 are the individual card bonuses sorted descending.
  - Race/Element/Size bonuses are multiplicative with each other
  - Cards that modify ATK% are additive with other ATK% modifiers
  - All card bonuses apply to physical attacks only unless specified
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Final, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums for card bonus targets
# ---------------------------------------------------------------------------

class CardBonusType(str, enum.Enum):
    """The type of bonus a card provides."""
    RACE = "race"            # e.g. +20% to DemiHuman
    ELEMENT = "element"      # e.g. +20% to Water element
    SIZE = "size"            # e.g. +15% to Medium size
    ATK_PERCENT = "atk%"     # e.g. +5% ATK
    PHYSICAL_DAMAGE = "phys_dmg"  # e.g. +10% physical damage
    IGNORE_DEFENSE = "ignore_def"  # e.g. ignore 5% defense
    CRITICAL = "critical"    # e.g. +7 critical rate


class CardSlot(str, enum.Enum):
    """Where the card can be equipped."""
    WEAPON = "weapon"
    SHIELD = "shield"
    ARMOR = "armor"
    HEADGEAR = "headgear"
    GARMENT = "garment"
    SHOES = "shoes"
    ACCESSORY = "accessory"


# ---------------------------------------------------------------------------
# Card definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Card:
    """Definition of a single RO card.

    All bonus values are in percentage points (e.g. race_bonus=20 means +20%).
    """

    name: str
    slot: CardSlot = CardSlot.WEAPON

    # Race bonus: {race_name: percentage_add}
    race_bonus: dict[str, int] = field(default_factory=dict)

    # Element bonus: {element_name: percentage_add}
    element_bonus: dict[str, int] = field(default_factory=dict)

    # Size bonus: {size_name: percentage_add}
    size_bonus: dict[str, int] = field(default_factory=dict)

    # Flat ATK% modifier (additive with other ATK% cards/buffs)
    atk_percent: int = 0

    # Physical damage% modifier (multiplicative with race/element/size)
    phys_damage_percent: int = 0

    # Description for display
    description: str = ""

    @property
    def race_multiplier(self) -> float:
        """Get the race bonus as a multiplier (e.g. 1.20 for +20%)."""
        if not self.race_bonus:
            return 1.0
        total = sum(self.race_bonus.values())
        return 1.0 + (total / 100.0)

    @property
    def element_multiplier(self) -> float:
        """Get the element bonus as a multiplier."""
        if not self.element_bonus:
            return 1.0
        total = sum(self.element_bonus.values())
        return 1.0 + (total / 100.0)

    @property
    def size_multiplier(self) -> float:
        """Get the size bonus as a multiplier."""
        if not self.size_bonus:
            return 1.0
        total = sum(self.size_bonus.values())
        return 1.0 + (total / 100.0)


# ---------------------------------------------------------------------------
# Built-in card definitions
# ---------------------------------------------------------------------------

# Common weapon cards that provide race/element/size damage bonuses
# Values are based on official iRO / rAthena rates

CARDS: dict[str, Card] = {
    # ---- Race-specific cards (weapon) ----
    "Hydra Card": Card(
        name="Hydra Card",
        slot=CardSlot.WEAPON,
        race_bonus={"DemiHuman": 20},
        description="+20% damage to DemiHuman race monsters.",
    ),
    "Desert Wolf Card": Card(
        name="Desert Wolf Card",
        slot=CardSlot.WEAPON,
        race_bonus={"Brute": 20},
        description="+20% damage to Brute race monsters.",
    ),
    "Kukre Card": Card(
        name="Kukre Card",
        slot=CardSlot.WEAPON,
        race_bonus={"Fish": 20},
        description="+20% damage to Fish race monsters.",
    ),
    "Drainliar Card": Card(
        name="Drainliar Card",
        slot=CardSlot.WEAPON,
        race_bonus={"Brute": 20},
        description="+20% damage to Brute race monsters.",
    ),
    "Mandragora Card": Card(
        name="Mandragora Card",
        slot=CardSlot.WEAPON,
        race_bonus={"Plant": 20},
        description="+20% damage to Plant race monsters.",
    ),
    "Skeleton Worker Card": Card(
        name="Skeleton Worker Card",
        slot=CardSlot.WEAPON,
        size_bonus={"Medium": 15},
        description="+15% damage to Medium size monsters.",
    ),
    "Minor Behoth Card": Card(
        name="Minor Behoth Card",
        slot=CardSlot.WEAPON,
        size_bonus={"Large": 15},
        description="+15% damage to Large size monsters.",
    ),
    "Andre Card": Card(
        name="Andre Card",
        slot=CardSlot.WEAPON,
        race_bonus={"Insect": 20},
        description="+20% damage to Insect race monsters.",
    ),
    "Pecopeco Card": Card(
        name="Pecopeco Card",
        slot=CardSlot.WEAPON,
        race_bonus={"Formless": 20},
        description="+20% damage to Formless race monsters.",
    ),
    "Bathory Card": Card(
        name="Bathory Card",
        slot=CardSlot.WEAPON,
        race_bonus={"Demon": 20},
        description="+20% damage to Demon race monsters.",
    ),
    "Dokkaebi Card": Card(
        name="Dokkaebi Card",
        slot=CardSlot.WEAPON,
        race_bonus={"Angel": 20},
        description="+20% damage to Angel race monsters.",
    ),
    "Dragon Tamer Card": Card(
        name="Dragon Tamer Card",
        slot=CardSlot.WEAPON,
        race_bonus={"Dragon": 20},
        description="+20% damage to Dragon race monsters.",
    ),
    "Marina Card": Card(
        name="Marina Card",
        slot=CardSlot.WEAPON,
        race_bonus={"Fish": 10},
        description="+10% damage to Fish race monsters.",
    ),

    # ---- Element-specific cards (weapon) ----
    "Vadon Card": Card(
        name="Vadon Card",
        slot=CardSlot.WEAPON,
        element_bonus={"Water": 20},
        description="+20% damage to Water element monsters.",
    ),
    "Pasana Card": Card(
        name="Pasana Card",
        slot=CardSlot.WEAPON,
        element_bonus={"Fire": 20},
        description="+20% damage to Fire element monsters.",
    ),
    "Kaho Card": Card(
        name="Kaho Card",
        slot=CardSlot.WEAPON,
        element_bonus={"Fire": 20},
        description="+20% damage to Fire element monsters.",
    ),
    "Marduk Card": Card(
        name="Marduk Card",
        slot=CardSlot.WEAPON,
        element_bonus={"Wind": 20},
        description="+20% damage to Wind element monsters.",
    ),
    "Marc Card": Card(
        name="Marc Card",
        slot=CardSlot.WEAPON,
        element_bonus={"Water": 20},
        description="+20% damage to Water element monsters.",
    ),

    # ---- Size-specific cards ----
    "Ragged Zombie Card": Card(
        name="Ragged Zombie Card",
        slot=CardSlot.WEAPON,
        size_bonus={"Small": 15},
        description="+15% damage to Small size monsters.",
    ),

    # ---- ATK% cards ----
    "Abysmal Knight Card": Card(
        name="Abysmal Knight Card",
        slot=CardSlot.WEAPON,
        race_bonus={"DemiHuman": 15, "Brute": 10},
        atk_percent=5,
        description="+15% to DemiHuman, +10% to Brute, +5% ATK.",
    ),

    # ---- Physical damage cards ----
    "Turtle General Card": Card(
        name="Turtle General Card",
        slot=CardSlot.WEAPON,
        phys_damage_percent=20,
        atk_percent=20,
        description="+20% physical damage, +20% ATK.",
    ),
}


# ---------------------------------------------------------------------------
# CardDatabase class
# ---------------------------------------------------------------------------


class CardDatabase:
    """Thread-safe card database with correct RO stacking and multiplier calculation.

    Usage:
        db = CardDatabase()
        # Query cards
        hydra = db.get_card("Hydra Card")
        # Get combined multiplier for equipped cards against a monster
        mult = db.get_total_multiplier(
            cards=["Hydra Card", "Hydra Card", "Hydra Card", "Hydra Card"],
            race="DemiHuman", size="Large", element="Neutral",
        )
        # Returns 1.0 + (0.20 + 0.20*0.8 + 0.20*0.6 + 0.20*0.4) = 1.56
    """

    # Diminishing returns coefficients for card stacking
    # First card: 100%, second: 80%, third: 60%, fourth: 40%
    DIMINISHING_COEFFS = [1.0, 0.8, 0.6, 0.4]

    def __init__(self) -> None:
        self._lock = RLock()
        self._cards: dict[str, Card] = dict(CARDS)

    def get_card(self, name: str) -> Optional[Card]:
        """Look up a card by name."""
        with self._lock:
            return self._cards.get(name)

    def register_card(self, card: Card) -> None:
        """Register a custom card at runtime."""
        with self._lock:
            self._cards[card.name] = card

    def list_cards(self, slot: Optional[CardSlot] = None) -> list[Card]:
        """List all known cards, optionally filtered by slot."""
        with self._lock:
            cards = list(self._cards.values())
        if slot:
            return [c for c in cards if c.slot == slot]
        return cards

    # ------------------------------------------------------------------
    # Multiplier calculation with correct RO diminishing returns
    # ------------------------------------------------------------------

    def _calculate_stacked_bonus(self, bonuses: list[int]) -> float:
        """Calculate the total multiplier from a list of card bonuses with diminishing returns.

        RO formula: total = 1 + (b1*1.0 + b2*0.8 + b3*0.6 + b4*0.4)
        where b1-b4 are the individual card bonuses sorted descending.

        Args:
            bonuses: List of percentage bonuses (e.g. [20, 20, 20, 20])

        Returns:
            Combined multiplier (e.g. 1.56 for 4x +20% cards)
        """
        if not bonuses:
            return 1.0

        # Sort descending so the largest bonus gets the full coefficient
        sorted_bonuses = sorted(bonuses, reverse=True)

        total_pct = 0.0
        for i, bonus in enumerate(sorted_bonuses):
            if i < len(self.DIMINISHING_COEFFS):
                total_pct += bonus * self.DIMINISHING_COEFFS[i]
            else:
                # Beyond 4 cards, use 0.2 (very diminished)
                total_pct += bonus * 0.2

        return 1.0 + (total_pct / 100.0)

    def calculate_card_damage_bonus(
        self,
        cards: list[str],
        target_race: str,
        target_element: str,
        target_size: str,
    ) -> float:
        """Calculate the combined card damage bonus with correct RO diminishing returns.

        Each card type (race, element, size) is stacked independently with
        diminishing returns, then the three types are multiplied together.

        Args:
            cards: Card names equipped on weapon slots.
            target_race: Target monster's race (e.g. "DemiHuman").
            target_element: Target monster's element (e.g. "Water").
            target_size: Target monster's size (e.g. "Large").

        Returns:
            Combined card damage multiplier.
        """
        with self._lock:
            race_bonuses: list[int] = []
            element_bonuses: list[int] = []
            size_bonuses: list[int] = []
            atk_total = 0
            phys_total = 0

            for card_name in cards:
                card = self._cards.get(card_name)
                if card is None:
                    logger.warning("Unknown card: %s", card_name)
                    continue

                # Collect race bonuses that match
                if target_race in card.race_bonus:
                    race_bonuses.append(card.race_bonus[target_race])

                # Collect element bonuses that match
                if target_element in card.element_bonus:
                    element_bonuses.append(card.element_bonus[target_element])

                # Collect size bonuses that match
                if target_size in card.size_bonus:
                    size_bonuses.append(card.size_bonus[target_size])

                # ATK% adds up across cards (additive)
                atk_total += card.atk_percent

                # Physical damage% adds up across cards (additive)
                phys_total += card.phys_damage_percent

        # Apply diminishing returns to each bonus type independently
        race_mult = self._calculate_stacked_bonus(race_bonuses)
        element_mult = self._calculate_stacked_bonus(element_bonuses)
        size_mult = self._calculate_stacked_bonus(size_bonuses)

        # ATK% and phys_dmg% are additive across cards
        atk_mult = 1.0 + (atk_total / 100.0)
        phys_mult = 1.0 + (phys_total / 100.0)

        # Race * Element * Size * Phys_Dmg (multiplicative across types)
        # ATK% is a separate multiplier on top
        return race_mult * element_mult * size_mult * phys_mult * atk_mult

    def get_total_multiplier(
        self,
        cards: list[str],
        race: str,
        size: str,
        element: str,
    ) -> float:
        """Compute the combined card damage multiplier against a target.

        RO mechanics (corrected):
          - Same-type bonuses stack with diminishing returns:
            total = 1 + (c1*1.0 + c2*0.8 + c3*0.6 + c4*0.4)
          - Race, element, and size multipliers multiply each other
          - ATK% bonuses are additive with each other, then multiplicative
            with the race/element/size product

        Args:
            cards: Card names equipped on weapon slots.
            race: Target monster's race (e.g. "DemiHuman").
            size: Target monster's size (e.g. "Large").
            element: Target monster's element (e.g. "Water").

        Returns:
            Combined multiplier (e.g. 1.56 for 4x +20% cards).
        """
        return self.calculate_card_damage_bonus(cards, race, element, size)

    def get_card_multiplier_breakdown(
        self,
        cards: list[str],
        race: str,
        size: str,
        element: str,
    ) -> dict[str, float]:
        """Return a detailed breakdown of card multiplier components.

        Returns a dict with keys:
          race_mult, element_mult, size_mult, atk_mult, phys_mult, total
        """
        with self._lock:
            race_bonuses: list[int] = []
            element_bonuses: list[int] = []
            size_bonuses: list[int] = []
            atk_total = 0
            phys_total = 0

            for card_name in cards:
                card = self._cards.get(card_name)
                if card is None:
                    continue
                if race in card.race_bonus:
                    race_bonuses.append(card.race_bonus[race])
                if element in card.element_bonus:
                    element_bonuses.append(card.element_bonus[element])
                if size in card.size_bonus:
                    size_bonuses.append(card.size_bonus[size])
                atk_total += card.atk_percent
                phys_total += card.phys_damage_percent

        race_mult = self._calculate_stacked_bonus(race_bonuses)
        element_mult = self._calculate_stacked_bonus(element_bonuses)
        size_mult = self._calculate_stacked_bonus(size_bonuses)
        atk_mult = 1.0 + (atk_total / 100.0)
        phys_mult = 1.0 + (phys_total / 100.0)

        total = race_mult * element_mult * size_mult * phys_mult * atk_mult

        return {
            "race_mult": race_mult,
            "element_mult": element_mult,
            "size_mult": size_mult,
            "atk_mult": atk_mult,
            "phys_mult": phys_mult,
            "total": total,
        }


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_card_db_instance: Optional[CardDatabase] = None
_card_db_lock: Final[RLock] = RLock()


def get_card_database() -> CardDatabase:
    """Return the global CardDatabase singleton (thread-safe)."""
    global _card_db_instance  # noqa: PLW0603
    with _card_db_lock:
        if _card_db_instance is None:
            _card_db_instance = CardDatabase()
        return _card_db_instance


def get_total_card_multiplier(
    cards: list[str],
    race: str,
    size: str,
    element: str,
) -> float:
    """Convenience function: get combined card multiplier from global singleton."""
    return get_card_database().get_total_multiplier(cards, race, size, element)


def calculate_card_damage_bonus(
    cards: list[str],
    target_race: str,
    target_element: str,
    target_size: str,
) -> float:
    """Convenience function: calculate card damage bonus with diminishing returns."""
    return get_card_database().calculate_card_damage_bonus(cards, target_race, target_element, target_size)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _test_card_database() -> None:
    """Run a quick self-test of card multiplier calculations."""
    db = get_card_database()

    print("\n" + "=" * 60)
    print("Card Database Self-Test (Correct RO Stacking)")
    print("=" * 60)

    # Test 1: Single Hydra Card vs DemiHuman
    hydra = db.get_card("Hydra Card")
    assert hydra is not None, "Hydra Card should exist"
    print(f"\nHydra Card: {hydra.description}")
    print(f"  race_multiplier = {hydra.race_multiplier}")

    mult = db.get_total_multiplier(["Hydra Card"], "DemiHuman", "Medium", "Neutral")
    print(f"  vs DemiHuman: {mult:.4f}")
    assert abs(mult - 1.20) < 0.01, f"Expected 1.20, got {mult}"

    # Test 2: 4x Hydra with diminishing returns = 1 + (0.20 + 0.16 + 0.12 + 0.08) = 1.56
    mult4 = db.get_total_multiplier(
        ["Hydra Card", "Hydra Card", "Hydra Card", "Hydra Card"],
        "DemiHuman", "Medium", "Neutral",
    )
    print(f"\n4x Hydra Card vs DemiHuman: {mult4:.4f}")
    expected = 1.0 + (20 + 20*0.8 + 20*0.6 + 20*0.4) / 100.0
    assert abs(mult4 - expected) < 0.01, f"Expected {expected}, got {mult4}"
    print(f"  Correct! 1 + (20 + 16 + 12 + 8)/100 = {expected}")

    # Test 3: Hydra vs non-DemiHuman (no bonus)
    mult_no = db.get_total_multiplier(["Hydra Card"], "Brute", "Medium", "Neutral")
    print(f"\nHydra Card vs Brute: {mult_no:.4f} (should be 1.0)")
    assert abs(mult_no - 1.0) < 0.01, f"Expected 1.0, got {mult_no}"

    # Test 4: Skeleton Worker vs Medium
    mult_sw = db.get_total_multiplier(["Skeleton Worker Card"], "DemiHuman", "Medium", "Neutral")
    print(f"\nSkeleton Worker Card vs Medium: {mult_sw:.4f} (15% size)")
    assert abs(mult_sw - 1.15) < 0.01, f"Expected 1.15, got {mult_sw}"

    # Test 5: Hydra + Skeleton Worker = 1.20 * 1.15 = 1.38
    mult_comb = db.get_total_multiplier(
        ["Hydra Card", "Skeleton Worker Card"],
        "DemiHuman", "Medium", "Neutral",
    )
    print(f"\nHydra + Skeleton Worker vs DemiHuman/Medium: {mult_comb:.4f} (should be 1.38)")
    assert abs(mult_comb - 1.38) < 0.01, f"Expected 1.38, got {mult_comb}"

    # Test 6: 2x Hydra + 2x Vadon = race(20+16) * element(20+16) = 1.36 * 1.36 = 1.8496
    mult_mixed = db.get_total_multiplier(
        ["Hydra Card", "Hydra Card", "Vadon Card", "Vadon Card"],
        "DemiHuman", "Medium", "Water",
    )
    race_bonus = 1.0 + (20 + 20*0.8) / 100.0  # 1.36
    elem_bonus = 1.0 + (20 + 20*0.8) / 100.0   # 1.36
    expected_mixed = race_bonus * elem_bonus
    print(f"\n2x Hydra + 2x Vadon vs DemiHuman/Water: {mult_mixed:.4f}")
    print(f"  Race: {race_bonus:.4f}, Element: {elem_bonus:.4f}, Combined: {expected_mixed:.4f}")
    assert abs(mult_mixed - expected_mixed) < 0.01, f"Expected {expected_mixed}, got {mult_mixed}"

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    _test_card_database()
