"""Item Value Database — knows what items are actually worth farming vs vendor trash.

A pro player doesn't loot Jellopys (1z vendor trash). They know exactly which
items are worth picking up, which are worth keeping, and which are worth selling
to players vs NPCs. This module provides that knowledge using knowledge.json data.

Key metrics:
  - Market value: what players actually pay (estimated from buy price)
  - Vendor value: what NPCs pay (sell price)
  - Value density: value per weight unit (for inventory management)
  - Farmability: how farmable an item is (drop rate × monster density)
  - Category: card, material, consumable, equipment, etc.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

# Value thresholds (in zeny)
VENDOR_TRASH_THRESHOLD = 100  # Items worth less than this are vendor trash
LOW_VALUE_THRESHOLD = 1000  # Items worth 100-1000z are low value
MEDIUM_VALUE_THRESHOLD = 10000  # Items worth 1k-10k are medium
HIGH_VALUE_THRESHOLD = 100000  # Items worth 10k-100k are high
PREMIUM_THRESHOLD = 1000000  # Items worth 1M+ are premium

# Weight thresholds (for value density)
LIGHT_WEIGHT = 10  # Items <= 10 weight are light
MEDIUM_WEIGHT = 50  # Items <= 50 weight are medium
HEAVY_WEIGHT = 200  # Items <= 200 weight are heavy

# Drop rate thresholds (rate is in 0.01% units in knowledge.json)
COMMON_DROP_RATE = 1000  # 10%+
UNCOMMON_DROP_RATE = 100  # 1%+
RARE_DROP_RATE = 10  # 0.1%+
VERY_RARE_DROP_RATE = 1  # 0.01%+

# Value density thresholds (value per weight)
EXCELLENT_DENSITY = 1000  # 1000z per weight
GOOD_DENSITY = 100  # 100z per weight
POOR_DENSITY = 10  # 10z per weight


# ── Dataclasses ───────────────────────────────────────────────────────────


@dataclass
class ItemValuation:
    """Complete valuation for a single item."""
    name: str
    aegis_name: str
    buy_price: int  # NPC buy price (what you'd pay at NPC shop)
    sell_price: int  # NPC sell price (what NPC pays you)
    market_value: int  # Estimated player market value
    weight: int
    value_density: float  # market_value / weight (0 if weight=0)
    category: str  # Weapon, Armor, Card, Usable, Etc, Healing, etc.
    subcategory: str  # More specific type
    is_vendor_trash: bool  # True if worth < 100z
    is_valuable: bool  # True if worth farming
    is_premium: bool  # True if worth > 1M
    drop_rate_class: str  # common, uncommon, rare, very_rare
    typical_drop_rate: float  # 0.01% units
    farmability_score: float  # 0-100, how farmable
    recommendation: str  # sell_npc, sell_player, keep, craft, vendor_trash


@dataclass
class MonsterDropValue:
    """Value of a monster's drops."""
    monster_name: str
    monster_level: int
    monster_hp: int
    drops: list[dict[str, Any]]
    total_drop_value: float  # Expected zeny per kill from drops
    best_drop: str  # Name of the most valuable drop
    best_drop_value: float  # Expected value of best drop
    efficiency_score: float  # total_drop_value / monster_hp * 1000
    is_mvp: bool


# ── Item Value Database ───────────────────────────────────────────────────


@dataclass(slots=True)
class ItemValueDB:
    """Database of item valuations from knowledge.json.

    Thread-safe. Provides lookups, rankings, and farming recommendations.
    """

    _lock: RLock = field(default_factory=RLock)
    _items: dict[str, ItemValuation] = field(default_factory=dict)  # aegis_name -> valuation
    _items_by_name: dict[str, ItemValuation] = field(default_factory=dict)  # Name -> valuation
    _monsters: list[dict[str, Any]] = field(default_factory=list)
    _monster_drop_values: dict[str, MonsterDropValue] = field(default_factory=dict)
    _valuable_items: list[ItemValuation] = field(default_factory=list)
    _premium_items: list[ItemValuation] = field(default_factory=list)
    _vendor_trash_items: list[ItemValuation] = field(default_factory=list)
    _stats: dict[str, int] = field(default_factory=lambda: {"items_loaded": 0, "monsters_loaded": 0, "queries": 0})

    def __post_init__(self) -> None:
        self._load_knowledge()
        self._seed_market_prices()

    def _seed_market_prices(self) -> None:
        """Seed realistic market prices from market_seeder on top of knowledge.json data."""
        try:
            from ai_sidecar.economy.market_seeder import MarketSeeder
            _seeder = MarketSeeder()
            _prices = _seeder.seed_prices()
            _count = 0
            for _name, _price in _prices.items():
                _low = _name.lower()
                for _item in self._items.values():
                    if _item.name.lower() == _low or _item.aegis_name.lower() == _low:
                        _item.market_value = _price
                        _item.value_density = _price / max(_item.weight, 1)
                        _count += 1
                        break
            logger.info("market_prices_seeded: %d items updated from market_seeder", _count)
        except Exception as exc:
            logger.debug("market_seeder_skipped: %s", exc)

    def _load_knowledge(self) -> None:
        """Load knowledge.json and build item valuations."""
        for candidate in [
            str(Path(__file__).parent.parent.parent.parent / "knowledge" / "knowledge.json"),
            str(Path(__file__).parent.parent.parent / "knowledge" / "knowledge.json"),
            "knowledge/knowledge.json",
        ]:
            if candidate and Path(candidate).exists():
                try:
                    data = json.loads(Path(candidate).read_text(encoding="utf-8"))
                    self._process_items(data)
                    self._process_monsters(data)
                    logger.info(
                        "item_value_db_loaded: %d items, %d monsters",
                        self._stats["items_loaded"],
                        self._stats["monsters_loaded"],
                    )
                    return
                except Exception as exc:
                    logger.warning("item_value_db_load_failed: %s", exc)
                    continue
        logger.warning("item_value_db: knowledge.json not found")

    def _process_items(self, data: dict[str, Any]) -> None:
        """Process all items from knowledge.json."""
        all_items = data.get("items", {}).get("all", [])
        if not all_items:
            return

        for item in all_items:
            if not isinstance(item, dict):
                continue

            name = str(item.get("Name", "") or "")
            aegis_name = str(item.get("AegisName", "") or "")
            if not name and not aegis_name:
                continue

            buy_price = int(item.get("Buy", 0) or 0)
            sell_price = int(item.get("Sell", 0) or 0)
            weight = int(item.get("Weight", 0) or 0)
            category = str(item.get("Type", "") or "")
            subcategory = str(item.get("SubType", "") or "")

            # Market value estimation:
            # - If buy_price > 0, market value is typically 1.5-3x buy price
            # - If only sell_price > 0, market value is sell_price * 2
            # - Cards have their own valuation
            if category == "Card":
                market_value = self._estimate_card_value(name, buy_price)
            elif buy_price > 0:
                # Player market is typically 2-3x NPC buy for in-demand items
                market_value = buy_price * 2
            elif sell_price > 0:
                market_value = sell_price * 3
            else:
                market_value = 0

            # Value density
            value_density = market_value / max(weight, 1)

            # Classification
            is_vendor_trash = market_value < VENDOR_TRASH_THRESHOLD
            is_valuable = market_value >= MEDIUM_VALUE_THRESHOLD and not is_vendor_trash
            is_premium = market_value >= PREMIUM_THRESHOLD

            # Drop rate class (estimated from item type)
            drop_rate_class = self._estimate_drop_rate_class(category, market_value)

            # Farmability score (0-100)
            farmability_score = self._compute_farmability(
                market_value, weight, category, drop_rate_class
            )

            # Recommendation
            recommendation = self._recommend_action(
                market_value, weight, category, is_vendor_trash, is_premium
            )

            valuation = ItemValuation(
                name=name,
                aegis_name=aegis_name,
                buy_price=buy_price,
                sell_price=sell_price,
                market_value=market_value,
                weight=weight,
                value_density=round(value_density, 2),
                category=category,
                subcategory=subcategory,
                is_vendor_trash=is_vendor_trash,
                is_valuable=is_valuable,
                is_premium=is_premium,
                drop_rate_class=drop_rate_class,
                typical_drop_rate=0.0,
                farmability_score=farmability_score,
                recommendation=recommendation,
            )

            self._items[aegis_name] = valuation
            if name:
                self._items_by_name[name] = valuation

            if is_premium:
                self._premium_items.append(valuation)
            elif is_valuable:
                self._valuable_items.append(valuation)
            elif is_vendor_trash:
                self._vendor_trash_items.append(valuation)

        self._stats["items_loaded"] = len(self._items)

        # Sort valuable items by market value descending
        self._valuable_items.sort(key=lambda v: v.market_value, reverse=True)
        self._premium_items.sort(key=lambda v: v.market_value, reverse=True)

    def _process_monsters(self, data: dict[str, Any]) -> None:
        """Process monsters and compute drop values."""
        monsters = data.get("monsters", [])
        if not monsters:
            return

        self._monsters = monsters

        for mob in monsters:
            if not isinstance(mob, dict):
                continue

            mob_name = mob.get("Name", "") or ""
            if not mob_name:
                continue

            level = int(mob.get("Level", 0) or 0)
            hp = int(mob.get("Hp", 0) or 0)
            drops = mob.get("Drops", []) or []

            total_value = 0.0
            best_drop_name = ""
            best_drop_value = 0.0
            processed_drops = []

            for drop in drops:
                item_aegis = drop.get("Item", "")
                rate = float(drop.get("Rate", 0) or 0)

                valuation = self._items.get(item_aegis)
                if valuation:
                    market_val = valuation.market_value
                    expected_value = market_val * (rate / 10000.0)
                    total_value += expected_value

                    if expected_value > best_drop_value:
                        best_drop_value = expected_value
                        best_drop_name = valuation.name

                    processed_drops.append({
                        "item": valuation.name,
                        "aegis": item_aegis,
                        "rate": rate,
                        "market_value": market_val,
                        "expected_value": round(expected_value, 2),
                    })

            # Efficiency: expected zeny per kill relative to HP
            efficiency = (total_value / max(hp, 1)) * 1000 if hp > 0 else 0

            # Check if MVP (high HP relative to level)
            is_mvp = hp > 100000 and level > 30

            self._monster_drop_values[mob_name] = MonsterDropValue(
                monster_name=mob_name,
                monster_level=level,
                monster_hp=hp,
                drops=processed_drops,
                total_drop_value=round(total_value, 2),
                best_drop=best_drop_name,
                best_drop_value=round(best_drop_value, 2),
                efficiency_score=round(efficiency, 4),
                is_mvp=is_mvp,
            )

        self._stats["monsters_loaded"] = len(self._monster_drop_values)

    def _estimate_card_value(self, name: str, buy_price: int) -> int:
        """Estimate market value of a card based on name and typical prices."""
        # Cards are typically worth much more than their buy price
        # Use name-based heuristics
        name_lower = name.lower()

        # Premium cards (1M+)
        if any(kw in name_lower for kw in ["thara frog", "hydra", "ghostring", "deviling",
                                             "golden thief", "orc hero", "orc lord",
                                             "phreeoni", "maya", "moonlight", "edga",
                                             "turtle general", "drake", "stormy knight",
                                             "mistress", "doppelganger", "baphomet",
                                             "osiris", "gloom", "kiel", "thanatos"]):
            return 5000000

        # High value cards (500k-1M)
        if any(kw in name_lower for kw in ["marc", "vadon", "drainliar", "skeleton worker",
                                             "raydric", "pasana", "anolian", "sky pete",
                                             "alarm", "rideword", "bloody knight",
                                             "abysmal knight", "knight of abyss"]):
            return 800000

        # Medium value cards (100k-500k)
        if any(kw in name_lower for kw in ["savage", "mantis", "undead", "demon",
                                             "brute", "fish", "insect", "plant",
                                             "pecopeco", "hode", "zenorc", "archer skeleton",
                                             "soldier skeleton", "cramp", "caramel",
                                             "willow", "horn", "familiar", "poison spore",
                                             "spore", "chonchon", "condor", "picky",
                                             "lunatic", "poring", "drops", "poporing"]):
            return 200000

        # Generic card: use buy price * 10 as baseline
        if buy_price > 0:
            return buy_price * 10

        return 50000  # Default card value

    def _estimate_drop_rate_class(self, category: str, market_value: int) -> str:
        """Estimate drop rate class from item category and value."""
        if category == "Card":
            return "very_rare"
        if market_value > HIGH_VALUE_THRESHOLD:
            return "rare"
        if market_value > MEDIUM_VALUE_THRESHOLD:
            return "uncommon"
        return "common"

    def _compute_farmability(self, market_value: int, weight: int,
                              category: str, drop_rate_class: str) -> float:
        """Compute farmability score (0-100)."""
        score = 0.0

        # Value contribution (up to 40 points)
        if market_value > PREMIUM_THRESHOLD:
            score += 40
        elif market_value > HIGH_VALUE_THRESHOLD:
            score += 30
        elif market_value > MEDIUM_VALUE_THRESHOLD:
            score += 20
        elif market_value > LOW_VALUE_THRESHOLD:
            score += 10
        else:
            score += 2

        # Weight contribution (up to 20 points)
        if weight <= LIGHT_WEIGHT:
            score += 20
        elif weight <= MEDIUM_WEIGHT:
            score += 15
        elif weight <= HEAVY_WEIGHT:
            score += 10
        else:
            score += 2

        # Category contribution (up to 20 points)
        if category == "Card":
            score += 20  # Cards are always worth picking up
        elif category in ("Healing", "Usable"):
            score += 15  # Consumables have steady demand
        elif category in ("Weapon", "Armor"):
            score += 10
        else:
            score += 5

        # Drop rate contribution (up to 20 points)
        if drop_rate_class == "common":
            score += 20
        elif drop_rate_class == "uncommon":
            score += 15
        elif drop_rate_class == "rare":
            score += 10
        else:
            score += 5

        return min(100.0, score)

    def _recommend_action(self, market_value: int, weight: int,
                           category: str, is_vendor_trash: bool,
                           is_premium: bool) -> str:
        """Recommend what to do with an item."""
        if is_vendor_trash:
            return "vendor_trash"
        if is_premium:
            return "sell_player"
        if category == "Card":
            return "sell_player"
        if market_value > HIGH_VALUE_THRESHOLD:
            return "sell_player"
        if market_value > MEDIUM_VALUE_THRESHOLD and weight <= MEDIUM_WEIGHT:
            return "sell_player"
        if market_value > LOW_VALUE_THRESHOLD:
            return "sell_npc"
        return "vendor_trash"

    # ── Public API ─────────────────────────────────────────────────────

    def get_item(self, name_or_aegis: str) -> ItemValuation | None:
        """Get valuation for an item by name or AegisName."""
        with self._lock:
            self._stats["queries"] += 1
            # Try exact match first
            result = self._items.get(name_or_aegis)
            if result:
                return result
            result = self._items_by_name.get(name_or_aegis)
            if result:
                return result
            # Case-insensitive search
            lower = name_or_aegis.lower()
            for key, val in self._items_by_name.items():
                if key.lower() == lower:
                    return val
            for key, val in self._items.items():
                if key.lower() == lower:
                    return val
            return None

    def get_monster_drop_value(self, monster_name: str) -> MonsterDropValue | None:
        """Get drop value analysis for a monster."""
        with self._lock:
            self._stats["queries"] += 1
            result = self._monster_drop_values.get(monster_name)
            if result:
                return result
            # Case-insensitive
            lower = monster_name.lower()
            for key, val in self._monster_drop_values.items():
                if key.lower() == lower:
                    return val
            return None

    def get_valuable_items(self, min_value: int = MEDIUM_VALUE_THRESHOLD,
                            max_weight: int = 0) -> list[ItemValuation]:
        """Get items worth farming, optionally filtered by weight."""
        with self._lock:
            self._stats["queries"] += 1
            if max_weight > 0:
                return [v for v in self._valuable_items
                        if v.market_value >= min_value and v.weight <= max_weight]
            return [v for v in self._valuable_items if v.market_value >= min_value]

    def get_premium_items(self) -> list[ItemValuation]:
        """Get premium items (1M+ market value)."""
        with self._lock:
            self._stats["queries"] += 1
            return list(self._premium_items)

    def get_best_farming_targets(self, level: int, max_weight: int = 100,
                                  top_n: int = 20) -> list[dict[str, Any]]:
        """Get the best items to farm at a given level.

        Returns items sorted by farmability score, filtered by level-appropriate
        monsters that drop them.
        """
        with self._lock:
            self._stats["queries"] += 1
            targets: list[dict[str, Any]] = []

            # Find monsters near this level
            level_range = range(max(1, level - 10), level + 10)
            nearby_monsters = [
                m for m in self._monster_drop_values.values()
                if m.monster_level in level_range and not m.is_mvp
            ]

            # Score each monster by expected zeny per kill
            for mdv in nearby_monsters:
                if mdv.total_drop_value <= 0:
                    continue
                # Adjust for kill speed (lower HP = faster kills)
                kill_speed_factor = max(0.1, 1.0 - (mdv.monster_hp / 10000.0))
                adjusted_value = mdv.total_drop_value * kill_speed_factor

                targets.append({
                    "monster": mdv.monster_name,
                    "level": mdv.monster_level,
                    "hp": mdv.monster_hp,
                    "expected_zeny_per_kill": mdv.total_drop_value,
                    "adjusted_value": round(adjusted_value, 2),
                    "best_drop": mdv.best_drop,
                    "best_drop_value": mdv.best_drop_value,
                    "drop_count": len(mdv.drops),
                })

            targets.sort(key=lambda t: -t["adjusted_value"])
            return targets[:top_n]

    def get_best_monsters_for_item(self, item_name: str,
                                    top_n: int = 10) -> list[dict[str, Any]]:
        """Find which monsters drop a given item, sorted by drop rate."""
        with self._lock:
            self._stats["queries"] += 1
            # Resolve to aegis name
            valuation = self.get_item(item_name)
            if not valuation:
                return []

            target_aegis = valuation.aegis_name
            results: list[dict[str, Any]] = []

            for mdv in self._monster_drop_values.values():
                for drop in mdv.drops:
                    if drop.get("aegis") == target_aegis:
                        results.append({
                            "monster": mdv.monster_name,
                            "level": mdv.monster_level,
                            "hp": mdv.monster_hp,
                            "drop_rate": drop.get("rate", 0),
                            "expected_value": drop.get("expected_value", 0),
                            "is_mvp": mdv.is_mvp,
                        })
                        break

            results.sort(key=lambda r: -r["drop_rate"])
            return results[:top_n]

    def is_vendor_trash(self, item_name: str) -> bool:
        """Check if an item is vendor trash (not worth picking up)."""
        valuation = self.get_item(item_name)
        if valuation:
            return valuation.is_vendor_trash
        # Unknown items: assume not trash (better safe)
        return False

    def get_farming_recommendation(self, level: int, zeny: int,
                                    weight_capacity: int) -> dict[str, Any]:
        """Get a complete farming recommendation.

        Returns:
            Best items to farm, best monsters to kill, and what to do with loot.
        """
        with self._lock:
            self._stats["queries"] += 1

            # Best items to target
            targets = self.get_best_farming_targets(level, weight_capacity, 10)

            # Best monsters
            monsters = []
            for t in targets[:5]:
                monster_info = self._monster_drop_values.get(t["monster"])
                if monster_info:
                    monsters.append({
                        "name": monster_info.monster_name,
                        "level": monster_info.monster_level,
                        "hp": monster_info.monster_hp,
                        "expected_zeny_per_kill": monster_info.total_drop_value,
                        "efficiency": monster_info.efficiency_score,
                    })

            return {
                "level": level,
                "zeny": zeny,
                "weight_capacity": weight_capacity,
                "top_targets": targets[:5],
                "top_monsters": monsters[:5],
                "advice": self._generate_advice(level, zeny, targets),
            }

    def _generate_advice(self, level: int, zeny: int,
                          targets: list[dict[str, Any]]) -> str:
        """Generate human-readable farming advice."""
        if not targets:
            return "No profitable targets found for your level."

        best = targets[0]
        lines = [
            f"Best monster to farm: {best['monster']} (Lv{best['level']})",
            f"Expected zeny per kill: {best['expected_zeny_per_kill']:.0f}z",
            f"Best drop: {best['best_drop']}",
        ]

        # Check if player can afford potions for sustained farming
        if zeny < 5000:
            lines.append("Low on zeny — focus on low-HP monsters for quick cash")

        return " | ".join(lines)

    def get_item_value_summary(self, top_n: int = 20) -> str:
        """Get a formatted summary of top valuable items."""
        with self._lock:
            lines = ["── Item Value Database ──"]
            lines.append(f"Items loaded: {self._stats['items_loaded']}")
            lines.append(f"Monsters analyzed: {self._stats['monsters_loaded']}")
            lines.append("")

            lines.append("Top valuable items (by market value):")
            for i, v in enumerate(self._valuable_items[:top_n]):
                lines.append(
                    f"  {i+1}. {v.name}: {v.market_value:,}z "
                    f"(density={v.value_density:.0f}z/wt, "
                    f"farm={v.farmability_score:.0f}/100, "
                    f"action={v.recommendation})"
                )

            lines.append("")
            lines.append(f"Premium items (1M+): {len(self._premium_items)}")
            lines.append(f"Vendor trash items: {len(self._vendor_trash_items)}")

            return "\n".join(lines)

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Global Singleton ─────────────────────────────────────────────────────

_item_value_db: ItemValueDB | None = None
_item_value_db_lock = RLock()


def get_item_value_db() -> ItemValueDB:
    """Get the global ItemValueDB singleton."""
    global _item_value_db
    with _item_value_db_lock:
        if _item_value_db is None:
            _item_value_db = ItemValueDB()
        return _item_value_db
