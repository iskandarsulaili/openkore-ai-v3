"""Loot Discipline Engine — designated looter, pickup filters, coordination.

In a party, everyone grabbing everything causes chaos. A real party:
- Designates one primary looter (usually tank or highest STR)
- Sets pickup filters per member (cards only for designated looter)
- Coordinates item vacuuming (Rogue zips around collecting everything)
"""
from __future__ import annotations
from typing import Any
import logging
from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class LootDisciplineEngine:
    """Manages loot distribution in a party.
    
    Key rules:
    - Cards: only designated looter picks up
    - Valuable equipment: only designated looter picks up
    - Consumables/ammo: anyone can grab, but prefer looter
    - Junk: anyone grabs, sell later
    """
    
    # Item classifications for loot priority
    LOOT_PRIORITIES = {
        "card": 1,           # Highest priority — designated looter only
        "equipment": 2,      # Valuable — designated looter preferred
        "consumable": 3,     # Potions/food — anyone can grab
        "ammo": 4,           # Arrows/traps — anyone can grab
        "material": 5,       # Crafting — designated looter for even split
        "junk": 6,           # Vendor trash — anyone can grab
    }
    
    def __init__(self):
        self._designated_looters: dict[str, str] = {}  # party_id -> bot_id
    
    def get_designated_looter(self, party_id: str, members: list[dict]) -> str:
        """Get the designated looter for a party.
        Prefers: Tank > Rogue > Highest STR > First member.
        """
        if party_id in self._designated_looters:
            return self._designated_looters[party_id]
        
        # Role priority for looting
        for role_order in ["tank", "rogue", "thief", "melee", "ranged", "magic", "support"]:
            for member in members:
                if isinstance(member, dict):
                    job = str(member.get("job", member.get("name", "")) or "").lower()
                    if role_order in job:
                        looter = member.get("id", member.get("name", members[0].get("id", "")))
                        self._designated_looters[party_id] = looter
                        return looter
        
        # Fallback to first member
        if members:
            first = members[0]
            lid = first.get("id", first.get("name", ""))
            self._designated_looters[party_id] = lid
            return lid
        return ""
    
    def classify_item(self, item_name: str) -> str:
        """Classify an item by loot priority."""
        name_lower = item_name.lower()
        if "card" in name_lower:
            return "card"
        if any(w in name_lower for w in ["_potion", "_juice", "_food", "_berry"]):
            return "consumable"
        if any(w in name_lower for w in ["arrow", "trap", "bullet"]):
            return "ammo"
        if any(w in name_lower for w in ["_hammer", "_anvil", "_dust", "elunium", "oridecon", "rough"]):
            return "material"
        if any(w in name_lower for w in ["sword", "staff", "bow", "mace", "knife", "dagger", "shield", "armor", "boots", "muffler", "robe", "hat", "helm", "goggles"]):
            return "equipment"
        return "junk"
    
    def should_pickup(self, item_name: str, bot_id: str, party_members: list[dict], party_id: str = "") -> bool:
        """Check if this bot should pick up an item."""
        item_class = self.classify_item(item_name)
        looter = self.get_designated_looter(party_id, party_members) if party_members else bot_id
        
        if item_class == "card" or item_class == "equipment":
            # Only designated looter picks up
            return bot_id == looter
        
        if item_class == "material":
            # Designated looter preferred, but anyone can grab
            if bot_id == looter:
                return True
            # If looter is far away or hasn't picked up in 5s, anyone can grab
            return True  # Default: anyone can grab materials
        
        # Consumables, ammo, junk: anyone can grab
        return True
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        """Run loot discipline assessment."""
        party = signals.get("party", {}) or {}
        party_members = party.get("members", []) if isinstance(party, dict) else []
        party_id = str(party.get("id", party.get("name", "")) if isinstance(party, dict) else "")
        
        if len(party_members) < 2:
            return
        
        # Determine designated looter
        looter = self.get_designated_looter(party_id, party_members)
        
        # Log loot discipline setup
        actions.append(HeuristicAction(
            kind="command",
            command=f"loot_designate {looter}",
            confidence=0.8,
            reason=f"Loot discipline: {looter} is primary looter for party {party_id}",
            domain="party",
        ))
        
        # If this bot is the looter, higher pickup priority
        if bot_id == looter:
            actions.append(HeuristicAction(
                kind="command",
                command="itemsTakeAuto 2",  # Aggressive pickup
                confidence=0.9,
                reason=f"Designated looter: aggressive pickup mode",
                domain="party",
            ))
        else:
            actions.append(HeuristicAction(
                kind="command",
                command="itemsTakeAuto 0",  # Only grab what's nearby
                confidence=0.7,
                reason=f"Non-looter: passive pickup mode",
                domain="party",
            ))


class EventDetector:
    """Detects server events and redirects farming.
    
    rAthena servers run events:
    - Valentine's Day: double drops on certain maps
    - Easter: egg hunt NPC
    - Halloween: special monsters
    - Christmas: gift quests
    - WoE: castle siege
    - Double EXP weekends
    """
    
    EVENTS = {
        "valentines": {"months": [2], "maps": ["prt_fild08", "pay_fild11"], "description": "Valentine's Day event"},
        "easter": {"months": [3, 4], "maps": ["izlude"], "description": "Easter egg hunt"},
        "halloween": {"months": [10], "maps": ["gef_dun00", "gef_dun01"], "description": "Halloween event"},
        "christmas": {"months": [12], "maps": ["xmas", "xmas_dun01"], "description": "Christmas event"},
        "woe": {"days": [2, 3, 5], "hours": [12, 13, 14], "description": "War of Emperium"},  # Wed,Thu,Sat 20-22 UTC+8
    }
    
    @staticmethod
    def get_active_events() -> list[dict]:
        """Get currently active server events."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        active = []
        
        for name, config in EventDetector.EVENTS.items():
            # Check month
            months = config.get("months", [])
            days = config.get("days", [])
            hours = config.get("hours", [])
            
            if months and now.month not in months:
                continue
            if days and now.weekday() not in days:
                continue
            if hours and now.hour not in hours:
                continue
            
            active.append({"name": name, **config})
        
        return active
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        active = self.get_active_events()
        
        if active:
            for event in active:
                actions.append(HeuristicAction(
                    kind="log",
                    command=f"event_active {event['name']} maps={event.get('maps', [])}",
                    confidence=0.8,
                    reason=f"Server event active: {event['description']}",
                    domain="world",
                ))
                
                # Redirect farming to event maps if we're not on one
                current_map = str(signals.get("map", "") or "")
                event_maps = event.get("maps", [])
                if event_maps and current_map not in event_maps:
                    actions.append(HeuristicAction(
                        kind="command",
                        command=f"move_to_event {event_maps[0]}",
                        confidence=0.6,
                        reason=f"Event active: redirecting to {event_maps[0]} for {event['name']}",
                        domain="navigation",
                    ))


class LiveMarketScanner:
    """Scans player shops for dynamic pricing.
    
    Rather than relying on static YAML prices, this module:
    - Scans player vending in town
    - Tracks buy/sell prices per item
    - Adjusts fair market value based on actual trades
    - Detects arbitrage opportunities (buy low, sell high)
    """
    
    def __init__(self):
        self._market_data: dict[str, dict] = {}  # item_name -> {buy, sell, spread, trend}
    
    def record_trade(self, item_name: str, price: int, is_buy: bool) -> None:
        """Record a player trade to update market data."""
        if item_name not in self._market_data:
            self._market_data[item_name] = {
                "buys": [],
                "sells": [],
                "last_price": price,
                "volume": 0,
            }
        
        entry = self._market_data[item_name]
        if is_buy:
            entry["buys"].append(price)
        else:
            entry["sells"].append(price)
        entry["last_price"] = price
        entry["volume"] += 1
        
        # Keep only last 10 trades
        if len(entry["buys"]) > 10:
            entry["buys"] = entry["buys"][-10:]
        if len(entry["sells"]) > 10:
            entry["sells"] = entry["sells"][-10:]
    
    def get_fair_price(self, item_name: str) -> dict:
        """Get fair market price based on observed trades.
        
        Returns {buy, sell, spread, confidence}.
        Falls back to static YAML if no trades observed.
        """
        try:
            from ai_sidecar.domains.economy.database import MarketPriceDB
            static = MarketPriceDB.get_market_price(item_name)
        except Exception:
            static = {}
        
        entry = self._market_data.get(item_name)
        if not entry or not entry["buys"] or not entry["sells"]:
            # No trades observed — use static prices
            return {
                "buy": static.get("market_buy", 0),
                "sell": static.get("market_sell", 0),
                "confidence": 0.3,
                "source": "static",
            }
        
        avg_buy = sum(entry["buys"]) / len(entry["buys"])
        avg_sell = sum(entry["sells"]) / len(entry["sells"])
        confidence = min(0.9, entry["volume"] / 10.0)
        
        return {
            "buy": int(avg_buy),
            "sell": int(avg_sell),
            "spread": int(avg_sell - avg_buy),
            "confidence": confidence,
            "source": "live",
            "volume": entry["volume"],
        }
    
    def assess(self, signals: dict[str, Any], actions: list[HeuristicAction], bot_id: str) -> None:
        """Scan for arbitrage opportunities."""
        player_shops = signals.get("player_shops", signals.get("vendors", [])) or []
        in_town = bool(signals.get("in_town", False))
        
        if in_town and player_shops:
            # Report market activity
            actions.append(HeuristicAction(
                kind="log",
                command=f"market_shops {len(player_shops)} vendors in town",
                confidence=0.5,
                reason=f"Market: {len(player_shops)} player shops detected",
                domain="economy",
            ))
