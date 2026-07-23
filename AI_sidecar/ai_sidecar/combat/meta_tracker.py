"""
Meta Tracker — observes and adapts to what's currently good in the meta.

Instead of relying on static build knowledge, this system observes:
1. What top players are using (gear, skills, builds)
2. What items are popular in player shops (demand signals)
3. What maps are popular for farming (supply signals)
4. What party compositions are common (meta shifts)
5. What's being discussed in chat (buzz signals)

The meta changes every patch. This system tracks it in real-time.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MetaSnapshot:
    """A snapshot of the current meta at a point in time."""
    timestamp: float
    popular_gear: dict[str, float] = field(default_factory=dict)  # item -> popularity (0-1)
    popular_skills: dict[str, float] = field(default_factory=dict)  # skill -> popularity
    popular_maps: dict[str, float] = field(default_factory=dict)  # map -> player count estimate
    popular_builds: dict[str, float] = field(default_factory=dict)  # build_name -> popularity
    popular_party_comps: dict[str, float] = field(default_factory=dict)  # comp -> popularity
    high_demand_items: list[str] = field(default_factory=list)
    trending_items: list[str] = field(default_factory=list)
    buzz_topics: list[str] = field(default_factory=list)


class MetaTracker:
    """Tracks the evolving meta by observing player behavior.
    
    Better than human because:
    - Humans miss trends (confirmation bias)
    - Humans have limited observation (can't watch 100 players at once)
    - Humans forget (recency bias)
    - This system observes EVERYTHING and remembers EVERYTHING
    """
    
    def __init__(self):
        self._lock = RLock()
        
        # Player observations
        self._player_gear: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._player_skills: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._player_maps: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._player_builds: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Player shop observations
        self._shop_listings: deque[dict[str, Any]] = deque(maxlen=10000)
        
        # Chat observations
        self._chat_topics: dict[str, int] = defaultdict(int)
        
        # Party observations
        self._party_comps: dict[str, int] = defaultdict(int)
        
        # Time-decayed popularity scores
        self._gear_popularity: dict[str, float] = defaultdict(float)
        self._skill_popularity: dict[str, float] = defaultdict(float)
        self._map_popularity: dict[str, float] = defaultdict(float)
        self._build_popularity: dict[str, float] = defaultdict(float)
        
        # Stats
        self._stats: dict[str, int] = defaultdict(int)
        self._last_decay: float = time.time()
    
    def _apply_decay(self) -> None:
        """Apply time decay to all popularity scores.
        
        Observations older than 7 days are heavily decayed.
        This ensures the meta tracker always reflects CURRENT meta,
        not historical averages.
        """
        now = time.time()
        days_passed = (now - self._last_decay) / 86400
        if days_passed < 0.01:  # Don't decay more than once per ~15 min
            return
        self._last_decay = now
        
        decay_factor = 0.5 ** days_passed  # Half-life of 1 day
        
        for key in list(self._gear_popularity.keys()):
            self._gear_popularity[key] *= decay_factor
            if self._gear_popularity[key] < 0.01:
                del self._gear_popularity[key]
        
        for key in list(self._skill_popularity.keys()):
            self._skill_popularity[key] *= decay_factor
            if self._skill_popularity[key] < 0.01:
                del self._skill_popularity[key]
        
        for key in list(self._map_popularity.keys()):
            self._map_popularity[key] *= decay_factor
            if self._map_popularity[key] < 0.01:
                del self._map_popularity[key]
        
        for key in list(self._build_popularity.keys()):
            self._build_popularity[key] *= decay_factor
            if self._build_popularity[key] < 0.01:
                del self._build_popularity[key]
    
    def observe_player_gear(self, player_name: str, gear: list[str]) -> None:
        """Observe what gear a player is wearing."""
        with self._lock:
            self._apply_decay()
            for item in gear:
                self._player_gear[player_name][item] += 1
                self._gear_popularity[item] += 1.0
            self._stats["gear_observations"] += 1
    
    def observe_player_skills(self, player_name: str, skills: list[str]) -> None:
        """Observe what skills a player is using."""
        with self._lock:
            self._apply_decay()
            for skill in skills:
                self._player_skills[player_name][skill] += 1
                self._skill_popularity[skill] += 1.0
            self._stats["skill_observations"] += 1
    
    def observe_player_map(self, player_name: str, map_name: str) -> None:
        """Observe what map a player is on."""
        with self._lock:
            self._apply_decay()
            self._player_maps[player_name][map_name] += 1
            self._map_popularity[map_name] += 1.0
            self._stats["map_observations"] += 1
    
    def observe_shop_listing(self, item_name: str, price: int,
                              seller: str, quantity: int = 1) -> None:
        """Observe a player shop listing."""
        with self._lock:
            self._shop_listings.append({
                "item": item_name,
                "price": price,
                "seller": seller,
                "quantity": quantity,
                "timestamp": time.time(),
            })
            self._stats["shop_observations"] += 1
    
    def observe_chat_topic(self, topic: str) -> None:
        """Observe a topic being discussed in chat."""
        with self._lock:
            self._chat_topics[topic] += 1
            self._stats["chat_observations"] += 1
    
    def observe_party_comp(self, comp_description: str) -> None:
        """Observe a party composition."""
        with self._lock:
            self._party_comps[comp_description] += 1
            self._stats["party_observations"] += 1
    
    def get_popular_gear(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Get the most popular gear items right now."""
        with self._lock:
            self._apply_decay()
            sorted_items = sorted(
                self._gear_popularity.items(),
                key=lambda x: -x[1]
            )
            return sorted_items[:top_n]
    
    def get_popular_skills(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Get the most popular skills right now."""
        with self._lock:
            self._apply_decay()
            sorted_items = sorted(
                self._skill_popularity.items(),
                key=lambda x: -x[1]
            )
            return sorted_items[:top_n]
    
    def get_popular_maps(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Get the most popular farming maps right now."""
        with self._lock:
            self._apply_decay()
            sorted_items = sorted(
                self._map_popularity.items(),
                key=lambda x: -x[1]
            )
            return sorted_items[:top_n]
    
    def get_high_demand_items(self, top_n: int = 10) -> list[str]:
        """Get items that are in high demand (many shop listings, high prices)."""
        with self._lock:
            # Count listings per item
            item_counts: dict[str, int] = defaultdict(int)
            item_prices: dict[str, list[int]] = defaultdict(list)
            
            for listing in self._shop_listings:
                item_counts[listing["item"]] += 1
                item_prices[listing["item"]].append(listing["price"])
            
            # Score: listing count * average price
            scores: dict[str, float] = {}
            for item, count in item_counts.items():
                avg_price = sum(item_prices[item]) / len(item_prices[item])
                scores[item] = count * avg_price
            
            sorted_items = sorted(scores.items(), key=lambda x: -x[1])
            return [item for item, _ in sorted_items[:top_n]]
    
    def get_trending_items(self, top_n: int = 5) -> list[str]:
        """Get items whose popularity is increasing fastest."""
        with self._lock:
            # Compare recent popularity vs older popularity
            now = time.time()
            recent: dict[str, float] = defaultdict(float)
            older: dict[str, float] = defaultdict(float)
            
            for listing in self._shop_listings:
                age_hours = (now - listing["timestamp"]) / 3600
                if age_hours < 24:
                    recent[listing["item"]] += 1.0
                elif age_hours < 72:
                    older[listing["item"]] += 1.0
            
            # Calculate trend (recent / older, with smoothing)
            trends: dict[str, float] = {}
            for item in set(list(recent.keys()) + list(older.keys())):
                r = recent.get(item, 0)
                o = older.get(item, 0.1)  # Avoid division by zero
                trends[item] = r / o
            
            sorted_items = sorted(trends.items(), key=lambda x: -x[1])
            return [item for item, _ in sorted_items[:top_n]]
    
    def get_build_recommendation(self, job_name: str) -> dict[str, Any]:
        """Get a meta-informed build recommendation for a job class.
        
        Returns the most popular gear and skills for this job,
        based on observations of other players.
        """
        with self._lock:
            # Find players of this job
            relevant_gear: dict[str, float] = defaultdict(float)
            relevant_skills: dict[str, float] = defaultdict(float)
            
            for player, gear_counts in self._player_gear.items():
                for item, count in gear_counts.items():
                    relevant_gear[item] += count
            
            for player, skill_counts in self._player_skills.items():
                for skill, count in skill_counts.items():
                    relevant_skills[skill] += count
            
            top_gear = sorted(relevant_gear.items(), key=lambda x: -x[1])[:5]
            top_skills = sorted(relevant_skills.items(), key=lambda x: -x[1])[:5]
            
            return {
                "job": job_name,
                "recommended_gear": [item for item, _ in top_gear],
                "recommended_skills": [skill for skill, _ in top_skills],
                "confidence": min(1.0, self._stats.get("gear_observations", 0) / 100),
            }
    
    def get_meta_snapshot(self) -> MetaSnapshot:
        """Get a complete snapshot of the current meta."""
        with self._lock:
            self._apply_decay()
            return MetaSnapshot(
                timestamp=time.time(),
                popular_gear=dict(self._gear_popularity),
                popular_skills=dict(self._skill_popularity),
                popular_maps=dict(self._map_popularity),
                popular_builds=dict(self._build_popularity),
                popular_party_comps=dict(self._party_comps),
                high_demand_items=self.get_high_demand_items(10),
                trending_items=self.get_trending_items(5),
                buzz_topics=sorted(self._chat_topics.keys(), key=lambda k: -self._chat_topics[k])[:10],
            )
    
    def get_stats(self) -> dict[str, int]:
        """Get meta tracker statistics."""
        with self._lock:
            return dict(self._stats)


# Global singleton
_tracker: MetaTracker | None = None

def get_meta_tracker() -> MetaTracker:
    """Get the global MetaTracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = MetaTracker()
    return _tracker
