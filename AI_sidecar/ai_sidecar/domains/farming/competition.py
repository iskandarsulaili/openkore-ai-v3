"""Player Classifier — detect bots vs humans by movement analysis.

On a bot-allowed server, most "players" are bots. Your bot should:
1. Detect other bots (competitive, efficient spawn-clearing)
2. Prefer farming near real players (inefficient, leave monsters)
3. Avoid bot-dense areas (competition for spawns)

Classification signals:
- Movement pattern (grid vs natural)
- Attack interval variance (fixed vs varying)
- Pause frequency (never stops vs stops to type/look)
- Response to environment (walks through monsters vs avoids)
"""
from __future__ import annotations
import time
import math
import logging
from typing import Any
from collections import deque

from ai_sidecar.actions import HeuristicAction

logger = logging.getLogger(__name__)


class PlayerClassifier:
    """Classifies observed players as bot or human based on behavior.
    
    Tracks each player for ~30 seconds before classifying.
    More data = more accurate classification.
    """
    
    def __init__(self):
        self._players: dict[str, dict[str, Any]] = {}
        self._observation_window = 30  # seconds of observation needed
        self._movement_buffer = 10  # positions to keep
    
    def observe(self, player_id: str, data: dict[str, Any]) -> None:
        """Observe a player's behavior for one tick.
        
        Args:
            player_id: Unique player identifier
            data: Observation data with keys:
                - pos_x, pos_y: Current position
                - action: What they're doing (move, attack, sit, idle)
                - attack_interval_ms: Time since last attack
                - distance_to_monster: Distance to nearest monster
        """
        now = time.time()
        
        if player_id not in self._players:
            self._players[player_id] = {
                "first_seen": now,
                "last_seen": now,
                "positions": deque(maxlen=self._movement_buffer),
                "attack_times": [],
                "action_changes": 0,
                "last_action": None,
                "last_pos": None,
                "pause_count": 0,
                "classification": "unknown",
                "confidence": 0.0,
            }
        
        p = self._players[player_id]
        p["last_seen"] = now
        
        # Track positions
        x = data.get("pos_x")
        y = data.get("pos_y")
        if x is not None and y is not None:
            p["positions"].append((x, y, now))
        
        # Track action changes (natural players change actions more)
        action = data.get("action", "")
        if action and action != p["last_action"]:
            p["action_changes"] += 1
        p["last_action"] = action
        
        # Track attack timing variance
        interval = data.get("attack_interval_ms", 0)
        if interval > 0:
            p["attack_times"].append(interval)
        
        # Track movement path
        if p["last_pos"] and x is not None and y is not None:
            dx = abs(x - p["last_pos"][0])
            dy = abs(y - p["last_pos"][1])
            # Bot-like: moves in straight grid lines (dx == 0 or dy == 0)
            # Human-like: moves diagonally or variably
            p["_last_move_axial"] = (dx == 0 or dy == 0)
        
        p["last_pos"] = (x, y) if x is not None and y is not None else p["last_pos"]
    
    def classify(self, player_id: str) -> tuple[str, float]:
        """Classify a player as 'bot', 'human', or 'unknown'.
        
        Returns:
            (classification, confidence) tuple
        """
        p = self._players.get(player_id)
        if not p:
            return ("unknown", 0.0)
        
        observation_time = p["last_seen"] - p["first_seen"]
        if observation_time < self._observation_window:
            return ("unknown", min(0.5, observation_time / self._observation_window))
        
        # Signals for bot detection
        bot_signals = 0
        human_signals = 0
        total_signals = 0
        
        # 1. Attack timing variance
        if p["attack_times"]:
            total_signals += 1
            times = p["attack_times"][-10:]  # Last 10 attacks
            if len(times) >= 3:
                variance = sum(abs(times[i] - times[i-1]) for i in range(1, len(times))) / len(times)
                if variance < 50:  # Very consistent timing = bot
                    bot_signals += 1
                elif variance > 200:  # Variable timing = human
                    human_signals += 1
                else:
                    human_signals += 0.5
        
        # 2. Action changes per minute
        total_signals += 1
        action_rate = p["action_changes"] / max(1, observation_time / 60)
        if action_rate < 0.5:  # Barely changes actions = bot
            bot_signals += 1
        elif action_rate > 3:  # Frequent action changes = human
            human_signals += 1
        else:
            human_signals += 0.5
        
        # 3. Movement pattern
        if len(p["positions"]) >= 5:
            total_signals += 1
            axial_moves = sum(1 for i in range(1, len(p["positions"])) 
                            if abs(p["positions"][i][0] - p["positions"][i-1][0]) == 0 
                            or abs(p["positions"][i][1] - p["positions"][i-1][1]) == 0)
            axial_ratio = axial_moves / max(1, len(p["positions"]) - 1)
            if axial_ratio > 0.8:  # Mostly axial movement = bot
                bot_signals += 1
            elif axial_ratio < 0.4:  # Diagonal movement = human
                human_signals += 1
        
        # 4. Pauses (humans pause to type, think)
        total_signals += 1
        if p["action_changes"] > 10:  # Lots of action changes suggests pauses
            human_signals += 0.5
        
        if total_signals == 0:
            return ("unknown", 0.0)
        
        score = (human_signals - bot_signals) / total_signals
        
        if score > 0.3:
            classification = "human"
        elif score < -0.3:
            classification = "bot"
        else:
            classification = "unknown"
        
        confidence = abs(score)
        
        p["classification"] = classification
        p["confidence"] = confidence
        
        return (classification, confidence)
    
    def get_bot_density(self) -> float:
        """Get estimated bot density in current area (0.0-1.0)."""
        bots = 0
        total = 0
        for pid, p in self._players.items():
            c, _ = self.classify(pid)
            if c != "unknown":
                total += 1
                if c == "bot":
                    bots += 1
        return bots / max(1, total)
    
    def get_bot_hotspots(self) -> list[tuple[float, float]]:
        """Get coordinates of known bot activity hotspots."""
        hotspots = []
        for p in self._players.values():
            if p["classification"] == "bot" and p["last_pos"]:
                hotspots.append(p["last_pos"])
        return hotspots
    
    def get_stats(self) -> dict[str, Any]:
        classified = sum(1 for p in self._players.values() if p["classification"] != "unknown")
        bots = sum(1 for p in self._players.values() if p["classification"] == "bot")
        humans = sum(1 for p in self._players.values() if p["classification"] == "human")
        return {
            "tracked_players": len(self._players),
            "classified": classified,
            "bots": bots,
            "humans": humans,
            "bot_density": self.get_bot_density(),
            "unknown": len(self._players) - classified,
        }


class CompetitionAwareFarming:
    """Adjusts farming strategy based on competition level.
    
    Uses PlayerClassifier to determine current map competition:
    - Peak hours (high bot density): switch to less popular maps
    - Off-peak (low bot density): farm best spawn maps
    - Avoid areas where bots outnumber humans 3:1
    """
    
    def __init__(self, classifier: PlayerClassifier | None = None):
        self._classifier = classifier or PlayerClassifier()
        self._map_stats: dict[str, dict[str, Any]] = {}
    
    def assess(self, signals: dict[str, Any], actions: list[Any], bot_id: str) -> None:
        """Assess competition level and adjust farming strategy."""
        current_map = signals.get("map", "")
        other_players = signals.get("other_players", [])
        
        # Observe other players
        for player in other_players:
            pid = player.get("name") or player.get("id", str(id(player)))
            self._classifier.observe(pid, player)
        
        # Track per-map stats
        if current_map:
            if current_map not in self._map_stats:
                self._map_stats[current_map] = {"visits": 0, "bot_density_samples": []}
            ms = self._map_stats[current_map]
            ms["visits"] += 1
            ms["bot_density_samples"].append(self._classifier.get_bot_density())
            # Keep last 20 samples
            ms["bot_density_samples"] = ms["bot_density_samples"][-20:]
        
        # Log competition level
        bot_density = self._classifier.get_bot_density()
        stats = self._classifier.get_stats()
        
        if bot_density > 0.6:
            actions.append(HeuristicAction(
                kind="log",
                command=f"high_competition density={bot_density:.0%} bots={stats['bots']} humans={stats['humans']}",
                confidence=0.6,
                reason="High bot density detected, consider map change",
                domain="farming"
            ))
        
        # Clean up stale observations
        now = time.time()
        stale = [pid for pid, p in self._classifier._players.items() 
                if now - p["last_seen"] > 300]  # 5 min timeout
        for pid in stale:
            del self._classifier._players[pid]
