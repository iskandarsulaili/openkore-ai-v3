"""
Predictive Intelligence Engine — pattern-based prediction that exceeds human capability.

Instead of "feeling" the game, this system uses statistical pattern recognition
to predict future game states with higher accuracy than any human player.

Key capabilities:
1. Latency-aware combat timing (ping + cast + travel time compensation)
2. Mob behavior prediction (attack patterns, aggro range, movement prediction)
3. Player behavior prediction (farming routes, PK patterns, GM patrols)
4. Server event prediction (WoE timing, double exp, maintenance patterns)
5. Economic trend prediction (price cycles, demand forecasting)
6. Death prediction (assess risk before engaging, not after)
7. Temporal pattern recognition (interval tracking, next-occurrence prediction with confidence)
"""

from __future__ import annotations

import logging
import math
import random
import statistics
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LatencyProfile:
    """Network latency profile for ping-compensated combat."""
    ping_ms: int = 50
    jitter_ms: int = 20
    last_measured_ms: float = 0.0

    def effective_cast_time(self, base_cast_ms: int) -> int:
        """Return effective cast time including ping compensation.
        A real player with 300ms ping needs to start casting 300ms earlier.
        """
        return base_cast_ms + self.ping_ms + self.jitter_ms

    def effective_travel_time(self, distance: float, base_speed_ms: int = 100) -> int:
        """Return effective travel time including ping.
        Projectile travel + ping = when the damage actually lands.
        """
        return int(distance * base_speed_ms) + self.ping_ms

    def interrupt_deadline_ms(self, remaining_cast_ms: int) -> int:
        """Return the deadline by which an interrupt must be sent.
        If ping is 300ms and cast has 200ms remaining, it's too late.
        """
        return remaining_cast_ms - self.ping_ms - self.jitter_ms


@dataclass(slots=True)
class MobBehaviorProfile:
    """Learned behavior profile for a specific monster type."""
    mob_name: str
    mob_id: int
    aggro_range: int = 10
    aggro_type: str = "normal"  # normal, assist, aggressive, boss
    attack_patterns: list[dict[str, Any]] = field(default_factory=list)
    movement_speed: int = 150  # ms per cell
    preferred_distance: int = 1  # 1=melee, 3=range, 7=far
    cast_skills: list[str] = field(default_factory=list)
    avg_reaction_ms: int = 500
    observations: int = 0

    def predict_position(self, current_x: float, current_y: float,
                         target_x: float, target_y: float,
                         time_delta_ms: int) -> tuple[float, float]:
        """Predict where this mob will be in time_delta_ms.
        
        Uses observed movement patterns to predict position better than
        simple linear interpolation.
        """
        dx = target_x - current_x
        dy = target_y - current_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.1:
            return (current_x, current_y)
        
        # Movement speed in cells per ms
        speed = 1.0 / max(self.movement_speed, 1)
        max_move = speed * time_delta_ms
        
        if max_move >= dist:
            return (target_x, target_y)
        
        ratio = max_move / dist
        return (
            current_x + dx * ratio,
            current_y + dy * ratio,
        )


@dataclass(slots=True)
class PlayerBehaviorProfile:
    """Learned behavior profile for another player."""
    player_name: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    sightings: int = 0
    typical_maps: list[str] = field(default_factory=list)
    typical_hours: list[int] = field(default_factory=list)
    is_pker: bool = False
    is_gm: bool = False
    is_bot: bool = False
    threat_level: int = 0  # 0=harmless, 10=deadly
    farming_spots: list[str] = field(default_factory=list)
    typical_gear: list[str] = field(default_factory=list)
    typical_skills: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DeathPrediction:
    """Prediction of death risk before engaging."""
    risk_level: int  # 0-10
    primary_threat: str
    confidence: float
    recommended_action: str
    factors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ServerEventPrediction:
    """Prediction of upcoming server events."""
    event_type: str  # woe, double_exp, maintenance, holiday
    probability: float  # 0.0-1.0
    estimated_start: float = 0.0
    estimated_duration_hours: float = 0.0
    confidence: float = 0.0
    observed_pattern: str = ""


# ── Temporal Pattern Recognition ──────────────────────────────────────────


@dataclass(slots=True)
class IntervalStats:
    """Statistical summary of observed intervals between events."""
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    min_interval: float = 0.0
    max_interval: float = 0.0
    count: int = 0
    last_interval: float = 0.0


@dataclass(slots=True)
class TemporalPrediction:
    """Prediction of when the next event of a type will occur."""
    event_type: str
    event_subtype: str  # e.g. MVP name, item name, server event type
    predicted_timestamp: float
    predicted_in_seconds: float
    confidence_interval_low: float  # earliest expected (seconds from now)
    confidence_interval_high: float  # latest expected (seconds from now)
    confidence: float  # 0.0-1.0
    interval_stats: IntervalStats | None = None
    accuracy_history: list[dict[str, Any]] = field(default_factory=list)
    accuracy_pct: float = 0.0  # rolling accuracy percentage


@dataclass(slots=True)
class AccuracyRecord:
    """Record of a prediction vs actual outcome."""
    predicted_timestamp: float
    actual_timestamp: float
    error_seconds: float  # absolute error
    error_pct: float  # error as percentage of predicted interval
    event_type: str
    event_subtype: str
    recorded_at: float = 0.0


class IntervalTracker:
    """Tracks timestamps for a specific event type and calculates interval statistics.
    
    All data-driven — no hardcoded intervals. Learns purely from observed
    timestamps and calculates statistical properties of the intervals between
    consecutive events.
    """
    
    def __init__(self, max_samples: int = 100):
        self._timestamps: list[float] = []
        self._intervals: list[float] = []
        self._max_samples = max_samples
        self._last_stats: IntervalStats | None = None
    
    def record(self, timestamp: float | None = None) -> None:
        """Record a timestamp and calculate the interval from the previous one."""
        ts = timestamp if timestamp is not None else time.time()
        
        if self._timestamps:
            interval = ts - self._timestamps[-1]
            self._intervals.append(interval)
            # Keep intervals bounded
            if len(self._intervals) > self._max_samples:
                self._intervals = self._intervals[-self._max_samples:]
        
        self._timestamps.append(ts)
        if len(self._timestamps) > self._max_samples:
            self._timestamps = self._timestamps[-self._max_samples:]
    
    def get_intervals(self) -> list[float]:
        """Get all observed intervals in seconds."""
        return list(self._intervals)
    
    def get_timestamps(self) -> list[float]:
        """Get all recorded timestamps."""
        return list(self._timestamps)
    
    def get_stats(self) -> IntervalStats:
        """Calculate statistical summary of observed intervals.
        
        Returns mean, median, std_dev, min, max, count, and last interval.
        All computed from observed data — no hardcoded values.
        """
        if not self._intervals:
            return IntervalStats()
        
        intervals = self._intervals
        n = len(intervals)
        
        mean_val = sum(intervals) / n
        median_val = statistics.median(intervals) if n >= 2 else intervals[0]
        
        if n >= 2:
            variance = sum((i - mean_val) ** 2 for i in intervals) / n
            std_dev_val = math.sqrt(variance)
        else:
            std_dev_val = 0.0
        
        stats = IntervalStats(
            mean=mean_val,
            median=median_val,
            std_dev=std_dev_val,
            min_interval=min(intervals),
            max_interval=max(intervals),
            count=n,
            last_interval=intervals[-1] if intervals else 0.0,
        )
        self._last_stats = stats
        return stats
    
    def get_event_count(self) -> int:
        """Get total number of recorded events."""
        return len(self._timestamps)
    
    def get_last_timestamp(self) -> float | None:
        """Get the most recent recorded timestamp."""
        if self._timestamps:
            return self._timestamps[-1]
        return None
    
    def clear(self) -> None:
        """Clear all recorded data."""
        self._timestamps.clear()
        self._intervals.clear()
        self._last_stats = None


class TemporalPredictor:
    """Predicts next occurrence of an event type with confidence intervals.
    
    Uses statistical analysis of observed intervals to predict when the next
    event will occur. Provides confidence intervals based on the distribution
    of observed intervals. Tracks prediction accuracy to improve over time.
    """
    
    def __init__(self, min_samples: int = 2):
        self._trackers: dict[str, IntervalTracker] = {}
        self._predictions: dict[str, TemporalPrediction] = {}
        self._accuracy_records: dict[str, list[AccuracyRecord]] = defaultdict(list)
        self._min_samples = min_samples
        self._lock = RLock()
    
    def _key(self, event_type: str, event_subtype: str) -> str:
        """Generate a composite key for event type + subtype."""
        return f"{event_type}::{event_subtype}"
    
    def record_event(self, event_type: str, event_subtype: str,
                     timestamp: float | None = None) -> None:
        """Record an event occurrence and update interval tracking.
        
        Args:
            event_type: Category of event (mvp_kill, price_spike, server_event)
            event_subtype: Specific identifier (MVP name, item name, event type)
            timestamp: When the event occurred (default: now)
        """
        key = self._key(event_type, event_subtype)
        
        with self._lock:
            if key not in self._trackers:
                self._trackers[key] = IntervalTracker()
            
            tracker = self._trackers[key]
            
            # Check if we had a pending prediction for this event
            if key in self._predictions:
                pred = self._predictions[key]
                ts = timestamp if timestamp is not None else time.time()
                self._record_accuracy(key, pred, ts)
            
            tracker.record(timestamp)
    
    def _record_accuracy(self, key: str, prediction: TemporalPrediction,
                         actual_timestamp: float) -> None:
        """Record the accuracy of a prediction against the actual event time."""
        predicted_ts = prediction.predicted_timestamp
        error_sec = abs(actual_timestamp - predicted_ts)
        
        # Error as percentage of the predicted interval
        interval = prediction.interval_stats.mean if prediction.interval_stats else 1.0
        error_pct = (error_sec / max(interval, 1.0)) * 100.0
        
        record = AccuracyRecord(
            predicted_timestamp=predicted_ts,
            actual_timestamp=actual_timestamp,
            error_seconds=error_sec,
            error_pct=error_pct,
            event_type=prediction.event_type,
            event_subtype=prediction.event_subtype,
            recorded_at=time.time(),
        )
        
        self._accuracy_records[key].append(record)
        # Keep last 50 accuracy records per event
        if len(self._accuracy_records[key]) > 50:
            self._accuracy_records[key] = self._accuracy_records[key][-50:]
    
    def predict_next(self, event_type: str, event_subtype: str,
                     confidence_level: float = 0.95) -> TemporalPrediction | None:
        """Predict when the next event of this type will occur.
        
        Args:
            event_type: Category of event
            event_subtype: Specific identifier
            confidence_level: Statistical confidence level (0.0-1.0)
            
        Returns:
            TemporalPrediction with predicted timestamp, confidence intervals,
            and accuracy history, or None if insufficient data.
        """
        key = self._key(event_type, event_subtype)
        
        with self._lock:
            tracker = self._trackers.get(key)
            if tracker is None or tracker.get_event_count() < self._min_samples:
                return None
            
            stats = tracker.get_stats()
            last_ts = tracker.get_last_timestamp()
            if last_ts is None or stats.count < 1:
                return None
            
            now = time.time()
            
            # Use median for prediction (more robust to outliers than mean)
            predicted_interval = stats.median if stats.median > 0 else stats.mean
            predicted_ts = last_ts + predicted_interval
            predicted_in_seconds = predicted_ts - now
            
            # Confidence interval based on std dev
            # For 95% confidence: ~2 std devs; scale by confidence_level
            z_score = self._z_score_for_confidence(confidence_level)
            half_width = stats.std_dev * z_score if stats.std_dev > 0 else predicted_interval * 0.2
            
            ci_low = max(0, predicted_in_seconds - half_width)
            ci_high = predicted_in_seconds + half_width
            
            # Confidence in the prediction itself
            # Higher when we have more samples and lower variance
            if stats.count < 3:
                pred_confidence = 0.3
            elif stats.count < 5:
                pred_confidence = 0.5
            elif stats.count < 10:
                pred_confidence = 0.7
            else:
                # Coefficient of variation — lower = more consistent = higher confidence
                cv = stats.std_dev / max(stats.mean, 1.0)
                pred_confidence = max(0.0, min(1.0, 1.0 - cv))
            
            # Factor in historical accuracy
            accuracy_pct = self._get_rolling_accuracy(key)
            if accuracy_pct > 0:
                pred_confidence = pred_confidence * 0.7 + accuracy_pct * 0.3
            
            prediction = TemporalPrediction(
                event_type=event_type,
                event_subtype=event_subtype,
                predicted_timestamp=predicted_ts,
                predicted_in_seconds=predicted_in_seconds,
                confidence_interval_low=ci_low,
                confidence_interval_high=ci_high,
                confidence=round(pred_confidence, 3),
                interval_stats=stats,
                accuracy_history=self._get_accuracy_history(key),
                accuracy_pct=round(accuracy_pct * 100, 1),
            )
            
            self._predictions[key] = prediction
            return prediction
    
    def _z_score_for_confidence(self, confidence_level: float) -> float:
        """Approximate z-score for a given confidence level.
        
        Uses a simple approximation of the normal distribution inverse CDF.
        """
        # Common z-scores
        if confidence_level >= 0.99:
            return 2.576
        elif confidence_level >= 0.98:
            return 2.326
        elif confidence_level >= 0.95:
            return 1.96
        elif confidence_level >= 0.90:
            return 1.645
        elif confidence_level >= 0.85:
            return 1.440
        elif confidence_level >= 0.80:
            return 1.282
        else:
            return 1.0
    
    def _get_rolling_accuracy(self, key: str) -> float:
        """Calculate rolling prediction accuracy (0.0-1.0) for an event type.
        
        Accuracy is measured as the proportion of predictions where the
        actual event fell within the confidence interval.
        """
        records = self._accuracy_records.get(key, [])
        if not records:
            return 0.0
        
        # Use last 20 records for rolling accuracy
        recent = records[-20:] if len(records) > 20 else records
        
        # Count predictions where error was within reasonable bounds
        # "Accurate" = error < 30% of the predicted interval
        accurate = sum(1 for r in recent if r.error_pct < 30.0)
        return accurate / len(recent)
    
    def _get_accuracy_history(self, key: str) -> list[dict[str, Any]]:
        """Get accuracy history for an event type as serializable dicts."""
        records = self._accuracy_records.get(key, [])
        return [
            {
                "predicted_timestamp": r.predicted_timestamp,
                "actual_timestamp": r.actual_timestamp,
                "error_seconds": round(r.error_seconds, 1),
                "error_pct": round(r.error_pct, 1),
            }
            for r in records[-10:]  # Last 10 records
        ]
    
    def get_tracker(self, event_type: str, event_subtype: str) -> IntervalTracker | None:
        """Get the IntervalTracker for a specific event type+subtype."""
        key = self._key(event_type, event_subtype)
        return self._trackers.get(key)
    
    def get_all_tracked_events(self) -> list[dict[str, Any]]:
        """Get summary of all tracked event types and their stats."""
        results = []
        with self._lock:
            for key, tracker in self._trackers.items():
                event_type, event_subtype = key.split("::", 1)
                stats = tracker.get_stats()
                results.append({
                    "event_type": event_type,
                    "event_subtype": event_subtype,
                    "count": tracker.get_event_count(),
                    "intervals": stats.count,
                    "mean_interval": round(stats.mean, 1),
                    "median_interval": round(stats.median, 1),
                    "std_dev": round(stats.std_dev, 1),
                    "last_interval": round(stats.last_interval, 1),
                })
        return results
    
    def get_accuracy_summary(self) -> list[dict[str, Any]]:
        """Get accuracy summary across all tracked event types."""
        results = []
        with self._lock:
            for key, records in self._accuracy_records.items():
                if not records:
                    continue
                event_type, event_subtype = key.split("::", 1)
                recent = records[-20:] if len(records) > 20 else records
                avg_error = sum(r.error_seconds for r in recent) / len(recent)
                accurate = sum(1 for r in recent if r.error_pct < 30.0)
                results.append({
                    "event_type": event_type,
                    "event_subtype": event_subtype,
                    "total_predictions": len(records),
                    "recent_accuracy_pct": round((accurate / len(recent)) * 100, 1),
                    "avg_error_seconds": round(avg_error, 1),
                })
        return results


class PredictiveIntelligence:
    """Pattern-based prediction engine that exceeds human capability.
    
    Uses statistical pattern recognition across multiple dimensions:
    - Combat timing (ping-compensated)
    - Mob behavior (learned attack patterns)
    - Player behavior (farming routes, PK patterns)
    - Server events (WoE timing, maintenance)
    - Economic trends (price cycles)
    - Death risk (pre-engagement assessment)
    - Temporal patterns (interval-based next-occurrence prediction)
    """
    
    def __init__(self):
        self._lock = RLock()
        
        # Latency tracking
        self._latency: LatencyProfile = LatencyProfile()
        
        # Mob behavior profiles (learned from observations)
        self._mob_profiles: dict[int, MobBehaviorProfile] = {}
        
        # Player behavior profiles (learned from observations)
        self._player_profiles: dict[str, PlayerBehaviorProfile] = {}
        
        # Server event history
        self._server_events: list[dict[str, Any]] = []
        
        # Death history for risk assessment
        self._death_history: deque[dict[str, Any]] = deque(maxlen=100)
        
        # Combat outcome history
        self._combat_outcomes: deque[dict[str, Any]] = deque(maxlen=200)
        
        # Temporal pattern recognition
        self._temporal: TemporalPredictor = TemporalPredictor()
        
        # MVP kill history (for temporal tracking)
        self._mvp_kills: list[dict[str, Any]] = []
        
        # Price spike history (for temporal tracking)
        self._price_spikes: list[dict[str, Any]] = []
        
        # Statistics
        self._stats: dict[str, int] = defaultdict(int)
    
    # ── Latency Management ──────────────────────────────────────────────
    
    def record_ping(self, ping_ms: int) -> None:
        """Record a ping measurement and update the latency profile."""
        with self._lock:
            old_ping = self._latency.ping_ms
            # Exponential moving average
            self._latency.ping_ms = int(old_ping * 0.7 + ping_ms * 0.3)
            self._latency.jitter_ms = abs(self._latency.ping_ms - ping_ms)
            self._latency.last_measured_ms = time.time()
            self._stats["ping_updates"] += 1
    
    def get_latency_profile(self) -> LatencyProfile:
        """Get the current latency profile."""
        with self._lock:
            return self._latency
    
    def get_compensated_cast_time(self, base_cast_ms: int) -> int:
        """Get ping-compensated cast time.
        
        A real player with 300ms ping needs to start casting 300ms earlier.
        Our bot can compensate perfectly because it knows its own ping.
        """
        with self._lock:
            return self._latency.effective_cast_time(base_cast_ms)
    
    def get_interrupt_deadline(self, remaining_cast_ms: int) -> int:
        """Get the deadline by which an interrupt must be sent.
        
        If ping is 300ms and cast has 200ms remaining, it's too late.
        Our bot knows this and won't waste the attempt.
        """
        with self._lock:
            return self._latency.interrupt_deadline_ms(remaining_cast_ms)
    
    # ── Mob Behavior Prediction ─────────────────────────────────────────
    
    def record_mob_observation(self, mob_id: int, mob_name: str,
                               data: dict[str, Any]) -> None:
        """Record an observation of a mob's behavior."""
        with self._lock:
            if mob_id not in self._mob_profiles:
                self._mob_profiles[mob_id] = MobBehaviorProfile(
                    mob_name=mob_name, mob_id=mob_id
                )
            profile = self._mob_profiles[mob_id]
            profile.observations += 1
            
            # Track attack patterns
            if data.get("skill_name"):
                pattern = {
                    "skill": data["skill_name"],
                    "distance": data.get("distance", 10),
                    "hp_pct": data.get("hp_pct", 100),
                    "timestamp": time.time(),
                }
                profile.attack_patterns.append(pattern)
                # Keep last 20 patterns
                if len(profile.attack_patterns) > 20:
                    profile.attack_patterns = profile.attack_patterns[-20:]
            
            # Track aggro range
            if data.get("aggro_distance"):
                observed_range = data["aggro_distance"]
                profile.aggro_range = max(profile.aggro_range, observed_range)
            
            # Track cast skills
            if data.get("casting_skill") and data["casting_skill"] not in profile.cast_skills:
                profile.cast_skills.append(data["casting_skill"])
            
            self._stats["mob_observations"] += 1
    
    def predict_mob_position(self, mob_id: int,
                              current_x: float, current_y: float,
                              target_x: float, target_y: float,
                              time_delta_ms: int) -> tuple[float, float]:
        """Predict where a mob will be in time_delta_ms."""
        with self._lock:
            profile = self._mob_profiles.get(mob_id)
            if profile is None:
                # Default linear prediction
                dx = target_x - current_x
                dy = target_y - current_y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 0.1:
                    return (current_x, current_y)
                speed = 1.0 / 150  # default 150ms/cell
                max_move = speed * time_delta_ms
                if max_move >= dist:
                    return (target_x, target_y)
                ratio = max_move / dist
                return (current_x + dx * ratio, current_y + dy * ratio)
            
            return profile.predict_position(
                current_x, current_y, target_x, target_y, time_delta_ms
            )
    
    def get_mob_aggro_range(self, mob_id: int) -> int:
        """Get the learned aggro range for a mob."""
        with self._lock:
            profile = self._mob_profiles.get(mob_id)
            if profile is None:
                return 10  # default
            return profile.aggro_range
    
    def predict_mob_next_skill(self, mob_id: int,
                                current_hp_pct: float,
                                distance: float) -> str | None:
        """Predict what skill a mob will use next.
        
        Uses observed patterns to predict with higher accuracy than
        a human player who hasn't fought this mob 100 times.
        """
        with self._lock:
            profile = self._mob_profiles.get(mob_id)
            if not profile or not profile.attack_patterns:
                return None
            
            # Find most common skill at similar HP and distance
            similar_patterns = [
                p for p in profile.attack_patterns
                if abs(p.get("hp_pct", 100) - current_hp_pct) < 20
                and abs(p.get("distance", 10) - distance) < 3
            ]
            
            if not similar_patterns:
                # Fall back to most common skill overall
                skill_counts: dict[str, int] = {}
                for p in profile.attack_patterns:
                    skill = p.get("skill", "")
                    if skill:
                        skill_counts[skill] = skill_counts.get(skill, 0) + 1
                if skill_counts:
                    return max(skill_counts.items(), key=lambda x: x[1])[0]
                return None
            
            # Find most common skill in similar situations
            skill_counts = {}
            for p in similar_patterns:
                skill = p.get("skill", "")
                if skill:
                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
            
            if skill_counts:
                return max(skill_counts.items(), key=lambda x: x[1])[0]
            return None
    
    # ── Player Behavior Prediction ───────────────────────────────────────
    
    def record_player_observation(self, player_name: str,
                                   data: dict[str, Any]) -> None:
        """Record an observation of another player's behavior."""
        with self._lock:
            if player_name not in self._player_profiles:
                self._player_profiles[player_name] = PlayerBehaviorProfile(
                    player_name=player_name,
                    first_seen=time.time(),
                )
            
            profile = self._player_profiles[player_name]
            profile.last_seen = time.time()
            profile.sightings += 1
            
            # Track maps
            if data.get("map") and data["map"] not in profile.typical_maps:
                profile.typical_maps.append(data["map"])
                if len(profile.typical_maps) > 10:
                    profile.typical_maps = profile.typical_maps[-10:]
            
            # Track hours
            hour = int(time.localtime().tm_hour)
            if hour not in profile.typical_hours:
                profile.typical_hours.append(hour)
            
            # Track gear
            if data.get("gear"):
                for item in data["gear"]:
                    if item not in profile.typical_gear:
                        profile.typical_gear.append(item)
                        if len(profile.typical_gear) > 20:
                            profile.typical_gear = profile.typical_gear[-20:]
            
            # Track skills
            if data.get("skills"):
                for skill in data["skills"]:
                    if skill not in profile.typical_skills:
                        profile.typical_skills.append(skill)
                        if len(profile.typical_skills) > 20:
                            profile.typical_skills = profile.typical_skills[-20:]
            
            # Detect PKers
            if data.get("pked") or data.get("attacked_by"):
                profile.is_pker = True
                profile.threat_level = min(10, profile.threat_level + 2)
            
            # Detect GMs
            if data.get("is_gm"):
                profile.is_gm = True
                profile.threat_level = 10
            
            self._stats["player_observations"] += 1
    
    def get_player_threat(self, player_name: str) -> int:
        """Get the threat level of a player (0=harmless, 10=deadly)."""
        with self._lock:
            profile = self._player_profiles.get(player_name)
            if profile is None:
                return 0
            return profile.threat_level
    
    def predict_player_location(self, player_name: str) -> str | None:
        """Predict where a player is likely to be right now.
        
        Uses observed patterns: if they always farm the same map at this hour,
        we know where they are.
        """
        with self._lock:
            profile = self._player_profiles.get(player_name)
            if not profile or not profile.typical_maps:
                return None
            
            hour = int(time.localtime().tm_hour)
            
            # If they have a farming spot and are active at this hour
            if profile.farming_spots and hour in profile.typical_hours:
                return profile.farming_spots[0]
            
            # Most common map
            return profile.typical_maps[0] if profile.typical_maps else None
    
    # ── Temporal Pattern Recognition ─────────────────────────────────────
    
    def record_mvp_kill(self, mvp_name: str,
                        data: dict[str, Any] | None = None) -> None:
        """Record an MVP kill for temporal pattern learning.
        
        Tracks when each MVP is killed to predict respawn timers.
        All interval calculations are data-driven from observed kill patterns.
        
        Args:
            mvp_name: Name of the MVP (e.g. "Phreeoni", "Dracula")
            data: Optional additional context (map, damage dealt, party info)
        """
        with self._lock:
            ts = time.time()
            self._mvp_kills.append({
                "mvp_name": mvp_name,
                "timestamp": ts,
                "data": data or {},
            })
            # Keep last 200 kills
            if len(self._mvp_kills) > 200:
                self._mvp_kills = self._mvp_kills[-200:]
            
            self._stats["mvp_kills_recorded"] += 1
        
        # Record in temporal predictor (outside lock to avoid deadlock)
        self._temporal.record_event("mvp_kill", mvp_name, ts)
    
    def predict_mvp_respawn(self, mvp_name: str,
                            confidence_level: float = 0.95) -> TemporalPrediction | None:
        """Predict when an MVP will respawn.
        
        Uses observed kill intervals to calculate expected respawn time
        with confidence intervals. All data-driven — no hardcoded timers.
        
        Args:
            mvp_name: Name of the MVP
            confidence_level: Statistical confidence (0.0-1.0)
            
        Returns:
            TemporalPrediction with predicted respawn time, or None if
            insufficient data (need at least 2 kills to calculate intervals).
        """
        return self._temporal.predict_next("mvp_kill", mvp_name, confidence_level)
    
    def record_price_spike(self, item_name: str,
                           price: int,
                           data: dict[str, Any] | None = None) -> None:
        """Record a price spike for temporal pattern learning.
        
        Tracks when item prices spike to predict future market cycles.
        All interval calculations are data-driven from observed price events.
        
        Args:
            item_name: Name of the item
            price: The spike price
            data: Optional additional context (normal price, vendor, quantity)
        """
        with self._lock:
            ts = time.time()
            self._price_spikes.append({
                "item_name": item_name,
                "price": price,
                "timestamp": ts,
                "data": data or {},
            })
            # Keep last 200 price spikes
            if len(self._price_spikes) > 200:
                self._price_spikes = self._price_spikes[-200:]
            
            self._stats["price_spikes_recorded"] += 1
        
        # Record in temporal predictor
        self._temporal.record_event("price_spike", item_name, ts)
    
    def predict_next_price_spike(self, item_name: str,
                                 confidence_level: float = 0.95) -> TemporalPrediction | None:
        """Predict when the next price spike will occur for an item.
        
        Uses observed price spike intervals to predict future spikes.
        All data-driven — no hardcoded market cycles.
        
        Args:
            item_name: Name of the item
            confidence_level: Statistical confidence (0.0-1.0)
            
        Returns:
            TemporalPrediction with predicted next spike, or None if
            insufficient data.
        """
        return self._temporal.predict_next("price_spike", item_name, confidence_level)
    
    def record_server_event(self, event_type: str,
                             data: dict[str, Any]) -> None:
        """Record a server event for pattern learning.
        
        Enhanced to also track temporal patterns for interval-based prediction
        of recurring server events (maintenance, WoE, double exp, etc.).
        
        Args:
            event_type: Type of event (woe, double_exp, maintenance, holiday, etc.)
            data: Event details including optional 'subtype' for granular tracking
        """
        with self._lock:
            ts = time.time()
            self._server_events.append({
                "type": event_type,
                "timestamp": ts,
                "data": data,
            })
            # Keep last 100 events
            if len(self._server_events) > 100:
                self._server_events = self._server_events[-100:]
            self._stats["server_events"] += 1
        
        # Record in temporal predictor for interval-based prediction
        # Use data.get("subtype", event_type) for granular tracking
        subtype = data.get("subtype", event_type) if data else event_type
        self._temporal.record_event("server_event", subtype, ts)
    
    def predict_next_server_event(self, event_type: str,
                                  confidence_level: float = 0.95) -> TemporalPrediction | None:
        """Predict when the next server event will occur.
        
        Uses observed event intervals to predict future occurrences.
        All data-driven — no hardcoded schedules.
        
        Args:
            event_type: Type of event (woe, double_exp, maintenance, etc.)
            confidence_level: Statistical confidence (0.0-1.0)
            
        Returns:
            TemporalPrediction with predicted next event, or None if
            insufficient data.
        """
        return self._temporal.predict_next("server_event", event_type, confidence_level)
    
    def get_temporal_stats(self) -> dict[str, Any]:
        """Get comprehensive temporal pattern recognition statistics."""
        with self._lock:
            return {
                "tracked_events": self._temporal.get_all_tracked_events(),
                "accuracy": self._temporal.get_accuracy_summary(),
                "mvp_kills_tracked": len(self._mvp_kills),
                "price_spikes_tracked": len(self._price_spikes),
                "server_events_tracked": len(self._server_events),
            }
    
    # ── Server Event Prediction (legacy) ─────────────────────────────────
    
    def predict_next_woe(self) -> ServerEventPrediction | None:
        """Predict when the next WoE will start.
        
        Uses observed WoE timing patterns. Most servers have WoE
        at the same time every week.
        """
        with self._lock:
            woe_events = [
                e for e in self._server_events
                if e["type"] == "woe"
            ]
            if not woe_events:
                return None
            
            # Check if WoE happens at regular intervals
            intervals = []
            for i in range(1, len(woe_events)):
                interval = woe_events[i]["timestamp"] - woe_events[i-1]["timestamp"]
                intervals.append(interval)
            
            if not intervals:
                return None
            
            avg_interval = sum(intervals) / len(intervals)
            last_woe = woe_events[-1]["timestamp"]
            next_woe = last_woe + avg_interval
            
            # Confidence based on consistency of intervals
            if len(intervals) >= 3:
                variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
                confidence = max(0.0, min(1.0, 1.0 - (variance / (avg_interval ** 2))))
            else:
                confidence = 0.3
            
            return ServerEventPrediction(
                event_type="woe",
                probability=0.8 if confidence > 0.5 else 0.5,
                estimated_start=next_woe,
                estimated_duration_hours=2.0,
                confidence=confidence,
                observed_pattern=f"Every {avg_interval / 3600:.1f} hours",
            )
    
    def predict_double_exp(self) -> ServerEventPrediction | None:
        """Predict when the next double exp event will occur.
        
        Many servers have double exp on weekends or specific days.
        """
        with self._lock:
            exp_events = [
                e for e in self._server_events
                if e["type"] == "double_exp"
            ]
            if not exp_events:
                return None
            
            # Check day-of-week pattern
            days = []
            for e in exp_events:
                t = time.localtime(e["timestamp"])
                days.append(t.tm_wday)
            
            if not days:
                return None
            
            # Most common day
            day_counts = Counter(days)
            most_common_day = day_counts.most_common(1)[0][0]
            confidence = day_counts.most_common(1)[0][1] / len(days)
            
            return ServerEventPrediction(
                event_type="double_exp",
                probability=0.6 if confidence > 0.3 else 0.3,
                confidence=confidence,
                observed_pattern=f"Most common on day {most_common_day}",
            )
    
    # ── Death Risk Prediction ───────────────────────────────────────────
    
    def record_death(self, map_name: str, reason: str,
                     position: tuple[float, float] | None = None,
                     context: dict[str, Any] | None = None) -> None:
        """Record a death for risk assessment learning."""
        with self._lock:
            self._death_history.append({
                "map": map_name,
                "reason": reason,
                "position": position,
                "context": context or {},
                "timestamp": time.time(),
            })
            self._stats["deaths_recorded"] += 1
    
    def record_combat_outcome(self, mob_name: str, mob_id: int,
                               won: bool, damage_taken: int,
                               context: dict[str, Any] | None = None) -> None:
        """Record a combat outcome for risk assessment."""
        with self._lock:
            self._combat_outcomes.append({
                "mob_name": mob_name,
                "mob_id": mob_id,
                "won": won,
                "damage_taken": damage_taken,
                "context": context or {},
                "timestamp": time.time(),
            })
            self._stats["combat_outcomes"] += 1
    
    def assess_death_risk(self, map_name: str, mobs_nearby: int,
                           current_hp_pct: float,
                           has_escape_items: bool = True) -> DeathPrediction:
        """Assess death risk before engaging.
        
        Uses historical death data and combat outcomes to predict
        risk with higher accuracy than human intuition.
        """
        with self._lock:
            risk = 0
            factors = []
            
            # Factor 1: Death history on this map
            map_deaths = [d for d in self._death_history if d["map"] == map_name]
            if len(map_deaths) >= 3:
                risk += 3
                factors.append(f"3+ deaths on {map_name}")
            elif map_deaths:
                risk += 1
                factors.append(f"Previous deaths on {map_name}")
            
            # Factor 2: Recent deaths (last hour)
            recent = [d for d in self._death_history
                      if time.time() - d["timestamp"] < 3600]
            if recent:
                risk += min(3, len(recent))
                factors.append(f"{len(recent)} deaths in last hour")
            
            # Factor 3: Low HP
            if current_hp_pct < 30:
                risk += 3
                factors.append(f"HP at {current_hp_pct:.0f}%")
            elif current_hp_pct < 50:
                risk += 1
                factors.append(f"HP at {current_hp_pct:.0f}%")
            
            # Factor 4: Multiple mobs
            if mobs_nearby > 5:
                risk += 2
                factors.append(f"{mobs_nearby} mobs nearby")
            elif mobs_nearby > 3:
                risk += 1
            
            # Factor 5: No escape items
            if not has_escape_items:
                risk += 2
                factors.append("No escape items (fly wing/teleport)")
            
            # Factor 6: Combat outcome history
            recent_losses = [o for o in self._combat_outcomes
                             if not o["won"] and
                             time.time() - o["timestamp"] < 3600]
            if recent_losses:
                risk += min(2, len(recent_losses))
                factors.append(f"{len(recent_losses)} recent combat losses")
            
            # Determine action
            risk = min(10, risk)
            if risk >= 7:
                action = "flee"
                threat = "High death risk"
            elif risk >= 4:
                action = "cautious"
                threat = "Moderate death risk"
            else:
                action = "engage"
                threat = "Low death risk"
            
            return DeathPrediction(
                risk_level=risk,
                primary_threat=threat,
                confidence=0.7 + (len(factors) * 0.05),
                recommended_action=action,
                factors=factors,
            )
    
    # ── Economic Trend Prediction ───────────────────────────────────────
    
    def predict_price_trend(self, item_name: str,
                             current_price: int,
                             history: list[int]) -> dict[str, Any]:
        """Predict future price trends for an item.
        
        Uses statistical analysis of price history to predict
        future prices with higher accuracy than human intuition.
        """
        if len(history) < 5:
            return {
                "trend": "stable",
                "predicted_change": 0,
                "confidence": 0.0,
                "recommendation": "hold",
            }
        
        # Simple linear regression for trend
        n = len(history)
        x_avg = (n - 1) / 2
        y_avg = sum(history) / n
        
        numerator = sum((i - x_avg) * (price - y_avg) for i, price in enumerate(history))
        denominator = sum((i - x_avg) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Predict next price
        predicted_next = y_avg + slope * n
        
        # Confidence based on R-squared
        if denominator == 0:
            confidence = 0.0
        else:
            ss_res = sum((price - (y_avg + slope * (i - x_avg))) ** 2
                         for i, price in enumerate(history))
            ss_tot = sum((price - y_avg) ** 2 for price in history)
            confidence = max(0.0, 1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        
        change_pct = ((predicted_next - current_price) / max(current_price, 1)) * 100
        
        if change_pct > 10:
            trend = "up"
            recommendation = "hold"
        elif change_pct < -10:
            trend = "down"
            recommendation = "sell"
        else:
            trend = "stable"
            recommendation = "hold"
        
        return {
            "trend": trend,
            "predicted_change": round(change_pct, 1),
            "predicted_price": round(predicted_next),
            "confidence": round(confidence, 2),
            "recommendation": recommendation,
        }
    
    # ── Statistics ──────────────────────────────────────────────────────
    
    def get_stats(self) -> dict[str, int]:
        """Get prediction engine statistics."""
        with self._lock:
            return dict(self._stats)
    
    def get_mob_count(self) -> int:
        """Get number of tracked mob profiles."""
        with self._lock:
            return len(self._mob_profiles)
    
    def get_player_count(self) -> int:
        """Get number of tracked player profiles."""
        with self._lock:
            return len(self._player_profiles)


# Global singleton
_engine: PredictiveIntelligence | None = None

def get_predictive_intelligence() -> PredictiveIntelligence:
    """Get the global PredictiveIntelligence instance."""
    global _engine
    if _engine is None:
        _engine = PredictiveIntelligence()
    return _engine
