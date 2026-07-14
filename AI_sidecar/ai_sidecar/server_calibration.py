"""
Server Calibration Engine — learns the server's economy, population patterns,
GM patrol routes, and community norms. Adapts to the specific server.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ServerProfile:
    """A profile of the server's characteristics."""
    server_name: str = ""
    economy_multiplier: float = 1.0
    population_peak_hours: list[int] = field(default_factory=list)
    population_offpeak_hours: list[int] = field(default_factory=list)
    gm_activity_level: str = "medium"  # low, medium, high
    gm_patrol_maps: list[str] = field(default_factory=list)
    community_friendliness: str = "neutral"  # friendly, neutral, toxic
    has_custom_events: bool = False
    custom_event_schedule: str = ""
    notes: str = ""


@dataclass
class PriceBaseline:
    """A baseline price for an item on this server."""
    item_name: str
    avg_price: int = 0
    min_price: int = 0
    max_price: int = 0
    volatility: float = 0.0
    sample_count: int = 0


class ServerCalibration:
    """Learns and adapts to the specific server."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._profile: ServerProfile = ServerProfile()
        self._price_baselines: dict[str, PriceBaseline] = {}
        self._population_log: list[tuple[float, int]] = []  # (timestamp, player_count)
        self._gm_sightings: list[tuple[float, str]] = []  # (timestamp, map_name)
        self._max_log: int = 1000
        self._learning_days: int = 0
        self._start_time: float = time.time()

    # ── Public API ──

    def record_price(self, item_name: str, price: int) -> None:
        """Record a price observation to build baselines."""
        with self._lock:
            if item_name not in self._price_baselines:
                self._price_baselines[item_name] = PriceBaseline(item_name=item_name)
            baseline = self._price_baselines[item_name]
            if baseline.sample_count == 0:
                baseline.avg_price = price
                baseline.min_price = price
                baseline.max_price = price
            else:
                baseline.avg_price = (baseline.avg_price * baseline.sample_count + price) // (baseline.sample_count + 1)
                baseline.min_price = min(baseline.min_price, price)
                baseline.max_price = max(baseline.max_price, price)
            baseline.sample_count += 1

            # Update volatility
            if baseline.sample_count >= 10:
                baseline.volatility = (baseline.max_price - baseline.min_price) / max(baseline.avg_price, 1)

    def record_population(self, player_count: int) -> None:
        """Record a population observation."""
        with self._lock:
            self._population_log.append((time.time(), player_count))
            if len(self._population_log) > self._max_log:
                self._population_log = self._population_log[-self._max_log:]
            self._update_peak_hours()

    def _update_peak_hours(self) -> None:
        """Update peak/off-peak hour detection."""
        if len(self._population_log) < 50:
            return

        hour_counts: dict[int, list[int]] = defaultdict(list)
        for ts, count in self._population_log:
            hour = int(time.strftime("%H", time.localtime(ts)))
            hour_counts[hour].append(count)

        peak: list[int] = []
        offpeak: list[int] = []
        for hour, counts in hour_counts.items():
            avg = sum(counts) / len(counts)
            overall_avg = sum(c for _, c in self._population_log) / len(self._population_log)
            if avg > overall_avg * 1.2:
                peak.append(hour)
            elif avg < overall_avg * 0.8:
                offpeak.append(hour)

        self._profile.population_peak_hours = sorted(peak)
        self._profile.population_offpeak_hours = sorted(offpeak)

    def record_gm_sighting(self, map_name: str) -> None:
        """Record a GM sighting."""
        with self._lock:
            self._gm_sightings.append((time.time(), map_name))
            if len(self._gm_sightings) > self._max_log:
                self._gm_sightings = self._gm_sightings[-self._max_log:]

            # Update GM patrol maps
            map_counts: dict[str, int] = defaultdict(int)
            for _, m in self._gm_sightings:
                map_counts[m] += 1
            self._profile.gm_patrol_maps = sorted(map_counts, key=lambda x: -map_counts[x])[:5]

            # Update GM activity level
            sightings_per_day = len(self._gm_sightings) / max(self._learning_days, 1)
            if sightings_per_day > 5:
                self._profile.gm_activity_level = "high"
            elif sightings_per_day > 1:
                self._profile.gm_activity_level = "medium"
            else:
                self._profile.gm_activity_level = "low"

    def get_price_baseline(self, item_name: str) -> PriceBaseline | None:
        with self._lock:
            return self._price_baselines.get(item_name)

    def is_price_good(self, item_name: str, price: int, as_buyer: bool = True) -> bool:
        """Check if a price is good compared to the baseline."""
        with self._lock:
            baseline = self._price_baselines.get(item_name)
            if not baseline or baseline.sample_count < 3:
                return True
            if as_buyer:
                return price <= baseline.avg_price * 0.9  # 10% below avg = good buy
            else:
                return price >= baseline.avg_price * 1.1  # 10% above avg = good sell

    def get_recommended_behavior(self) -> str:
        """Get recommended behavior based on server calibration."""
        with self._lock:
            now = time.localtime()
            current_hour = now.tm_hour

            if current_hour in self._profile.population_peak_hours:
                return "caution"  # More players = more GMs, more competition
            elif current_hour in self._profile.population_offpeak_hours:
                return "aggressive"  # Fewer players = safer farming
            else:
                return "normal"

    def get_calibration_summary(self) -> str:
        with self._lock:
            lines = [f"── Server Calibration ──"]
            lines.append(f"Learning days: {self._learning_days}")
            lines.append(f"Items calibrated: {len(self._price_baselines)}")
            lines.append(f"GM activity: {self._profile.gm_activity_level}")
            if self._profile.gm_patrol_maps:
                lines.append(f"GM patrol maps: {', '.join(self._profile.gm_patrol_maps[:3])}")
            lines.append(f"Peak hours: {self._profile.population_peak_hours}")
            lines.append(f"Off-peak hours: {self._profile.population_offpeak_hours}")
            lines.append(f"Recommended: {self.get_recommended_behavior()}")
            return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._profile = ServerProfile()
            self._price_baselines.clear()
            self._population_log.clear()
            self._gm_sightings.clear()
            self._learning_days = 0
            self._start_time = time.time()


# ── Global Singleton ──

_server_cal: ServerCalibration | None = None
_server_cal_lock = RLock()


def get_server_calibration() -> ServerCalibration:
    global _server_cal
    with _server_cal_lock:
        if _server_cal is None:
            _server_cal = ServerCalibration()
        return _server_cal
