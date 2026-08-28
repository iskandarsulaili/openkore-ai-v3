"""Integration Bus — brain layer that wires all modules together.

Each subsystem exists independently. This bus connects them:
- LearningFeedbackLoop → HighFreqReflex thresholds
- CombatIntelligence → HighFreqReflex flee decisions  
- EconomyEngine → Cold start planner
- MapIntelligence → Navigation routing
- EdgeCaseHandler → Action queue

Called from _emit_heuristic_actions in the PDCA loop (every 5s)
AND from HighFreqReflex._tick() (every 50ms).
"""

from __future__ import annotations

import logging
import time
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)


class IntegrationBus:
    """Central coordinator for all subsystem communication.
    
    Initialized once with references to all subsystems.
    Provides orchestration methods called from PDCA and HFR.
    No hardcoded values — all thresholds and decisions from subsystems.
    """
    
    def __init__(
        self,
        highfreq_reflex=None,
        learning_loop=None,
        combat_intel=None,
        economy_engine=None,
        map_intel=None,
        edge_handler=None,
    ) -> None:
        self._lock = RLock()
        self._hfr = highfreq_reflex
        self._learning = learning_loop
        self._combat = combat_intel
        self._economy = economy_engine
        self._maps = map_intel
        self._edges = edge_handler
        
        # Tracking
        self._last_learning_review = 0.0
        self._last_economy_review = 0.0
        self._stats: dict[str, int] = {
            "learning_reviews": 0, "economy_actions": 0,
            "edge_triggers": 0, "flee_decisions": 0,
        }
    
    # ── Called from HighFreqReflex._tick() every 50ms ──
    
    def evaluate_combat_threat(
        self,
        bot_id: str,
        hp: int,
        max_hp: int,
        monster_hp: int,
        monster_atk: int,
        avg_damage: int,
        aggro_count: int = 1,
    ) -> dict[str, Any]:
        """Evaluate combat threat using CombatIntelligence.
        Called from HighFreqReflex._tick() every 50ms.
        Returns flee recommendation with predictive analysis.
        """
        ci = self._combat
        if ci is None:
            return {"should_flee": False, "reason": "no_combat_intel"}
        
        recommendation = ci.flee_recommendation(
            my_hp=hp, my_max_hp=max_hp,
            monster_hp=monster_hp, monster_atk=monster_atk,
            avg_damage=avg_damage, aggro_count=aggro_count,
        )
        
        with self._lock:
            self._stats["flee_decisions"] += 1
        
        return recommendation
    
    def get_reflex_thresholds(self, bot_id: str) -> dict[str, float] | None:
        """Get current reflex thresholds (adjusted by learning loop).
        Returns None if HFR not available — caller uses defaults.
        """
        hfr = self._hfr
        if hfr is None:
            return None
        return hfr.get_thresholds()
    
    # ── Called from PDCA loop every 5s (via _emit_heuristic_actions) ──
    
    def periodic_review(
        self,
        bot_id: str,
        snapshot: dict[str, Any],
        action_queue,
    ) -> int:
        """Called from PDCA loop. Runs all subsystem checks.
        Returns count of actions enqueued.
        """
        actions_enqueued = 0
        now = time.time()
        
        # 1. Edge case check — runs every cycle
        actions_enqueued += self._check_edge_cases(bot_id, snapshot, action_queue)
        
        # 2. Economy review — every 60s
        if now - self._last_economy_review > 60.0:
            actions_enqueued += self._review_economy(bot_id, snapshot, action_queue)
            self._last_economy_review = now
        
        # 3. Learning review — every 120s
        if now - self._last_learning_review > 120.0:
            actions_enqueued += self._review_learning(bot_id, snapshot)
            self._last_learning_review = now
        
        return actions_enqueued
    
    def _check_edge_cases(self, bot_id: str, snapshot: dict, action_queue) -> int:
        """Run EdgeCaseHandler and enqueue any triggered actions."""
        handler = self._edges
        if handler is None:
            return 0
        
        try:
            proposals = handler.check_all(bot_id=bot_id, bot_state=snapshot)
            if proposals:
                count = 0
                for proposal in proposals:
                    if action_queue and proposal:
                        action_queue.enqueue(bot_id, proposal)
                        count += 1
                with self._lock:
                    self._stats["edge_triggers"] += count
                return count
        except Exception as e:
            logger.warning("integration_edge_check_failed: %s", e)
        return 0
    
    def _review_economy(self, bot_id: str, snapshot: dict, action_queue) -> int:
        """Review economy and enqueue buy/sell actions if needed."""
        economy = self._economy
        if economy is None or action_queue is None:
            return 0
        
        try:
            zeny = int(snapshot.get("zeny", snapshot.get("inventory", {}).get("zeny", 0)) or 0)
            level = int(snapshot.get("base_level", 1) or 1)
            job = str(snapshot.get("job", "") or "")
            inv_items = snapshot.get("inventory_items", [])
            
            # Get budget recommendation
            budget = economy.budget_planning(
                zeny=zeny, level=level, job=job,
                inventory={"items": inv_items} if isinstance(inv_items, list) else {},
            )
            
            count = 0
            
            # Enqueue buy actions from budget
            for item in budget.get("buy", []):
                buy_cmd = item.get("buy_command", "")
                if buy_cmd:
                    from datetime import datetime, timedelta, UTC
                    from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
                    action_queue.enqueue(bot_id, ActionProposal(
                        action_id=f"eco_buy_{bot_id}_{int(time.time()*1000)}",
                        kind="command", command=buy_cmd,
                        priority_tier=ActionPriorityTier.tactical,
                        source="economy_engine",
                        created_at=datetime.now(UTC),
                        expires_at=datetime.now(UTC) + timedelta(seconds=30),
                        idempotency_key=f"eco:buy:{item.get('name','')}:{int(time.time()/30)}",
                    ))
                    count += 1
            
            with self._lock:
                self._stats["economy_actions"] += count
            return count
            
        except Exception as e:
            logger.warning("integration_economy_review_failed: %s", e)
        return 0
    
    def _review_learning(self, bot_id: str, snapshot: dict) -> int:
        """Review learning data and update system parameters."""
        learning = self._learning
        if learning is None:
            return 0
        
        try:
            hfr = self._hfr
            
            # 1. Record map time
            current_map = str(snapshot.get("map", "") or "")
            if current_map:
                learning.record_map_time(current_map, 120.0)  # 120s since last review
            
            # 2. Get death rate and adjust thresholds
            deaths_this_session = int(self._stats.get("edge_triggers", 0))
            total_time = time.time() - getattr(learning, "_session_start", time.time())
            
            adjustments = learning.adjust_reflex_thresholds(
                recent_deaths=deaths_this_session,
                total_time=total_time,
            )
            
            # Apply adjustments to HighFreqReflex
            if hfr is not None and adjustments:
                reason = learning.get_last_adjustment().get("reason", "periodic_review")
                hfr.update_thresholds(adjustments, reason=reason)
                logger.info(
                    "integration_learning_applied: bot=%s death_rate=%.1f/hr adjustments=%s reason=%s",
                    bot_id,
                    learning.get_last_adjustment().get("death_rate", 0),
                    len(adjustments),
                    reason,
                )
            
            # 3. Record map death/kill stats if available
            events = snapshot.get("events", [])
            for event in events:
                if isinstance(event, dict):
                    etype = event.get("type", "")
                    if etype == "kill" and current_map:
                        learning.record_map_kill(
                            current_map,
                            loot_value=float(event.get("loot_value", 0)),
                        )
                    elif etype == "death" and current_map:
                        learning.record_map_death(current_map)
            
            # 4. Flush to DB periodically
            learning.flush()
            
            with self._lock:
                self._stats["learning_reviews"] += 1
            
        except Exception as e:
            logger.warning("integration_learning_review_failed: %s", e)
        return 0
    
    # ── Called from cold start planner ──
    
    def get_hunting_plan(self, level: int, job: str, current_map: str) -> dict[str, Any]:
        """Get hunting plan from MapIntelligence.
        Returns best zone recommendation for current level.
        """
        mi = self._maps
        if mi is None:
            return {"map": current_map, "reason": "no_map_intel"}
        
        try:
            recommendation = mi.next_hunting_zone(level=level, job=job)
            return recommendation or {"map": current_map, "reason": "no_recommendation"}
        except Exception:
            return {"map": current_map, "reason": "map_intel_error"}
    
    def get_weapon_priority(self, zeny: int, level: int, job: str) -> dict[str, Any]:
        """Get weapon-first purchase priority from EconomyEngine.
        Returns what to buy first: weapon before potions before gear.
        """
        economy = self._economy
        if economy is None:
            return {"priority_items": [], "assessment": "no_economy_engine"}
        
        try:
            budget = economy.budget_planning(
                zeny=zeny, level=level, job=job, inventory={},
            )
            return budget or {"priority_items": [], "assessment": "budget_planning_empty"}
        except Exception:
            return {"priority_items": [], "assessment": "budget_planning_failed"}
    
    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic stats."""
        with self._lock:
            return dict(self._stats)


def create_integration_bus(**kwargs) -> IntegrationBus:
    """Factory function for dependency injection."""
    return IntegrationBus(**kwargs)
