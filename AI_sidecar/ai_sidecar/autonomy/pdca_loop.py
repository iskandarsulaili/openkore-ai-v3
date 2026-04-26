"""PDCA autonomy loop — continuous Plan-Do-Check-Act cycle."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ai_sidecar.autonomy.plan_executor import PlanExecutor
from ai_sidecar.autonomy.progress_tracker import ProgressTracker
from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.crewai import CrewStrategizeRequest
from ai_sidecar.contracts.autonomy import GoalStackState
from ai_sidecar.planner.schemas import PlannerResponse, StrategicPlan, TacticalIntentBundle
from ai_sidecar.planner.schemas import PlanHorizon, PlannerPlanRequest
from ai_sidecar.contracts.state import BotStateSnapshot
from ai_sidecar.reflex.circuit_breaker import ReflexCircuitBreaker

logger = logging.getLogger(__name__)


class Horizon(Enum):
    SHORT_TERM = "short_term"      # 5s  — tactical movement, combat
    MEDIUM_TERM = "medium_term"    # 30s — zone clearing, quest step
    LONG_TERM = "long_term"        # 120s — map transition, gear upgrade


@dataclass
class PDCAResult:
    """Outcome of a single PDCA cycle iteration."""

    horizon: Horizon
    plan_id: str | None
    actions_queued: int
    progress_pct: float
    stuck: bool
    re_planned: bool
    cycle_ms: float
    force_replan: bool = False
    replan_reasons: list[str] = field(default_factory=list)
    objective: str = ""
    selected_goal: str = ""
    error: str | None = None


@dataclass
class PDCAConfig:
    """Configuration for the PDCA loop."""

    short_term_interval_s: float = 5.0
    medium_term_interval_s: float = 30.0
    long_term_interval_s: float = 120.0
    max_stuck_cycles: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_s: float = 60.0
    plan_timeout_s: float = 30.0
    max_actions_per_cycle: int = 5


class PDCALoop:
    """Continuous Plan-Do-Check-Act autonomy loop.

    Runs three nested horizons:
      SHORT_TERM  — every 5s,  tactical decisions (move, attack, loot)
      MEDIUM_TERM — every 30s, tactical bundles (clear zone, quest step)
      LONG_TERM   — every 120s, strategic plans (map change, gear upgrade)
    """

    def __init__(
        self,
        runtime_state: Any,  # RuntimeState from lifecycle
        config: PDCAConfig | None = None,
    ) -> None:
        self._runtime = runtime_state
        self._config = config or PDCAConfig()
        self._plan_executor = PlanExecutor(runtime_state)
        self._progress_tracker = ProgressTracker(runtime_state)
        self._circuit_breaker = ReflexCircuitBreaker()
        self._breaker_bot_id = "pdca"
        self._breaker_key = "queue.default"
        self._breaker_family = "queue"
        self._default_bot_id = "openkoreai"
        self._last_bot_id: str | None = None

        # Per-horizon state
        self._active_plan: dict[Horizon, StrategicPlan | TacticalIntentBundle | None] = {
            h: None for h in Horizon
        }
        self._last_plan_time: dict[Horizon, float] = {h: 0.0 for h in Horizon}
        self._stuck_counter: dict[Horizon, int] = {h: 0 for h in Horizon}
        self._objective_rotation_index: dict[Horizon, int] = {h: 0 for h in Horizon}
        self._last_objective_switch_at: dict[Horizon, float] = {h: 0.0 for h in Horizon}
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._cycle_count: int = 0

    # ── Public API ──────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._running

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def start(self) -> None:
        """Start the PDCA loop in a background task."""
        if self._running:
            logger.warning("PDCALoop already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("PDCALoop started")

    async def stop(self) -> None:
        """Stop the PDCA loop gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("PDCALoop stopped")

    async def get_status(self) -> dict[str, Any]:
        """Return current loop status as a dict."""
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "circuit_breaker_tripped": self._circuit_breaker_tripped(),
            "horizons": {
                h.value: {
                    "has_active_plan": self._active_plan[h] is not None,
                    "stuck_cycles": self._stuck_counter[h],
                    "last_plan_seconds_ago": time.time() - self._last_plan_time[h]
                    if self._last_plan_time[h] > 0
                    else None,
                }
                for h in Horizon
            },
        }

    # ── Internal loop ───────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Main async loop — runs until stopped."""
        logger.info("PDCALoop _run_loop entered")
        while self._running:
            try:
                now = time.time()
                for horizon in Horizon:
                    if self._circuit_breaker_tripped():
                        logger.warning("Circuit breaker tripped — skipping all horizons")
                        await asyncio.sleep(1.0)
                        continue

                    interval = self._interval_for(horizon)
                    if now - self._last_plan_time[horizon] >= interval:
                        result = await self._run_one_cycle(horizon)
                        self._cycle_count += 1
                        self._last_plan_time[horizon] = time.time()

                        if result.error:
                            self._circuit_breaker.record_failure(
                                bot_id=self._breaker_bot_id,
                                key=self._breaker_key,
                                family=self._breaker_family,
                                reason=result.error,
                            )
                            logger.error("PDCA cycle error [%s]: %s", horizon.value, result.error)
                        else:
                            self._circuit_breaker.record_success(
                                bot_id=self._breaker_bot_id,
                                key=self._breaker_key,
                                family=self._breaker_family,
                            )

                        # Log cycle result
                        logger.info(
                            "PDCA [%s] plan=%s actions=%d progress=%.1f%% stuck=%s replan=%s force=%s goal=%s objective=%s reasons=%s cycle_ms=%.1f",
                            horizon.value,
                            result.plan_id,
                            result.actions_queued,
                            result.progress_pct * 100,
                            result.stuck,
                            result.re_planned,
                            result.force_replan,
                            result.selected_goal,
                            result.objective,
                            ",".join(result.replan_reasons),
                            result.cycle_ms,
                        )

                # Sleep a short interval before re-checking horizons
                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                logger.info("PDCALoop cancelled")
                break
            except Exception:
                logger.exception("PDCALoop unhandled error")
                await asyncio.sleep(5.0)

    async def _run_one_cycle(self, horizon: Horizon) -> PDCAResult:
        """Execute one PDCA cycle for the given horizon."""
        start = time.monotonic()
        plan_id: str | None = self._artifact_id(self._active_plan[horizon])
        actions_queued = 0
        re_planned = False
        force_replan = False
        objective = ""
        selected_goal = ""
        replan_reasons: list[str] = []

        try:
            # ── CHECK phase ──────────────────────────────────────
            latest_snapshot = self._get_latest_snapshot()
            progress = self._progress_tracker.evaluate(
                horizon=horizon,
                active_plan=self._active_plan[horizon],
                snapshot=latest_snapshot,
            )

            stuck = progress.stuck_cycles >= self._config.max_stuck_cycles
            replan_reasons = self._collect_replan_reasons(
                horizon=horizon,
                progress=progress,
                snapshot=latest_snapshot,
            )
            force_replan = bool(replan_reasons)

            decision_meta = ContractMeta(source="pdca_loop", bot_id=self._resolve_bot_id(latest_snapshot))
            goal_state = self._select_goal_state(
                meta=decision_meta,
                horizon=horizon,
                replan_reasons=replan_reasons,
            )
            if goal_state is not None:
                selected_goal = goal_state.selected_goal.goal_key.value
                objective = goal_state.selected_goal.objective

            # ── PLAN phase ───────────────────────────────────────
            if self._active_plan[horizon] is None or force_replan:
                objective_override = self._select_objective(
                    horizon=horizon,
                    snapshot=latest_snapshot,
                    replan_reasons=replan_reasons,
                    goal_state=goal_state,
                )
                objective = objective_override or self._objective_for(horizon=horizon, snapshot=latest_snapshot)
                plan = await self._generate_plan(
                    horizon,
                    latest_snapshot,
                    force_replan=force_replan and self._active_plan[horizon] is not None,
                    objective_override=objective_override,
                )
                if plan:
                    self._active_plan[horizon] = plan
                    plan_id = self._artifact_id(plan) or f"plan_{int(time.time())}"
                    re_planned = True
                    self._stuck_counter[horizon] = 0
                else:
                    return PDCAResult(
                        horizon=horizon,
                        plan_id=None,
                        actions_queued=0,
                        progress_pct=progress.progress_pct,
                        stuck=stuck,
                        re_planned=False,
                        force_replan=force_replan,
                        replan_reasons=replan_reasons,
                        objective=objective,
                        selected_goal=selected_goal,
                        cycle_ms=(time.monotonic() - start) * 1000,
                        error="plan generation returned None",
                    )
            else:
                objective = objective or self._objective_for(horizon=horizon, snapshot=latest_snapshot)

            # ── DO phase ─────────────────────────────────────────
            if self._active_plan[horizon] is not None:
                actions_queued = await self._plan_executor.execute(
                    plan=self._active_plan[horizon],
                    horizon=horizon,
                    max_actions=self._config.max_actions_per_cycle,
                )

            # ── ACT phase ────────────────────────────────────────
            if force_replan and re_planned:
                self._stuck_counter[horizon] = 0
                logger.info(
                    "Re-planned [%s] after %d stuck cycles reasons=%s",
                    horizon.value,
                    progress.stuck_cycles,
                    ",".join(replan_reasons),
                )
            elif progress.stuck_cycles > 0:
                self._stuck_counter[horizon] = progress.stuck_cycles

            plan_id = plan_id or self._artifact_id(self._active_plan[horizon])

            return PDCAResult(
                horizon=horizon,
                plan_id=plan_id,
                actions_queued=actions_queued,
                progress_pct=progress.progress_pct,
                stuck=stuck,
                re_planned=re_planned,
                force_replan=force_replan,
                replan_reasons=replan_reasons,
                objective=objective,
                selected_goal=selected_goal,
                cycle_ms=(time.monotonic() - start) * 1000,
            )

        except Exception as e:
            logger.exception("PDCA cycle failed [%s]", horizon.value)
            return PDCAResult(
                horizon=horizon,
                plan_id=plan_id,
                actions_queued=actions_queued,
                progress_pct=0.0,
                stuck=False,
                re_planned=re_planned,
                force_replan=force_replan,
                replan_reasons=replan_reasons,
                objective=objective,
                selected_goal=selected_goal,
                cycle_ms=(time.monotonic() - start) * 1000,
                error=str(e),
            )

    def _artifact_id(self, artifact: StrategicPlan | TacticalIntentBundle | None) -> str | None:
        if artifact is None:
            return None
        return (
            getattr(artifact, "plan_id", None)
            or getattr(artifact, "bundle_id", None)
            or getattr(artifact, "id", None)
        )

    def _interval_for(self, horizon: Horizon) -> float:
        if horizon == Horizon.SHORT_TERM:
            return self._config.short_term_interval_s
        elif horizon == Horizon.MEDIUM_TERM:
            return self._config.medium_term_interval_s
        return self._config.long_term_interval_s

    def _get_latest_snapshot(self) -> BotStateSnapshot | None:
        """Get the latest snapshot from the runtime's snapshot cache."""
        try:
            bot_id = self._resolve_bot_id()
            cache = getattr(self._runtime, "snapshot_cache", None)
            if cache and hasattr(cache, "get"):
                snapshot = cache.get(bot_id)
                if snapshot is not None:
                    self._last_bot_id = bot_id
                    return snapshot
        except Exception:
            logger.exception("Failed to get latest snapshot")
        return None

    async def _generate_plan(
        self,
        horizon: Horizon,
        snapshot: BotStateSnapshot | None,
        *,
        force_replan: bool = False,
        objective_override: str | None = None,
    ) -> StrategicPlan | TacticalIntentBundle | None:
        """Generate a plan using planner or crewAI depending on horizon."""
        try:
            bot_id = self._resolve_bot_id(snapshot)
            objective = objective_override or self._objective_for(horizon=horizon, snapshot=snapshot)
            if horizon == Horizon.LONG_TERM:
                # Use crewAI strategize for long-term strategic plans
                result = await self._runtime.crewai_strategize(
                    CrewStrategizeRequest(
                        meta=ContractMeta(source="pdca_loop", bot_id=bot_id),
                        objective=objective,
                        horizon=PlanHorizon.strategic,
                        force_replan=force_replan,
                        max_steps=12,
                        context_overrides=self._context_overrides(snapshot),
                    )
                )
                planner_response = getattr(result, "planner_response", None)
                if result and getattr(result, "ok", False) and planner_response is not None:
                    return planner_response.strategic_plan or planner_response.tactical_bundle
                return None
            elif horizon == Horizon.MEDIUM_TERM:
                # Use planner for medium-term tactical bundles
                result = await self._runtime.planner_plan(
                    PlannerPlanRequest(
                        meta=ContractMeta(source="pdca_loop", bot_id=bot_id),
                        objective=objective,
                        horizon=PlanHorizon.tactical,
                        force_replan=force_replan,
                        max_steps=8,
                    )
                )
                if result and getattr(result, "ok", False) and result.tactical_bundle is not None:
                    return result.tactical_bundle
                return None
            else:
                # SHORT_TERM: use planner for immediate actions
                result = await self._runtime.planner_plan(
                    PlannerPlanRequest(
                        meta=ContractMeta(source="pdca_loop", bot_id=bot_id),
                        objective=objective,
                        horizon=PlanHorizon.tactical,
                        force_replan=force_replan,
                        max_steps=4,
                    )
                )
                if result and getattr(result, "ok", False) and result.tactical_bundle is not None:
                    return result.tactical_bundle
                return None
        except Exception:
            logger.exception("Plan generation failed [%s]", horizon.value)
            return None

    def _circuit_breaker_tripped(self) -> bool:
        allowed, _state = self._circuit_breaker.allow(
            bot_id=self._breaker_bot_id,
            key=self._breaker_key,
            family=self._breaker_family,
        )
        return not allowed

    def _resolve_bot_id(self, snapshot: BotStateSnapshot | None = None) -> str:
        if snapshot is not None and getattr(snapshot, "meta", None) is not None:
            bot_id = getattr(snapshot.meta, "bot_id", None)
            if bot_id:
                self._last_bot_id = str(bot_id)
                return self._last_bot_id

        if self._last_bot_id:
            return self._last_bot_id

        try:
            if hasattr(self._runtime, "list_bots"):
                bots = self._runtime.list_bots()
                if bots:
                    first = bots[0]
                    bot_id = first.get("bot_id") if isinstance(first, dict) else getattr(first, "bot_id", None)
                    if bot_id:
                        self._last_bot_id = str(bot_id)
                        return self._last_bot_id
        except Exception:
            logger.exception("Failed to resolve active bot id for PDCA loop")

        return self._default_bot_id

    def _objective_for(self, *, horizon: Horizon, snapshot: BotStateSnapshot | None) -> str:
        current_map = getattr(getattr(snapshot, "position", None), "map", None) or "unknown"
        if horizon == Horizon.LONG_TERM:
            return f"advance long-term progression safely from {current_map}"
        if horizon == Horizon.MEDIUM_TERM:
            return f"progress tactical objective safely on {current_map}"
        return f"execute immediate tactical actions safely on {current_map}"

    def _collect_replan_reasons(
        self,
        *,
        horizon: Horizon,
        progress: Any,
        snapshot: BotStateSnapshot | None,
    ) -> list[str]:
        reasons: list[str] = []

        progress_reasons = list(getattr(progress, "reasons", []) or [])
        for item in progress_reasons:
            if item and item not in reasons:
                reasons.append(str(item))

        if bool(getattr(progress, "force_replan_hint", False)) is False and progress.stuck_cycles >= self._config.max_stuck_cycles:
            reasons.append("stuck_cycles")

        if snapshot is not None:
            if self._snapshot_disconnected(snapshot):
                reasons.append("disconnect_recovery")
            reconnect_age_s = self._snapshot_reconnect_age_s(snapshot)
            if reconnect_age_s is not None and reconnect_age_s >= self._policy_float("reconnect_grace_s", 20.0):
                reasons.append("reconnect_stale")
            if self._overweight_ratio(snapshot) >= 0.90:
                reasons.append("inventory_overweight_pressure")

        fleet_status = self._fleet_status()
        fleet_central_enabled = bool(fleet_status.get("central_enabled", True))
        if fleet_central_enabled and bool(fleet_status.get("stale", False)):
            reasons.append("fleet_central_stale")
        if fleet_central_enabled and bool(fleet_status.get("central_available", True)) is False:
            reasons.append("fleet_central_unavailable")

        # Limit trigger aggression for long-term horizon to hard stale/failure reasons.
        if horizon == Horizon.LONG_TERM:
            hard = {
                "stale_progress",
                "objective_aged_out",
                "death_loop_detected",
                "disconnect_recovery",
                "reconnect_stale",
                "fleet_central_stale",
                "fleet_central_unavailable",
                "stuck_cycles",
            }
            reasons = [item for item in reasons if item in hard]

        deduped: list[str] = []
        seen: set[str] = set()
        for item in reasons:
            key = str(item).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    def _select_objective(
        self,
        *,
        horizon: Horizon,
        snapshot: BotStateSnapshot | None,
        replan_reasons: list[str],
        goal_state: GoalStackState | None = None,
    ) -> str | None:
        if goal_state is not None:
            objective = str(goal_state.selected_goal.objective or "").strip()
            if objective:
                return objective

        if horizon == Horizon.LONG_TERM:
            return None

        ranked = self._ranked_objectives()
        if not ranked:
            return None

        preferred: str | None = None
        if "inventory_overweight_pressure" in replan_reasons:
            preferred = "economy"
        elif "death_loop_detected" in replan_reasons or "disconnect_recovery" in replan_reasons or "reconnect_stale" in replan_reasons:
            preferred = "recovery"
        elif "objective_aged_out" in replan_reasons and "quest" in ranked:
            preferred = "quest"
        elif "stale_progress" in replan_reasons or "map_dwell_no_gain" in replan_reasons or "route_churn_no_position_gain" in replan_reasons:
            preferred = "grind"

        choice = self._pick_ranked_objective(horizon=horizon, ranked=ranked, preferred=preferred, force_rotate=bool(replan_reasons))
        if choice is None:
            return None

        current_map = getattr(getattr(snapshot, "position", None), "map", None) or "unknown"
        if choice == "recovery":
            return f"recover safely and re-establish operational posture on {current_map}"
        if choice == "economy":
            return f"stabilize inventory and economy pressure safely from {current_map}"
        if choice == "quest":
            return f"advance active quest objectives safely near {current_map}"
        return f"resume efficient grind and loot progression safely on {current_map}"

    def _select_goal_state(
        self,
        *,
        meta: ContractMeta,
        horizon: Horizon,
        replan_reasons: list[str],
    ) -> GoalStackState | None:
        if not hasattr(self._runtime, "autonomy_decide"):
            return None
        try:
            return self._runtime.autonomy_decide(
                meta=meta,
                horizon=horizon.value,
                replan_reasons=replan_reasons,
            )
        except Exception:
            logger.exception(
                "pdca_goal_decision_failed",
                extra={
                    "event": "pdca_goal_decision_failed",
                    "bot_id": meta.bot_id,
                    "horizon": horizon.value,
                },
            )
            return None

    def _pick_ranked_objective(
        self,
        *,
        horizon: Horizon,
        ranked: list[str],
        preferred: str | None,
        force_rotate: bool,
    ) -> str | None:
        if not ranked:
            return None

        now = time.time()
        current_index = int(self._objective_rotation_index.get(horizon, 0))
        last_switch = float(self._last_objective_switch_at.get(horizon, 0.0))
        cooldown_s = self._policy_float("objective_rotation_cooldown_s", 20.0)

        if preferred in ranked:
            current_index = ranked.index(preferred)
            self._objective_rotation_index[horizon] = current_index
            self._last_objective_switch_at[horizon] = now
            return ranked[current_index]

        if last_switch <= 0.0:
            self._objective_rotation_index[horizon] = current_index
            self._last_objective_switch_at[horizon] = now
            return ranked[current_index]

        if force_rotate or (now - last_switch) >= cooldown_s:
            current_index = (current_index + 1) % len(ranked)
            self._objective_rotation_index[horizon] = current_index
            self._last_objective_switch_at[horizon] = now

        return ranked[current_index]

    def _ranked_objectives(self) -> list[str]:
        policy = getattr(self._runtime, "autonomy_policy", {})
        ranked = []
        if isinstance(policy, dict):
            raw = policy.get("ranked_objectives")
            if isinstance(raw, list):
                ranked = [str(item).strip().lower() for item in raw if str(item).strip()]
            elif isinstance(raw, str):
                ranked = [item.strip().lower() for item in raw.split(",") if item.strip()]

        if not ranked:
            ranked = ["grind", "recovery", "economy", "quest"]
        return ranked

    def _policy_float(self, key: str, default: float) -> float:
        policy = getattr(self._runtime, "autonomy_policy", {})
        if isinstance(policy, dict):
            try:
                return float(policy.get(key, default))
            except (TypeError, ValueError):
                return default
        return default

    def _fleet_status(self) -> dict[str, object]:
        state = getattr(self._runtime, "fleet_constraint_state", None)
        if state is not None and hasattr(state, "status"):
            try:
                data = state.status()
                if isinstance(data, dict):
                    return data
            except Exception:
                logger.exception("Failed to read fleet status for PDCA")
        return {
            "mode": "local",
            "central_enabled": True,
            "central_available": False,
            "stale": True,
            "last_sync_at": None,
            "doctrine_version": "local",
            "last_error": "fleet_constraint_state_unavailable",
        }

    def _snapshot_disconnected(self, snapshot: BotStateSnapshot) -> bool:
        raw = getattr(snapshot, "raw", {})
        if not isinstance(raw, dict):
            return False
        if raw.get("in_game") is False:
            return True
        status = str(raw.get("status") or raw.get("state") or raw.get("net_state") or "").strip().lower()
        return status in {
            "offline",
            "disconnected",
            "disconnect",
            "reconnecting",
            "connecting",
            "not_connected",
        }

    def _snapshot_reconnect_age_s(self, snapshot: BotStateSnapshot) -> float | None:
        raw = getattr(snapshot, "raw", {})
        if not isinstance(raw, dict):
            return None
        for key in ("reconnect_age_s", "disconnect_age_s", "offline_age_s"):
            value = raw.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    continue
        return None

    def _overweight_ratio(self, snapshot: BotStateSnapshot) -> float:
        vitals = getattr(snapshot, "vitals", None)
        weight = getattr(vitals, "weight", None)
        weight_max = getattr(vitals, "weight_max", None)
        if not isinstance(weight, int) or not isinstance(weight_max, int) or weight_max <= 0:
            return 0.0
        return max(0.0, min(2.0, float(weight) / float(weight_max)))

    def _context_overrides(self, snapshot: BotStateSnapshot | None) -> dict[str, object]:
        if snapshot is None:
            return {}
        return {
            "map": getattr(getattr(snapshot, "position", None), "map", None),
            "tick_id": getattr(snapshot, "tick_id", None),
        }
