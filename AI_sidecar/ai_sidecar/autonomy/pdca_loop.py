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
_STARTUP_GATE_MIN_EVENTS = 2
_STARTUP_GATE_MAX_CREW_FAILURES = 2


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
        self._startup_gate_defaults = {
            "grace_s": max(20.0, self._policy_float("reconnect_grace_s", 20.0)),
            "min_events": _STARTUP_GATE_MIN_EVENTS,
            "max_crewai_failures": _STARTUP_GATE_MAX_CREW_FAILURES,
        }

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
                goal_metadata = goal_state.selected_goal.metadata if isinstance(goal_state.selected_goal.metadata, dict) else {}
                if bool(goal_metadata.get("mission_force_replan")):
                    if "mission_agent_replan" not in replan_reasons:
                        replan_reasons.append("mission_agent_replan")
                    force_replan = True

            startup_gate = self._evaluate_startup_gate(
                bot_id=decision_meta.bot_id,
                horizon=horizon,
                snapshot=latest_snapshot,
                goal_state=goal_state,
                replan_reasons=replan_reasons,
            )
            if not bool(startup_gate.get("gate_open", False)):
                logger.info(
                    "pdca_startup_gate_blocked",
                    extra={
                        "event": "pdca_startup_gate_blocked",
                        "bot_id": decision_meta.bot_id,
                        "horizon": horizon.value,
                        "reason": startup_gate.get("reason"),
                        "mode": startup_gate.get("mode"),
                        "snapshot_ready": startup_gate.get("snapshot_ready"),
                        "history_ready": startup_gate.get("history_ready"),
                        "continuity_goal_state_present": startup_gate.get("continuity_goal_state_present"),
                        "recent_event_count": startup_gate.get("recent_event_count"),
                    },
                )
                return PDCAResult(
                    horizon=horizon,
                    plan_id=plan_id,
                    actions_queued=0,
                    progress_pct=progress.progress_pct,
                    stuck=stuck,
                    re_planned=False,
                    force_replan=force_replan,
                    replan_reasons=replan_reasons,
                    objective=objective,
                    selected_goal=selected_goal,
                    cycle_ms=(time.monotonic() - start) * 1000,
                    error=None,
                )

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
                    startup_gate=startup_gate,
                )
                if plan:
                    self._active_plan[horizon] = plan
                    plan_id = self._artifact_id(plan) or f"plan_{int(time.time())}"
                    re_planned = True
                    self._stuck_counter[horizon] = 0
                else:
                    logger.warning(
                        "pdca_plan_generation_unavailable",
                        extra={
                            "event": "pdca_plan_generation_unavailable",
                            "bot_id": decision_meta.bot_id,
                            "horizon": horizon.value,
                            "mode": startup_gate.get("mode"),
                            "reason": startup_gate.get("reason"),
                            "selected_goal": selected_goal,
                            "objective": objective,
                        },
                    )
            else:
                objective = objective or self._objective_for(horizon=horizon, snapshot=latest_snapshot)

            # ── DO phase ─────────────────────────────────────────
            if self._active_plan[horizon] is not None or goal_state is not None:
                actions_queued = await self._plan_executor.execute(
                    plan=self._active_plan[horizon],
                    horizon=horizon,
                    max_actions=self._config.max_actions_per_cycle,
                    goal_state=goal_state,
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
        startup_gate: dict[str, object] | None = None,
    ) -> StrategicPlan | TacticalIntentBundle | None:
        """Generate a plan using planner or crewAI depending on horizon."""
        try:
            bot_id = self._resolve_bot_id(snapshot)
            objective = objective_override or self._objective_for(horizon=horizon, snapshot=snapshot)
            if horizon == Horizon.LONG_TERM:
                # Use crewAI strategize for long-term strategic plans
                crewai_reason = "crewai_unusable"
                startup_mode = str((startup_gate or {}).get("mode") or "pending")
                startup_reason = str((startup_gate or {}).get("reason") or "")
                if startup_mode not in {"conscious", "degraded"}:
                    logger.warning(
                        "pdca_long_term_startup_gate_unexpected_mode",
                        extra={
                            "event": "pdca_long_term_startup_gate_unexpected_mode",
                            "bot_id": bot_id,
                            "mode": startup_mode,
                            "reason": startup_reason,
                        },
                    )
                try:
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
                    crew_ok = bool(getattr(result, "ok", False))
                    crew_message = str(getattr(result, "message", "") or "")
                    crewai_errors = [str(item) for item in list(getattr(result, "errors", []) or [])]
                    crew_signals = [crew_message, *crewai_errors]
                    crew_degraded_signal = any(
                        token in signal.lower()
                        for signal in crew_signals
                        for token in ("crewai_disabled", "crewai_unavailable", "crewai_pipeline_disabled")
                    )

                    planner_response = getattr(result, "planner_response", None)
                    if crew_ok and not crew_degraded_signal and planner_response is not None:
                        artifact = planner_response.strategic_plan or planner_response.tactical_bundle
                        if artifact is not None:
                            self._record_startup_gate_success(bot_id=bot_id)
                            return artifact
                    crewai_reason = ",".join([crewai_reason, *crewai_errors]).strip(",") or "crewai_unusable"
                    self._record_startup_gate_failure(bot_id=bot_id, reason=crewai_reason)
                except Exception as exc:
                    crewai_reason = f"crewai_exception:{type(exc).__name__}"
                    self._record_startup_gate_failure(bot_id=bot_id, reason=crewai_reason)

                planner_fn = getattr(self._runtime, "planner_plan", None)
                if callable(planner_fn):
                    fallback_state = self._startup_gate_status(bot_id=bot_id)
                    fallback_mode = str(fallback_state.get("mode") or "warmup")
                    fallback_reason = str(fallback_state.get("reason") or "")
                    if fallback_mode != "degraded":
                        logger.error(
                            "pdca_long_term_conscious_required_before_fallback",
                            extra={
                                "event": "pdca_long_term_conscious_required_before_fallback",
                                "bot_id": bot_id,
                                "reason": crewai_reason,
                                "startup_mode": fallback_mode,
                                "startup_reason": fallback_reason,
                            },
                        )
                        return None
                    logger.info(
                        "pdca_long_term_fallback_to_planner",
                        extra={
                            "event": "pdca_long_term_fallback_to_planner",
                            "bot_id": bot_id,
                            "objective": objective,
                            "reason": crewai_reason,
                            "startup_mode": fallback_mode,
                            "startup_reason": fallback_reason,
                        },
                    )
                    fallback = await planner_fn(
                        PlannerPlanRequest(
                            meta=ContractMeta(source="pdca_loop", bot_id=bot_id),
                            objective=objective,
                            horizon=PlanHorizon.strategic,
                            force_replan=force_replan,
                            max_steps=12,
                        )
                    )
                    if fallback and getattr(fallback, "ok", False):
                        artifact = fallback.strategic_plan or fallback.tactical_bundle
                        if artifact is not None:
                            return artifact
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
            return self._fallback_goal_state(meta=meta, horizon=horizon)
        try:
            decided = self._runtime.autonomy_decide(
                meta=meta,
                horizon=horizon.value,
                replan_reasons=replan_reasons,
            )
            if decided is not None:
                return decided
            return self._fallback_goal_state(meta=meta, horizon=horizon)
        except Exception:
            logger.exception(
                "pdca_goal_decision_failed",
                extra={
                    "event": "pdca_goal_decision_failed",
                    "bot_id": meta.bot_id,
                    "horizon": horizon.value,
                },
            )
            return self._fallback_goal_state(meta=meta, horizon=horizon)

    def _fallback_goal_state(self, *, meta: ContractMeta, horizon: Horizon) -> GoalStackState | None:
        latest_fn = getattr(self._runtime, "latest_goal_state", None)
        if not callable(latest_fn):
            return None
        try:
            restored = latest_fn(bot_id=meta.bot_id)
        except Exception:
            logger.exception(
                "pdca_goal_state_restore_failed",
                extra={
                    "event": "pdca_goal_state_restore_failed",
                    "bot_id": meta.bot_id,
                    "horizon": horizon.value,
                },
            )
            return None
        if restored is None:
            return None
        logger.info(
            "pdca_goal_state_restored_from_runtime_cache",
            extra={
                "event": "pdca_goal_state_restored_from_runtime_cache",
                "bot_id": meta.bot_id,
                "horizon": horizon.value,
                "restored_horizon": restored.horizon,
                "decision_version": restored.decision_version,
                "tick_id": restored.tick_id,
                "selected_goal": restored.selected_goal.goal_key.value,
            },
        )
        return restored

    def _startup_gate_status(self, *, bot_id: str) -> dict[str, object]:
        status_fn = getattr(self._runtime, "startup_gate_status", None)
        if callable(status_fn):
            try:
                status = status_fn(bot_id=bot_id)
                if isinstance(status, dict):
                    return status
            except Exception:
                logger.exception(
                    "pdca_startup_gate_status_failed",
                    extra={
                        "event": "pdca_startup_gate_status_failed",
                        "bot_id": bot_id,
                    },
                )

        return {
            "bot_id": bot_id,
            "gate_open": True,
            "mode": "degraded",
            "reason": "startup_gate_runtime_unavailable",
            "failure_count": 0,
            "last_error": "",
            "elapsed_s": 0.0,
            "grace_s": float(self._startup_gate_defaults["grace_s"]),
            "min_events": int(self._startup_gate_defaults["min_events"]),
            "snapshot_ready": True,
            "history_ready": True,
            "continuity_goal_state_present": True,
            "recent_event_count": int(self._startup_gate_defaults["min_events"]),
        }

    def _update_startup_gate(
        self,
        *,
        bot_id: str,
        gate_open: bool,
        mode: str,
        reason: str,
        failure_count: int,
        last_error: str,
        grace_s: float,
        min_events: int,
        major_reasons: list[str] | None = None,
    ) -> dict[str, object]:
        update_fn = getattr(self._runtime, "update_startup_gate", None)
        if callable(update_fn):
            try:
                status = update_fn(
                    bot_id=bot_id,
                    gate_open=gate_open,
                    mode=mode,
                    reason=reason,
                    failure_count=failure_count,
                    last_error=last_error,
                    grace_s=grace_s,
                    min_events=min_events,
                    major_reasons=major_reasons,
                )
                if isinstance(status, dict):
                    return status
            except Exception:
                logger.exception(
                    "pdca_startup_gate_update_failed",
                    extra={
                        "event": "pdca_startup_gate_update_failed",
                        "bot_id": bot_id,
                        "mode": mode,
                        "reason": reason,
                    },
                )
        return self._startup_gate_status(bot_id=bot_id)

    def _evaluate_startup_gate(
        self,
        *,
        bot_id: str,
        horizon: Horizon,
        snapshot: BotStateSnapshot | None,
        goal_state: GoalStackState | None,
        replan_reasons: list[str],
    ) -> dict[str, object]:
        status = self._startup_gate_status(bot_id=bot_id)
        snapshot_ready = bool(snapshot is not None and str(getattr(snapshot, "tick_id", "") or "").strip())
        if snapshot_ready:
            map_name = str(getattr(getattr(snapshot, "position", None), "map", "") or "").strip()
            snapshot_ready = bool(map_name)
        bot_ready = bool(snapshot is not None and snapshot_ready and not self._snapshot_disconnected(snapshot))

        continuity_goal_state_present = bool(goal_state is not None or status.get("continuity_goal_state_present", False))
        history_ready = bool(continuity_goal_state_present or status.get("history_ready", False))
        minimum_readiness = bool(bot_ready and history_ready)

        fleet_status = self._fleet_status()
        fleet_enabled = bool(fleet_status.get("central_enabled", True))
        fleet_stale = bool(fleet_status.get("stale", False))
        fleet_available = bool(fleet_status.get("central_available", False))

        planner_degraded = False
        planner_reason = ""
        planner_fn = getattr(self._runtime, "planner_status", None)
        if callable(planner_fn):
            try:
                planner = planner_fn(bot_id=bot_id)
                planner_healthy = bool(getattr(planner, "planner_healthy", False))
                planner_updated_at = getattr(planner, "updated_at", None)
                stale_seconds: float | None = None
                if isinstance(planner_updated_at, datetime):
                    if planner_updated_at.tzinfo is None:
                        planner_updated_at = planner_updated_at.replace(tzinfo=UTC)
                    stale_seconds = max(0.0, (datetime.now(UTC) - planner_updated_at.astimezone(UTC)).total_seconds())
                stale_threshold = max(float(getattr(self._runtime, "planner_stale_threshold_s", 60.0) or 60.0), 1.0)
                planner_stale = (not planner_healthy) or stale_seconds is None or stale_seconds > stale_threshold
                if planner_stale:
                    planner_degraded = True
                    planner_reason = "planner_stale"
            except Exception:
                planner_degraded = True
                planner_reason = "planner_status_unavailable"

        crew_degraded = False
        crew_reason = ""
        crew_status_fn = getattr(self._runtime, "crewai_status", None)
        if callable(crew_status_fn):
            try:
                crew_status = crew_status_fn()
                crew_available = bool(getattr(crew_status, "crew_available", False))
                crew_enabled = bool(getattr(crew_status, "crewai_enabled", True))
                if not crew_enabled:
                    crew_degraded = True
                    crew_reason = "crewai_disabled"
                elif not crew_available:
                    crew_degraded = True
                    crew_reason = "crewai_unavailable"
            except Exception:
                crew_degraded = True
                crew_reason = "crewai_status_unavailable"

        startup_failures = int(status.get("failure_count", 0) or 0)
        startup_last_error = str(status.get("last_error") or "").strip()

        degraded_reasons: list[str] = []
        if fleet_enabled and fleet_stale:
            degraded_reasons.append("fleet_central_stale")
        if fleet_enabled and not fleet_available:
            degraded_reasons.append("fleet_central_unavailable")
        if planner_degraded and planner_reason:
            degraded_reasons.append(planner_reason)
        if crew_degraded and crew_reason:
            degraded_reasons.append(crew_reason)
        if startup_failures > 0:
            degraded_reasons.append("crewai_failures_present")
        if startup_last_error:
            degraded_reasons.append(startup_last_error)

        deduped_reasons: list[str] = []
        seen: set[str] = set()
        for item in degraded_reasons:
            key = str(item).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped_reasons.append(key)

        grace_s = float(status.get("grace_s", self._startup_gate_defaults["grace_s"]) or self._startup_gate_defaults["grace_s"])
        min_events = int(status.get("min_events", self._startup_gate_defaults["min_events"]) or self._startup_gate_defaults["min_events"])

        if not minimum_readiness:
            wait_reasons: list[str] = []
            if not snapshot_ready:
                wait_reasons.append("snapshot_unavailable")
            if snapshot_ready and not bot_ready:
                wait_reasons.append("bot_not_ready")
            if not history_ready:
                wait_reasons.append("history_unavailable")
            reason = f"startup_gate_waiting_minimum_live_state:{','.join(wait_reasons) or 'minimum_state_unavailable'}"
            self._update_startup_gate(
                bot_id=bot_id,
                gate_open=False,
                mode="warmup",
                reason=reason,
                failure_count=int(status.get("failure_count", 0) or 0),
                last_error=str(status.get("last_error") or ""),
                grace_s=grace_s,
                min_events=min_events,
                major_reasons=wait_reasons,
            )
            return self._startup_gate_status(bot_id=bot_id)

        mode = "degraded" if deduped_reasons else "conscious"
        reason = (
            f"startup_gate_open_degraded_optional_subsystems:{','.join(deduped_reasons)}"
            if deduped_reasons
            else "startup_gate_open_minimum_live_state_ready"
        )
        opened = self._update_startup_gate(
            bot_id=bot_id,
            gate_open=True,
            mode=mode,
            reason=reason,
            failure_count=int(status.get("failure_count", 0) or 0),
            last_error=str(status.get("last_error") or ""),
            grace_s=grace_s,
            min_events=min_events,
            major_reasons=deduped_reasons,
        )
        logger.info(
            "pdca_startup_gate_ready",
            extra={
                "event": "pdca_startup_gate_ready",
                "bot_id": bot_id,
                "horizon": horizon.value,
                "mode": mode,
                "reason": reason,
                "snapshot_ready": snapshot_ready,
                "bot_ready": bot_ready,
                "history_ready": history_ready,
                "continuity_goal_state_present": continuity_goal_state_present,
                "degraded_reasons": deduped_reasons,
                "replan_reasons": list(replan_reasons),
            },
        )
        return opened

    def _record_startup_gate_success(self, *, bot_id: str) -> None:
        status = self._startup_gate_status(bot_id=bot_id)
        self._update_startup_gate(
            bot_id=bot_id,
            gate_open=bool(status.get("gate_open", False)),
            mode=str(status.get("mode") or "conscious"),
            reason=str(status.get("reason") or "startup_gate_open_state_history_ready"),
            failure_count=0,
            last_error="",
            grace_s=float(status.get("grace_s", self._startup_gate_defaults["grace_s"]) or self._startup_gate_defaults["grace_s"]),
            min_events=int(status.get("min_events", self._startup_gate_defaults["min_events"]) or self._startup_gate_defaults["min_events"]),
        )

    def _record_startup_gate_failure(self, *, bot_id: str, reason: str) -> None:
        status = self._startup_gate_status(bot_id=bot_id)
        failures = int(status.get("failure_count", 0) or 0) + 1
        elapsed_s = float(status.get("elapsed_s", 0.0) or 0.0)
        grace_s = float(status.get("grace_s", self._startup_gate_defaults["grace_s"]) or self._startup_gate_defaults["grace_s"])
        max_failures = int(self._startup_gate_defaults["max_crewai_failures"])
        normalized_reason = str(reason or "crewai_unusable").strip()
        reason_lower = normalized_reason.lower()
        immediate_tokens = ("crewai_disabled", "crewai_unavailable", "crewai_pipeline_disabled")
        degrade = failures >= max_failures or elapsed_s >= grace_s or any(token in reason_lower for token in immediate_tokens)

        mode = "degraded" if degrade else str(status.get("mode") or "conscious")
        gate_reason = "startup_gate_degraded_after_bounded_crewai_failures" if degrade else str(status.get("reason") or "startup_gate_open_minimum_live_state_ready")
        major_reasons = [
            str(item).strip()
            for item in list(status.get("major_reasons") or [])
            if str(item).strip()
        ]
        if normalized_reason:
            major_reasons.append(normalized_reason)
        self._update_startup_gate(
            bot_id=bot_id,
            gate_open=bool(status.get("gate_open", True)),
            mode=mode,
            reason=gate_reason,
            failure_count=failures,
            last_error=normalized_reason,
            grace_s=grace_s,
            min_events=int(status.get("min_events", self._startup_gate_defaults["min_events"]) or self._startup_gate_defaults["min_events"]),
            major_reasons=major_reasons,
        )
        if degrade:
            logger.warning(
                "pdca_startup_gate_degraded",
                extra={
                    "event": "pdca_startup_gate_degraded",
                    "bot_id": bot_id,
                    "failure_count": failures,
                    "elapsed_s": elapsed_s,
                    "grace_s": grace_s,
                    "reason": reason,
                },
            )

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
        runtime_status_fn = getattr(self._runtime, "_fleet_status", None)
        if callable(runtime_status_fn):
            try:
                data = runtime_status_fn()
                if isinstance(data, dict):
                    return data
            except Exception:
                logger.exception("Failed to read runtime fleet status for PDCA")

        state = getattr(self._runtime, "fleet_constraint_state", None)
        if state is not None and hasattr(state, "status"):
            try:
                data = state.status()
                if isinstance(data, dict):
                    return data
            except Exception:
                logger.exception("Failed to read fleet status for PDCA")

        fleet_client = getattr(self._runtime, "fleet_sync_client", None)
        central_enabled = bool(getattr(fleet_client, "enabled", True))
        return {
            "mode": "local",
            "central_enabled": central_enabled,
            "central_available": False,
            "stale": (True if central_enabled else False),
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
