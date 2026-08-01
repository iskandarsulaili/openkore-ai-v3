"""
Failure Pipeline Wiring — Wires the FailureReasoningEngine into all subsystems.

This module provides a single entry point to connect the failure reasoning
pipeline with the degradation manager, P2P knowledge sharing, PDCA loop,
and combat loop. It does NOT modify those files directly — instead it
provides hooks and callbacks that can be registered at startup.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def wire_failure_pipeline(runtime_state: Any) -> Any:
    """Wire the failure reasoning pipeline into the runtime at startup.

    This function:
    1. Gets or creates the FailureReasoningEngine singleton
    2. Registers it with the degradation manager
    3. Sets up P2P failure sharing
    4. Registers failure handlers in the PDCA loop
    5. Registers failure callbacks in the combat loop

    Args:
        runtime_state: The runtime state object (app.state.runtime)

    Returns:
        The FailureReasoningEngine instance
    """
    from ai_sidecar.learning.failure_reasoning import get_failure_reasoning_engine

    engine = get_failure_reasoning_engine()

    # Wire shared DB
    try:
        from ai_sidecar.learning.shared_learning_db import get_shared_learning_db
        shared_db = get_shared_learning_db()
        engine._shared_db = shared_db
        logger.info("failure_pipeline: wired shared_learning_db")
    except Exception:
        logger.warning("failure_pipeline: shared_learning_db not available")

    # Wire P2P node
    try:
        p2p_node = getattr(runtime_state, "p2p_node", None)
        if p2p_node is None:
            p2p_node = getattr(runtime_state, "p2p_knowledge", None)
        if p2p_node is None:
            p2p_node = getattr(runtime_state, "p2p_knowledge_node", None)
        if p2p_node is not None:
            engine._p2p_node = p2p_node
            logger.info("failure_pipeline: wired p2p_node")
    except Exception:
        logger.warning("failure_pipeline: p2p_node not available")

    # Wire server adaptation
    try:
        sa = getattr(runtime_state, "server_adaptation", None)
        if sa is not None:
            engine._server_adaptation = sa
            logger.info("failure_pipeline: wired server_adaptation")
    except Exception:
        logger.warning("failure_pipeline: server_adaptation not available")

    # Register with degradation manager
    try:
        dm = getattr(runtime_state, "degradation_manager", None)
        if dm is not None and hasattr(dm, "register_module"):
            dm.register_module(
                "failure_reasoning",
                engine,
                health_check=lambda: True,
            )
            logger.info("failure_pipeline: registered with degradation_manager")
    except Exception:
        logger.warning("failure_pipeline: degradation_manager registration failed")

    # Store engine on runtime for access by other subsystems
    runtime_state.failure_reasoning_engine = engine

    logger.info("failure_pipeline: wired successfully")
    return engine


def get_failure_callback(runtime_state: Any) -> Any:
    """Get a failure capture callback for use in combat_loop and other subsystems.

    Returns a callable that captures a failure via the FailureReasoningEngine.
    """
    engine = getattr(runtime_state, "failure_reasoning_engine", None)
    if engine is None:
        try:
            engine = wire_failure_pipeline(runtime_state)
        except Exception:
            return None

    def capture(
        category: str,
        subcategory: str | None = None,
        context: dict | None = None,
        bot_id: str = "default",
    ) -> str | None:
        try:
            return engine.capture_failure(
                category=category,
                subcategory=subcategory,
                context=context,
                bot_id=bot_id,
            )
        except Exception:
            logger.exception("failure_callback_failed: category=%s", category)
            return None

    return capture


def get_recurring_failures_check(runtime_state: Any) -> Any:
    """Get a recurring failures check callback for use in the PDCA loop.

    Returns a callable that checks for recurring failures and applies
    config adjustments if any have count >= 3.
    """
    from datetime import timedelta
    engine = getattr(runtime_state, "failure_reasoning_engine", None)
    if engine is None:
        return None

    def check_and_adjust(server_id: str | None = None) -> list[dict]:
        try:
            recurring = engine.get_recurring_failures(
                server_id=server_id, min_count=3, limit=10,
            )
            applied: list[dict] = []
            aq = getattr(runtime_state, "action_queue", None)
            for failure in recurring:
                from ai_sidecar.learning.failure_reasoning import FailureRecord
                record = FailureRecord(
                    id=failure.get("id", ""),
                    server_id=failure.get("server_id", "default"),
                    bot_id=failure.get("bot_id", ""),
                    category=failure.get("category", "unknown"),
                    subcategory=failure.get("subcategory"),
                    timestamp=failure.get("timestamp", 0.0),
                    context={},
                    recurrence_count=failure.get("recurrence_count", 1),
                )
                config_changes = engine._apply_config_adjustment(record)
                if config_changes:
                    logger.info(
                        "failure_pipeline: auto-adjusting config for %s/%s: %s",
                        record.category, record.subcategory, config_changes,
                    )
                    # ACTUALLY APPLY: enqueue `set <key> <value>` commands
                    # through the bridge's `set` config path (was recorded
                    # but never applied — a dormant incomplete loop).
                    if aq is not None and record.bot_id:
                        from ai_sidecar.contracts.actions import ActionProposal, ActionPriorityTier
                        from ai_sidecar.contracts.common import utc_now
                        for change in config_changes:
                            change = change.strip()
                            if not change:
                                continue
                            _key = change.split(maxsplit=1)[0]
                            _now = utc_now()
                            aq.enqueue(record.bot_id, ActionProposal(
                                action_id=f"failure-{record.id[:20]}-{_key}",
                                bot_id=record.bot_id,
                                action_type="set",
                                command=f"set {change}",
                                priority_tier=ActionPriorityTier.strategic,
                                source="failure_reasoning",
                                conflict_key=f"failure_adjust.{_key}",
                                created_at=_now,
                                expires_at=_now + timedelta(seconds=300),
                                idempotency_key=f"failure:{record.id}:{_key}",
                                metadata={
                                    "failure_id": record.id,
                                    "category": record.category,
                                    "subcategory": record.subcategory or "",
                                },
                            ))
                    applied.append({
                        "id": record.id,
                        "category": record.category,
                        "changes": config_changes,
                        "bot_id": record.bot_id,
                    })
            return applied or recurring
        except Exception:
            logger.exception("recurring_failures_check_failed")
            return []

    return check_and_adjust
