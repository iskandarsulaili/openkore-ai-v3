from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.contracts.reflex import (
    ReflexBreakerListResponse,
    ReflexRuleEnableRequest,
    ReflexRuleEnableResponse,
    ReflexRuleListResponse,
    ReflexRuleUpsertRequest,
    ReflexRuleUpsertResponse,
    ReflexTriggerListResponse,
)
from ai_sidecar.lifecycle import RuntimeState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2/reflex", tags=["reflex"])


@router.post("/rules", response_model=ReflexRuleUpsertResponse)
def upsert_rule(
    payload: ReflexRuleUpsertRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> ReflexRuleUpsertResponse:
    ok, message = runtime.upsert_reflex_rule(bot_id=payload.meta.bot_id, rule=payload.rule)
    logger.info(
        "v2_reflex_rule_upsert",
        extra={
            "event": "v2_reflex_rule_upsert",
            "bot_id": payload.meta.bot_id,
            "rule_id": payload.rule.rule_id,
            "ok": ok,
        },
    )
    return ReflexRuleUpsertResponse(ok=True, saved=ok, rule_id=payload.rule.rule_id, message=message)


@router.get("/rules/{bot_id}", response_model=ReflexRuleListResponse)
def list_rules(
    bot_id: str,
    runtime: RuntimeState = Depends(get_runtime),
) -> ReflexRuleListResponse:
    return ReflexRuleListResponse(ok=True, bot_id=bot_id, rules=runtime.list_reflex_rules(bot_id=bot_id))


@router.post("/rules/{rule_id}/enable", response_model=ReflexRuleEnableResponse)
def set_rule_enabled(
    rule_id: str,
    payload: ReflexRuleEnableRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> ReflexRuleEnableResponse:
    updated = runtime.enable_reflex_rule(bot_id=payload.meta.bot_id, rule_id=rule_id, enabled=payload.enabled)
    logger.info(
        "v2_reflex_rule_enable",
        extra={
            "event": "v2_reflex_rule_enable",
            "bot_id": payload.meta.bot_id,
            "rule_id": rule_id,
            "enabled": payload.enabled,
            "updated": updated,
        },
    )
    return ReflexRuleEnableResponse(
        ok=True,
        updated=updated,
        rule_id=rule_id,
        enabled=payload.enabled,
        message="updated" if updated else "rule_not_found",
    )


@router.get("/triggers/{bot_id}", response_model=ReflexTriggerListResponse)
def list_triggers(
    bot_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    runtime: RuntimeState = Depends(get_runtime),
) -> ReflexTriggerListResponse:
    rows = runtime.recent_reflex_triggers(bot_id=bot_id, limit=limit)
    return ReflexTriggerListResponse(ok=True, bot_id=bot_id, triggers=rows)


@router.get("/breakers/{bot_id}", response_model=ReflexBreakerListResponse)
def list_breakers(
    bot_id: str,
    runtime: RuntimeState = Depends(get_runtime),
) -> ReflexBreakerListResponse:
    rows = runtime.reflex_breakers(bot_id=bot_id)
    return ReflexBreakerListResponse(ok=True, bot_id=bot_id, breakers=rows)
