"""Capabilities API — query what the AI system can do.

Lets agents, LLM planners, and external tooling introspect the full capability
surface of openkore-ai-v3's sidecar (RULE.md §18 crowdsource/delegate and §19
self-adapt/self-learn). The LLM uses this to decide what it can delegate,
plan, execute, and tool-call.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/capabilities", tags=["capabilities"])


@router.get("")
def get_capabilities() -> dict[str, Any]:
    """Return the full structured capability registry."""
    from ai_sidecar.capabilities import capabilities_to_json
    return capabilities_to_json()


@router.get("/prompt")
def get_capabilities_prompt() -> dict[str, str]:
    """Return the capability context as a rendered LLM prompt block."""
    from ai_sidecar.capabilities import get_capabilities_prompt_block
    return {"block": get_capabilities_prompt_block()}


@router.get("/domains")
def get_execution_domains() -> dict[str, Any]:
    """Return just the in-game execution domains."""
    from ai_sidecar.capabilities import get_capabilities_registry
    return get_capabilities_registry().get("execution_domains", {})


@router.get("/roots")
def get_command_roots() -> dict[str, list[str]]:
    """Return the bridge-safe direct command roots."""
    from ai_sidecar.capabilities import get_capabilities_registry
    return {"direct_command_roots": get_capabilities_registry().get("direct_command_roots", [])}
