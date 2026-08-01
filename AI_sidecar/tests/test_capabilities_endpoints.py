"""Tests for the capabilities API + prompt-block consumer (RULE.md §18/§19).

Verifies the four /v1/capabilities endpoints return the declared surface and
that the registry feeds prompt_invariants through _system_capabilities_context.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_sidecar.api.routers.capabilities import router as caps_router
from ai_sidecar.capabilities import build_capabilities_registry, get_capabilities_prompt_block
from ai_sidecar.autonomy.ro_knowledge import _system_capabilities_context, prompt_invariants


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(caps_router)  # router carries its own /v1/capabilities prefix
    return TestClient(app)


def test_capabilities_registry_well_formed() -> None:
    reg = build_capabilities_registry()
    assert reg["system"] == "openkore-ai-v3 sidecar"
    assert isinstance(reg["execution_domains"], dict) and reg["execution_domains"]
    assert isinstance(reg["direct_command_roots"], list) and reg["direct_command_roots"]
    assert isinstance(reg["knowledge_systems"], dict)
    assert isinstance(reg["learning_systems"], dict)
    assert isinstance(reg["fleet_systems"], dict)
    assert isinstance(reg["api_surface"], dict)


def test_capabilities_endpoints() -> None:
    client = _client()
    r = client.get("/v1/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert body["system"] == "openkore-ai-v3 sidecar"
    assert "execution_domains" in body
    assert "direct_command_roots" in body

    rp = client.get("/v1/capabilities/prompt")
    assert rp.status_code == 200
    assert "SYSTEM CAPABILITIES" in rp.json()["block"]

    rd = client.get("/v1/capabilities/domains")
    assert rd.status_code == 200
    assert isinstance(rd.json(), dict) and rd.json()

    rr = client.get("/v1/capabilities/roots")
    assert rr.status_code == 200
    assert isinstance(rr.json()["direct_command_roots"], list)


def test_capabilities_consumed_by_prompt_invariants() -> None:
    # _system_capabilities_context must return the registry (non-empty)
    ctx = _system_capabilities_context()
    assert isinstance(ctx, dict) and ctx.get("system") == "openkore-ai-v3 sidecar"

    # prompt_invariants must include the capabilities block (no NameError,
    # real production path — regression guard for the old context_assembler bug)
    inv = prompt_invariants()
    assert isinstance(inv, dict)
    joined = str(inv)
    assert "SYSTEM CAPABILITIES" in joined or "capabilities" in joined.lower()

    # prompt block renders without error
    block = get_capabilities_prompt_block()
    assert "Execution domains:" in block
    assert "Direct command roots" in block
