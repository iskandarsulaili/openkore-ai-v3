"""Regression test: the bridge poll contract must accept max_actions.

Since c36b32dc7 the bridge sends `max_actions` in every /v1/actions/next
poll. NextActionRequest had extra="forbid" and no max_actions field, so
EVERY poll returned 422 and NO queued action ever executed live — the
fleet ran on native OpenKore AI + bridge-side reflex only (proven by
[ai_action]=0 across all 8 bot logs). This test locks the contract:
the field must be accepted and clamped.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ai_sidecar.contracts.actions import NextActionRequest
from ai_sidecar.contracts.common import ContractMeta


def _meta() -> ContractMeta:
    return ContractMeta(
        contract_version="v1",
        source="openkore-bridge",
        bot_id="kicapmasin11",
        trace_id="t",
        emitted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_next_action_request_accepts_bridge_max_actions() -> None:
    req = NextActionRequest(
        meta=_meta(),
        poll_id="poll-1",
        max_actions=5,
    )
    assert req.max_actions == 5


def test_next_action_request_defaults_max_actions_to_one() -> None:
    req = NextActionRequest(meta=_meta(), poll_id="poll-2")
    assert req.max_actions == 1


def test_next_action_request_clamps_max_actions_range() -> None:
    # 0 is below the minimum -> rejected
    with pytest.raises(ValidationError):
        NextActionRequest(meta=_meta(), poll_id="poll-3", max_actions=0)
    # 11 is above the maximum -> rejected
    with pytest.raises(ValidationError):
        NextActionRequest(meta=_meta(), poll_id="poll-4", max_actions=11)
    # boundary values accepted
    assert NextActionRequest(meta=_meta(), poll_id="poll-5", max_actions=1).max_actions == 1
    assert NextActionRequest(meta=_meta(), poll_id="poll-6", max_actions=10).max_actions == 10


def test_next_action_request_still_rejects_unknown_fields() -> None:
    # extra="forbid" must remain in force for genuinely unknown fields
    with pytest.raises(ValidationError):
        NextActionRequest(
            meta=_meta(),
            poll_id="poll-7",
            max_actions=5,
            unknown_field="x",  # type: ignore[call-arg]
        )
