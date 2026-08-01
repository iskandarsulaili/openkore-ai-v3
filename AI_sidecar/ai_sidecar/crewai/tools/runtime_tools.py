"""Runtime tool facade for CrewAI integration.

Bridges the CrewAI agent layer to the ML subconscious service (v2) through a
small synchronous facade. Promised in the original architecture but never
implemented (tests/test_ml_subconscious.py imports CrewToolFacade) — this
module delivers the real capability: shadow prediction for the ML model
families, validated against the ModelFamily enum, with a generic execute()
dispatcher for agent tool calls.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from ai_sidecar.contracts.common import ContractMeta
from ai_sidecar.contracts.ml_subconscious import MLPredictRequest, ModelFamily

logger = logging.getLogger(__name__)

# Tool names the facade can dispatch. Each maps to a method on this class.
_TOOL_REGISTRY: dict[str, str] = {
    "ml_shadow_predict": "ml_shadow_predict",
}


class CrewToolFacade:
    """Synchronous facade exposing ML subconscious tools to CrewAI agents."""

    def __init__(self, *, runtime: Any) -> None:
        # runtime must expose ml_predict(payload: MLPredictRequest) -> MLPredictResponse
        self._runtime = runtime

    # ── Public tools ────────────────────────────────────────────────────────

    def ml_shadow_predict(
        self,
        *,
        bot_id: str,
        model_family: str,
        objective: str = "",
        planner_choice: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a shadow prediction for the given model family.

        Returns a dict (agent-friendly, JSON-safe):
          ok=True  -> {"ok", "family", "model_version", "recommendation", "confidence", "shadow"}
          ok=False -> {"ok": False, "allowed_families": [...]} for an unknown family.
        """
        try:
            family = ModelFamily(model_family)
        except ValueError:
            return {
                "ok": False,
                "allowed_families": [item.value for item in ModelFamily],
            }

        payload = MLPredictRequest(
            meta=ContractMeta(
                contract_version="v1",
                source="crewai",
                bot_id=bot_id,
                trace_id=f"trace-crewai-{uuid.uuid4().hex[:12]}",
            ),
            model_family=family,
            state_features={},
            context={"objective": objective},
            planner_choice=dict(planner_choice or {}),
        )
        result = self._runtime.ml_predict(payload)
        family_value = result.model_family.value if hasattr(result.model_family, "value") else str(result.model_family)
        return {
            "ok": bool(getattr(result, "ok", True)),
            "family": family_value,
            "model_version": getattr(result, "model_version", ""),
            "recommendation": dict(getattr(result, "recommendation", {})),
            "confidence": float(getattr(result, "confidence", 0.0)),
            "shadow": dict(getattr(result, "shadow", {})),
        }

    def execute(
        self,
        *,
        bot_id: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generic dispatcher for agent tool calls.

        Maps tool_name -> facade method and forwards the supported keyword
        arguments. Unknown tools return an ok=False dict instead of raising,
        so the agent can degrade gracefully.
        """
        method_name = _TOOL_REGISTRY.get(tool_name)
        if method_name is None:
            return {"ok": False, "error": f"unknown_tool:{tool_name}", "known_tools": sorted(_TOOL_REGISTRY)}
        method = getattr(self, method_name, None)
        if method is None:
            return {"ok": False, "error": f"tool_not_bound:{tool_name}"}

        args = dict(arguments or {})
        if tool_name == "ml_shadow_predict":
            return method(
                bot_id=bot_id,
                model_family=str(args.get("model_family", "")),
                objective=str(args.get("objective", "")),
                planner_choice=args.get("planner_choice") if isinstance(args.get("planner_choice"), dict) else {},
            )
        return {"ok": False, "error": f"tool_dispatch_unhandled:{tool_name}"}
