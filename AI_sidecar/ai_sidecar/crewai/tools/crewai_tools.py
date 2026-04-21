from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ai_sidecar.crewai.tools.runtime_tools import CrewToolFacade


class GetBotStateInput(BaseModel):
    bot_id: str = Field(..., min_length=1, max_length=128)


class QueryMemoryInput(BaseModel):
    bot_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1, max_length=512)
    limit: int = Field(default=10, ge=1, le=50)


class CheckReflexRulesInput(BaseModel):
    bot_id: str = Field(..., min_length=1, max_length=128)
    trigger_type: str = Field(default="", max_length=128)


class GenerateMacroTemplateInput(BaseModel):
    bot_id: str = Field(..., min_length=1, max_length=128)
    action_sequence: list[str] = Field(default_factory=list)


class EvaluatePlanFeasibilityInput(BaseModel):
    bot_id: str = Field(..., min_length=1, max_length=128)
    plan: dict[str, object] = Field(default_factory=dict)


class CoordinateWithFleetInput(BaseModel):
    bot_id: str = Field(..., min_length=1, max_length=128)
    action: str = Field(..., min_length=1, max_length=256)
    target_bots: list[str] = Field(default_factory=list)


class MLShadowPredictInput(BaseModel):
    bot_id: str = Field(..., min_length=1, max_length=128)
    model_family: str = Field(..., min_length=1, max_length=128)
    objective: str = Field(default="", max_length=512)
    planner_choice: dict[str, object] = Field(default_factory=dict)


def build_crewai_tools(*, facade: CrewToolFacade) -> dict[str, Any]:
    from crewai.tools import BaseTool

    class GetBotStateTool(BaseTool):
        name: str = "get_bot_state"
        description: str = "Retrieve current enriched bot state and queue depth by bot id."
        args_schema: type[BaseModel] = GetBotStateInput

        def _run(self, bot_id: str) -> str:
            return str(facade.get_bot_state(bot_id=bot_id))

    class QueryMemoryTool(BaseTool):
        name: str = "query_memory"
        description: str = "Search semantic memory and recent episodes for a bot."
        args_schema: type[BaseModel] = QueryMemoryInput

        def _run(self, bot_id: str, query: str, limit: int = 10) -> str:
            return str(facade.query_memory(bot_id=bot_id, query=query, limit=limit))

    class CheckReflexRulesTool(BaseTool):
        name: str = "check_reflex_rules"
        description: str = "Inspect active reflex rules and optionally filter by trigger type."
        args_schema: type[BaseModel] = CheckReflexRulesInput

        def _run(self, bot_id: str, trigger_type: str = "") -> str:
            return str(facade.check_reflex_rules(bot_id=bot_id, trigger_type=trigger_type))

    class GenerateMacroTemplateTool(BaseTool):
        name: str = "generate_macro_template"
        description: str = "Generate a macro template draft from an action sequence."
        args_schema: type[BaseModel] = GenerateMacroTemplateInput

        def _run(self, bot_id: str, action_sequence: list[str]) -> str:
            return str(facade.generate_macro_template(bot_id=bot_id, action_sequence=action_sequence))

    class EvaluatePlanFeasibilityTool(BaseTool):
        name: str = "evaluate_plan_feasibility"
        description: str = "Evaluate plan feasibility against queue pressure and risk constraints."
        args_schema: type[BaseModel] = EvaluatePlanFeasibilityInput

        def _run(self, bot_id: str, plan: dict[str, object]) -> str:
            return str(facade.evaluate_plan_feasibility(bot_id=bot_id, plan=plan))

    class CoordinateWithFleetTool(BaseTool):
        name: str = "coordinate_with_fleet"
        description: str = "Coordinate a command action across target bots by queueing strategic actions."
        args_schema: type[BaseModel] = CoordinateWithFleetInput

        def _run(self, bot_id: str, action: str, target_bots: list[str]) -> str:
            return str(facade.coordinate_with_fleet(bot_id=bot_id, action=action, target_bots=target_bots))

    class MLShadowPredictTool(BaseTool):
        name: str = "ml_shadow_predict"
        description: str = "Get ML subconscious recommendation in shadow mode for comparison against planner choice."
        args_schema: type[BaseModel] = MLShadowPredictInput

        def _run(self, bot_id: str, model_family: str, objective: str = "", planner_choice: dict[str, object] | None = None) -> str:
            return str(
                facade.ml_shadow_predict(
                    bot_id=bot_id,
                    model_family=model_family,
                    objective=objective,
                    planner_choice=dict(planner_choice or {}),
                )
            )

    return {
        "get_bot_state": GetBotStateTool(),
        "query_memory": QueryMemoryTool(),
        "check_reflex_rules": CheckReflexRulesTool(),
        "generate_macro_template": GenerateMacroTemplateTool(),
        "evaluate_plan_feasibility": EvaluatePlanFeasibilityTool(),
        "coordinate_with_fleet": CoordinateWithFleetTool(),
        "ml_shadow_predict": MLShadowPredictTool(),
    }
