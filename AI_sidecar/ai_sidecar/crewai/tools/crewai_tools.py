from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ai_sidecar.crewai.tools.runtime_tools import CrewToolFacade


class GetBotStateInput(BaseModel):
    bot_id: str = Field(..., min_length=1, max_length=128)


class GetEnrichedStateInput(BaseModel):
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


class ListActiveMacrosInput(BaseModel):
    bot_id: str = Field(..., min_length=1, max_length=128)


class ProposeActionsInput(BaseModel):
    bot_id: str = Field(..., min_length=1, max_length=128)
    intents: list[dict[str, object]] = Field(default_factory=list)


class PublishMacroInput(BaseModel):
    bot_id: str = Field(..., min_length=1, max_length=128)
    macro_bundle: dict[str, object] = Field(default_factory=dict)


class EvaluatePlanFeasibilityInput(BaseModel):
    bot_id: str = Field(..., min_length=1, max_length=128)
    plan: dict[str, object] = Field(default_factory=dict)


class CoordinateWithFleetInput(BaseModel):
    bot_id: str = Field(..., min_length=1, max_length=128)
    action: str = Field(..., min_length=1, max_length=256)
    target_bots: list[str] = Field(default_factory=list)


class GetFleetConstraintsInput(BaseModel):
    bot_id: str = Field(..., min_length=1, max_length=128)


class WriteReflectionInput(BaseModel):
    bot_id: str = Field(..., min_length=1, max_length=128)
    episode: dict[str, object] = Field(default_factory=dict)


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

    class GetEnrichedStateTool(BaseTool):
        name: str = "get_enriched_state"
        description: str = "Retrieve the full enriched state graph for a bot."
        args_schema: type[BaseModel] = GetEnrichedStateInput

        def _run(self, bot_id: str) -> str:
            return str(facade.get_enriched_state(bot_id=bot_id))

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

    class ListActiveMacrosTool(BaseTool):
        name: str = "list_active_macros"
        description: str = "List the latest macro publication and manifest for a bot."
        args_schema: type[BaseModel] = ListActiveMacrosInput

        def _run(self, bot_id: str) -> str:
            return str(facade.list_active_macros(bot_id=bot_id))

    class ProposeActionsTool(BaseTool):
        name: str = "propose_actions"
        description: str = "Queue proposed action intents into the sidecar action queue."
        args_schema: type[BaseModel] = ProposeActionsInput

        def _run(self, bot_id: str, intents: list[dict[str, object]]) -> str:
            return str(facade.propose_actions(bot_id=bot_id, intents=intents))

    class PublishMacroTool(BaseTool):
        name: str = "publish_macro"
        description: str = "Publish macro bundles for a bot using the sidecar macro pipeline."
        args_schema: type[BaseModel] = PublishMacroInput

        def _run(self, bot_id: str, macro_bundle: dict[str, object]) -> str:
            return str(facade.publish_macro(bot_id=bot_id, macro_bundle=macro_bundle))

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

    class GetFleetConstraintsTool(BaseTool):
        name: str = "get_fleet_constraints"
        description: str = "Retrieve fleet coordination constraints and doctrine metadata for a bot."
        args_schema: type[BaseModel] = GetFleetConstraintsInput

        def _run(self, bot_id: str) -> str:
            return str(facade.get_fleet_constraints(bot_id=bot_id))

    class WriteReflectionTool(BaseTool):
        name: str = "write_reflection"
        description: str = "Write a reflection episode into memory with semantic indexing."
        args_schema: type[BaseModel] = WriteReflectionInput

        def _run(self, bot_id: str, episode: dict[str, object]) -> str:
            return str(facade.write_reflection(bot_id=bot_id, episode=episode))

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
        "get_enriched_state": GetEnrichedStateTool(),
        "get_bot_state": GetBotStateTool(),
        "query_memory": QueryMemoryTool(),
        "check_reflex_rules": CheckReflexRulesTool(),
        "generate_macro_template": GenerateMacroTemplateTool(),
        "list_active_macros": ListActiveMacrosTool(),
        "propose_actions": ProposeActionsTool(),
        "publish_macro": PublishMacroTool(),
        "evaluate_plan_feasibility": EvaluatePlanFeasibilityTool(),
        "coordinate_with_fleet": CoordinateWithFleetTool(),
        "get_fleet_constraints": GetFleetConstraintsTool(),
        "write_reflection": WriteReflectionTool(),
        "ml_shadow_predict": MLShadowPredictTool(),
    }
