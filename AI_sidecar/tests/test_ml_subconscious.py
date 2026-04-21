from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.api.routers import ml_subconscious_v2
from ai_sidecar.contracts.ml_subconscious import (
    MLDistillMacroRequest,
    MLDistillMacroResponse,
    MLModelsResponse,
    MLModelFamilyView,
    MLModelVersionView,
    MLObserveRequest,
    MLObserveResponse,
    MLPerformanceResponse,
    MLPredictRequest,
    MLPredictResponse,
    MLPromoteRequest,
    MLPromoteResponse,
    MLTrainRequest,
    MLTrainResponse,
    ModelFamily,
)
from ai_sidecar.crewai.tools.runtime_tools import CrewToolFacade


class _MLRouterRuntime:
    def ml_observe(self, payload: MLObserveRequest) -> MLObserveResponse:
        return MLObserveResponse(
            ok=True,
            message="observed",
            trace_id=payload.meta.trace_id,
            episode_id=payload.episode.episode_id,
            bot_id=payload.episode.bot_id,
            reward=0.42,
            reward_breakdown={"objective": 0.42},
            labels_generated=2,
        )

    def ml_train(self, payload: MLTrainRequest) -> MLTrainResponse:
        return MLTrainResponse(
            ok=True,
            message="trained",
            trace_id=payload.meta.trace_id,
            model_family=payload.model_family,
            model_version="encounter_classifier-v1",
            trained_samples=128,
            metrics={"accuracy": 0.91},
            ab_test={"candidate": "encounter_classifier-v1", "control": "encounter_classifier-v0"},
        )

    def ml_models(self) -> MLModelsResponse:
        return MLModelsResponse(
            ok=True,
            models=[
                MLModelFamilyView(
                    family=ModelFamily.encounter_classifier,
                    active_version="encounter_classifier-v1",
                    versions=[
                        MLModelVersionView(
                            version="encounter_classifier-v1",
                            created_at=MLModelsResponse().generated_at,
                            active=True,
                            metrics={"accuracy": 0.91},
                            path="/tmp/encounter_classifier-v1.pkl",
                        )
                    ],
                )
            ],
        )

    def ml_predict(self, payload: MLPredictRequest) -> MLPredictResponse:
        return MLPredictResponse(
            ok=True,
            message="predicted",
            trace_id=payload.meta.trace_id,
            model_family=payload.model_family,
            model_version="encounter_classifier-v1",
            recommendation={"action": "kite", "target": "mob:poring"},
            confidence=0.88,
            shadow={"disagrees_with_planner": False},
        )

    def ml_promote(self, payload: MLPromoteRequest) -> MLPromoteResponse:
        return MLPromoteResponse(
            ok=True,
            message="promotion_updated",
            trace_id=payload.meta.trace_id,
            model_family=payload.model_family,
            promotion={
                "model_version": payload.model_version,
                "canary_percentage": payload.canary_percentage,
                "rollback_threshold": payload.rollback_threshold,
            },
        )

    def ml_performance(self) -> MLPerformanceResponse:
        return MLPerformanceResponse(
            ok=True,
            shadow_metrics={"encounter_classifier": {"samples": 10, "accuracy": 0.8}},
            promotion_metrics={"encounter_classifier": {"canary_active": True}},
            training_metrics={"encounter_classifier": {"last_samples": 128}},
        )

    def ml_distill_macro(self, payload: MLDistillMacroRequest) -> MLDistillMacroResponse:
        bot_id = payload.bot_id or payload.meta.bot_id
        return MLDistillMacroResponse(
            ok=True,
            message="distilled",
            trace_id=payload.meta.trace_id,
            bot_id=bot_id,
            proposal_id="proposal-1",
            support=5,
            success_rate=0.9,
            macro={"name": "auto_farm_combo", "lines": ["attack", "loot"]},
            automacro={"name": "on_auto_farm_combo", "conditions": ["OnCharLogIn"]},
            publication=None,
        )


def test_ml_subconscious_v2_router_endpoints() -> None:
    runtime = _MLRouterRuntime()
    app = FastAPI()
    app.include_router(ml_subconscious_v2.router)
    app.dependency_overrides[get_runtime] = lambda: runtime

    with TestClient(app) as client:
        observe_resp = client.post(
            "/v2/ml/observe",
            json={
                "meta": {
                    "contract_version": "v1",
                    "source": "pytest",
                    "bot_id": "bot:ml",
                    "trace_id": "trace-ml-observe",
                },
                "episode": {
                    "episode_id": "ep-1",
                    "bot_id": "bot:ml",
                    "state_features": {"hp_pct": 0.95},
                    "decision_source": "llm",
                    "decision_payload": {"intent": "kite"},
                    "executed_action": {"kind": "command", "command": "move 10 20"},
                    "outcome": {"success": True, "reward": 0.2, "latency_ms": 25.0, "side_effects": []},
                    "safety_flags": [],
                    "macro_version": "",
                },
            },
        )
        assert observe_resp.status_code == 200
        assert observe_resp.json()["ok"] is True
        assert observe_resp.json()["episode_id"] == "ep-1"

        train_resp = client.post(
            "/v2/ml/train",
            json={
                "meta": {
                    "contract_version": "v1",
                    "source": "pytest",
                    "bot_id": "bot:ml",
                    "trace_id": "trace-ml-train",
                },
                "model_family": "encounter_classifier",
                "bot_id": "bot:ml",
                "incremental": True,
                "max_samples": 200,
            },
        )
        assert train_resp.status_code == 200
        assert train_resp.json()["ok"] is True
        assert train_resp.json()["model_family"] == "encounter_classifier"

        models_resp = client.get("/v2/ml/models")
        assert models_resp.status_code == 200
        assert models_resp.json()["ok"] is True
        assert models_resp.json()["models"][0]["active_version"] == "encounter_classifier-v1"

        predict_resp = client.post(
            "/v2/ml/predict",
            json={
                "meta": {
                    "contract_version": "v1",
                    "source": "pytest",
                    "bot_id": "bot:ml",
                    "trace_id": "trace-ml-predict",
                },
                "model_family": "encounter_classifier",
                "state_features": {"hp_pct": 0.9, "mobs_near": 3},
                "context": {"objective": "grind safely"},
                "planner_choice": {"action": "attack"},
            },
        )
        assert predict_resp.status_code == 200
        assert predict_resp.json()["ok"] is True
        assert predict_resp.json()["confidence"] == 0.88

        promote_resp = client.post(
            "/v2/ml/promote",
            json={
                "meta": {
                    "contract_version": "v1",
                    "source": "pytest",
                    "bot_id": "bot:ml",
                    "trace_id": "trace-ml-promote",
                },
                "model_family": "encounter_classifier",
                "model_version": "encounter_classifier-v1",
                "canary_percentage": 15.0,
                "rollback_threshold": 0.25,
                "scope": {"map": "prt_fild08"},
            },
        )
        assert promote_resp.status_code == 200
        assert promote_resp.json()["ok"] is True
        assert promote_resp.json()["promotion"]["canary_percentage"] == 15.0

        performance_resp = client.get("/v2/ml/performance")
        assert performance_resp.status_code == 200
        assert performance_resp.json()["ok"] is True
        assert "shadow_metrics" in performance_resp.json()

        distill_resp = client.post(
            "/v2/ml/distill-macro",
            json={
                "meta": {
                    "contract_version": "v1",
                    "source": "pytest",
                    "bot_id": "bot:ml",
                    "trace_id": "trace-ml-distill",
                },
                "bot_id": "bot:ml",
                "episode_ids": ["ep-1", "ep-2", "ep-3"],
                "min_support": 3,
                "max_steps": 6,
                "enqueue_reload": False,
            },
        )
        assert distill_resp.status_code == 200
        assert distill_resp.json()["ok"] is True
        assert distill_resp.json()["proposal_id"] == "proposal-1"


class _ToolRuntime:
    def ml_predict(self, payload: MLPredictRequest) -> MLPredictResponse:
        return MLPredictResponse(
            ok=True,
            message="predicted",
            trace_id=payload.meta.trace_id,
            model_family=payload.model_family,
            model_version="encounter_classifier-v1",
            recommendation={"action": "kite"},
            confidence=0.73,
            shadow={"planner_action": payload.planner_choice.get("action", "")},
        )


def test_crewai_tool_facade_ml_shadow_predict() -> None:
    facade = CrewToolFacade(runtime=_ToolRuntime())

    invalid = facade.ml_shadow_predict(bot_id="bot:ml", model_family="not_a_family", objective="safe grind", planner_choice={})
    assert invalid["ok"] is False
    assert "allowed_families" in invalid

    valid = facade.ml_shadow_predict(
        bot_id="bot:ml",
        model_family="encounter_classifier",
        objective="safe grind",
        planner_choice={"action": "attack"},
    )
    assert valid["ok"] is True
    assert valid["family"] == "encounter_classifier"
    assert valid["confidence"] == 0.73
    assert valid["shadow"]["planner_action"] == "attack"

    dispatched = facade.execute(
        bot_id="bot:ml",
        tool_name="ml_shadow_predict",
        arguments={
            "model_family": "encounter_classifier",
            "objective": "safe grind",
            "planner_choice": {"action": "retreat"},
        },
    )
    assert dispatched["ok"] is True
    assert dispatched["shadow"]["planner_action"] == "retreat"

