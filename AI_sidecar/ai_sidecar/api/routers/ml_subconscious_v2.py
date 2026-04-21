from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.contracts.ml_subconscious import (
    MLDistillMacroRequest,
    MLDistillMacroResponse,
    MLModelsResponse,
    MLObserveRequest,
    MLObserveResponse,
    MLPerformanceResponse,
    MLPredictRequest,
    MLPredictResponse,
    MLPromoteRequest,
    MLPromoteResponse,
    MLTrainRequest,
    MLTrainResponse,
)
from ai_sidecar.lifecycle import RuntimeState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2/ml", tags=["ml-subconscious-v2"])


@router.post("/observe", response_model=MLObserveResponse)
def observe(
    payload: MLObserveRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> MLObserveResponse:
    result = runtime.ml_observe(payload)
    logger.info(
        "ml_observe_recorded",
        extra={
            "event": "ml_observe_recorded",
            "bot_id": payload.episode.bot_id,
            "trace_id": payload.meta.trace_id,
            "ok": result.ok,
            "episode_id": result.episode_id,
            "labels_generated": result.labels_generated,
        },
    )
    return result


@router.post("/train", response_model=MLTrainResponse)
def train(
    payload: MLTrainRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> MLTrainResponse:
    result = runtime.ml_train(payload)
    logger.info(
        "ml_training_completed",
        extra={
            "event": "ml_training_completed",
            "bot_id": payload.bot_id or payload.meta.bot_id,
            "trace_id": payload.meta.trace_id,
            "family": payload.model_family.value,
            "ok": result.ok,
            "version": result.model_version,
            "trained_samples": result.trained_samples,
        },
    )
    return result


@router.get("/models", response_model=MLModelsResponse)
def models(runtime: RuntimeState = Depends(get_runtime)) -> MLModelsResponse:
    return runtime.ml_models()


@router.post("/predict", response_model=MLPredictResponse)
def predict(
    payload: MLPredictRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> MLPredictResponse:
    result = runtime.ml_predict(payload)
    logger.info(
        "ml_shadow_prediction_completed",
        extra={
            "event": "ml_shadow_prediction_completed",
            "bot_id": payload.meta.bot_id,
            "trace_id": payload.meta.trace_id,
            "family": payload.model_family.value,
            "ok": result.ok,
            "confidence": result.confidence,
        },
    )
    return result


@router.post("/promote", response_model=MLPromoteResponse)
def promote(
    payload: MLPromoteRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> MLPromoteResponse:
    result = runtime.ml_promote(payload)
    logger.warning(
        "ml_promotion_updated",
        extra={
            "event": "ml_promotion_updated",
            "bot_id": payload.meta.bot_id,
            "trace_id": payload.meta.trace_id,
            "family": payload.model_family.value,
            "ok": result.ok,
            "version": payload.model_version,
            "canary": payload.canary_percentage,
        },
    )
    return result


@router.get("/performance", response_model=MLPerformanceResponse)
def performance(runtime: RuntimeState = Depends(get_runtime)) -> MLPerformanceResponse:
    return runtime.ml_performance()


@router.post("/distill-macro", response_model=MLDistillMacroResponse)
def distill_macro(
    payload: MLDistillMacroRequest,
    runtime: RuntimeState = Depends(get_runtime),
) -> MLDistillMacroResponse:
    result = runtime.ml_distill_macro(payload)
    logger.info(
        "ml_macro_distillation_completed",
        extra={
            "event": "ml_macro_distillation_completed",
            "bot_id": payload.bot_id or payload.meta.bot_id,
            "trace_id": payload.meta.trace_id,
            "ok": result.ok,
            "proposal_id": result.proposal_id,
            "support": result.support,
        },
    )
    return result

