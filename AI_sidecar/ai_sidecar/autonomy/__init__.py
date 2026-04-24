"""Autonomy module — PDCA continuous improvement loop."""
from ai_sidecar.autonomy.pdca_loop import PDCALoop, PDCAResult, PDCAConfig, Horizon
from ai_sidecar.autonomy.plan_executor import PlanExecutor
from ai_sidecar.autonomy.progress_tracker import ProgressTracker, ProgressEvaluation

__all__ = [
    "PDCALoop",
    "PDCAResult",
    "PDCAConfig",
    "Horizon",
    "PlanExecutor",
    "ProgressTracker",
    "ProgressEvaluation",
]
