"""Prediction Systems — skill prediction, path prediction, spawn tracking,
MVP finisher timing, and server tick alignment.

All modules use self-learning models that improve with experience rather
than hardcoded formulae. Each module adapts to the specific server and
player behaviors it observes.
"""

from __future__ import annotations

from ai_sidecar.prediction.self_learning_skill_predictor import (
    SelfLearningSkillPredictor,
    SkillPrediction as SkillPrediction,
    SkillSignature as SkillSignature,
)
from ai_sidecar.prediction.path_predictor import PathPredictor, PositionPrediction
from ai_sidecar.prediction.self_learning_spawn_tracker import (
    SelfLearningSpawnTracker,
    SpawnEvent as SpawnEvent,
    SpawnTimerModel as SpawnTimerModel,
)
from ai_sidecar.prediction.mvp_finisher import (
    MvpFinisher,
    MvpModel,
    MvpFinisherObservation,
)
from ai_sidecar.prediction.server_tick_synchronizer import ServerTickSynchronizer
from ai_sidecar.prediction.persistence import (
    save_module_state, load_module_state,
    save_all_prediction_state, get_persist_cycle_checker,
)

__all__ = [
    "SelfLearningSkillPredictor",
    "SkillPrediction",
    "SkillSignature",
    "PathPredictor",
    "PositionPrediction",
    "SelfLearningSpawnTracker",
    "SpawnEvent",
    "SpawnTimerModel",
    "MvpFinisher",
    "MvpModel",
    "MvpFinisherObservation",
    "ServerTickSynchronizer",
]
