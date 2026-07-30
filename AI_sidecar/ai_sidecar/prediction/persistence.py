"""Prediction State Persistence — saves and loads learned prediction state.

Each prediction module can persist its learned state to disk so it survives
sidecar restarts. This module provides save/load helpers for all prediction
modules using JSON serialization.

Usage:
    from ai_sidecar.prediction.persistence import save_all_prediction_state, load_all_prediction_state
    
    # Save all prediction module states
    save_all_prediction_state()
    
    # Load all prediction module states (call at startup)
    load_all_prediction_state()
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# Data directory relative to AI_sidecar/
_DATA_DIR = Path(os.environ.get(
    "OPENKORE_AI_DATA_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "data"),
))
_DATA_DIR.mkdir(parents=True, exist_ok=True)

# How often to persist (3600 cycles ~= 1 hour at 1s/cycle)
PERSIST_INTERVAL = 3600


def _module_state_path(module_name: str) -> Path:
    """Get the JSON state file path for a prediction module."""
    return _DATA_DIR / f"learned_{module_name}.json"


def save_module_state(module_name: str, state: dict[str, Any]) -> None:
    """Save a prediction module's learned state to disk."""
    try:
        path = _module_state_path(module_name)
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.debug(f"Saved prediction state: {module_name} ({len(state)} keys)")
    except Exception as e:
        logger.warning(f"Failed to save {module_name} state: {e}")


def load_module_state(module_name: str) -> dict[str, Any] | None:
    """Load a prediction module's learned state from disk."""
    try:
        path = _module_state_path(module_name)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            logger.info(f"Loaded prediction state: {module_name} ({len(data)} keys)")
            return data
    except Exception as e:
        logger.warning(f"Failed to load {module_name} state: {e}")
    return None


def save_all_prediction_state(
    skill_predictor: Any = None,
    path_predictor: Any = None,
    spawn_tracker: Any = None,
    mvp_finisher: Any = None,
    tick_sync: Any = None,
) -> None:
    """Save all prediction modules' state to disk."""
    modules = {
        "skill_predictor": skill_predictor,
        "path_predictor": path_predictor,
        "spawn_tracker": spawn_tracker,
        "mvp_finisher": mvp_finisher,
        "tick_sync": tick_sync,
    }
    for name, mod in modules.items():
        if mod and hasattr(mod, "to_dict"):
            try:
                state = mod.to_dict()
                if state:
                    save_module_state(name, state)
            except Exception as e:
                logger.warning(f"Failed to save {name}: {e}")


def get_persist_cycle_checker() -> callable:
    """Returns a callable that returns True every PERSIST_INTERVAL calls.
    
    Usage:
        should_save = get_persist_cycle_checker()
        # in main loop:
        if should_save():
            save_all_prediction_state(...)
    """
    cycle = [0]
    lock = RLock()
    
    def _check() -> bool:
        with lock:
            cycle[0] += 1
            if cycle[0] >= PERSIST_INTERVAL:
                cycle[0] = 0
                return True
            return False
    
    return _check
