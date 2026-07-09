"""CrewAI orchestration layer — behavior profiles (no CrewAI SDK dependency)."""

from typing import TYPE_CHECKING

# Backward-compatible import: CrewManager is available at ai_sidecar.crewai.CrewManager
from ai_sidecar.crewai.crew_manager import CrewManager  # noqa: F401

__all__ = ["CrewManager", "get_crew_manager"]


def get_crew_manager(*args, **kwargs):
    """Lazy-imported factory for use in circular-import scenarios."""
    from ai_sidecar.crewai.crew_manager import CrewManager
    return CrewManager(*args, **kwargs)
