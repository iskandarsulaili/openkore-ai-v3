"""CrewAI orchestration layer — behavior profiles (no CrewAI SDK dependency)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_sidecar.crewai.crew_manager import CrewManager

__all__ = ["CrewManager"]


def get_crew_manager(*args, **kwargs):
    """Lazy-imported factory to avoid circular imports with agents/__init__.py."""
    from ai_sidecar.crewai.crew_manager import CrewManager
    return CrewManager(*args, **kwargs)
