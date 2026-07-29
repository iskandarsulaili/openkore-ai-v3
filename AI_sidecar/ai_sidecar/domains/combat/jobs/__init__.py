"""Jobs package — job definitions and registry for all 45+ RO classes."""
from ai_sidecar.domains.combat.jobs.registry import JobRegistry, get_job_registry, get_tactics_for_job

__all__ = [
    "JobRegistry",
    "get_job_registry",
    "get_tactics_for_job",
]
