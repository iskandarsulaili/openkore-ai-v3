"""Skills Curator — background maintenance for agent-created skills.

Runs periodically to:
1. Mark unused skills as stale (excluded from context)
2. Archive long-stale skills (moved to .archive/)
3. (Optional) Consolidate related stale skills via LLM review
4. Pre-run backup of skills state

Inspired by Hermes Agent's agent/curator.py pattern.
"""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from ai_sidecar import skills_usage

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).resolve().parent / "skills"

# ── Config (inline defaults, can be overridden via config) ──

_CONF = {
    "enabled": True,
    "interval_hours": 168,  # weekly
    "min_idle_hours": 1,
    "stale_after_days": 7,
    "archive_after_days": 14,
    "consolidate": False,
    "backup_enabled": True,
    "backup_path": None,  # defaults to skills/.curator_backups/
}

_CURATOR_LOCK = Lock()
_LAST_RUN_AT: Optional[str] = None


# ── Config ──


def configure(**kwargs) -> None:
    """Override curator config defaults."""
    _CONF.update(kwargs)


def get_config() -> Dict[str, Any]:
    return dict(_CONF)


def last_run_at() -> Optional[str]:
    global _LAST_RUN_AT
    return _LAST_RUN_AT


# ── Backup ──


def _backup_dir() -> Path:
    path = _CONF.get("backup_path")
    if path:
        p = Path(path)
    else:
        p = _SKILLS_DIR / ".curator_backups"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _create_backup() -> Optional[Path]:
    """Create a tar.gz backup of all skills (excluding .archive and hidden)."""
    backup_dir = _backup_dir()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    backup_path = backup_dir / f"skills-{timestamp}.tar.gz"

    try:
        with tarfile.open(backup_path, "w:gz") as tar:
            for item in _SKILLS_DIR.iterdir():
                if item.name.startswith("."):
                    continue
                if item.name == ".archive":
                    continue
                tar.add(item, arcname=item.name)
        return backup_path
    except (OSError, PermissionError) as exc:
        logger.warning("Backup failed: %s", exc)
        return None


def list_backups() -> List[Dict[str, Any]]:
    """List available backup files."""
    backup_dir = _backup_dir()
    backups: List[Dict[str, Any]] = []
    for f in sorted(backup_dir.glob("skills-*.tar.gz"), reverse=True):
        backups.append({
            "path": str(f),
            "size": f.stat().st_size,
            "created": f.stem.replace("skills-", ""),
        })
    return backups


# ── Main Curator Run ──


def run_curator(
    stale_after_days: Optional[int] = None,
    archive_after_days: Optional[int] = None,
    consolidate: Optional[bool] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run one curator cycle. Returns summary dict.

    Args:
        stale_after_days: Override default stale threshold
        archive_after_days: Override default archive threshold
        consolidate: Override default consolidate flag
        dry_run: If True, only report what would happen

    Returns:
        Dict with marked_stale, archived, backed_up, errors
    """
    global _LAST_RUN_AT

    if not _CONF.get("enabled"):
        return {"skipped": True, "reason": "curator disabled"}

    result: Dict[str, Any] = {
        "marked_stale": [],
        "archived": [],
        "backed_up": None,
        "errors": [],
        "dry_run": dry_run,
    }

    # Step 1: Backup (unless dry-run)
    if not dry_run and _CONF.get("backup_enabled", True):
        backup = _create_backup()
        if backup:
            result["backed_up"] = str(backup)

    # Step 2: Mark stale
    sad = stale_after_days or _CONF.get("stale_after_days", 7)
    if dry_run:
        # Simulate by reading current state
        usage = skills_usage.list_skills()
        for name, record in usage.items():
            if record.get("state") != "active":
                continue
            if record.get("pinned", False):
                continue
            last_str = record.get("last_activity_at")
            if not last_str:
                continue
            try:
                last = datetime.fromisoformat(last_str)
                now = datetime.now(timezone.utc)
                days_unused = (now - last).total_seconds() / 86400
                if days_unused > sad:
                    result["marked_stale"].append(name)
            except (ValueError, TypeError):
                continue
    else:
        marked = skills_usage.mark_stale_if_unused(stale_after_days=sad)
        result["marked_stale"] = marked

    # Step 3: Archive stale
    aad = archive_after_days or _CONF.get("archive_after_days", 14)
    if not dry_run:
        archived = skills_usage.mark_archived_if_stale(archive_after_days=aad)
        result["archived"] = archived

    # Step 4: Consolidation (LLM review) — opt-in
    cons = consolidate if consolidate is not None else _CONF.get("consolidate", False)
    if cons and not dry_run:
        try:
            cons_result = _run_consolidation()
            result["consolidation"] = cons_result
        except Exception as exc:
            logger.warning("Consolidation failed: %s", exc)
            result["consolidation_error"] = str(exc)

    _LAST_RUN_AT = datetime.now(timezone.utc).isoformat()
    return result


def _run_consolidation() -> Dict[str, Any]:
    """LLM-driven consolidation of stale skills (opt-in).
    Placeholder — actual LLM call would be implemented in Phase 2."""
    # This would ask the Pro RO LLM to review stale skills and suggest merges/deletions
    stale = skills_usage.list_skills()
    stale_names = [n for n, r in stale.items() if r.get("state") == "stale"]
    return {
        "reviewed": stale_names,
        "merged": [],
        "deleted": [],
        "note": "LLM consolidation not yet implemented — Phase 2 feature",
    }


def should_run_now(min_idle_hours: Optional[float] = None) -> bool:
    """Check if enough time has passed since last curator run."""
    global _LAST_RUN_AT
    if _LAST_RUN_AT is None:
        return True
    mih = min_idle_hours or _CONF.get("min_idle_hours", 1)
    try:
        last = datetime.fromisoformat(_LAST_RUN_AT)
        elapsed = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        return elapsed >= mih
    except (ValueError, TypeError):
        return True


def force_run() -> Dict[str, Any]:
    """Force a curator run regardless of interval."""
    return run_curator()
