"""Skill usage telemetry + lifecycle state tracking for the AI Sidecar.

Tracks per-skill usage metadata in .usage.json, keyed by skill name.
Every skills_manager or skills_loader call bumps counters.

Lifecycle states:
    active    -> default, loaded into context when triggers match
    stale     -> unused > stale_after_days, excluded from context
    archived  -> unused > archive_after_days, moved to .archive/

Inspired by Hermes Agent's skill_usage.py pattern.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

# ── Paths ──

_SKILLS_DIR = Path(__file__).resolve().parent / "skills"
_USAGE_FILE = _SKILLS_DIR / ".usage.json"
_ARCHIVE_DIR = _SKILLS_DIR / ".archive"

_USAGE_LOCK = Lock()

# ── Constants ──

STATE_ACTIVE = "active"
STATE_STALE = "stale"
STATE_ARCHIVED = "archived"
_VALID_STATES = {STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED}

DEFAULT_STALE_AFTER_DAYS = 7
DEFAULT_ARCHIVE_AFTER_DAYS = 14


# ── Helpers ──


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _skills_dir() -> Path:
    _SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    return _SKILLS_DIR


def _archive_dir() -> Path:
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    return _ARCHIVE_DIR


def _usage_file() -> Path:
    _skills_dir()
    return _USAGE_FILE


def _default_record() -> Dict[str, Any]:
    return {
        "state": STATE_ACTIVE,
        "created_at": _now_iso(),
        "last_activity_at": _now_iso(),
        "use_count": 0,
        "view_count": 0,
        "patch_count": 0,
        "confidence": 0.5,
        "provenance": "foreground",
        "pinned": False,
    }


# ── Read/Write Usage Data ──


def _read_usage() -> Dict[str, Any]:
    """Read .usage.json, returning empty dict if missing or corrupt."""
    path = _usage_file()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Corrupt .usage.json: %s", exc)
        return {}


def _write_usage(data: Dict[str, Any]) -> None:
    """Atomically write .usage.json via tempfile + os.replace."""
    path = _usage_file()
    try:
        fd, tmp = tempfile.mkstemp(
            suffix=".tmp", prefix=".usage_", dir=str(path.parent)
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        os.replace(tmp, str(path))
    except OSError as exc:
        logger.debug("Failed to write .usage.json: %s", exc)


# ── Public API ──


def list_skills() -> Dict[str, Any]:
    """Return all usage records (read-only snapshot)."""
    with _USAGE_LOCK:
        return dict(_read_usage())


def get_skill(name: str) -> Optional[Dict[str, Any]]:
    """Return a single skill's usage record, or None."""
    with _USAGE_LOCK:
        return _read_usage().get(name)


def bump(name: str, event: str = "use") -> None:
    """Increment a counter: use, view, or patch. Creates record if missing."""
    with _USAGE_LOCK:
        data = _read_usage()
        record = data.get(name)
        if record is None:
            record = _default_record()
            data[name] = record
        if event == "use":
            record["use_count"] = record.get("use_count", 0) + 1
        elif event == "view":
            record["view_count"] = record.get("view_count", 0) + 1
        elif event == "patch":
            record["patch_count"] = record.get("patch_count", 0) + 1
        record["last_activity_at"] = _now_iso()
        record["state"] = STATE_ACTIVE  # any activity re-activates
        _write_usage(data)


def set_state(name: str, state: str) -> bool:
    """Manually set a skill's lifecycle state. Returns True on success."""
    if state not in _VALID_STATES:
        return False
    with _USAGE_LOCK:
        data = _read_usage()
        record = data.get(name)
        if record is None:
            return False
        record["state"] = state
        record["last_activity_at"] = _now_iso()
        _write_usage(data)
        return True


def set_pinned(name: str, pinned: bool) -> bool:
    """Pin or unpin a skill. Pinned skills are exempt from auto-archive."""
    with _USAGE_LOCK:
        data = _read_usage()
        record = data.get(name)
        if record is None:
            return False
        record["pinned"] = pinned
        _write_usage(data)
        return True


def set_provenance(name: str, provenance: str) -> bool:
    """Set provenance: 'foreground' (user/manual) or 'background_review' (agent)."""
    with _USAGE_LOCK:
        data = _read_usage()
        record = data.get(name)
        if record is None:
            return False
        record["provenance"] = provenance
        _write_usage(data)
        return True


def update_confidence(name: str, delta: float) -> bool:
    """Adjust confidence score by delta, clamped to [0.0, 1.0]."""
    with _USAGE_LOCK:
        data = _read_usage()
        record = data.get(name)
        if record is None:
            return False
        current = record.get("confidence", 0.5)
        record["confidence"] = max(0.0, min(1.0, current + delta))
        _write_usage(data)
        return True


def remove_skill(name: str) -> bool:
    """Remove a skill's usage record entirely."""
    with _USAGE_LOCK:
        data = _read_usage()
        if name not in data:
            return False
        del data[name]
        _write_usage(data)
        return True


# ── Lifecycle Transitions ──


def mark_stale_if_unused(
    stale_after_days: int = DEFAULT_STALE_AFTER_DAYS,
) -> list[str]:
    """Mark skills as stale if they haven't been used in N days.
    Returns list of affected skill names. Skips pinned skills."""
    now = datetime.now(timezone.utc)
    affected: list[str] = []
    with _USAGE_LOCK:
        data = _read_usage()
        for name, record in data.items():
            if record.get("state") != STATE_ACTIVE:
                continue
            if record.get("pinned", False):
                continue
            if record.get("provenance") == "bundled":
                continue
            last_str = record.get("last_activity_at")
            if not last_str:
                continue
            try:
                last = datetime.fromisoformat(last_str)
            except (ValueError, TypeError):
                continue
            days_unused = (now - last).total_seconds() / 86400
            if days_unused > stale_after_days:
                record["state"] = STATE_STALE
                affected.append(name)
        if affected:
            _write_usage(data)
    return affected


def mark_archived_if_stale(
    archive_after_days: int = DEFAULT_ARCHIVE_AFTER_DAYS,
) -> list[str]:
    """Move stale skills to .archive/ if they've been stale > N days.
    Returns list of archived skill names."""
    now = datetime.now(timezone.utc)
    affected: list[str] = []
    with _USAGE_LOCK:
        data = _read_usage()
        for name, record in list(data.items()):
            if record.get("state") != STATE_STALE:
                continue
            if record.get("pinned", False):
                continue
            last_str = record.get("last_activity_at")
            if not last_str:
                continue
            try:
                last = datetime.fromisoformat(last_str)
            except (ValueError, TypeError):
                continue
            days_since = (now - last).total_seconds() / 86400
            if days_since > archive_after_days:
                # Move skill dir to .archive/
                skill_dir = _SKILLS_DIR / name
                if skill_dir.exists():
                    dest = _archive_dir() / f"{name}__{_now_iso().replace(':', '-')}"
                    try:
                        skill_dir.rename(dest)
                    except OSError as exc:
                        logger.warning("Failed to archive %s: %s", name, exc)
                        continue
                record["state"] = STATE_ARCHIVED
                affected.append(name)
        if affected:
            _write_usage(data)
    return affected


def get_active_skills() -> list[str]:
    """Return names of all active (non-stale, non-archived) skills."""
    with _USAGE_LOCK:
        data = _read_usage()
        return [n for n, r in data.items() if r.get("state") == STATE_ACTIVE]


def get_skills_by_domain(domain: str) -> list[str]:
    """Return active skills in a given domain category."""
    active = get_active_skills()
    return [n for n in active if _SKILLS_DIR.joinpath(n).parent.name == domain]
