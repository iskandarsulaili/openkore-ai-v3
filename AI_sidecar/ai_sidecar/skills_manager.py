"""Skill Manager — Create/read/update/delete/patch skill files.

Skills are stored as directories under AI_sidecar/ai_sidecar/skills/,
each containing a SKILL.md with YAML frontmatter and optional supporting
files (references/, templates/, assets/).

Inspired by Hermes Agent's skill_manager_tool.py pattern.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai_sidecar import skills_usage

logger = logging.getLogger(__name__)

# ── Paths ──

_SKILLS_DIR = Path(__file__).resolve().parent / "skills"

# ── Validation ──

_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9_-]{0,62}[a-z0-9])?$")
_MAX_NAME_LEN = 64
_MAX_DESC_LEN = 1024


def _validate_name(name: str) -> Optional[str]:
    if not name or len(name) > _MAX_NAME_LEN:
        return f"Name must be 1-{_MAX_NAME_LEN} characters"
    if not _NAME_RE.match(name):
        return "Name must start/end with alphanumeric, contain only lowercase letters, digits, hyphens, underscores"
    return None


def _validate_frontmatter(content: str) -> Optional[str]:
    """Check that content has valid YAML frontmatter with name and description."""
    stripped = content.strip()
    if not stripped.startswith("---"):
        return "Content must start with --- frontmatter"
    end = stripped.find("---", 3)
    if end == -1:
        return "Unclosed --- frontmatter block"
    front = stripped[3:end]
    if "name:" not in front:
        return "Frontmatter must contain 'name:' field"
    return None


def _validate_content_size(content: str, max_chars: int = 50000) -> Optional[str]:
    if len(content) > max_chars:
        return f"Content exceeds {max_chars} characters"
    return None


def _validate_file_path(file_path: str) -> Optional[str]:
    if not file_path:
        return "file_path is required"
    # Prevent path traversal
    resolved = Path(file_path)
    if ".." in resolved.parts:
        return "Path traversal not allowed"
    return None


# ── Path Resolution ──


def _skills_dir() -> Path:
    _SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    return _SKILLS_DIR


def _resolve_skill_dir(name: str, category: str = None) -> Path:
    base = _skills_dir()
    if category:
        base = base / category
        base.mkdir(parents=True, exist_ok=True)
    return base / name


def _find_skill(name: str) -> Optional[Dict[str, Any]]:
    """Find a skill by name, searching categories if needed."""
    # Direct lookup
    skill_dir = _skills_dir() / name
    if skill_dir.exists() and (skill_dir / "SKILL.md").exists():
        return {"name": name, "path": skill_dir, "category": ""}
    # Search categories
    for cat_dir in _skills_dir().iterdir():
        if not cat_dir.is_dir() or cat_dir.name.startswith("."):
            continue
        skill_dir = cat_dir / name
        if skill_dir.exists() and (skill_dir / "SKILL.md").exists():
            return {"name": name, "path": skill_dir, "category": cat_dir.name}
    return None


def _read_skill_md(skill_dir: Path) -> Optional[str]:
    path = skill_dir / "SKILL.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _atomic_write_text(file_path: Path, content: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        suffix=".tmp", prefix=f".{file_path.name}_", dir=str(file_path.parent)
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, str(file_path))


# ── Public API ──


def create_skill(
    name: str,
    content: str,
    category: str = None,
    provenance: str = "foreground",
) -> Dict[str, Any]:
    """Create a new skill (SKILL.md + directory). Returns result dict."""
    err = _validate_name(name)
    if err:
        return {"success": False, "error": err}
    err = _validate_frontmatter(content)
    if err:
        return {"success": False, "error": err}
    err = _validate_content_size(content)
    if err:
        return {"success": False, "error": err}

    if _find_skill(name):
        return {"success": False, "error": f"Skill '{name}' already exists"}

    skill_dir = _resolve_skill_dir(name, category)
    skill_dir.mkdir(parents=True, exist_ok=True)

    try:
        _atomic_write_text(skill_dir / "SKILL.md", content)
    except OSError as exc:
        return {"success": False, "error": f"Failed to write SKILL.md: {exc}"}

    # Create usage record
    skills_usage.bump(name, event="use")
    skills_usage.set_provenance(name, provenance)

    return {
        "success": True,
        "name": name,
        "path": str(skill_dir),
        "category": category or "",
    }


def edit_skill(name: str, content: str) -> Dict[str, Any]:
    """Replace the SKILL.md of an existing skill (full rewrite)."""
    err = _validate_frontmatter(content)
    if err:
        return {"success": False, "error": err}
    err = _validate_content_size(content)
    if err:
        return {"success": False, "error": err}

    existing = _find_skill(name)
    if not existing:
        return {"success": False, "error": f"Skill '{name}' not found"}

    skill_md = existing["path"] / "SKILL.md"
    try:
        _atomic_write_text(skill_md, content)
    except OSError as exc:
        return {"success": False, "error": f"Failed to write: {exc}"}

    skills_usage.bump(name, event="patch")
    return {"success": True, "name": name}


def patch_skill(
    name: str,
    old_string: str,
    new_string: str,
    file_path: str = "SKILL.md",
) -> Dict[str, Any]:
    """Find-and-replace within SKILL.md or a supporting file."""
    if not old_string:
        return {"success": False, "error": "old_string is required"}
    if new_string is None:
        new_string = ""

    existing = _find_skill(name)
    if not existing:
        return {"success": False, "error": f"Skill '{name}' not found"}

    target = existing["path"] / file_path
    if not target.exists():
        return {"success": False, "error": f"File '{file_path}' not found in skill '{name}'"}

    try:
        original = target.read_text(encoding="utf-8")
    except OSError as exc:
        return {"success": False, "error": f"Failed to read: {exc}"}

    if old_string not in original:
        return {"success": False, "error": "old_string not found in file"}

    updated = original.replace(old_string, new_string)
    try:
        _atomic_write_text(target, updated)
    except OSError as exc:
        return {"success": False, "error": f"Failed to write: {exc}"}

    skills_usage.bump(name, event="patch")
    return {"success": True, "name": name, "file": file_path}


def delete_skill(name: str) -> Dict[str, Any]:
    """Remove a skill directory entirely."""
    existing = _find_skill(name)
    if not existing:
        return {"success": False, "error": f"Skill '{name}' not found"}

    record = skills_usage.get_skill(name)
    if record and record.get("pinned", False):
        return {"success": False, "error": f"Skill '{name}' is pinned — unpin before deleting"}

    try:
        shutil.rmtree(existing["path"])
    except OSError as exc:
        return {"success": False, "error": f"Failed to delete: {exc}"}

    skills_usage.remove_skill(name)
    return {"success": True, "name": name}


def write_file(
    name: str,
    file_path: str,
    file_content: str,
) -> Dict[str, Any]:
    """Add or overwrite a supporting file (reference, template, script, asset)."""
    existing = _find_skill(name)
    if not existing:
        return {"success": False, "error": f"Skill '{name}' not found"}

    target = existing["path"] / file_path
    try:
        _atomic_write_text(target, file_content)
    except OSError as exc:
        return {"success": False, "error": f"Failed to write: {exc}"}

    return {"success": True, "name": name, "file": file_path}


def remove_file(name: str, file_path: str) -> Dict[str, Any]:
    """Remove a supporting file from a skill."""
    existing = _find_skill(name)
    if not existing:
        return {"success": False, "error": f"Skill '{name}' not found"}

    target = existing["path"] / file_path
    if not target.exists():
        return {"success": False, "error": f"File '{file_path}' not found"}

    try:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    except OSError as exc:
        return {"success": False, "error": f"Failed to remove: {exc}"}

    return {"success": True, "name": name, "file": file_path}


def _recurse_list(
    directory: Path,
    usage_data: Dict[str, Any],
    result: List[Dict[str, Any]],
    category: str = "",
) -> None:
    """Recursively scan a directory for skills (SKILL.md files)."""
    for item in directory.iterdir():
        if item.name.startswith("."):
            continue
        if not item.is_dir():
            continue
        skill_md = item / "SKILL.md"
        if not skill_md.exists():
            continue
        skill_name = item.name
        content = skill_md.read_text(encoding="utf-8")
        meta = _parse_frontmatter(content)
        record = usage_data.get(skill_name, {})
        result.append({
            "name": skill_name,
            "description": meta.get("description", ""),
            "tags": meta.get("metadata", {}).get("hermes", {}).get("tags", []),
            "domain": meta.get("metadata", {}).get("domain", category),
            "state": record.get("state", "active"),
            "confidence": record.get("confidence", 0.5),
            "use_count": record.get("use_count", 0),
            "last_used": record.get("last_activity_at", ""),
        })

def list_skills(category: str = None) -> List[Dict[str, Any]]:
    """List all skills with metadata. Returns list of {name, description, tags, state}."""
    result: List[Dict[str, Any]] = []
    usage_data = skills_usage.list_skills()

    # Scan skills dir
    skills_dir = _skills_dir()
    if not skills_dir.exists():
        return result

    for item in skills_dir.iterdir():
        if not item.is_dir() or item.name.startswith("."):
            continue
        skill_name = item.name
        skill_md = item / "SKILL.md"
        if not skill_md.exists():
                # Not a skill — might be a category dir, recurse
                if not (skill_md.parent / "SKILL.md").exists():
                    _recurse_list(item, usage_data, result, category=item.name)
                continue
        meta = _parse_frontmatter(content)

        record = usage_data.get(skill_name, {})
        result.append({
            "name": skill_name,
            "description": meta.get("description", ""),
            "tags": meta.get("metadata", {}).get("hermes", {}).get("tags", []),
            "domain": meta.get("metadata", {}).get("domain", ""),
            "state": record.get("state", "active"),
            "confidence": record.get("confidence", 0.5),
            "use_count": record.get("use_count", 0),
            "last_used": record.get("last_activity_at", ""),
        })

    # Sort: active first, then by use_count desc
    result.sort(key=lambda x: (
        0 if x["state"] == "active" else 1,
        -x["use_count"],
    ))

    if category:
        result = [s for s in result if s.get("domain") == category]

    return result


def view_skill(name: str) -> Optional[Dict[str, Any]]:
    """Load full skill content: SKILL.md + linked files list.
    Returns None if not found. Bumps view_count."""
    existing = _find_skill(name)
    if not existing:
        return None

    content = _read_skill_md(existing["path"])
    if content is None:
        return None

    # List supporting files
    supporting: List[Dict[str, Any]] = []
    for sub in existing["path"].iterdir():
        if sub.name == "SKILL.md":
            continue
        if sub.is_file():
            supporting.append({
                "path": sub.name,
                "size": sub.stat().st_size,
            })
        elif sub.is_dir():
            supporting.append({
                "path": sub.name + "/",
                "type": "directory",
                "files": [f.name for f in sub.iterdir() if f.is_file()],
            })

    skills_usage.bump(name, event="view")
    return {
        "name": name,
        "content": content,
        "path": str(existing["path"]),
        "category": existing.get("category", ""),
        "supporting": supporting,
    }


def _parse_frontmatter(content: str) -> Dict[str, Any]:
    """Minimal frontmatter parser (YAML without pyyaml dependency)."""
    result: Dict[str, Any] = {}
    stripped = content.strip()
    if not stripped.startswith("---"):
        return result
    end = stripped.find("---", 3)
    if end == -1:
        return result
    front = stripped[3:end]
    current_key = None
    for line in front.split("\n"):
        line = line.rstrip()  # preserve leading spaces for indentation
        if not line:
            continue
        if ":" in line and not line.startswith("  "):
            parts = line.split(":", 1)
            current_key = parts[0].strip()
            val = parts[1].strip()
            # Handle nested (e.g., metadata.hermes.tags)
            if current_key == "metadata":
                result[current_key] = {}
            elif current_key == "description":
                result[current_key] = val
            elif current_key == "name":
                result[current_key] = val
            elif current_key == "triggers":
                result[current_key] = [t.strip().strip(chr(34)+chr(39)) for t in val.split(",")] if val else []
            elif current_key == "when_to_use":
                result[current_key] = [t.strip().strip(chr(34)+chr(39)) for t in val.split(",")] if val else []
            else:
                result[current_key] = val
        elif line.startswith("  ") and current_key:
            # Handle indented list items
            val = line.strip().strip("- ").strip()
            if isinstance(result.get(current_key), list):
                result[current_key].append(val)
    return result
