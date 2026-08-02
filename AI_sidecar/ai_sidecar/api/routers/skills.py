"""Skills API Router — manage skill lifecycle via HTTP.

Endpoints:
  POST /v1/skills/manage — create, edit, patch, delete, write_file, remove_file
  GET  /v1/skills/list   — list skills with metadata
  GET  /v1/skills/view   — view full skill content
  POST /v1/skills/curate — trigger curator run
  GET  /v1/skills/status — curator status
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from ai_sidecar import skills_manager, skills_usage, skills_curator, skills_loader

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/skills", tags=["skills"])


# ── Schemas (inline — no pydantic dependency for Phase 1) ──


class ManageRequest:
    def __init__(self, data: dict):
        self.action = data.get("action", "")
        self.name = data.get("name", "")
        self.content = data.get("content", "")
        self.category = data.get("category")
        self.file_path = data.get("file_path", "SKILL.md")
        self.file_content = data.get("file_content", "")
        self.old_string = data.get("old_string", "")
        self.new_string = data.get("new_string", "")
        self.provenance = data.get("provenance", "foreground")
        self.confidence_delta = float(data.get("confidence_delta", 0) or 0)


# ── Endpoints ──


@router.post("/manage")
async def manage_skill(data: dict) -> dict:
    """Create, edit, patch, delete, write_file, or remove_file for skills."""
    req = ManageRequest(data)

    if req.action == "create":
        return skills_manager.create_skill(provenance=req.provenance, 
            name=req.name,
            content=req.content,
            category=req.category,
        )
    elif req.action == "edit":
        return skills_manager.edit_skill(
            name=req.name,
            content=req.content,
        )
    elif req.action == "patch":
        return skills_manager.patch_skill(provenance=req.provenance, 
            name=req.name,
            old_string=req.old_string,
            new_string=req.new_string,
            file_path=req.file_path,
        )
    elif req.action == "delete":
        return skills_manager.delete_skill(name=req.name, provenance=req.provenance)
    elif req.action == "write_file":
        return skills_manager.write_file(
            name=req.name,
            file_path=req.file_path,
            file_content=req.file_content,
        )
    elif req.action == "remove_file":
        return skills_manager.remove_file(
            name=req.name,
            file_path=req.file_path,
        )
    elif req.action == "adjust_confidence":
        ok = skills_usage.update_confidence(req.name, req.confidence_delta)
        return {"success": ok, "skill": req.name,
                "delta": req.confidence_delta,
                "error": None if ok else "skill not found in usage DB"}
    else:
        return {"success": False, "error": f"Unknown action: {req.action}"}


@router.get("/list")
async def list_skills(category: str = None, domain: str = None) -> List[dict]:
    """Return list of all skills with metadata.

    `category` filters by skill category dir; `domain` filters by metadata
    domain (routed through skills_usage.get_skills_by_domain).
    """
    result = skills_manager.list_skills(category=category)
    if domain:
        dset = set(skills_usage.get_skills_by_domain(domain))
        result = [r for r in result if r.get("name") in dset]
    return result


@router.get("/view")
async def view_skill(name: str = Query(...)) -> dict:
    """Return full skill content by name."""
    result = skills_manager.view_skill(name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    return result


@router.post("/curate")
async def curate(data: dict = None) -> dict:
    """Run a curator cycle. Pass dry_run=true to preview.

    Non-dry-run calls route through skills_curator.force_run() (an immediate,
    interval-independent run) so operators forcing a curation use the same
    entry point the alias exposes — the alias is wired, not dead.
    """
    if data is None:
        data = {}
    dry_run = data.get("dry_run", False)
    if dry_run:
        return skills_curator.run_curator(dry_run=True)
    return skills_curator.force_run()


@router.get("/status")
async def curator_status() -> dict:
    """Return curator status and skill statistics."""
    usage = skills_usage.list_skills()
    active = [n for n, r in usage.items() if r.get("state") == "active"]
    stale = [n for n, r in usage.items() if r.get("state") == "stale"]
    archived = [n for n, r in usage.items() if r.get("state") == "archived"]
    return {
        "enabled": skills_curator.get_config().get("enabled", True),
        "last_run_at": skills_curator.last_run_at(),
        "total_skills": len(usage),
        "active_count": len(active),
        "stale_count": len(stale),
        "archived_count": len(archived),
        "backups_available": len(skills_curator.list_backups()),
    }


@router.post("/load-context")
async def load_context(data: dict) -> dict:
    """Load skills matching a situation. Used by context_assembler."""
    situation = data.get("situation", {})
    max_skills = data.get("max_skills", 5)
    content = skills_loader.load_for_context(situation, max_skills=max_skills)
    return {"skills_loaded": len(content), "content": content}
