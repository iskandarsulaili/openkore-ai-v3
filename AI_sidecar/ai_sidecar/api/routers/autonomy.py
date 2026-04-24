"""Autonomy control API — start/stop/status the PDCA loop."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/autonomy", tags=["autonomy"])

# Global reference set by app startup
_pdca_loop: Any = None


def set_pdca_loop(loop: Any) -> None:
    global _pdca_loop
    _pdca_loop = loop


@router.post("/start")
async def start_autonomy() -> dict[str, Any]:
    """Start the PDCA autonomy loop."""
    if _pdca_loop is None:
        raise HTTPException(status_code=503, detail="PDCA loop not initialized")
    if _pdca_loop.running:
        return {"status": "already_running"}
    _pdca_loop.start()
    return {"status": "started"}


@router.post("/stop")
async def stop_autonomy() -> dict[str, Any]:
    """Stop the PDCA autonomy loop."""
    if _pdca_loop is None:
        raise HTTPException(status_code=503, detail="PDCA loop not initialized")
    if not _pdca_loop.running:
        return {"status": "already_stopped"}
    await _pdca_loop.stop()
    return {"status": "stopped"}


@router.get("/status")
async def autonomy_status() -> dict[str, Any]:
    """Get PDCA loop status."""
    if _pdca_loop is None:
        raise HTTPException(status_code=503, detail="PDCA loop not initialized")
    return await _pdca_loop.get_status()
