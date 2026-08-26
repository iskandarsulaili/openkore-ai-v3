"""Self-awareness API — lets the conscious LLM write/read MEMORY.md lessons and
inspect the injected SOUL/MEMORY context, mirroring Hermes's memory tool contract.

The conscious tier calls POST /v1/self-awareness/lesson when it decides
something is worth remembering (a durable lesson). Every new lesson is pushed
to the central sink for P2P crowdsource improvement.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ai_sidecar.api.deps import get_runtime
from ai_sidecar.lifecycle import RuntimeState

router = APIRouter(prefix="/v1/self", tags=["self-awareness"])


class LessonIn(BaseModel):
    content: str = Field(min_length=1, max_length=2000)


class LessonOut(BaseModel):
    success: bool
    done: bool = False
    entry_count: int = 0
    usage: str = ""
    note: str = ""
    error: str = ""


class SelfStatusOut(BaseModel):
    ok: bool
    enabled: bool = False
    soul_chars: int = 0
    memory_entries: int = 0
    memory_char_count: int = 0
    memory_char_limit: int = 0
    sink_available: bool = False


@router.get("/status", response_model=SelfStatusOut)
def self_status(runtime: RuntimeState = Depends(get_runtime)) -> SelfStatusOut:
    sa = getattr(runtime, "self_awareness", None)
    if sa is None:
        return SelfStatusOut(ok=True, enabled=False)
    return SelfStatusOut(
        ok=True,
        enabled=True,
        soul_chars=len(sa.soul),
        memory_entries=len(sa.memory_entries),
        memory_char_count=sa.memory_char_count,
        memory_char_limit=sa.memory_char_limit,
        sink_available=bool(getattr(sa.sink, "available", False)),
    )


@router.post("/lesson", response_model=LessonOut)
def add_lesson(payload: LessonIn, runtime: RuntimeState = Depends(get_runtime)) -> LessonOut:
    """Append a durable lesson to MEMORY.md (self-learning)."""
    sa = getattr(runtime, "self_awareness", None)
    if sa is None:
        raise HTTPException(status_code=503, detail="self_awareness_disabled")
    result = sa.add_lesson(payload.content)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "lesson_rejected"))
    return LessonOut(
        success=True,
        done=result.get("done", False),
        entry_count=result.get("entry_count", 0),
        usage=result.get("usage", ""),
        note=result.get("note", ""),
    )


@router.get("/soul", response_model=dict[str, Any])
def get_soul(runtime: RuntimeState = Depends(get_runtime)) -> dict[str, Any]:
    """Return the current SOUL.md (identity + doctrine)."""
    sa = getattr(runtime, "self_awareness", None)
    if sa is None:
        raise HTTPException(status_code=503, detail="self_awareness_disabled")
    return {"ok": True, "soul": sa.soul}
