from __future__ import annotations

from fastapi import Request

from ai_sidecar.lifecycle import RuntimeState


def get_runtime(request: Request) -> RuntimeState:
    return request.app.state.runtime

