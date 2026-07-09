from __future__ import annotations

import secrets
import logging

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ai_sidecar.config import settings

logger = logging.getLogger(__name__)

# Health endpoints that bypass auth
_OPEN_PATHS: frozenset[str] = frozenset({
    "/health/live", "/health/ready",
    "/v1/health/live", "/v1/health/ready",
    "/docs", "/redoc", "/openapi.json",
    "/docs/oauth2-redirect",
})

def _get_auth_token() -> bytes | None:
    """Read auth token from settings at call time, not import time."""
    if settings.api_auth_enabled and settings.api_auth_token:
        return settings.api_auth_token.encode("utf-8")
    return None


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        token_bytes = _get_auth_token()
        if not token_bytes or request.url.path in _OPEN_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Missing Authorization header"})
        token = auth_header.removeprefix("Bearer ").encode("utf-8")
        if not secrets.compare_digest(token, token_bytes):
            return JSONResponse(status_code=403, content={"detail": "Invalid API token"})

        return await call_next(request)


def add_auth_middleware(app):
    """Add auth middleware to a FastAPI app."""
    app.add_middleware(AuthMiddleware)
    logger.info("API auth middleware installed")
