from __future__ import annotations

import asyncio
import json
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi import Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response

from ai_sidecar.api.routers import (
    skills,
    discovery,
    npc_dialog,
    acknowledgements,
    actions,
    combat,
    conscious,
    control_domain,
    crewai_v2,
    fleet,
    fleet_coordinator,
    fleet_v2,
    health,
    ingest,
    ingest_v2,
    macros,
    ml_subconscious_v2,
    observability_v2,
    party,
    planner_v2,
    providers_v2,
    reflex,
    state_v2,
    telemetry,
)
from ai_sidecar.config import settings
from ai_sidecar.lifecycle import create_runtime, start_fleet_sync_loop
from ai_sidecar.logging_setup import configure_logging
from ai_sidecar.observability import install_fastapi_tracing
from ai_sidecar.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(level=settings.log_level, use_json=settings.log_json)
    
    # Validate auth config at startup
    if settings.api_auth_enabled:
        if not settings.api_auth_token:
            _generated = secrets.token_urlsafe(32)
            settings.api_auth_token = _generated
            # Write token to file so the Perl bridge can read it
            _token_path = Path.cwd() / "control" / "sidecar_auth_token.txt"
            try:
                _token_path.parent.mkdir(parents=True, exist_ok=True)
                _token_path.write_text(_generated)
                logger.info("api_auth_token written to %%s for bridge plugin", _token_path)
            except Exception:
                pass
            logger.warning("api_auth_token not set — generated: %%s", _generated)
        if not settings.api_auth_token:
            logger.warning("api_auth_enabled=True but api_auth_token is empty — auth will reject all requests")
        elif len(settings.api_auth_token) < 8:
            logger.warning("api_auth_token is very short (<8 chars) — consider a stronger token")
    app.state.runtime = create_runtime()
    runtime = app.state.runtime

    # Initialize and start PDCA loop
    from ai_sidecar.autonomy.pdca_loop import PDCALoop, PDCAConfig
    from ai_sidecar.api.routers.autonomy import set_pdca_loop

    pdca_config = PDCAConfig(
        short_term_interval_s=5.0,
        medium_term_interval_s=30.0,
        long_term_interval_s=120.0,
    )
    pdca_loop = PDCALoop(runtime_state=runtime, config=pdca_config)
    runtime.pdca_loop = pdca_loop
    set_pdca_loop(pdca_loop)
    # Auto-start the PDCA loop
    pdca_loop.start()
    logger.info("PDCA autonomy loop started (auto)")

    # Initialize Failure Reasoning Pipeline
    try:
        from ai_sidecar.learning.failure_wiring import wire_failure_pipeline
        _fre = wire_failure_pipeline(runtime)
        runtime.failure_reasoning = _fre
        logger.info("failure_reasoning_pipeline_wired")
    except Exception as e:
        logger.warning("failure_reasoning_wire_failed: %s", e)
    fleet_sync_task: asyncio.Task[None] | None = None
    fleet_sync_enabled = bool(getattr(runtime.fleet_sync_client, "enabled", False))
    if fleet_sync_enabled:
        fleet_sync_task = start_fleet_sync_loop(runtime)
        logger.info("fleet sync loop started")
    else:
        logger.info(
            "fleet sync loop disabled",
            extra={"event": "fleet_sync_loop_disabled", "fleet_central_enabled": False},
        )
    # Start skills curator background loop
    curator_task = None
    try:
        from ai_sidecar.skills_curator import run_curator, should_run_now
        async def _curator_loop():
            while True:
                try:
                    if should_run_now():
                        result = run_curator()
                        if result.get("marked_stale"):
                            logger.info("Curator marked %d skills stale", len(result["marked_stale"]))
                        if result.get("archived"):
                            logger.info("Curator archived %d skills", len(result["archived"]))
                except Exception as exc:
                    logger.debug("Curator cycle error: %s", exc)
                await asyncio.sleep(3600)  # run every hour
        curator_task = asyncio.create_task(_curator_loop())
        logger.info("Skills curator background loop started")
    except Exception as exc:
        logger.debug("Curator not available: %s", exc)
        curator_task = None
    # Start keep-alive loop if enabled
    runtime.keep_alive_enabled = settings.keep_alive_enabled
    runtime.keep_alive_timeout_minutes = settings.keep_alive_timeout_minutes
    runtime.keep_alive_poll_interval_s = settings.keep_alive_poll_interval_s
    if runtime.keep_alive_enabled:
        runtime.start_keep_alive()
        logger.info(
            "keep_alive_mode_enabled",
            extra={
                "event": "keep_alive_mode_enabled",
                "timeout_minutes": runtime.keep_alive_timeout_minutes,
                "poll_interval_s": runtime.keep_alive_poll_interval_s,
                "server": f"{settings.game_server_host}:{settings.game_server_port}",
            },
        )


    yield

    # Stop curator loop
    if curator_task is not None:
        curator_task.cancel()
        try:
            await curator_task
        except (asyncio.CancelledError, Exception):
            pass

    # Stop PDCA loop
    if pdca_loop.running:
        await pdca_loop.stop()
        logger.info("PDCA autonomy loop stopped")
    if fleet_sync_task is not None:
        fleet_sync_task.cancel()
        try:
            await fleet_sync_task
        except asyncio.CancelledError:
            logger.info("fleet sync loop cancelled")
        except Exception:
            logger.info("fleet sync loop stopped")
    try:
        await runtime.shutdown()
    except Exception:
        logger.exception("runtime_shutdown_failed")


def install_request_validation_logging(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request: Request, exc: RequestValidationError):
        trace_id = str(getattr(request.state, "trace_id", "") or "")
        body_preview = ""
        try:
            raw_body = await request.body()
            if raw_body:
                body_preview = raw_body.decode("utf-8", errors="replace")[:2048]
        except Exception as body_error:
            body_preview = f"<unavailable:{type(body_error).__name__}>"

        details = exc.errors()
        logger.debug(
            "http_request_validation_failed",
            extra={
                "event": "http_request_validation_failed_ignored",
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "errors": details,
                "errors_json": json.dumps(details, ensure_ascii=False)[:4096],
                "body_preview": body_preview,
            },
        )
        return await request_validation_exception_handler(request, exc)

# Auth token check middleware
if settings.api_auth_enabled and settings.api_auth_token:
    _api_token = settings.api_auth_token.encode("utf-8")

    class _AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):
            # Skip auth for health endpoints
            if request.url.path in ("/health/live", "/health/ready", "/v1/health/live", "/v1/health/ready"):
                return await call_next(request)
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
            token = auth_header.removeprefix("Bearer ").encode("utf-8")
            if not secrets.compare_digest(token, _api_token):
                raise HTTPException(status_code=403, detail="Invalid API token")
            return await call_next(request)

def create_app() -> FastAPI:
    docs_url = "/docs" if settings.enable_docs else None
    redoc_url = "/redoc" if settings.enable_docs else None

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
    )
    if settings.observability_enable_tracing:
        install_fastapi_tracing(app)
    install_request_validation_logging(app)
    app.include_router(health.router, prefix="/health")
    app.include_router(health.router, prefix="/v1/health")
    app.include_router(ingest.router)
    app.include_router(actions.router)
    app.include_router(skills.router)
    app.include_router(acknowledgements.router)
    app.include_router(macros.router)
    app.include_router(telemetry.router)
    app.include_router(fleet.router)
    app.include_router(ingest_v2.router)
    app.include_router(state_v2.router)
    app.include_router(reflex.router)
    app.include_router(control_domain.router)
    app.include_router(planner_v2.router)
    app.include_router(providers_v2.router)
    app.include_router(crewai_v2.router)
    app.include_router(ml_subconscious_v2.router)
    app.include_router(fleet_v2.router)
    app.include_router(observability_v2.router)
    app.include_router(combat.router)
    app.include_router(conscious.router)
    app.include_router(party.router)
    app.include_router(discovery.router)
    app.include_router(fleet_coordinator.router)
    app.include_router(npc_dialog.router)
    # Register autonomy router
    from ai_sidecar.api.routers.autonomy import router as autonomy_router, set_pdca_loop

    app.include_router(autonomy_router)
    if settings.api_auth_enabled and settings.api_auth_token:
        from ai_sidecar.api.middleware import add_auth_middleware
        add_auth_middleware(app)
    return app


app = create_app()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="openkore-ai-sidecar")
    parser.add_argument("--keep-alive", action="store_true", default=False,
                        help="Enable keep-alive mode: stay alive when no bots connected, poll game server, auto-restart bots")
    parser.add_argument("--keep-alive-timeout", type=int, default=30,
                        help="Keep-alive timeout in minutes (default: 30)")
    parser.add_argument("--keep-alive-poll", type=float, default=30.0,
                        help="Keep-alive poll interval in seconds (default: 30)")
    args = parser.parse_args()

    if args.keep_alive:
        settings.keep_alive_enabled = True
        settings.keep_alive_timeout_minutes = args.keep_alive_timeout
        settings.keep_alive_poll_interval_s = args.keep_alive_poll
        logger.info(
            "keep_alive_enabled_via_cli",
            extra={
                "event": "keep_alive_enabled_via_cli",
                "timeout_minutes": args.keep_alive_timeout,
                "poll_interval_s": args.keep_alive_poll,
            },
        )

    uvicorn.run(
        "ai_sidecar.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
