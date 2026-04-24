from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi import Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from ai_sidecar.api.routers import (
    acknowledgements,
    actions,
    crewai_v2,
    fleet,
    fleet_v2,
    health,
    ingest,
    ingest_v2,
    macros,
    ml_subconscious_v2,
    observability_v2,
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

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(level=settings.log_level, use_json=settings.log_json)
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
    set_pdca_loop(pdca_loop)
    # Auto-start the PDCA loop
    pdca_loop.start()
    logger.info("PDCA autonomy loop started (auto)")
    fleet_sync_task = start_fleet_sync_loop(runtime)
    logger.info("fleet sync loop started")
    yield

    # Stop PDCA loop
    if pdca_loop.running:
        await pdca_loop.stop()
        logger.info("PDCA autonomy loop stopped")
    fleet_sync_task.cancel()
    try:
        await fleet_sync_task
    except asyncio.CancelledError:
        logger.info("fleet sync loop cancelled")
    except Exception:
        logger.info("fleet sync loop stopped")


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
        logger.warning(
            "http_request_validation_failed",
            extra={
                "event": "http_request_validation_failed",
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "errors": details,
                "errors_json": json.dumps(details, ensure_ascii=False)[:4096],
                "body_preview": body_preview,
            },
        )
        return await request_validation_exception_handler(request, exc)


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
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(actions.router)
    app.include_router(acknowledgements.router)
    app.include_router(macros.router)
    app.include_router(telemetry.router)
    app.include_router(fleet.router)
    app.include_router(ingest_v2.router)
    app.include_router(state_v2.router)
    app.include_router(reflex.router)
    app.include_router(planner_v2.router)
    app.include_router(providers_v2.router)
    app.include_router(crewai_v2.router)
    app.include_router(ml_subconscious_v2.router)
    app.include_router(fleet_v2.router)
    app.include_router(observability_v2.router)
    # Register autonomy router
    from ai_sidecar.api.routers.autonomy import router as autonomy_router, set_pdca_loop

    app.include_router(autonomy_router)
    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "ai_sidecar.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
