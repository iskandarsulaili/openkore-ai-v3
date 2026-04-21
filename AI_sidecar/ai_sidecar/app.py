from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from ai_sidecar.api.routers import (
    acknowledgements,
    actions,
    fleet,
    health,
    ingest,
    ingest_v2,
    macros,
    planner_v2,
    providers_v2,
    reflex,
    state_v2,
    telemetry,
)
from ai_sidecar.config import settings
from ai_sidecar.lifecycle import create_runtime
from ai_sidecar.logging_setup import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(level=settings.log_level, use_json=settings.log_json)
    app.state.runtime = create_runtime()
    yield


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
