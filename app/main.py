from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi.middleware import SlowAPIASGIMiddleware

from app.api.routes import assistant, health
from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.db.sqlite import initialize_database
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.services.assistant import create_assistant_http_client
from app.services.disease_detection import create_disease_http_client
from app.services.sarvam_voice import create_sarvam_http_client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    await initialize_database()
    # One HTTP client per service, shared by all requests and closed on shutdown.
    async with AsyncExitStack() as stack:
        app.state.assistant_http = await stack.enter_async_context(
            create_assistant_http_client(settings)
        )
        app.state.sarvam_http = await stack.enter_async_context(
            create_sarvam_http_client(settings)
        )
        app.state.disease_http = await stack.enter_async_context(
            create_disease_http_client(settings)
        )
        yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered agricultural assistant for crop disease guidance.",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    # Added before CORS so CORS stays outermost and 429 responses still carry CORS headers.
    app.add_middleware(SlowAPIASGIMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])
    app.include_router(assistant.router, prefix=settings.api_prefix, tags=["assistant"])
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

    return app


app = create_app()
