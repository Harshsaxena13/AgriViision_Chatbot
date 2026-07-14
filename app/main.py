from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import assistant, health
from app.core.config import get_settings
from app.db.sqlite import initialize_database
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered agricultural assistant for crop disease guidance.",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    register_exception_handlers(app)
    app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])
    app.include_router(assistant.router, prefix=settings.api_prefix, tags=["assistant"])
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

    @app.on_event("startup")
    async def startup() -> None:
        await initialize_database()

    return app


app = create_app()
