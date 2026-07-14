import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.models.schemas import ErrorResponse
from app.services.assistant import AssistantServiceError
from app.services.disease_detection import DiseaseDetectionError


logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="http_error",
                detail=str(exc.detail),
                request_id=request.headers.get("x-request-id"),
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="validation_error",
                detail="Request validation failed.",
                request_id=request.headers.get("x-request-id"),
                fields=exc.errors(),
            ).model_dump(),
        )

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        logger.warning("Validation error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="validation_error",
                detail="Validation failed.",
                request_id=request.headers.get("x-request-id"),
            ).model_dump(),
        )

    @app.exception_handler(AssistantServiceError)
    async def assistant_service_exception_handler(
        request: Request, exc: AssistantServiceError
    ) -> JSONResponse:
        logger.exception("Assistant service failed")
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ErrorResponse(
                error="assistant_service_error",
                detail=str(exc),
                request_id=request.headers.get("x-request-id"),
            ).model_dump(),
        )

    @app.exception_handler(DiseaseDetectionError)
    async def disease_detection_exception_handler(
        request: Request, exc: DiseaseDetectionError
    ) -> JSONResponse:
        logger.exception("Disease detection failed")
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ErrorResponse(
                error="disease_detection_error",
                detail=str(exc),
                request_id=request.headers.get("x-request-id"),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_server_error",
                detail="An unexpected error occurred.",
                request_id=request.headers.get("x-request-id"),
            ).model_dump(),
        )
