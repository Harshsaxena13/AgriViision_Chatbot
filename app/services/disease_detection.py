import logging
from typing import Any, Optional

import httpx
from fastapi import UploadFile, status
from starlette.exceptions import HTTPException

from app.core.config import Settings, get_settings
from app.models.schemas import DiseaseDetectionResponse


logger = logging.getLogger(__name__)


class DiseaseDetectionError(RuntimeError):
    """Raised when the disease detection API cannot return a prediction."""


class WheatDiseaseDetectionService:
    _allowed_content_types = {"image/jpeg", "image/png", "image/webp"}

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def predict_upload(self, image: UploadFile) -> DiseaseDetectionResponse:
        content_type = image.content_type or "application/octet-stream"
        if content_type not in self._allowed_content_types:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Upload a JPEG, PNG, or WebP crop image.",
            )

        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image is empty.",
            )

        return await self.predict(
            filename=image.filename or "crop-image",
            content_type=content_type,
            image_bytes=image_bytes,
        )

    async def predict(
        self, filename: str, content_type: str, image_bytes: bytes
    ) -> DiseaseDetectionResponse:
        endpoint = self._build_endpoint()
        logger.info("Calling wheat disease API endpoint=%s filename=%s", endpoint, filename)

        try:
            async with httpx.AsyncClient(
                timeout=self._settings.disease_api_timeout_seconds
            ) as client:
                response = await client.post(
                    endpoint,
                    files={
                        self._settings.disease_api_file_field: (
                            filename,
                            image_bytes,
                            content_type,
                        )
                    },
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise DiseaseDetectionError(f"Wheat disease API request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise DiseaseDetectionError("Wheat disease API returned invalid JSON.") from exc

        disease, confidence = self._extract_prediction(payload)
        return DiseaseDetectionResponse(
            disease=disease,
            confidence=confidence,
            raw_response=payload,
        )

    def _build_endpoint(self) -> str:
        base_url = self._settings.disease_api_url.rstrip("/")
        path = self._settings.disease_api_predict_path.strip()
        if not path:
            return base_url
        return f"{base_url}/{path.lstrip('/')}"

    def _extract_prediction(self, payload: dict[str, Any]) -> tuple[str, Optional[float]]:
        disease_keys = ("disease", "class", "label", "prediction", "predicted_class")
        confidence_keys = ("confidence", "probability", "score")

        normalized_payload = self._normalize_payload(payload)
        disease = next(
            (
                str(normalized_payload[key])
                for key in disease_keys
                if key in normalized_payload and normalized_payload[key]
            ),
            "Unknown wheat disease",
        )
        confidence = None
        for key in confidence_keys:
            if key in normalized_payload and normalized_payload[key] is not None:
                try:
                    confidence = float(normalized_payload[key])
                except (TypeError, ValueError):
                    confidence = None
                break

        return disease, confidence

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        for key in ("result", "data", "prediction", "predictions"):
            value = payload.get(key)
            if isinstance(value, dict):
                return {**payload, **value}
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return {**payload, **value[0]}
        return payload


def get_disease_detection_service() -> WheatDiseaseDetectionService:
    return WheatDiseaseDetectionService(get_settings())
