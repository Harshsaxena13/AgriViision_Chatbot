import asyncio
import io
import logging

from fastapi import UploadFile, status
from pypdf import PdfReader
from starlette.exceptions import HTTPException


logger = logging.getLogger(__name__)

MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 10
MAX_PDF_CHARS = 8000


async def extract_pdf_upload(upload: UploadFile) -> str:
    """Return text from the first pages of an uploaded PDF, or raise an HTTP error."""
    data = await upload.read(MAX_PDF_BYTES + 1)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="PDF files must be 10 MB or smaller.",
        )
    if b"%PDF-" not in data[:1024]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    text = await asyncio.to_thread(_extract_text, data)
    logger.info("Extracted PDF text filename=%s characters=%d", upload.filename, len(text))
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No text could be read from this PDF. It may be a scanned document. "
                "Please send a photo instead."
            ),
        )
    return text


def _extract_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="This PDF is password-protected. Please upload an unlocked copy.",
            )
        pages: list[str] = []
        length = 0
        for page in reader.pages[:MAX_PDF_PAGES]:
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append(page_text)
                length += len(page_text)
            if length >= MAX_PDF_CHARS:
                break
    except HTTPException:
        raise
    except Exception as exc:  # pypdf raises many error types on malformed files
        logger.warning("Could not read PDF: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not read this PDF. The file may be damaged.",
        ) from exc
    return "\n\n".join(pages)[:MAX_PDF_CHARS].strip()
