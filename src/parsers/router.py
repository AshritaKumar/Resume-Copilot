from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import CONFIG
from src.core.errors import ParseError
from src.parsers.docx_parser import parse_docx
from src.parsers.ocr_parser import parse_image_with_ocr
from src.parsers.pdf_parser import parse_pdf


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


@dataclass
class ParseResult:
    file_path: Path
    parser_used: str
    confidence: float
    text: str
    status: str
    warning: str | None = None


def parse_resume(path: Path) -> ParseResult:
    suffix = path.suffix.lower()
    mime, _ = mimetypes.guess_type(str(path))
    try:
        if suffix == ".pdf":
            text, confidence = parse_pdf(path)
            if confidence < CONFIG.thresholds.min_pdf_confidence_for_no_ocr:
                return _attempt_ocr_fallback(path, text, confidence)
            return ParseResult(path, "pdf", confidence, text, "ok")
        if suffix == ".docx":
            text, confidence = parse_docx(path)
            return ParseResult(path, "docx", confidence, text, "ok")
        if suffix in SUPPORTED_IMAGE_SUFFIXES or (mime and mime.startswith("image/")):
            text, confidence = parse_image_with_ocr(path)
            return ParseResult(path, "ocr", confidence, text, "ok")
        return ParseResult(path, "none", 0.0, "", "unsupported", "Unsupported file type.")
    except ParseError as exc:
        return ParseResult(path, "none", 0.0, "", "parse_failed", str(exc))


def _attempt_ocr_fallback(path: Path, primary_text: str, primary_confidence: float) -> ParseResult:
    # OCR fallback only applies if the PDF has low extractable text signal.
    image_equivalent = path.with_suffix(".png")
    if not image_equivalent.exists():
        return ParseResult(
            file_path=path,
            parser_used="pdf",
            confidence=primary_confidence,
            text=primary_text,
            status="ok",
            warning="Low-confidence PDF extraction and no OCR fallback image found.",
        )
    try:
        ocr_text, ocr_confidence = parse_image_with_ocr(image_equivalent)
        if len(ocr_text) > len(primary_text):
            return ParseResult(path, "ocr_fallback", ocr_confidence, ocr_text, "ok")
    except ParseError:
        pass
    return ParseResult(path, "pdf", primary_confidence, primary_text, "ok")
