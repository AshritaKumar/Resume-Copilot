from __future__ import annotations

from pathlib import Path

import pytesseract
from PIL import Image

from src.core.errors import ParseError


def parse_image_with_ocr(path: Path) -> tuple[str, float]:
    try:
        text = pytesseract.image_to_string(Image.open(path)).strip()
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"Failed OCR parse for {path.name}: {exc}") from exc
    if not text:
        raise ParseError(f"Empty OCR text for {path.name}")
    return text, 0.6
