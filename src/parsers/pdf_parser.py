from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from src.config.settings import CONFIG
from src.core.errors import ParseError


def parse_pdf(path: Path) -> tuple[str, float]:
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"Failed PDF parse for {path.name}: {exc}") from exc

    text = "\n".join(pages).strip()
    if not text:
        raise ParseError(f"Empty PDF text for {path.name}")
    confidence = 0.95 if len(text) > CONFIG.thresholds.low_text_length_confidence_cutoff else 0.75
    return text, confidence
