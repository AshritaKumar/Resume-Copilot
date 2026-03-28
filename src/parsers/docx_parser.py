from __future__ import annotations

from pathlib import Path

from docx import Document

from src.config.settings import CONFIG
from src.core.errors import ParseError


def parse_docx(path: Path) -> tuple[str, float]:
    try:
        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"Failed DOCX parse for {path.name}: {exc}") from exc

    if not text:
        raise ParseError(f"Empty DOCX text for {path.name}")
    confidence = 0.95 if len(text) > CONFIG.thresholds.low_text_length_confidence_cutoff else 0.8
    return text, confidence
