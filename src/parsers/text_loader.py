from __future__ import annotations

from pathlib import Path

from src.parsers.docx_parser import parse_docx
from src.parsers.pdf_parser import parse_pdf


def load_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text()
    if suffix == ".pdf":
        return parse_pdf(path)[0]
    if suffix == ".docx":
        return parse_docx(path)[0]
    raise ValueError(f"Unsupported JD format: {path.suffix}")
