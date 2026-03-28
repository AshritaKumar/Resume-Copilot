from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Callable

import streamlit as st
from pypdf import PdfReader

from src.config.settings import CONFIG
from src.ui.upload_names import candidate_id_from_upload_name


def build_preflight_report(
    jd_file,
    uploaded_resumes,
    *,
    validate_upload_types: Callable[[object, list[object]], None],
) -> dict[str, object]:
    report: dict[str, object] = {"blocking": [], "warnings": [], "infos": [], "has_blockers": False}
    if not jd_file or not uploaded_resumes:
        return report
    try:
        validate_upload_types(jd_file, uploaded_resumes)
    except Exception as exc:  # noqa: BLE001
        report["blocking"] = [str(exc)]
        report["has_blockers"] = True
        return report

    if len(jd_file.getbuffer()) == 0:
        _append_message(report, "blocking", "JD file is empty. Upload a non-empty JD.")
    empty_resumes = [file.name for file in uploaded_resumes if len(file.getbuffer()) == 0]
    if empty_resumes:
        _append_message(report, "blocking", f"Empty resume file(s): {', '.join(empty_resumes)}.")

    duplicate_names = _detect_duplicate_resumes(uploaded_resumes)
    if duplicate_names:
        _append_message(
            report,
            "warnings",
            "Possible duplicate resumes detected: "
            + ", ".join(f"`{name}`" for name in duplicate_names)
            + ". Keep only one copy to improve shortlist quality.",
        )

    low_text_pdf_files = _detect_low_text_pdfs([jd_file] + list(uploaded_resumes))
    if low_text_pdf_files:
        _append_message(
            report,
            "warnings",
            "Low-text PDF(s) (possible OCR risk): "
            + ", ".join(f"`{name}`" for name in low_text_pdf_files)
            + ". Use searchable PDFs or DOCX for better extraction.",
        )

    _append_message(report, "infos", "Preflight complete. Run will proceed with current files.")
    report["has_blockers"] = bool(report["blocking"])
    return report


def render_preflight_report(report: dict[str, object]) -> None:
    with st.sidebar:
        st.markdown("**Preflight checks**")
        for item in report.get("blocking", []):
            st.error(str(item))
        for item in report.get("warnings", []):
            st.warning(str(item))
        for item in report.get("infos", []):
            st.caption(str(item))


def can_retry_failed(jd_file, uploaded_resumes, output: dict | None) -> bool:
    if not jd_file or not uploaded_resumes:
        return False
    output = output or {}
    return bool(output.get("rejected_files"))


def select_failed_resumes_for_retry(uploaded_resumes, output: dict | None) -> list[object]:
    output = output or {}
    failed_ids = {str(item[0]) for item in output.get("rejected_files", []) if item and item[0]}
    if not failed_ids:
        return []
    return [
        file for file in uploaded_resumes
        if candidate_id_from_upload_name(str(file.name)) in failed_ids
    ]


def _detect_duplicate_resumes(uploaded_resumes) -> list[str]:
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for file in uploaded_resumes:
        digest = hashlib.sha256(file.getbuffer()).hexdigest()
        if digest in seen:
            duplicates.append(file.name)
        else:
            seen[digest] = file.name
    return duplicates


def _detect_low_text_pdfs(files: list[object]) -> list[str]:
    low_text: list[str] = []
    threshold = max(120, int(CONFIG.thresholds.low_text_length_confidence_cutoff))
    for file in files:
        if Path(str(file.name)).suffix.lower() != ".pdf":
            continue
        try:
            reader = PdfReader(io.BytesIO(bytes(file.getbuffer())))
            extracted = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
            if len(extracted) < threshold:
                low_text.append(file.name)
        except Exception:  # noqa: BLE001
            low_text.append(file.name)
    return low_text


def _append_message(report: dict[str, object], key: str, message: str) -> None:
    bucket = report.get(key)
    if isinstance(bucket, list):
        bucket.append(message)
