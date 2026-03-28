from __future__ import annotations

import json
import re
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def _xml_esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _para(text: str, style: ParagraphStyle) -> Paragraph:
    safe = _xml_esc(text).replace("\n", "<br/>")
    return Paragraph(safe, style)


def write_outreach_pdf(pdf_path: Path, subject: str, body: str) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=inch * 0.85,
        leftMargin=inch * 0.85,
        topMargin=inch * 0.85,
        bottomMargin=inch * 0.85,
    )
    story: list = [
        Paragraph("<b>Outreach</b>", styles["Heading1"]),
        Spacer(1, 0.15 * inch),
        Paragraph(f"<b>Subject:</b> {_xml_esc(subject)}", styles["Normal"]),
        Spacer(1, 0.2 * inch),
    ]
    for block in re.split(r"\n\s*\n", body.strip() or "(no body)"):
        b = block.strip()
        if b:
            story.append(_para(b, styles["BodyText"]))
            story.append(Spacer(1, 0.12 * inch))
    doc.build(story)


def write_interview_kit_pdf(
    pdf_path: Path,
    candidate_id: str,
    candidate_name: str | None,
    payload: dict,
) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=inch * 0.85,
        leftMargin=inch * 0.85,
        topMargin=inch * 0.85,
        bottomMargin=inch * 0.85,
    )
    title_name = (candidate_name or candidate_id).strip()
    story: list = [
        Paragraph("<b>Interview kit</b>", styles["Heading1"]),
        Spacer(1, 0.1 * inch),
        Paragraph(f"<b>Candidate:</b> {_xml_esc(title_name)}", styles["Normal"]),
        Spacer(1, 0.2 * inch),
    ]
    questions = payload.get("questions") or []
    warnings = payload.get("warnings") or []
    if warnings:
        story.append(Paragraph("<b>Warnings</b>", styles["Heading2"]))
        for w in warnings:
            story.append(_para(str(w), styles["BodyText"]))
            story.append(Spacer(1, 0.08 * inch))
        story.append(Spacer(1, 0.15 * inch))
    for idx, q in enumerate(questions, start=1):
        if not isinstance(q, dict):
            continue
        qtext = str(q.get("question", "")).strip()
        criterion = str(q.get("criterion", "")).strip()
        story.append(Paragraph(f"<b>Question {idx}</b>", styles["Heading2"]))
        if qtext:
            story.append(_para(qtext, styles["BodyText"]))
        if criterion:
            story.append(
                Paragraph(f"<i>Focus area:</i> {_xml_esc(criterion)}", styles["Normal"])
            )
        strong = q.get("strong_answer_signals") or []
        weak = q.get("weak_answer_signals") or []
        if strong:
            story.append(Paragraph("<b>Strong signals</b>", styles["Heading3"]))
            for s in strong:
                story.append(_para(f"• {str(s)}", styles["BodyText"]))
        if weak:
            story.append(Paragraph("<b>Weak signals</b>", styles["Heading3"]))
            for s in weak:
                story.append(_para(f"• {str(s)}", styles["BodyText"]))
        story.append(Spacer(1, 0.2 * inch))
    if not questions:
        story.append(_para("No interview questions in kit.", styles["BodyText"]))
    doc.build(story)


def _parse_outreach_md(text: str) -> tuple[str, str]:
    lines = text.strip().split("\n", 1)
    subject = ""
    body = text.strip()
    if lines and lines[0].strip().lower().startswith("# subject:"):
        subject = lines[0].split(":", 1)[1].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
    return subject or "Outreach", body


def ensure_outreach_pdf(outreach_dir: Path, candidate_id: str) -> Path | None:
    pdf_path = outreach_dir / f"{candidate_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    md_path = outreach_dir / f"{candidate_id}.md"
    if not md_path.exists():
        return None
    subject, body = _parse_outreach_md(md_path.read_text(encoding="utf-8"))
    write_outreach_pdf(pdf_path, subject, body)
    return pdf_path


def ensure_interview_kit_pdf(
    interview_dir: Path,
    candidate_id: str,
    candidate_name: str | None = None,
) -> Path | None:
    pdf_path = interview_dir / f"{candidate_id}.pdf"
    if pdf_path.exists():
        return pdf_path
    json_path = interview_dir / f"{candidate_id}.json"
    if not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    write_interview_kit_pdf(pdf_path, candidate_id, candidate_name, payload)
    return pdf_path
