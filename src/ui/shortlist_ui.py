from __future__ import annotations

import html
import json
import re
from pathlib import Path

import streamlit as st
from openai import OpenAI

from src.config.settings import CONFIG
from src.output.pdf_export import ensure_interview_kit_pdf, ensure_outreach_pdf
from src.ui.json_utils import first_json_object


def render_run_summary(output: dict | None) -> None:
    if not output:
        st.info("Run the agentic pipeline from the sidebar to unlock chat.")
        return
    st.success(f"Run complete: `{output['run_id']}`")
    st.write(f"Output directory: `{output['output_dir']}`")
    _render_pipeline_stage_status(output)
    rejected = output.get("rejected_files", [])
    rows = output.get("shortlist_rows", [])
    failures = list(output.get("scoring_failures") or [])
    _render_summary_kpis(rows, rejected)
    if rejected:
        for name, reason in rejected:
            st.warning(f"**{name}** was skipped: {reason}")
    if failures:
        st.error(
            f"**{len(failures)} candidate(s) could not be scored** — they are missing from the ranked list. "
            "See below or `scoring_failures.json` in the run folder."
        )
        for item in failures[:25]:
            cid = html.escape(str(item.get("candidate_id", "?")))
            err = html.escape(str(item.get("error", ""))[:480])
            st.markdown(f"- **{cid}** — `{err}`")
    if rows or rejected or failures:
        st.caption(
            f"Ranked rows: {len(rows)} | Rejected files: {len(rejected)} | Scoring failures: {len(failures)}"
        )
    if rows:
        st.write("**Ranked candidates**")
        for rank, row in enumerate(rows, start=1):
            _render_candidate_card(rank, row)


def _render_summary_kpis(rows: list[dict], rejected: list[tuple[str, str]]) -> None:
    shortlisted = len(rows)
    avg_score = sum(float(row.get("total_score", 0.0) or 0.0) for row in rows) / shortlisted if shortlisted else 0.0
    cols = st.columns(3)
    metrics = [
        ("Scored candidates", str(shortlisted)),
        ("Average Score", f"{avg_score:.1f}"),
        ("Rejected Files", str(len(rejected))),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-label'>{label}</div>"
                f"<div class='kpi-value'>{value}</div></div>",
                unsafe_allow_html=True,
            )


def _render_pipeline_stage_status(output: dict) -> None:
    output_dir = Path(str(output.get("output_dir", "")))
    parse_done = (output_dir / "parse_summary.json").exists()
    shortlist_done = (output_dir / "shortlist.json").exists()
    outreach_done = (output_dir / "outreach").exists()
    interview_done = (output_dir / "interview_kits").exists()
    completed = sum([parse_done, shortlist_done, outreach_done and interview_done, bool(output.get("run_id"))])
    st.progress(min(completed / 4, 1.0), text="Pipeline stages completed")
    stage_labels = [
        ("Parsing", parse_done),
        ("Extraction/Scoring", shortlist_done),
        ("Artifacts", outreach_done and interview_done),
        ("Finalize", bool(output.get("run_id"))),
    ]
    st.caption(" | ".join(f"{name}: {'done' if done else 'pending'}" for name, done in stage_labels))


def _render_candidate_card(rank: int, row: dict) -> None:
    name = row.get("name") or row.get("candidate_id", "Unknown")
    score = row.get("total_score", 0.0)
    criterion_scores = row.get("criterion_scores", [])
    strengths = _expand_points(
        base_items=list(row.get("strengths", [])),
        criterion_scores=criterion_scores,
        mode="strength",
        max_points=5,
    )
    concerns = _expand_points(
        base_items=list(row.get("concerns", [])),
        criterion_scores=criterion_scores,
        mode="concern",
        max_points=5,
    )
    skills = row.get("additional_skills", [])
    reasoning = row.get("reasoning", "")
    review = _get_review_state(str(row.get("candidate_id", "")))
    effective_concerns = review["override_concerns"] or concerns
    candidate_id = str(row.get("candidate_id", "")).strip()
    output_dir_raw = (st.session_state.get("full_pipeline_output") or {}).get("output_dir")
    output_dir = Path(str(output_dir_raw)) if output_dir_raw else None

    with st.expander(f"#{rank}  {name}  —  Score: {score:.1f}", expanded=(rank == 1)):
        st.caption(
            f"AI recommendation: {'Strong fit' if score >= 30 else 'Moderate fit'} | "
            f"Human decision: {review['decision']}"
        )
        st.progress(min(max(score, 0.0) / 40.0, 1.0), text="Score strength")

        st.markdown("**Strengths**")
        _render_numbered_chips(strengths, "No strengths identified.")
        st.markdown("**Concerns**")
        if effective_concerns:
            for idx, concern in enumerate(effective_concerns[:5], start=1):
                _render_chip(title=f"{idx}. {concern}")
        else:
            st.caption("No concerns flagged.")

        signals = row.get("career_signals") or []
        if signals:
            st.markdown("**Résumé & career signals**")
            st.caption(
                "Heuristic checks (gaps, thin history, non-standard titles)—not a verdict. Confirm on the source file."
            )
            for sig in signals[:8]:
                st.markdown(f"- {html.escape(str(sig))}")

        st.markdown("**Skills**")
        if skills:
            st.markdown(", ".join(skills))
        else:
            st.caption("No additional skills listed.")

        if reasoning:
            st.markdown("**Summary**")
            st.caption(reasoning)

        st.divider()
        st.markdown("**Explainability**")
        _render_explainability_panel(row=row, criterion_scores=criterion_scores)

        st.divider()
        _render_pdf_download_row(output_dir=output_dir, candidate_id=candidate_id, display_name=name)

        st.divider()
        _render_human_review_controls(candidate_id, concerns)


def _render_numbered_chips(items: list[str], empty_caption: str, *, max_items: int = 5) -> None:
    if not items:
        st.caption(empty_caption)
        return
    for idx, item in enumerate(items[:max_items], start=1):
        _render_chip(title=f"{idx}. {item}")


def _reference_phrases_for_overlap(row: dict, criterion_scores: list[dict]) -> list[str]:
    """Phrases explainability must not echo (strengths, concerns, expanded rubric chips)."""
    refs: list[str] = []
    refs.extend(str(s).strip() for s in (row.get("strengths") or []) if str(s).strip())
    refs.extend(str(s).strip() for s in (row.get("concerns") or []) if str(s).strip())
    refs.extend(
        _expand_points(list(row.get("strengths") or []), criterion_scores, "strength", 5),
    )
    refs.extend(
        _expand_points(list(row.get("concerns") or []), criterion_scores, "concern", 5),
    )
    out: list[str] = []
    seen: set[str] = set()
    for r in refs:
        k = r.lower()
        if k not in seen and len(k) > 4:
            seen.add(k)
            out.append(r)
    return out


def _line_overlaps_reference(line: str, refs: list[str]) -> bool:
    """Flag near-duplicate of scorecard bullets—avoid nuking synthesis that merely mentions the same skill."""
    ln = line.lower().strip()
    if len(ln) < 16:
        return False
    for ref in refs:
        r = ref.lower().strip()
        if len(r) < 22:
            continue
        if r in ln:
            return True
        prefix = r[:32]
        if len(prefix) >= 22 and prefix in ln:
            return True
    return False


def _synthetic_why_lines(row: dict, criterion_scores: list[dict]) -> list[str]:
    """When the LLM lines were all near-duplicates of strengths, add rubric-combo lines."""
    ranked = sorted(
        criterion_scores,
        key=lambda item: float(item.get("weighted_score", 0.0) or 0.0),
        reverse=True,
    )
    if len(ranked) < 2:
        return []
    a, b = ranked[0], ranked[1]
    ca, cb = str(a.get("criterion", "")).strip(), str(b.get("criterion", "")).strip()
    if not ca or not cb:
        return []
    name = candidate_hint(row)
    return [
        f"For {name}, score is driven mainly by overlap on «{ca}» and «{cb}»—"
        f"see how those two themes interact in the excerpt-backed rubric, not isolated buzzwords."
    ]


def _strip_overlap_with_strengths_concerns(
    why_lines: list[str],
    missing_lines: list[str],
    *,
    row: dict,
    criterion_scores: list[dict],
) -> tuple[list[str], list[str]]:
    refs = _reference_phrases_for_overlap(row, criterion_scores)
    why_f = [ln for ln in why_lines if ln.strip() and not _line_overlaps_reference(ln, refs)]
    miss_refs = refs + [ln.lower() for ln in why_f]
    miss_f = [ln for ln in missing_lines if ln.strip() and not _line_overlaps_reference(ln, miss_refs)]
    return why_f[:4], miss_f[:4]


def _sanitize_resume_snippet(text: str) -> str:
    sanitized = text
    sanitized = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", sanitized)
    sanitized = re.sub(r"https?://\S+|www\.\S+", " ", sanitized)
    sanitized = re.sub(r"\+?\d[\d\-\s\(\)]{7,}\d", " ", sanitized)
    sanitized = re.sub(r"(?<!\w)@[A-Za-z0-9_\.]{2,}", " ", sanitized)
    sanitized = re.sub(r"[|]{1,}", " ", sanitized)
    sanitized = re.sub(r"\[[^\]]*\]", " ", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if len(re.findall(r"[A-Za-z]{3,}", sanitized)) < 6:
        return ""
    return sanitized


def _render_explainability_panel(row: dict, criterion_scores: list[dict]) -> None:
    if not criterion_scores:
        return
    judged = _get_or_generate_judged_explainability(row=row, criterion_scores=criterion_scores)
    if judged:
        why_raw = judged.get("why", [])
        missing_raw = judged.get("missing", [])
        keys = ("claim", "why", "statement", "line")
        why_lines = _normalize_explainability_lines(why_raw, claim_keys=keys)
        missing_lines = _normalize_explainability_lines(missing_raw, claim_keys=keys)
        why_lines, missing_lines = _strip_overlap_with_strengths_concerns(
            why_lines, missing_lines, row=row, criterion_scores=criterion_scores
        )
        if not why_lines and criterion_scores:
            why_lines = _synthetic_why_lines(row, criterion_scores)
        if why_lines or missing_lines:
            if why_lines:
                st.caption("Why this candidate")
                for idx, line in enumerate(why_lines[:4], start=1):
                    _render_chip(title=f"{idx}. {line}")
            if missing_lines:
                st.caption("What’s missing or unclear")
                for idx, line in enumerate(missing_lines[:4], start=1):
                    _render_chip(title=f"{idx}. {line}")
            return

    _render_explainability_panel_fallback(row, criterion_scores)


def _get_or_generate_judged_explainability(row: dict, criterion_scores: list[dict]) -> dict | None:
    candidate_id = str(row.get("candidate_id", "")).strip()
    run_id = str(st.session_state.get("active_run_id", "")).strip()
    if not candidate_id:
        return None
    cache = st.session_state.setdefault("explainability_judge_cache", {})
    cache_key = f"{run_id}:{candidate_id}:explain_synth_v1"
    if cache_key in cache:
        return cache[cache_key]

    judged = _judge_explainability_with_llm(row=row, criterion_scores=criterion_scores)
    if judged:
        cache[cache_key] = judged
    return judged


def _judge_explainability_with_llm(row: dict, criterion_scores: list[dict]) -> dict | None:
    candidate_name = str(row.get("name", "") or row.get("candidate_id", "Candidate")).strip()
    candidate_key = str(row.get("candidate_id", "")).strip()
    condensed = []
    ranked = sorted(
        criterion_scores,
        key=lambda item: float(item.get("weighted_score", 0.0) or 0.0),
        reverse=True,
    )[:12]
    for score in ranked:
        criterion = str(score.get("criterion", "")).strip()
        category = str(score.get("category", "")).strip()
        match_value = float(score.get("match", 0.0) or 0.0)
        weighted = float(score.get("weighted_score", 0.0) or 0.0)
        evidence = _sanitize_resume_snippet(str(score.get("evidence", "")).strip())
        if len(evidence) > 220:
            evidence = evidence[:220] + "..."
        condensed.append(
            {
                "criterion": criterion,
                "category": category,
                "match": round(match_value, 3),
                "weighted_score": round(weighted, 3),
                "evidence": evidence,
            }
        )

    strengths_ui = [str(s).strip() for s in (row.get("strengths") or []) if str(s).strip()][:8]
    concerns_ui = [str(s).strip() for s in (row.get("concerns") or []) if str(s).strip()][:8]
    skills = [str(s).strip() for s in (row.get("additional_skills") or []) if str(s).strip()][:12]
    summary = str(row.get("reasoning", "") or "").strip()[:400]

    prompt = f"""
You write the EXPLAINABILITY panel for ONE candidate in an hiring app. Profile key: {candidate_key}
Name: {candidate_name} | Overall score (automated): {float(row.get("total_score", 0.0) or 0.0):.1f}

The UI already shows **Strengths** and **Concerns** bullets. Your output must NOT repeat or lightly rephrase them.
Forbidden: copying the same opening as generic templates used for every applicant ("Strong Python skills", "Good fit", "Has experience with").

Rubric rows (only factual source; evidence is fragmentary):
{json.dumps(condensed, ensure_ascii=True)}

Scorecard strengths (DO NOT restate — subtract these themes from your wording):
{json.dumps(strengths_ui, ensure_ascii=True)}
Scorecard concerns (DO NOT restate — subtract these themes from your wording):
{json.dumps(concerns_ui, ensure_ascii=True)}
Skills tags: {json.dumps(skills, ensure_ascii=True)}
One-line model summary (context only; do not quote): {json.dumps(summary, ensure_ascii=True)}

Return STRICT JSON:
{{
  "why": [ "string", ... ],
  "missing": [ "string", ... ]
}}

**why** (max 4 strings, each 18–240 chars):
- Each line must SYNTHESIZE or give IMPLICATION: how two rubric signals interact, what profile shape this creates for THIS role, tradeoff you might accept, or non-obvious upside.
- Each line MUST include at least one **specific** hook from the evidence text above (proper noun, tool, domain, project type, or exact rubric criterion phrase) so wording differs between candidates.
- Vary sentence openings across the 4 lines (do not start three lines with the same first word).
- Never apologize or mention "limited evidence" or "confidence".

**missing** (max 4 strings, each 18–240 chars):
- Each line = what is still UNKNOWN, UNVERIFIED, or needs a hiring-manager decision — framed as investigation ("Still need to confirm…", "Unclear whether…", "Depth of … not established…").
- Must NOT duplicate concern bullets; if the same theme appears, change angle (e.g. production vs academic depth, ownership vs participation, timeline, team size).
- Each line should tie to a concrete rubric criterion name OR a concrete gap in evidence for this person—not generic "soft skills" unless in rubric.
- Vary openings; include at least one specific hook from evidence or criterion names.

If rubric evidence is too thin for a sincere line, omit that slot (return fewer than 4 items) rather than inventing.
"""
    try:
        response = OpenAI().responses.create(
            model=CONFIG.models.high_stakes_llm,
            input=prompt,
            temperature=0.15,
        )
        text = response.output_text.strip()
        payload = first_json_object(text)
        if not payload:
            return None
        why_items = payload.get("why", [])
        missing_items = payload.get("missing", [])
        if not isinstance(why_items, list) or not isinstance(missing_items, list):
            return None
        return {
            "why": _normalize_explainability_lines(why_items, claim_keys=("claim", "why", "statement", "line")),
            "missing": _normalize_explainability_lines(missing_items, claim_keys=("gap", "missing", "statement", "line")),
        }
    except Exception:  # noqa: BLE001
        return None


def _normalize_explainability_lines(raw: list, claim_keys: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for item in raw[:8]:
        if isinstance(item, str):
            line = item.strip()
            if line:
                out.append(line[:260])
            continue
        if isinstance(item, dict):
            for key in claim_keys:
                val = item.get(key)
                if val is not None and str(val).strip():
                    out.append(str(val).strip()[:260])
                    break
            else:
                text = str(item.get("text", "")).strip()
                if text:
                    out.append(text[:260])
    deduped: list[str] = []
    seen: set[str] = set()
    for line in out:
        key = line.lower()
        if len(key) < 14:
            continue
        if key not in seen:
            seen.add(key)
            deduped.append(line)
            if len(deduped) >= 4:
                break
    return deduped


def _render_explainability_panel_fallback(row: dict, criterion_scores: list[dict]) -> None:
    """Non-LLM path: combine rubric rows so lines are not identical to strength/concern chips."""
    st.caption("Why this candidate")
    ranked = sorted(
        criterion_scores,
        key=lambda item: float(item.get("weighted_score", 0.0) or 0.0),
        reverse=True,
    )
    top = [r for r in ranked if float(r.get("match", 0.0) or 0.0) >= 0.42][:2]
    if len(top) < 2 and ranked:
        top = ranked[:2]
    lines_why: list[str] = []
    if len(top) >= 2:
        a, b = top[0], top[1]
        ca, cb = str(a.get("criterion", "")).strip(), str(b.get("criterion", "")).strip()
        ea, eb = _sanitize_resume_snippet(str(a.get("evidence", ""))[:90]), _sanitize_resume_snippet(str(b.get("evidence", ""))[:90])
        lines_why.append(
            f"Combined rubric lift: «{ca}» and «{cb}» jointly drive the score"
            + (f" (excerpts mention: {ea}; {eb})" if ea or eb else ".")
        )
    elif top:
        r0 = top[0]
        lines_why.append(
            f"Primary score driver is «{str(r0.get('criterion', '')).strip()}» (match {float(r0.get('match', 0) or 0):.2f})."
        )
    for r in ranked[:3]:
        if len(lines_why) >= 4:
            break
        cat = str(r.get("category", "")).replace("_", " ")
        c = str(r.get("criterion", "")).strip()
        m = float(r.get("match", 0.0) or 0.0)
        if not c:
            continue
        tip = f"For {candidate_hint(row)}: {cat or 'criterion'} «{c}» contributes with match {m:.2f}."
        if tip.lower() not in {x.lower() for x in lines_why}:
            lines_why.append(tip)
    lines_why = [ln for ln in lines_why if not _line_overlaps_reference(ln, _reference_phrases_for_overlap(row, criterion_scores))][:4]
    if lines_why:
        for idx, line in enumerate(lines_why, start=1):
            _render_chip(title=f"{idx}. {line}")
    else:
        st.caption("Run with OpenAI available for richer synthesis, or expand rubric evidence.")

    st.caption("What’s missing or unclear")
    weak = sorted(
        criterion_scores,
        key=lambda item: (float(item.get("match", 0.0) or 0.0), -float(item.get("weight", 0.0) or 0.0)),
    )[:4]
    miss_frames = (
        "Interview should pin down real-world depth on «{c}» (match {m:.2f})—portfolio vs production, team size, and who owned delivery.",
        "Automated match {m:.2f} on «{c}»—still need timeline, scale, and whether this was coursework, internship, or full-time scope.",
        "«{c}» is weak in excerpts (match {m:.2f}); clarify stack, monitoring, deploy path, and on-call if you care about this for the role.",
        "For «{c}», match {m:.2f}: verify seniority signal—lead vs contributor—because vector snippets rarely spell that out.",
    )
    miss_lines: list[str] = []
    for i, r in enumerate(weak):
        c = str(r.get("criterion", "")).strip()
        if not c:
            continue
        m = float(r.get("match", 0.0) or 0.0)
        miss_lines.append(miss_frames[i % len(miss_frames)].format(c=c, m=m))
    miss_lines = [ln for ln in miss_lines if not _line_overlaps_reference(ln, _reference_phrases_for_overlap(row, criterion_scores))][:4]
    if miss_lines:
        for idx, line in enumerate(miss_lines, start=1):
            _render_chip(title=f"{idx}. {line}")
    else:
        st.caption("No major gaps flagged in rubric.")


def candidate_hint(row: dict) -> str:
    return str(row.get("name") or row.get("candidate_id") or "this candidate").strip()


def _render_pdf_download_row(output_dir: Path | None, candidate_id: str, display_name: str) -> None:
    st.markdown("**Outreach & interview PDFs**")
    if not output_dir or not candidate_id:
        st.caption("PDFs appear here after a run that wrote outreach and interview artifacts.")
        return
    outreach_dir = output_dir / "outreach"
    interview_dir = output_dir / "interview_kits"
    outreach_pdf = ensure_outreach_pdf(outreach_dir, candidate_id)
    interview_pdf = ensure_interview_kit_pdf(interview_dir, candidate_id, candidate_name=display_name)
    c1, c2 = st.columns(2)
    with c1:
        if outreach_pdf and outreach_pdf.exists():
            st.download_button(
                label="Download outreach PDF",
                data=outreach_pdf.read_bytes(),
                file_name=f"{candidate_id}_outreach.pdf",
                mime="application/pdf",
                key=f"pdf_outreach_{candidate_id}",
                use_container_width=True,
            )
        else:
            st.caption("Outreach PDF not available.")
    with c2:
        if interview_pdf and interview_pdf.exists():
            st.download_button(
                label="Download interview PDF",
                data=interview_pdf.read_bytes(),
                file_name=f"{candidate_id}_interview.pdf",
                mime="application/pdf",
                key=f"pdf_interview_{candidate_id}",
                use_container_width=True,
            )
        else:
            st.caption("Interview PDF not available.")


def _render_human_review_controls(candidate_id: str, ai_concerns: list[str]) -> None:
    if not candidate_id:
        return
    with st.expander("Reviewer controls", expanded=False):
        existing = _get_review_state(candidate_id)
        decision = st.radio(
            "Decision",
            options=["pending", "approved", "rejected"],
            index=["pending", "approved", "rejected"].index(existing["decision"]),
            horizontal=True,
            key=f"decision_{candidate_id}",
        )
        note = st.text_area(
            "Reviewer note",
            value=existing["note"],
            key=f"note_{candidate_id}",
            placeholder="Why you approve/reject this recommendation.",
        )
        override_text = st.text_area(
            "Override concerns (one per line, optional)",
            value="\n".join(existing["override_concerns"]),
            key=f"override_{candidate_id}",
            placeholder="Leave blank to keep AI concerns, or enter your own concern lines.",
        )
        override_concerns = [line.strip() for line in override_text.splitlines() if line.strip()]
        st.session_state.review_decisions[candidate_id] = {
            "decision": decision,
            "note": note.strip(),
            "override_concerns": override_concerns,
        }
        _persist_reviewer_decisions()


def _get_review_state(candidate_id: str) -> dict[str, object]:
    stored = st.session_state.review_decisions.get(candidate_id, {})
    return {
        "decision": str(stored.get("decision", "pending")),
        "note": str(stored.get("note", "")),
        "override_concerns": list(stored.get("override_concerns", [])),
    }


def _persist_reviewer_decisions() -> None:
    output = st.session_state.full_pipeline_output or {}
    output_dir = output.get("output_dir")
    if not output_dir:
        return
    path = Path(str(output_dir)) / "review_decisions.json"
    try:
        path.write_text(json.dumps(st.session_state.review_decisions, indent=2, ensure_ascii=True))
    except Exception:  # noqa: BLE001
        pass


def _render_chip(title: str, subtitle: str = "") -> None:
    title_esc = html.escape(title)
    subtitle_esc = html.escape(subtitle) if subtitle else ""
    subtitle_html = f"<div class='chip-subtle'>{subtitle_esc}</div>" if subtitle else ""
    st.markdown(
        f"<div class='chip'><div class='chip-title'>{title_esc}</div>{subtitle_html}</div>",
        unsafe_allow_html=True,
    )


def _expand_points(
    base_items: list[str],
    criterion_scores: list[dict],
    mode: str,
    max_points: int,
) -> list[str]:
    cleaned = [str(item).strip() for item in base_items if str(item).strip()]
    if len(cleaned) >= max_points:
        return cleaned[:max_points]

    if mode == "strength":
        candidates = sorted(
            [score for score in criterion_scores if float(score.get("match", 0.0) or 0.0) >= 0.55],
            key=lambda item: float(item.get("weighted_score", 0.0) or 0.0),
            reverse=True,
        )
        for score in candidates:
            criterion = str(score.get("criterion", "")).strip()
            if not criterion:
                continue
            suggestion = f"Evidence of {criterion.lower()}"
            if suggestion.lower() not in {item.lower() for item in cleaned}:
                cleaned.append(suggestion)
            if len(cleaned) >= max_points:
                break
    else:
        candidates = sorted(
            [score for score in criterion_scores if float(score.get("match", 0.0) or 0.0) < 0.45],
            key=lambda item: float(item.get("weighted_score", 0.0) or 0.0),
            reverse=True,
        )
        for score in candidates:
            criterion = str(score.get("criterion", "")).strip()
            if not criterion:
                continue
            suggestion = f"Limited evidence for {criterion.lower()}"
            if suggestion.lower() not in {item.lower() for item in cleaned}:
                cleaned.append(suggestion)
            if len(cleaned) >= max_points:
                break

    return cleaned[:max_points]
