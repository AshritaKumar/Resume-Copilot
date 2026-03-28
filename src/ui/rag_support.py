"""RAG answer prompts, JSON repair, and markdown formatting (no Streamlit)."""

from __future__ import annotations

import re
from typing import Any

from openai import OpenAI

from src.config.settings import CONFIG
from src.retrieval.vector_store import RetrievedChunk
from src.ui.json_utils import first_json_object

_JSON_SUFFIX_DEFAULT = (
    "\n\n---\nOutput: STRICT JSON with keys answer, rationale, and optionally "
    "needs_human_review (boolean), human_review_summary (one short sentence for executives). "
    "Write answer and rationale for a CEO or hiring lead: plain language, no [1] or bracketed chunk references—"
    "use candidate names from the evidence. Prefer a short recommendation and tradeoffs over a long roster of "
    "everyone’s weaknesses; only use a name-by-name gap list if the user clearly asked for screening-out, "
    "comparison tables, or “who doesn’t fit.” Keep tone factual, not punitive. "
    "Set needs_human_review true when comparison, prioritization, or missing job-description detail requires a human decision."
)
_HUMAN_REVIEW_INTRO = (
    "*The system surfaced résumé excerpts; what follows needs a hiring decision—not more automation.* "
    "*Escalate final calls to your hiring manager or recruiter; confirm against the full CV and interviews.*"
)


def rag_pipeline_params() -> tuple[float, float, int, int]:
    pc = CONFIG.pipeline
    return (
        float(getattr(pc, "rag_max_distance", 0.62)),
        float(getattr(pc, "rag_near_best_margin", 0.18)),
        int(getattr(pc, "rag_evidence_char_limit", 900)),
        int(getattr(pc, "rag_context_max_chunks", 12)),
    )


def rag_answer_json_suffix(prompt_manager: object) -> str:
    name = (CONFIG.prompts.rag_answer_json_suffix_prompt_name or "").strip()
    if not name:
        return _JSON_SUFFIX_DEFAULT
    try:
        text, _ = prompt_manager.get_prompt(name)
        t = (text or "").strip()
        if t:
            return "\n\n" + t if not t.startswith("\n") else t
    except Exception:
        pass
    return _JSON_SUFFIX_DEFAULT


def _json_repair_fallback(broken: str) -> str:
    return (
        "Repair the following model output into STRICT JSON only.\n"
        "Required keys: answer (plain language for executives), rationale (short, why you believe the answer).\n"
        "Optional keys: needs_human_review (boolean), human_review_summary (one short sentence: what a human must decide).\n"
        "Answer and rationale must be prose only—no chunk numbers like [1] in the text; refer to candidates by name.\n"
        "Do not add markdown fences.\n\nOriginal output:\n" + broken
    )


def json_repair_prompt(prompt_manager: object, broken: str) -> str:
    name = (CONFIG.prompts.rag_json_repair_prompt_name or "").strip()
    if name:
        try:
            text, _ = prompt_manager.get_prompt(name, variables={"model_output": broken})
            if (text or "").strip():
                return text.strip()
        except Exception:
            pass
    return _json_repair_fallback(broken)


def call_llm_json_with_repair(
    prompt: str, model: str, *, prompt_manager: object
) -> tuple[dict[str, Any] | None, list[str]]:
    notes: list[str] = []
    client = OpenAI()
    raw = client.responses.create(model=model, input=prompt, temperature=0).output_text.strip()
    parsed0 = first_json_object(raw)
    if parsed0:
        return parsed0, notes
    notes.append("model_json_invalid")
    repair = json_repair_prompt(prompt_manager, raw)
    fixed = first_json_object(client.responses.create(model=model, input=repair, temperature=0).output_text.strip())
    if fixed:
        notes.append("model_json_repaired")
        return fixed, notes
    notes.append("model_json_repair_failed")
    return None, notes


def generate_from_evidence(
    query: str,
    results: list[RetrievedChunk],
    template: str,
    prompt_manager: object,
) -> tuple[str, str, bool, str]:
    _, _, snip, max_chunks = rag_pipeline_params()
    cap = min(max_chunks, len(results))
    top = results[:cap]
    block = "\n\n".join(
        f"[{i + 1}] candidate={c.candidate_id}, similarity={1 - c.distance:.3f} (higher = more relevant)\n{c.text[:snip]}"
        for i, c in enumerate(top)
    )
    full = (
        template.replace("{{query}}", query).replace("{query}", query)
        .replace("{{evidence_block}}", block).replace("{evidence_block}", block)
        + rag_answer_json_suffix(prompt_manager)
    )
    parsed, _ = call_llm_json_with_repair(full, CONFIG.models.primary_llm, prompt_manager=prompt_manager)
    if parsed:
        ans = str(parsed.get("answer", "")).strip()
        if ans:
            return (
                ans,
                str(parsed.get("rationale", "")).strip() or "Summary is based on the resume excerpts shown to the model.",
                bool(parsed.get("needs_human_review")),
                str(parsed.get("human_review_summary", "")).strip(),
            )
    t0 = top[0].text[:400].strip()
    return (
        f"The assistant could not format a full answer. The strongest retrieved resume excerpt starts: {t0}…",
        "Fallback to the top matching excerpt only; treat as preliminary.",
        True,
        "Have a recruiter validate against the full resume and JD.",
    )


def normalize_bullet_lines(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        lines.append("- " + s[1:].lstrip() if s.startswith("•") else line)
    return "\n\n".join(x for x in lines if x.strip())


def strip_bracket_citations(text: str) -> str:
    t = re.sub(r"\s*\[\d{1,2}\]\s*", " ", text)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def heuristic_human_review(answer: str, rationale: str) -> tuple[bool, str]:
    blob = f"{answer}\n{rationale}".lower()
    checks: list[tuple[bool, str]] = [
        (any(w in blob for w in ("uncertain", "cannot determine", "insufficient", "unclear if")), "Answer leans tentative or incomplete."),
        ("not provided" in blob or "not in the evidence" in blob or "lack of explicit" in blob, "Key facts (e.g. JD details) are missing from the excerpts."),
        (any(w in blob for w in ("compare", "versus", "vs ", "trade-off", "phone-screen", "who would you")), "Comparison or prioritization needs your judgment on role fit."),
        (any(w in blob for w in ("recommend", "prioritize", "if the role", "depends on")), "Outcome depends on how you weight the role."),
        (any(w in blob for w in ("verify", "confirm with", "should be validated")), "Suggests validating against sources or interviews."),
    ]
    reasons = [m for ok, m in checks if ok]
    if not reasons:
        return False, ""
    return True, " ".join(f"• {r}" for r in reasons[:4])


def format_executive_answer(
    answer: str,
    rationale: str,
    needs_human_review: bool,
    human_review_note: str,
    *,
    follow_up_questions: list[str] | None = None,
) -> str:
    body = strip_bracket_citations(normalize_bullet_lines(answer)).strip()
    why = strip_bracket_citations(rationale).strip()
    parts = ["#### Takeaway", "", body, "", "---", "", "#### Supporting detail", "", why]
    if needs_human_review:
        note = human_review_note or (
            "The documents leave room for interpretation; someone who knows the role should confirm before a final call."
        )
        parts += ["", "---", "", "#### Human review", "", _HUMAN_REVIEW_INTRO, "", note]
        qs = [q.strip() for q in (follow_up_questions or []) if str(q).strip()]
        if qs:
            parts += ["", "**Suggested questions**", ""] + [f"- {q}" for q in qs[:6]]
    return "\n".join(parts)


def format_simple_markdown(title: str, body: str) -> str:
    return f"#### {title}\n\n{body.strip()}"


def dedupe_preserve(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def compact_shortlist_hint(rows: list[dict]) -> str:
    bits: list[str] = []
    for row in rows[:4]:
        name = str(row.get("name") or row.get("candidate_id") or "").strip()
        skills = row.get("additional_skills") or []
        if isinstance(skills, list) and skills:
            sk = ", ".join(str(s) for s in skills[:5] if s)
            if name and sk:
                bits.append(f"{name}: {sk}")
    return "; ".join(bits)[:900]


def evidence_digest_for_judge(chunks: list[RetrievedChunk], *, snip: int) -> str:
    return "\n\n".join(
        f"[{i}] candidate={c.candidate_id} section={c.section}\n{c.text[:snip]}" for i, c in enumerate(chunks, 1)
    )
