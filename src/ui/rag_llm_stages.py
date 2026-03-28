"""Query rewrite + answer judge LLM calls. Prompt text is loaded from Langfuse in rag_chat."""

from __future__ import annotations

import re

from openai import OpenAI

from src.config.settings import CONFIG
from src.ui.json_utils import first_json_object

_VAGUE_HINT = re.compile(
    r"\b(best|better|who'?s?\s+the\s+best|strongest|top\s+(pick|choice|candidates?)|"
    r"who\s+should|recommend|compare|vs\.?|versus|fit|thoughts\??|your\s+pick)\b",
    re.IGNORECASE,
)


def is_vague_query(text: str) -> bool:
    t = text.strip()
    if len(t) < 12:
        return True
    if len(t.split()) <= 4 and "?" not in t:
        return True
    return bool(_VAGUE_HINT.search(t))


def rewrite_query_for_retrieval(prompt_text: str, model: str | None = None) -> str:
    """Langfuse prompt must require STRICT JSON: {\"retrieval_query\": \"...\"}."""
    chosen = model or CONFIG.models.primary_llm
    text = OpenAI().responses.create(model=chosen, input=prompt_text.strip(), temperature=0).output_text.strip()
    data = first_json_object(text)
    if not data:
        return ""
    return str(data.get("retrieval_query", "")).strip()


def judge_answer_and_suggestions(prompt_text: str, model: str | None = None) -> dict:
    """Langfuse prompt must require JSON: verdict, explanation, needs_human_review, follow_up_questions."""
    chosen = model or CONFIG.models.high_stakes_llm
    text = OpenAI().responses.create(model=chosen, input=prompt_text.strip(), temperature=0).output_text.strip()
    data = first_json_object(text)
    if not data:
        return {
            "verdict": "partial",
            "explanation": "Judge step did not return valid JSON; treat the answer as unverified.",
            "needs_human_review": True,
            "follow_up_questions": [
                "Which parts of the answer are you most unsure about relative to the full resumes?",
                "What single criterion should break a tie between final candidates?",
            ],
        }
    verdict = str(data.get("verdict", "partial")).lower().strip()
    if verdict not in {"supported", "partial", "unsupported"}:
        verdict = "partial"
    expl = str(data.get("explanation", "")).strip() or "No explanation provided."
    nhr = bool(data.get("needs_human_review", verdict != "supported"))
    raw_fq = data.get("follow_up_questions", [])
    follow: list[str] = []
    if isinstance(raw_fq, list):
        for item in raw_fq[:4]:
            s = str(item).strip()
            if s:
                follow.append(s)
    return {
        "verdict": verdict,
        "explanation": expl,
        "needs_human_review": nhr or verdict != "supported",
        "follow_up_questions": follow,
    }
