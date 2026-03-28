from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from openai import OpenAI

from src.config.settings import CONFIG
from src.core.retry import with_backoff
from src.extract.candidate_schema import CandidateProfile
from src.scoring.scorer import CandidateScoreCard


@dataclass
class InterviewQuestion:
    question: str
    criterion: str
    evidence_reference: str
    strong_answer_signals: list[str] = field(default_factory=list)
    weak_answer_signals: list[str] = field(default_factory=list)


@dataclass
class InterviewKit:
    candidate_id: str
    candidate_name: str | None
    questions: list[InterviewQuestion]
    warnings: list[str]


def generate_interview_kit(
    candidate_id: str,
    candidate: CandidateProfile,
    scorecard: CandidateScoreCard,
    prompt_template: str,
    model: str | None = None,
) -> InterviewKit:
    chosen_model = model or CONFIG.models.high_stakes_llm
    client = OpenAI()
    criteria = [item.criterion for item in scorecard.criterion_scores[:8]]
    prompt = _render_interview_prompt(prompt_template, candidate, scorecard, criteria)

    def _call() -> list[InterviewQuestion]:
        response = client.responses.create(model=chosen_model, input=prompt, temperature=0.2)
        payload = json.loads(_extract_json_block(response.output_text))
        return [
            InterviewQuestion(
                question=item["question"],
                criterion=item["criterion"],
                evidence_reference=item.get("evidence_reference", "unknown"),
                strong_answer_signals=item.get("strong_answer_signals", []),
                weak_answer_signals=item.get("weak_answer_signals", []),
            )
            for item in payload
        ]

    try:
        questions = with_backoff(_call, max_attempts=CONFIG.generation.max_question_regen_attempts)
        return InterviewKit(candidate_id=candidate_id, candidate_name=candidate.name, questions=questions, warnings=[])
    except Exception as exc:
        fallback = [
            InterviewQuestion(
                question=f"Walk me through your experience with {c}.",
                criterion=c,
                evidence_reference="heuristic fallback",
                strong_answer_signals=["concrete examples", "measurable outcomes"],
                weak_answer_signals=["vague or no examples"],
            )
            for c in criteria[:5]
        ]
        return InterviewKit(
            candidate_id=candidate_id,
            candidate_name=candidate.name,
            questions=fallback,
            warnings=[f"LLM generation failed; fallback questions used: {exc}"],
        )


def _extract_json_block(text: str) -> str:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON array in response.")
    return match.group(0)


def _render_interview_prompt(
    prompt_template: str,
    candidate: CandidateProfile,
    scorecard: CandidateScoreCard,
    criteria: list[str],
) -> str:
    variables = {
        "candidate_name": candidate.name or "unknown",
        "skills": ", ".join(candidate.skills),
        "concerns": ", ".join(scorecard.concerns),
        "criteria": ", ".join(criteria),
    }
    rendered = prompt_template
    for key, value in variables.items():
        rendered = rendered.replace("{{" + key + "}}", value).replace("{" + key + "}", value)
    return rendered
