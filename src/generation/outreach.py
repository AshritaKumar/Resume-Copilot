from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from src.config.settings import CONFIG
from src.core.retry import with_backoff
from src.extract.candidate_schema import CandidateProfile


@dataclass
class OutreachDraft:
    subject: str
    body: str
    grounded: bool
    warnings: list[str]


def generate_outreach_email(
    candidate: CandidateProfile,
    role_title: str,
    prompt_template: str,
    model: str | None = None,
) -> OutreachDraft:
    chosen_model = model or CONFIG.models.primary_llm
    client = OpenAI()
    prompt = _render_outreach_prompt(prompt_template, candidate, role_title)
    try:
        response = with_backoff(
            lambda: client.responses.create(model=chosen_model, input=prompt, temperature=0.5),
            max_attempts=CONFIG.generation.max_email_regen_attempts,
        )
        subject, body = _split_subject_body(response.output_text.strip())
        return OutreachDraft(subject=subject, body=body, grounded=True, warnings=[])
    except Exception as exc:
        name = candidate.name or "Candidate"
        fallback_body = (
            f"Hi {name.split()[0]},\n\n"
            f"We're reaching out about a {role_title} opportunity that aligns with your background.\n"
            "Would you be open to a quick 20-minute call this week?\n\n"
            "Confido Hiring Team"
        )
        return OutreachDraft(
            subject=f"{role_title} opportunity — {name}",
            body=fallback_body,
            grounded=False,
            warnings=[f"LLM generation failed; fallback used: {exc}"],
        )


def _split_subject_body(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "Opportunity", ""
    if lines[0].lower().startswith("subject:"):
        return lines[0].split(":", 1)[1].strip(), "\n".join(lines[1:]).strip()
    return "Opportunity", "\n".join(lines)


def _render_outreach_prompt(
    prompt_template: str,
    candidate: CandidateProfile,
    role_title: str,
) -> str:
    best_achievement = (
        candidate.achievements[0].quote.strip()
        if candidate.achievements else ""
    )
    top_skills = candidate.skills[:5]
    skills_str = ", ".join(top_skills) if top_skills else "software development"
    experience_str = (
        f"{int(candidate.total_years_experience)} years" if candidate.total_years_experience else "several years"
    )

    variables = {
        "role_title": role_title,
        "candidate_name": (candidate.name or "there").split()[0],
        "current_title": candidate.current_title or "Software Engineer",
        "experience": experience_str,
        "top_skills": skills_str,
        "best_achievement": best_achievement,
        "sender_name": "Confido Hiring Team",
    }
    rendered = prompt_template
    for key, value in variables.items():
        rendered = rendered.replace("{{" + key + "}}", value).replace("{" + key + "}", value)
    return rendered
