from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from openai import OpenAI

from src.config.settings import CONFIG
from src.core.errors import ExtractionError
from src.core.logging import build_logger
from src.core.retry import with_backoff
from src.extract.candidate_schema import CandidateProfile, Evidence
from src.extract.career_signals import enrich_career_signals


@dataclass
class ExtractionOutput:
    profile: CandidateProfile
    warnings: list[str]


def extract_candidate_profile(
    resume_text: str,
    prompt_template: str,
    model: str | None = None,
) -> ExtractionOutput:
    logger = build_logger("candidate_extractor")
    warnings: list[str] = []

    if not resume_text.strip():
        raise ExtractionError("Resume text is empty.")


    resume_text = resume_text[:12000]
    fallback_profile = _heuristic_extract(resume_text)

    if not _can_call_openai():
        warnings.append("OPENAI_API_KEY not set; using heuristic extraction only.")
        enrich_career_signals(fallback_profile, resume_text)
        return ExtractionOutput(profile=fallback_profile, warnings=warnings)

    prompt_input = prompt_template.replace("{{resume_text}}", resume_text).replace("{resume_text}", resume_text)
    model = model or CONFIG.models.primary_llm

    def _call() -> CandidateProfile:
        return _call_structured_parse(OpenAI(), model, prompt_input)

    try:
        profile = with_backoff(_call, max_attempts=3)
    except Exception as exc:
        warnings.append(f"LLM extraction failed; fallback used. Error: {exc}")
        logger.warning("Candidate extraction failed: %s", exc)
        enrich_career_signals(fallback_profile, resume_text)
        return ExtractionOutput(profile=fallback_profile, warnings=warnings)

    if not profile.is_resume:
        raise ExtractionError("Document does not appear to be a resume (e.g. research paper, report).")

    profile = _post_process(profile, fallback_profile, warnings, resume_text)
    logger.info("Candidate extraction: name=%s skills=%d", profile.name, len(profile.skills))
    return ExtractionOutput(profile=profile, warnings=warnings)


def _post_process(
    profile: CandidateProfile,
    fallback: CandidateProfile,
    warnings: list[str],
    resume_text: str,
) -> CandidateProfile:
    profile.skills = _clean_list(profile.skills) or fallback.skills
    profile.tools = _clean_list(profile.tools) or fallback.tools
    if not profile.skills:
        warnings.append("Empty skills from LLM, fallback used")
    profile.email = profile.email or fallback.email
    profile.phone = profile.phone or fallback.phone
    enrich_career_signals(profile, resume_text)
    return profile


def _clean_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    seen: dict[str, None] = {}
    for v in values:
        key = v.strip().lower()
        if key:
            seen[key] = None
    return list(seen)


def _can_call_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _call_structured_parse(client: OpenAI, model: str, prompt_input: str) -> CandidateProfile:
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt_input}],
            response_format=CandidateProfile,
            temperature=0,
        )
        parsed = completion.choices[0].message.parsed
        if isinstance(parsed, CandidateProfile):
            return parsed
        return CandidateProfile.model_validate(parsed)
    except Exception:
        response = client.responses.create(model=model, input=prompt_input, temperature=0)
        payload = json.loads(_extract_json_block(response.output_text.strip()))
        return CandidateProfile.model_validate(payload)


def _extract_json_block(text: str) -> str:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ExtractionError("No JSON object found in extraction response.")
    return match.group(0)


def _heuristic_extract(resume_text: str) -> CandidateProfile:
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", resume_text)
    phone_match = re.search(r"(\+?\d[\d\-\(\)\s]{8,}\d)", resume_text)
    skills = _extract_skills_line(resume_text)
    achievements = [
        Evidence(quote=line, source_hint="heuristic_line", confidence=0.6)
        for line in lines
        if any(token in line.lower() for token in CONFIG.heuristics.candidate_achievement_tokens)
    ][:3]
    confidence = min(1.0, 0.4 + (0.2 if skills else 0) + (0.2 if email_match else 0) + (0.2 if achievements else 0))
    return CandidateProfile(
        name=lines[0] if lines else None,
        email=email_match.group(0).lower() if email_match else None,
        phone=re.sub(r"\s+", "", phone_match.group(0)) if phone_match else None,
        skills=_clean_list(skills),
        tools=_clean_list(skills),
        achievements=achievements,
        extraction_confidence=round(confidence, 2),
    )


def _extract_skills_line(resume_text: str) -> list[str]:
    lines = [
        line for line in resume_text.splitlines()
        if any(k in line.lower() for k in ["skill", "tech", "stack", "tools"])
    ]
    tokens = ",".join(lines).replace("|", ",").split(",")
    return [t.strip() for t in tokens if t.strip()][:20]
