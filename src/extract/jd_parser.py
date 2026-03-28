from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel, Field

from src.config.settings import CONFIG
from src.core.logging import build_logger
from src.core.retry import with_backoff


class JDInterpretation(BaseModel):
    role_title: str | None = None
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    communication_leadership: list[str] = Field(default_factory=list)


@dataclass
class JDParseOutput:
    interpretation: JDInterpretation
    warnings: list[str]


def parse_jd(
    jd_text: str,
    prompt_template: str,
    model: str | None = None,
) -> JDParseOutput:
    logger = build_logger("jd_parser")
    warnings: list[str] = []
    fallback = _heuristic_jd_parse(jd_text)
    if not _can_call_openai():
        warnings.append("OPENAI_API_KEY not set; using heuristic JD parsing.")
        return JDParseOutput(fallback, warnings)

    chosen_model = model or CONFIG.models.primary_llm
    prompt_input = prompt_template.replace("{{jd_text}}", jd_text).replace("{jd_text}", jd_text)

    def _call() -> JDInterpretation:
        client = OpenAI()
        return _call_structured_parse(client, chosen_model, prompt_input)

    try:
        interpretation = with_backoff(_call, max_attempts=3)
        if not interpretation.must_have:
            warnings.append("LLM returned no must-have criteria — rubric scoring may be unreliable. Check JD clarity.")
            logger.warning("JD parse returned empty must_have list")
        logger.info("JD parse succeeded")
        return JDParseOutput(interpretation, warnings)
    except Exception as exc:
        warnings.append(f"LLM JD parse failed, fallback used: {exc}")
        logger.warning("JD parse failed, fallback used: %s", exc)
        return JDParseOutput(fallback, warnings)


def _heuristic_jd_parse(jd_text: str) -> JDInterpretation:
    lines = [line.strip() for line in jd_text.splitlines() if line.strip()]
    must_have: list[str] = []
    nice_to_have: list[str] = []
    for line in lines:
        lower = line.lower()
        if any(token in lower for token in CONFIG.heuristics.jd_must_tokens):
            must_have.append(line)
        elif any(token in lower for token in CONFIG.heuristics.jd_preferred_tokens):
            nice_to_have.append(line)
    return JDInterpretation(
        role_title=lines[0] if lines else None,
        must_have=must_have[:8],
        nice_to_have=nice_to_have[:8],
        risk_flags=CONFIG.heuristics.default_jd_risk_flags if not must_have else [],
        communication_leadership=CONFIG.heuristics.default_jd_communication_leadership,
    )


def _extract_json_block(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object in response.")
    return match.group(0)


def _call_structured_parse(client: OpenAI, model: str, prompt_input: str) -> JDInterpretation:
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt_input}],
            response_format=JDInterpretation,
            temperature=0,
        )
        parsed = completion.choices[0].message.parsed
        if isinstance(parsed, JDInterpretation):
            return parsed
        if parsed is not None:
            return JDInterpretation.model_validate(parsed)
    except Exception:
        pass

    response = client.responses.create(model=model, input=prompt_input, temperature=0)
    payload = json.loads(_extract_json_block(response.output_text))
    return JDInterpretation.model_validate(payload)


def _can_call_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))
