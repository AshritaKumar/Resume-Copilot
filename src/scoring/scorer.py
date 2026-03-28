from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field

from openai import OpenAI

from src.config.settings import CONFIG
from src.core.logging import build_logger
from src.extract.candidate_schema import CandidateProfile
from src.retrieval.embedder import embed_texts
from src.retrieval.vector_store import VectorStore
from src.scoring.rubric_builder import Rubric

_STOP_WORDS = {"and", "or", "the", "a", "an", "in", "of", "with", "for", "to", "years", "experience", "strong", "good"}


@dataclass
class CriterionScore:
    criterion: str
    category: str
    weight: float
    match: float
    weighted_score: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class CandidateScoreCard:
    candidate_id: str
    name: str | None
    base_score: float
    criterion_scores: list[CriterionScore] = field(default_factory=list)
    final_score: float = 0.0
    reasoning: str = ""
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    additional_skills: list[str] = field(default_factory=list)
    career_signals: list[str] = field(default_factory=list)


def score_candidates(
    job_run_id: str,
    candidates: dict[str, CandidateProfile],
    rubric: Rubric,
    vector_store: VectorStore,
    jd_text: str | None = None,
    judge_system_prompt: str = "",
) -> tuple[list[CandidateScoreCard], list[dict[str, str]]]:
    logger = build_logger("scorer")
    sem = asyncio.Semaphore(max(1, min(CONFIG.pipeline.max_parallel_workers, len(candidates) or 1)))

    async def _run_all():
        async def _one(cid, prof):
            async with sem:
                return await asyncio.to_thread(
                    _score_one, cid, prof, job_run_id, rubric, vector_store, jd_text, judge_system_prompt
                )

        results = await asyncio.gather(*[_one(cid, p) for cid, p in candidates.items()], return_exceptions=True)
        cards: list[CandidateScoreCard] = []
        failures: list[dict[str, str]] = []
        for (cid, _), res in zip(candidates.items(), results, strict=True):
            if isinstance(res, Exception):
                msg = f"{type(res).__name__}: {res}"
                logger.warning("Scoring failed for %s: %s", cid, msg)
                failures.append({"candidate_id": cid, "error": msg[:2000]})
            else:
                cards.append(res)
        return sorted(cards, key=lambda c: c.final_score, reverse=True), failures

    return asyncio.run(_run_all())


def _score_one(candidate_id, profile, job_run_id, rubric, vector_store, jd_text, judge_system_prompt):
    sc = rubric.metadata.get("scoring", {})
    cd = rubric.metadata.get("criterion_defaults", {})

    criterion_scores = _compute_base_scores(candidate_id, job_run_id, profile, rubric, vector_store, sc, cd)
    base_score = round(sum(cs.weighted_score for cs in criterion_scores), 2)

    strength_floor = float(sc.get("strength_match_floor", 0.20))
    top_ratio = float(sc.get("strength_match_top_ratio", 0.70))
    concern_thresh = float(sc.get("concern_threshold", 0.30))

    sorted_desc = sorted(criterion_scores, key=lambda x: x.match, reverse=True)
    match_floor = max(strength_floor, sorted_desc[0].match * top_ratio) if sorted_desc else strength_floor

    card = CandidateScoreCard(
        candidate_id=candidate_id,
        name=profile.name,
        base_score=base_score,
        criterion_scores=criterion_scores,
        final_score=base_score,
        strengths=[cs.criterion for cs in sorted_desc if cs.match >= match_floor][:3],
        concerns=[cs.criterion for cs in sorted_desc[::-1] if cs.category == "must_have" and cs.match < concern_thresh][:3],
        additional_skills=list(dict.fromkeys(profile.skills or []))[:6],
        career_signals=list(dict.fromkeys(profile.career_gaps_or_risks or []))[:8],
    )

    _run_llm_judge(card, profile, jd_text, criterion_scores, judge_system_prompt)
    return card


def _compute_base_scores(candidate_id, job_run_id, profile, rubric, vector_store, sc, cd):
    k = max(1, int(CONFIG.pipeline.retrieval_top_k))
    evidence_multiplier = float(cd.get("evidence_multiplier", 1.0))
    missing_evidence_score = float(cd.get("missing_evidence_score", 0.10))
    semantic_weight = float(sc.get("semantic_weight", 0.7))
    lexical_weight = float(sc.get("lexical_weight", 0.3))
    candidate_tokens = _candidate_tokens(profile)

    criterion_queries = [f"{c.name} {c.category}" for c in rubric.criteria]
    embeddings = embed_texts(criterion_queries) if criterion_queries else []

    results = []
    for criterion, embedding in zip(rubric.criteria, embeddings, strict=True):
        chunks = vector_store.search_knn(job_run_id, embedding, candidate_id=candidate_id, k=k)

        semantic = (
            min(1.0, sum(max(0.0, 1.0 - c.distance) for c in chunks) / len(chunks) * evidence_multiplier)
            if chunks else missing_evidence_score
        )
        lexical = _lexical_match(criterion.name, candidate_tokens)
        match = round(min(1.0, semantic_weight * semantic + lexical_weight * lexical), 3)

        results.append(CriterionScore(
            criterion=criterion.name,
            category=criterion.category,
            weight=criterion.weight,
            match=match,
            weighted_score=round(criterion.weight * match, 2),
            evidence=[c.text[:280] for c in chunks[:2] if c.text],
        ))
    return results


def _candidate_tokens(profile: CandidateProfile) -> set[str]:
    tokens: set[str] = set()
    for text in [*(profile.skills or []), *(profile.tools or [])]:
        tokens.update(re.sub(r"[^\w\s]", "", text.lower()).split())
    if profile.current_title:
        tokens.update(re.sub(r"[^\w\s]", "", profile.current_title.lower()).split())
    return tokens - _STOP_WORDS


def _lexical_match(criterion_name: str, candidate_tokens: set[str]) -> float:
    keywords = set(re.sub(r"[^\w\s]", "", criterion_name.lower()).split()) - _STOP_WORDS
    return len(keywords & candidate_tokens) / len(keywords) if keywords else 0.0


def _run_llm_judge(card, profile, jd_text, criterion_scores, judge_system_prompt):
    if not CONFIG.pipeline.enable_llm_score_audit:
        return
    if not jd_text or not judge_system_prompt or not os.getenv("OPENAI_API_KEY"):
        return

    criteria_lines = "\n".join(f"  - {cs.criterion} [{cs.category}]: match={cs.match:.2f}" for cs in criterion_scores)
    evidence_lines = [f"  [{cs.criterion}] {cs.evidence[0].strip()[:200]}"
                      for cs in criterion_scores if cs.evidence and cs.evidence[0].strip()]
    prompt_parts = [
        f"## Candidate\nName: {profile.name or 'Unknown'}\nTitle: {profile.current_title or 'Unknown'}\n"
        f"Experience: {profile.total_years_experience or '?'} years\n"
        f"Skills: {', '.join((profile.skills or [])[:20]) or 'None listed'}\n"
        f"Tools: {', '.join((profile.tools or [])[:20]) or 'None listed'}",
        f"## Job Description\n{jd_text.strip()[:1500]}",
        f"## Automated Base Score: {card.base_score:.1f} / 100",
        f"## Per-Criterion Scores\n{criteria_lines}",
    ]
    if evidence_lines:
        prompt_parts.append("## Top Resume Evidence Per Criterion\n" + "\n".join(evidence_lines))

    try:
        response = OpenAI().chat.completions.create(
            model=CONFIG.models.high_stakes_llm,
            messages=[{"role": "system", "content": judge_system_prompt},
                      {"role": "user", "content": "\n\n".join(prompt_parts)}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content or "{}")
        card.final_score = float(max(0, min(100, data.get("final_score", card.base_score))))
        card.reasoning = str(data.get("reasoning", "")).strip()
        if data.get("strengths"):
            card.strengths = [str(s).strip() for s in data["strengths"] if s][:3]
        if data.get("concerns"):
            card.concerns = [str(c).strip() for c in data["concerns"] if c][:3]
    except Exception as exc:
        build_logger("scorer").warning("LLM judge failed for %s: %s", card.candidate_id, exc)
