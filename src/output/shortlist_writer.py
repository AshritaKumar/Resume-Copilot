from __future__ import annotations

import json
from pathlib import Path

from src.config.settings import CONFIG
from src.scoring.scorer import CandidateScoreCard


def write_shortlist(
    output_dir: Path,
    scorecards: list[CandidateScoreCard],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = int(CONFIG.pipeline.shortlist_size)
    ranked = list(scorecards)[:cap]
    json_path = output_dir / "shortlist.json"
    md_path = output_dir / "shortlist.md"

    json_payload = [
        {
            "candidate_id": item.candidate_id,
            "name": item.name,
            "base_score": item.base_score,
            "final_score": item.final_score,
            "reasoning": item.reasoning,
            "strengths": item.strengths,
            "concerns": item.concerns,
            "additional_skills": item.additional_skills,
            "career_signals": item.career_signals,
            "criterion_scores": [
                {
                    "criterion": cs.criterion,
                    "category": cs.category,
                    "weight": cs.weight,
                    "match": cs.match,
                    "weighted_score": cs.weighted_score,
                    "evidence": cs.evidence,
                }
                for cs in item.criterion_scores
            ],
        }
        for item in ranked
    ]
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=True))

    lines = [f"# Shortlist (top {len(ranked)} by score)", ""]
    for idx, item in enumerate(ranked, start=1):
        lines.append(f"## {idx}. {item.name or item.candidate_id} — Final Score: {item.final_score}")
        lines.append(f"- Base score: {item.base_score}")
        lines.append(f"- Strengths: {', '.join(item.strengths) or 'none'}")
        lines.append(f"- Concerns: {', '.join(item.concerns) or 'none'}")
        if item.career_signals:
            lines.append(f"- Résumé / title signals: {'; '.join(item.career_signals)}")
        if item.reasoning:
            lines.append(f"- Reasoning: {item.reasoning}")
        lines.append("")

    md_path.write_text("\n".join(lines))
    return json_path, md_path