"""Per-candidate outreach + interview artifacts (parallel-friendly)."""

from __future__ import annotations

import json
from pathlib import Path

from src.extract.candidate_schema import CandidateProfile
from src.generation.interview_kit import generate_interview_kit
from src.generation.outreach import generate_outreach_email
from src.output.pdf_export import write_interview_kit_pdf, write_outreach_pdf
from src.scoring.scorer import CandidateScoreCard
from src.storage.repository import Repository


def materialize_shortlist_candidate_artifacts(
    *,
    run_id: str,
    repository: Repository,
    sc: CandidateScoreCard,
    candidate: CandidateProfile,
    role_title: str,
    outreach_prompt: str,
    interview_prompt: str,
    outreach_dir: Path,
    interview_dir: Path,
) -> None:
    outreach = generate_outreach_email(candidate, role_title, outreach_prompt)
    repository.save_outreach(run_id, sc.candidate_id, outreach.subject, outreach.body, outreach.grounded, outreach.warnings)
    (outreach_dir / f"{sc.candidate_id}.md").write_text(f"# Subject: {outreach.subject}\n\n{outreach.body}\n")
    write_outreach_pdf(outreach_dir / f"{sc.candidate_id}.pdf", outreach.subject, outreach.body)

    kit = generate_interview_kit(sc.candidate_id, candidate, sc, interview_prompt)
    questions = [q.__dict__ for q in kit.questions]
    repository.save_interview_kit(run_id, sc.candidate_id, {"questions": questions}, kit.warnings)
    (interview_dir / f"{sc.candidate_id}.json").write_text(
        json.dumps({"questions": questions, "warnings": kit.warnings}, indent=2)
    )
    write_interview_kit_pdf(
        interview_dir / f"{sc.candidate_id}.pdf",
        sc.candidate_id,
        candidate.name,
        {"questions": questions, "warnings": kit.warnings},
    )
