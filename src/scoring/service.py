from __future__ import annotations

from src.core.ports import ScoringServicePort, VectorStorePort
from src.extract.candidate_schema import CandidateProfile
from src.scoring.rubric_builder import Rubric
from src.scoring.scorer import CandidateScoreCard, score_candidates


class ScoringService(ScoringServicePort):
    def __init__(self, vector_store: VectorStorePort) -> None:
        self._vector_store = vector_store

    def score(
        self,
        job_run_id: str,
        candidates: dict[str, CandidateProfile],
        rubric: Rubric,
        jd_text: str | None = None,
        judge_system_prompt: str = "",
    ) -> tuple[list[CandidateScoreCard], list[dict[str, str]]]:
        return score_candidates(
            job_run_id=job_run_id,
            candidates=candidates,
            rubric=rubric,
            vector_store=self._vector_store,
            jd_text=jd_text,
            judge_system_prompt=judge_system_prompt,
        )
