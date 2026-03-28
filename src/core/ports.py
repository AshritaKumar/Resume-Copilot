from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from src.extract.candidate_schema import CandidateProfile
from src.scoring.rubric_builder import Rubric
from src.scoring.scorer import CandidateScoreCard


class RepositoryPort(Protocol):
    def init_schema(self) -> None: ...
    def create_run(self, run_id: str, metadata: dict) -> None: ...
    def finish_run(self, run_id: str, status: str = "completed") -> None: ...
    def save_candidate(self, run_id: str, candidate_id: str, payload: dict) -> None: ...
    def save_scorecard(self, run_id: str, candidate_id: str, payload: dict) -> None: ...
    def save_outreach(self, run_id: str, candidate_id: str, subject: str, body: str, grounded: bool, warnings: list[str]) -> None: ...
    def save_interview_kit(self, run_id: str, candidate_id: str, payload: dict, warnings: list[str]) -> None: ...
    def save_artifact(self, run_id: str, artifact_name: str, artifact_path: str) -> None: ...
    def delete_run_data(self, run_id: str) -> None: ...
    def delete_all_data(self) -> None: ...


class VectorStorePort(Protocol):
    def init_schema(self) -> None: ...
    def upsert_chunks(
        self,
        job_run_id: str,
        candidate_id: str,
        section: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None: ...
    def search_knn(
        self,
        job_run_id: str,
        query_embedding: list[float],
        candidate_id: str | None = None,
        k: int = 10,
    ) -> list[Any]: ...
    def delete_run_data(self, job_run_id: str) -> None: ...
    def delete_all_data(self) -> None: ...


class PromptProviderPort(Protocol):
    def get_prompt(
        self,
        prompt_name: str,
        variables: dict[str, Any] | None = None,
    ) -> tuple[str, str]: ...


class ScoringServicePort(Protocol):
    def score(
        self,
        job_run_id: str,
        candidates: dict[str, CandidateProfile],
        rubric: Rubric,
        jd_text: str | None = None,
        judge_system_prompt: str = "",
    ) -> tuple[list[CandidateScoreCard], list[dict[str, str]]]: ...
