from __future__ import annotations

from src.core.ports import PromptProviderPort, RepositoryPort, ScoringServicePort, VectorStorePort
from src.prompts.manager import PromptManager
from src.retrieval.vector_store import VectorStore
from src.scoring.service import ScoringService
from src.storage.repository import Repository

_repository: Repository | None = None
_vector_store: VectorStore | None = None
_prompt_provider: PromptManager | None = None
_scoring_service: ScoringService | None = None


def get_repository() -> RepositoryPort:
    global _repository
    if _repository is None:
        _repository = Repository()
    return _repository


def get_vector_store() -> VectorStorePort:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_prompt_provider() -> PromptProviderPort:
    global _prompt_provider
    if _prompt_provider is None:
        _prompt_provider = PromptManager()
    return _prompt_provider


def get_scoring_service() -> ScoringServicePort:
    global _scoring_service
    if _scoring_service is None:
        _scoring_service = ScoringService(vector_store=get_vector_store())
    return _scoring_service
