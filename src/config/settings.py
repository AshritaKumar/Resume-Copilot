from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    primary_llm: str = os.getenv("PRIMARY_LLM_MODEL", "gpt-4.1-mini")
    high_stakes_llm: str = os.getenv("HIGH_STAKES_LLM_MODEL", "gpt-4.1")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


@dataclass(frozen=True)
class DatabaseConfig:
    dsn: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/confido_hiring")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))


@dataclass(frozen=True)
class ThresholdConfig:
    min_pdf_confidence_for_no_ocr: float = float(os.getenv("MIN_PDF_CONFIDENCE_FOR_NO_OCR", "0.80"))
    low_text_length_confidence_cutoff: int = int(os.getenv("LOW_TEXT_LENGTH_CONFIDENCE_CUTOFF", "300"))


@dataclass(frozen=True)
class PromptConfig:
    langfuse_host: str = os.getenv("LANGFUSE_HOST", os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"))
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_label: str = os.getenv("LANGFUSE_LABEL", "production")
    candidate_extraction_prompt_name: str = os.getenv("CANDIDATE_EXTRACTION_PROMPT_NAME")
    jd_parse_prompt_name: str = os.getenv("JD_PARSE_PROMPT_NAME")
    outreach_prompt_name: str = os.getenv("OUTREACH_PROMPT_NAME")
    interview_prompt_name: str = os.getenv("INTERVIEW_PROMPT_NAME")
    rag_answer_prompt_name: str = os.getenv("RAG_ANSWER_PROMPT_NAME")
    judge_system_prompt_name: str = os.getenv("JUDGE_SYSTEM_PROMPT_NAME")
    rag_query_rewrite_prompt_name: str = os.getenv("RAG_QUERY_REWRITE_PROMPT_NAME", "rag_query_rewrite")
    rag_answer_judge_prompt_name: str = os.getenv("RAG_ANSWER_JUDGE_PROMPT_NAME", "rag_answer_judge")
    # Optional: Langfuse overrides for text otherwise bundled in UI modules
    shortlist_explainability_prompt_name: str = os.getenv("SHORTLIST_EXPLAINABILITY_PROMPT_NAME", "")
    rag_answer_json_suffix_prompt_name: str = os.getenv("RAG_ANSWER_JSON_SUFFIX_PROMPT_NAME", "")
    rag_json_repair_prompt_name: str = os.getenv("RAG_JSON_REPAIR_PROMPT_NAME", "")


@dataclass(frozen=True)
class GenerationConfig:
    max_email_regen_attempts: int = int(os.getenv("MAX_EMAIL_REGEN_ATTEMPTS", "2"))
    max_question_regen_attempts: int = int(os.getenv("MAX_QUESTION_REGEN_ATTEMPTS", "2"))


@dataclass(frozen=True)
class HeuristicConfig:
    candidate_achievement_tokens: list[str] = field(
        default_factory=lambda: _parse_csv_env(
            "HEURISTIC_CANDIDATE_ACHIEVEMENT_TOKENS",
            ["improved", "reduced", "%", "increased", "built"],
        )
    )
    jd_must_tokens: list[str] = field(
        default_factory=lambda: _parse_csv_env("HEURISTIC_JD_MUST_TOKENS", ["must", "required", "minimum"])
    )
    jd_preferred_tokens: list[str] = field(
        default_factory=lambda: _parse_csv_env("HEURISTIC_JD_PREFERRED_TOKENS", ["preferred", "nice to have", "plus"])
    )
    default_jd_risk_flags: list[str] = field(
        default_factory=lambda: _parse_csv_env("DEFAULT_JD_RISK_FLAGS", ["Missing clarity on evaluation criteria"])
    )
    default_jd_communication_leadership: list[str] = field(
        default_factory=lambda: _parse_csv_env("DEFAULT_JD_COMMUNICATION_LEADERSHIP", ["Communication quality", "Cross-functional collaboration"])
    )


@dataclass(frozen=True)
class PipelineConfig:
    shortlist_size: int = int(os.getenv("SHORTLIST_SIZE", "10"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    # Resume ingest indexing, scoring concurrency, shortlist LLM PDF parallelism.
    max_parallel_workers: int = int(os.getenv("MAX_PARALLEL_WORKERS", "8"))
    enable_llm_score_audit: bool = os.getenv("ENABLE_LLM_SCORE_AUDIT", "true").lower() == "true"
    rag_max_distance: float = float(os.getenv("RAG_MAX_DISTANCE", "0.62"))
    rag_near_best_margin: float = float(os.getenv("RAG_NEAR_BEST_MARGIN", "0.18"))
    rag_evidence_char_limit: int = int(os.getenv("RAG_EVIDENCE_CHAR_LIMIT", "900"))
    rag_context_max_chunks: int = int(os.getenv("RAG_CONTEXT_MAX_CHUNKS", "12"))
    # Fetch more from pgvector before lexical rerank; effective hybrid without a second index.
    rag_retrieval_pool_multiplier: float = float(os.getenv("RAG_RETRIEVAL_POOL_MULTIPLIER", "2.5"))
    # 0 = vector order only; ~0.25–0.4 boosts acronym/tool matches (SAP, K8s) in reranked list.
    rag_lexical_weight: float = float(os.getenv("RAG_LEXICAL_WEIGHT", "0.32"))
    rag_enable_query_rewrite: bool = os.getenv("RAG_ENABLE_QUERY_REWRITE", "true").lower() == "true"
    rag_enable_answer_judge: bool = os.getenv("RAG_ENABLE_ANSWER_JUDGE", "true").lower() == "true"


@dataclass(frozen=True)
class AppConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    heuristics: HeuristicConfig = field(default_factory=HeuristicConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def _parse_csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return default
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


CONFIG = AppConfig()
