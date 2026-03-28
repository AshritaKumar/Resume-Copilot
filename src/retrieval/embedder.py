from __future__ import annotations

import hashlib
import os

from openai import OpenAI

from src.config.settings import CONFIG
from src.core.logging import build_logger


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    logger = build_logger("embedder")
    if not text.strip():
        return []
    effective_chunk_size = chunk_size or CONFIG.chunking.chunk_size
    effective_overlap = overlap or CONFIG.chunking.chunk_overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + effective_chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - effective_overlap)
    logger.info("Chunked text into %s chunks (size=%s overlap=%s)", len(chunks), effective_chunk_size, effective_overlap)
    return chunks


def embed_texts(
    texts: list[str],
    model: str | None = None,
    allow_fallback: bool = True,
) -> list[list[float]]:
    logger = build_logger("embedder")
    if not texts:
        return []
    if not os.getenv("OPENAI_API_KEY"):
        if not allow_fallback:
            raise RuntimeError("OPENAI_API_KEY is required for strict embedding mode.")
        logger.warning("Using deterministic embedding fallback for %s texts", len(texts))
        return [_deterministic_embedding(text) for text in texts]
    client = OpenAI()
    chosen_model = model or CONFIG.models.embedding_model
    logger.info("Requesting embeddings for %s texts using model=%s", len(texts), chosen_model)
    response = client.embeddings.create(model=chosen_model, input=texts)
    logger.info("Received %s embeddings", len(response.data))
    return [item.embedding for item in response.data]


def _deterministic_embedding(text: str, dim: int | None = None) -> list[float]:
    dimension = dim or CONFIG.database.embedding_dimension
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for i in range(dimension):
        byte_value = digest[i % len(digest)]
        values.append((byte_value / 255.0) - 0.5)
    return values
