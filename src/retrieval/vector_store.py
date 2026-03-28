from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import psycopg

from src.config.settings import CONFIG
from src.core.errors import RetrievalError
from src.core.logging import build_logger


@dataclass
class RetrievedChunk:
    candidate_id: str
    section: str
    text: str
    distance: float
    metadata: dict
    chunk_id: int | None = None


class VectorStore:
    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or CONFIG.database.dsn
        self.logger = build_logger("vector_store")

    def init_schema(self) -> None:
        self.logger.info("Initializing pgvector schema")
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS resume_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    job_run_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    section TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}',
                    embedding vector({CONFIG.database.embedding_dimension}) NOT NULL
                );
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_resume_chunks_embedding_hnsw
                ON resume_chunks
                USING hnsw (embedding vector_cosine_ops);
                """
            )
            conn.commit()
        self.logger.info("pgvector schema initialization completed")

    def upsert_chunks(
        self,
        job_run_id: str,
        candidate_id: str,
        section: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise RetrievalError("Chunks and embeddings length mismatch.")
        self.logger.info(
            "Storing embeddings: run=%s candidate=%s section=%s chunks=%s",
            job_run_id,
            candidate_id,
            section,
            len(chunks),
        )
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True)):
                cur.execute(
                    """
                    INSERT INTO resume_chunks (job_run_id, candidate_id, section, chunk_text, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    """,
                    (
                        job_run_id,
                        candidate_id,
                        section,
                        chunk,
                        json.dumps({"chunk_index": idx}),
                        self._to_vector_literal(emb),
                    ),
                )
            conn.commit()
        self.logger.info(
            "Stored embeddings successfully: run=%s candidate=%s chunk_count=%s",
            job_run_id,
            candidate_id,
            len(chunks),
        )

    def search_knn(
        self,
        job_run_id: str,
        query_embedding: list[float],
        candidate_id: str | None = None,
        k: int = 10,
    ) -> list[RetrievedChunk]:
        self.logger.info(
            "KNN search: run=%s candidate_filter=%s k=%s",
            job_run_id,
            candidate_id if candidate_id else "none",
            k,
        )
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            if candidate_id:
                cur.execute(
                    """
                    SELECT id, candidate_id, section, chunk_text, metadata, embedding <=> %s::vector AS distance
                    FROM resume_chunks
                    WHERE job_run_id = %s AND candidate_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        self._to_vector_literal(query_embedding),
                        job_run_id,
                        candidate_id,
                        self._to_vector_literal(query_embedding),
                        k,
                    ),
                )
            else:
                cur.execute(
                    """
                    SELECT id, candidate_id, section, chunk_text, metadata, embedding <=> %s::vector AS distance
                    FROM resume_chunks
                    WHERE job_run_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (self._to_vector_literal(query_embedding), job_run_id, self._to_vector_literal(query_embedding), k),
                )
            rows = cur.fetchall()
        self.logger.info("KNN returned %s rows", len(rows))
        return [
            RetrievedChunk(
                chunk_id=int(row[0]),
                candidate_id=row[1],
                section=row[2],
                text=row[3],
                metadata=row[4] or {},
                distance=float(row[5]),
            )
            for row in rows
        ]

    def delete_run_data(self, job_run_id: str) -> None:
        self.logger.info("Deleting vector chunks for run=%s", job_run_id)
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM resume_chunks WHERE job_run_id = %s", (job_run_id,))
            conn.commit()

    def delete_all_data(self) -> None:
        self.logger.info("Deleting all vector chunks")
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM resume_chunks")
            conn.commit()

    @staticmethod
    def _to_vector_literal(embedding: list[float]) -> str:
        return "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"
