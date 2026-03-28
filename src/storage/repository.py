from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import psycopg

from src.config.settings import CONFIG
from src.core.logging import build_logger


@dataclass
class Repository:
    dsn: str = CONFIG.database.dsn

    def __post_init__(self) -> None:
        self.logger = build_logger("repository")

    def _execute(self, query: str, params: tuple = ()) -> None:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(query, params)
            conn.commit()

    def init_schema(self) -> None:
        self.logger.info("Initializing relational schema")

        queries = [
            """
            CREATE TABLE IF NOT EXISTS job_runs (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL,
                status TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY,
                job_run_id TEXT NOT NULL,
                parse_status TEXT NOT NULL,
                parser_used TEXT,
                parse_confidence DOUBLE PRECISION,
                profile JSONB NOT NULL DEFAULT '{}',
                warnings JSONB NOT NULL DEFAULT '[]'
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS criterion_scores (
                id BIGSERIAL PRIMARY KEY,
                job_run_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                payload JSONB NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS outreach_drafts (
                id BIGSERIAL PRIMARY KEY,
                job_run_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                grounded BOOLEAN NOT NULL,
                warnings JSONB NOT NULL DEFAULT '[]'
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS interview_kits (
                id BIGSERIAL PRIMARY KEY,
                job_run_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                payload JSONB NOT NULL,
                warnings JSONB NOT NULL DEFAULT '[]'
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                id BIGSERIAL PRIMARY KEY,
                job_run_id TEXT NOT NULL,
                artifact_name TEXT NOT NULL,
                artifact_path TEXT NOT NULL
            );
            """,
        ]

        for q in queries:
            self._execute(q)

        self.logger.info("Schema initialized")

    def create_run(self, run_id: str, metadata: dict) -> None:
        self.logger.info("Creating run: %s", run_id)

        self._execute(
            "INSERT INTO job_runs (id, created_at, status, metadata) VALUES (%s, %s, %s, %s)",
            (run_id, datetime.now(timezone.utc), "running", json.dumps(metadata)),
        )

    def finish_run(self, run_id: str, status: str = "completed") -> None:
        self.logger.info("Finishing run: %s (%s)", run_id, status)

        self._execute(
            "UPDATE job_runs SET status = %s WHERE id = %s",
            (status, run_id),
        )

    def save_candidate(self, run_id: str, candidate_id: str, payload: dict) -> None:
        self.logger.info("Saving candidate: %s", candidate_id)

        self._execute(
            """
            INSERT INTO candidates (id, job_run_id, parse_status, parser_used, parse_confidence, profile, warnings)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                parse_status = EXCLUDED.parse_status,
                parser_used = EXCLUDED.parser_used,
                parse_confidence = EXCLUDED.parse_confidence,
                profile = EXCLUDED.profile,
                warnings = EXCLUDED.warnings
            """,
            (
                candidate_id,
                run_id,
                payload["parse_status"],
                payload.get("parser_used"),
                payload.get("parse_confidence"),
                json.dumps(payload.get("profile", {})),
                json.dumps(payload.get("warnings", [])),
            ),
        )

    def save_scorecard(self, run_id: str, candidate_id: str, payload: dict) -> None:
        self.logger.info("Saving scorecard: %s", candidate_id)

        self._execute(
            "INSERT INTO criterion_scores (job_run_id, candidate_id, payload) VALUES (%s, %s, %s)",
            (run_id, candidate_id, json.dumps(payload)),
        )

    def save_outreach(self, run_id: str, candidate_id: str, subject: str, body: str, grounded: bool, warnings: list[str]) -> None:
        self.logger.info("Saving outreach: %s", candidate_id)

        self._execute(
            """
            INSERT INTO outreach_drafts (job_run_id, candidate_id, subject, body, grounded, warnings)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (run_id, candidate_id, subject, body, grounded, json.dumps(warnings)),
        )

    def save_interview_kit(self, run_id: str, candidate_id: str, payload: dict, warnings: list[str]) -> None:
        self.logger.info("Saving interview kit: %s", candidate_id)

        self._execute(
            """
            INSERT INTO interview_kits (job_run_id, candidate_id, payload, warnings)
            VALUES (%s, %s, %s, %s)
            """,
            (run_id, candidate_id, json.dumps(payload), json.dumps(warnings)),
        )

    def save_artifact(self, run_id: str, artifact_name: str, artifact_path: str) -> None:
        self.logger.info("Saving artifact: %s", artifact_name)

        self._execute(
            "INSERT INTO artifacts (job_run_id, artifact_name, artifact_path) VALUES (%s, %s, %s)",
            (run_id, artifact_name, artifact_path),
        )

    def delete_run_data(self, run_id: str) -> None:
        self.logger.info("Deleting relational records for run: %s", run_id)
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM criterion_scores WHERE job_run_id = %s", (run_id,))
            cur.execute("DELETE FROM outreach_drafts WHERE job_run_id = %s", (run_id,))
            cur.execute("DELETE FROM interview_kits WHERE job_run_id = %s", (run_id,))
            cur.execute("DELETE FROM artifacts WHERE job_run_id = %s", (run_id,))
            cur.execute("DELETE FROM candidates WHERE job_run_id = %s", (run_id,))
            cur.execute("DELETE FROM job_runs WHERE id = %s", (run_id,))
            conn.commit()

    def delete_all_data(self) -> None:
        self.logger.info("Deleting all relational records")
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM criterion_scores")
            cur.execute("DELETE FROM outreach_drafts")
            cur.execute("DELETE FROM interview_kits")
            cur.execute("DELETE FROM artifacts")
            cur.execute("DELETE FROM candidates")
            cur.execute("DELETE FROM job_runs")
            conn.commit()