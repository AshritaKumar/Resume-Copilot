from __future__ import annotations

import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import CONFIG
from src.core.container import get_prompt_provider, get_repository, get_scoring_service, get_vector_store
from src.core.input_validation import validate_path_input_types
from src.core.logging import TraceEvent, build_logger, write_trace_event
from src.extract.candidate_extractor import extract_candidate_profile
from src.extract.jd_parser import parse_jd
from src.output.shortlist_writer import write_shortlist
from src.pipeline.shortlist_artifacts import materialize_shortlist_candidate_artifacts
from src.parsers.router import parse_resume
from src.parsers.text_loader import load_text_file
from src.retrieval.embedder import chunk_text, embed_texts
from src.scoring.rubric_builder import build_rubric

@dataclass
class PipelineResult:
    run_id: str
    output_dir: Path
    shortlisted_candidate_ids: list[str]


@dataclass
class ProcessedResume:
    candidate_id: str
    status: str
    parser_used: str | None
    parse_confidence: float | None
    warning: str | None
    profile: object | None
    extraction_warnings: list[str]
    parse_text: str


def finalize_resume_ingestion(
    run_id: str,
    processed: list[ProcessedResume],
    repository,
    vector_store,
) -> tuple[dict[str, object], dict[str, str]]:
    """Persist parse rows, then embed + index successful resumes in parallel."""
    candidates: dict[str, object] = {}
    parse_summary: dict[str, str] = {}
    for p in processed:
        parse_summary[p.candidate_id] = p.status
        repository.save_candidate(run_id, p.candidate_id, {
            "parse_status": p.status,
            "parser_used": p.parser_used,
            "parse_confidence": p.parse_confidence,
            "warnings": (p.extraction_warnings or []) + ([p.warning] if p.warning else []),
            "profile": p.profile.model_dump() if p.profile else {},
        })
        if p.status == "ok" and p.profile is not None:
            candidates[p.candidate_id] = p.profile

    ok = [p for p in processed if p.status == "ok" and p.profile is not None]
    workers = max(1, min(len(ok), CONFIG.pipeline.max_parallel_workers))

    def _index_one(resume: ProcessedResume) -> None:
        chunks = chunk_text(resume.parse_text)
        vector_store.upsert_chunks(run_id, resume.candidate_id, "resume", chunks, embed_texts(chunks))

    if ok:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_index_one, p) for p in ok]
            for fut in as_completed(futs):
                fut.result()
    return candidates, parse_summary


def run_pipeline(jd_path: Path, resumes_dir: Path, output_dir: Path) -> PipelineResult:
    validate_path_input_types(jd_path, resumes_dir)
    logger = build_logger()
    run_id = uuid.uuid4().hex[:12]
    run_output_dir = output_dir / f"run_{run_id}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_output_dir / "trace.jsonl"

    repository = get_repository()
    vector_store = get_vector_store()
    prompt_manager = get_prompt_provider()
    scoring_service = get_scoring_service()
    repository.init_schema()
    vector_store.init_schema()
    repository.create_run(run_id, metadata={
        "primary_model": CONFIG.models.primary_llm,
        "high_stakes_model": CONFIG.models.high_stakes_llm,
        "embedding_model": CONFIG.models.embedding_model,
    })
    write_trace_event(trace_path, TraceEvent(run_id=run_id, stage="start", message="Pipeline started"))

    try:
        # Load prompts
        prompt_defs = [
            ("candidate", CONFIG.prompts.candidate_extraction_prompt_name),
            ("jd",        CONFIG.prompts.jd_parse_prompt_name),
            ("outreach",  CONFIG.prompts.outreach_prompt_name),
            ("interview", CONFIG.prompts.interview_prompt_name),
        ]
        if CONFIG.pipeline.enable_llm_score_audit:
            prompt_defs.append(("judge", CONFIG.prompts.judge_system_prompt_name))
        prompts: dict[str, str] = {}
        sources: dict[str, str] = {}
        for key, name in prompt_defs:
            prompts[key], sources[key] = prompt_manager.get_prompt(name)
        write_trace_event(trace_path, TraceEvent(run_id=run_id, stage="prompts", message="Prompts loaded", metadata=sources))

        # Parse JD and build rubric
        jd_text = load_text_file(jd_path)
        jd_parse = parse_jd(jd_text, prompt_template=prompts["jd"])
        rubric = build_rubric(jd_parse.interpretation, Path("config/rubric_defaults.yaml"))
        write_trace_event(trace_path, TraceEvent(run_id=run_id, stage="jd_parse", message="JD parsed and rubric built"))

        # Ingest resumes
        resume_files = [f for f in sorted(resumes_dir.iterdir()) if f.is_file()]
        processed = asyncio.run(process_resumes_async(resume_files, prompts["candidate"]))
        candidates, parse_summary = finalize_resume_ingestion(run_id, processed, repository, vector_store)

        if not candidates:
            raise RuntimeError("No candidates were successfully parsed.")

        # Score
        scorecards, score_failures = scoring_service.score(
            run_id, candidates, rubric, jd_text=jd_text,
            judge_system_prompt=prompts.get("judge", ""),
        )
        if score_failures:
            fail_path = run_output_dir / "scoring_failures.json"
            fail_path.write_text(json.dumps({"failures": score_failures}, indent=2, ensure_ascii=True))
            write_trace_event(
                trace_path,
                TraceEvent(
                    run_id=run_id,
                    stage="scoring",
                    message=f"{len(score_failures)} candidate(s) failed scoring",
                    metadata={"failures": score_failures[:50]},
                ),
            )
        for sc in scorecards:
            repository.save_scorecard(run_id, sc.candidate_id, {
                "base_score": sc.base_score, "final_score": sc.final_score,
                "reasoning": sc.reasoning, "strengths": sc.strengths,
                "concerns": sc.concerns, "additional_skills": sc.additional_skills,
                "career_signals": sc.career_signals,
                "criterion_scores": [s.__dict__ for s in sc.criterion_scores],
            })

        # Write shortlist
        shortlist_json, shortlist_md = write_shortlist(run_output_dir, scorecards)
        repository.save_artifact(run_id, "shortlist_json", str(shortlist_json))
        repository.save_artifact(run_id, "shortlist_md", str(shortlist_md))

        # Generate outreach + interview kits for shortlisted candidates
        role_title = jd_parse.interpretation.role_title or "the role"
        shortlisted = scorecards[:CONFIG.pipeline.shortlist_size]
        outreach_dir = run_output_dir / "outreach"
        interview_dir = run_output_dir / "interview_kits"
        outreach_dir.mkdir(parents=True, exist_ok=True)
        interview_dir.mkdir(parents=True, exist_ok=True)

        art_workers = max(1, min(len(shortlisted), CONFIG.pipeline.max_parallel_workers))

        def _artifacts_one(sc):
            materialize_shortlist_candidate_artifacts(
                run_id=run_id,
                repository=repository,
                sc=sc,
                candidate=candidates[sc.candidate_id],
                role_title=role_title,
                outreach_prompt=prompts["outreach"],
                interview_prompt=prompts["interview"],
                outreach_dir=outreach_dir,
                interview_dir=interview_dir,
            )

        with ThreadPoolExecutor(max_workers=art_workers) as pool:
            futs = [pool.submit(_artifacts_one, sc) for sc in shortlisted]
            for fut in as_completed(futs):
                fut.result()

        # Persist parse summary
        parse_summary_path = run_output_dir / "parse_summary.json"
        parse_summary_path.write_text(json.dumps(parse_summary, indent=2))
        repository.save_artifact(run_id, "parse_summary", str(parse_summary_path))

        write_trace_event(trace_path, TraceEvent(run_id=run_id, stage="finish", message="Pipeline completed",
                                                  metadata={"shortlisted_ids": [s.candidate_id for s in shortlisted]}))
        repository.finish_run(run_id, status="completed")
        logger.info("Run complete: %s", run_output_dir)
        return PipelineResult(run_id=run_id, output_dir=run_output_dir,
                              shortlisted_candidate_ids=[s.candidate_id for s in shortlisted])

    except Exception as exc:
        write_trace_event(trace_path, TraceEvent(run_id=run_id, stage="error", message=str(exc),
                                                  metadata={"exc_type": type(exc).__name__}))
        repository.finish_run(run_id, status="failed")
        raise


async def process_resumes_async(files: list[Path], candidate_extraction_prompt: str) -> list[ProcessedResume]:
    sem = asyncio.Semaphore(max(1, min(CONFIG.pipeline.max_parallel_workers, len(files) or 1)))

    def _process(file: Path) -> ProcessedResume:
        cid = file.stem.lower().replace(" ", "_")
        parsed = parse_resume(file)
        if parsed.status != "ok":
            return ProcessedResume(cid, parsed.status, parsed.parser_used, parsed.confidence, parsed.warning, None, [], "")
        try:
            ext = extract_candidate_profile(parsed.text, prompt_template=candidate_extraction_prompt)
        except Exception as exc:
            return ProcessedResume(cid, "not_a_resume", parsed.parser_used, parsed.confidence, str(exc), None, [], "")
        return ProcessedResume(cid, "ok", parsed.parser_used, parsed.confidence, parsed.warning, ext.profile, ext.warnings, parsed.text)

    async def _run(file: Path) -> ProcessedResume:
        async with sem:
            return await asyncio.to_thread(_process, file)

    return await asyncio.gather(*[_run(f) for f in files])
