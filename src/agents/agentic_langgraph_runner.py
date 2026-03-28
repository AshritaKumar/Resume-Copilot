from __future__ import annotations

import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.config.settings import CONFIG
from src.pipeline.shortlist_artifacts import materialize_shortlist_candidate_artifacts
from src.core.container import get_prompt_provider, get_repository, get_scoring_service, get_vector_store
from src.core.errors import UserFacingPipelineMessage
from src.core.input_validation import validate_path_input_types
from src.core.logging import TraceEvent, build_logger, log_prompt_usage, write_trace_event
from src.extract.jd_parser import parse_jd
from src.output.shortlist_writer import write_shortlist
from src.parsers.text_loader import load_text_file
from src.pipeline.orchestrator import PipelineResult, finalize_resume_ingestion, process_resumes_async
from src.scoring.rubric_builder import Rubric, build_rubric
from src.scoring.scorer import CandidateScoreCard

_logger = build_logger("agentic_langgraph_runner")


class AgenticState(TypedDict, total=False):
    jd_path: Path
    resumes_dir: Path
    run_id: str
    run_output_dir: Path
    trace_path: Path
    prompts: dict[str, str]
    jd_text: str
    role_title: str
    rubric: Rubric
    candidates: dict[str, object]
    scorecards: list[CandidateScoreCard]
    shortlist_ids: list[str]


def run_agentic_pipeline_with_langgraph(jd_path: Path, resumes_dir: Path, output_dir: Path) -> PipelineResult:
    if not jd_path.exists():
        raise RuntimeError(f"JD path does not exist: {jd_path}")
    if not resumes_dir.exists() or not resumes_dir.is_dir():
        raise RuntimeError(f"Invalid resumes dir: {resumes_dir}")
    validate_path_input_types(jd_path, resumes_dir)

    run_id = uuid.uuid4().hex[:12]
    run_output_dir = output_dir / f"run_{run_id}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_output_dir / "trace.jsonl"

    repo, vs = get_repository(), get_vector_store()
    repo.init_schema()
    vs.init_schema()
    repo.create_run(run_id, metadata={
        "mode": "agentic_langgraph",
        "primary_model": CONFIG.models.primary_llm,
        "high_stakes_model": CONFIG.models.high_stakes_llm,
        "embedding_model": CONFIG.models.embedding_model,
    })
    write_trace_event(trace_path, TraceEvent(run_id=run_id, stage="setup", message="Setup complete"))

    nodes = [("parse_jd", _parse_jd_node), ("ingest_candidates", _ingest_candidates_node),
             ("score_candidates", _score_candidates_node), ("generate_artifacts", _generate_artifacts_node)]
    graph = StateGraph(AgenticState)
    for name, fn in nodes:
        graph.add_node(name, fn)
    node_names = [n for n, _ in nodes]
    for a, b in zip([START] + node_names, node_names + [END]):
        graph.add_edge(a, b)

    _logger.info("Running agentic LangGraph pipeline")
    try:
        final = graph.compile().invoke({
            "jd_path": jd_path, "resumes_dir": resumes_dir,
            "run_id": run_id, "run_output_dir": run_output_dir, "trace_path": trace_path,
            "candidates": {}, "scorecards": [], "shortlist_ids": [],
        })
        repo.finish_run(run_id, status="completed")
    except UserFacingPipelineMessage as exc:
        repo.finish_run(run_id, status="failed")
        write_trace_event(trace_path, TraceEvent(run_id=run_id, stage="ingest_skipped", message=str(exc)))
        raise
    except Exception as exc:
        repo.finish_run(run_id, status="failed")
        write_trace_event(trace_path, TraceEvent(run_id=run_id, stage="error", message=str(exc)))
        raise

    return PipelineResult(run_id=final["run_id"], output_dir=final["run_output_dir"],
                          shortlisted_candidate_ids=final["shortlist_ids"])
def _parse_jd_node(state: AgenticState) -> AgenticState:
    run_id, trace_path = state["run_id"], state["trace_path"]
    pm = get_prompt_provider()
    prompt_defs = [
        ("candidate", CONFIG.prompts.candidate_extraction_prompt_name),
        ("jd",        CONFIG.prompts.jd_parse_prompt_name),
        ("outreach",  CONFIG.prompts.outreach_prompt_name),
        ("interview",  CONFIG.prompts.interview_prompt_name),
    ]
    if CONFIG.pipeline.enable_llm_score_audit:
        prompt_defs.append(("judge", CONFIG.prompts.judge_system_prompt_name))

    prompts: dict[str, str] = {}

    def _fetch_one(item: tuple[str, str]) -> tuple[str, str, str, str]:
        key, name = item
        text, source = pm.get_prompt(name)
        return key, name, text, source

    workers = max(1, min(len(prompt_defs), CONFIG.pipeline.max_parallel_workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_fetch_one, prompt_defs))
    for key, name, text, source in results:
        prompts[key] = text
        log_prompt_usage(_logger, prompt_name=name, prompt_source=source, prompt_text=text,
                         run_id=run_id, trace_path=trace_path, metadata={"prompt_key": key})

    jd_text = load_text_file(state["jd_path"])
    jd_parse = parse_jd(jd_text, prompt_template=prompts["jd"])
    rubric = build_rubric(jd_parse.interpretation, Path("config/rubric_defaults.yaml"))
    write_trace_event(trace_path, TraceEvent(run_id=run_id, stage="parse_jd", message="JD parsed"))
    return {**state, "prompts": prompts, "jd_text": jd_text,
            "role_title": jd_parse.interpretation.role_title or "the role", "rubric": rubric}


def _ingest_candidates_node(state: AgenticState) -> AgenticState:
    run_id, trace_path = state["run_id"], state["trace_path"]
    repo, vs = get_repository(), get_vector_store()
    candidates: dict[str, object] = {}
    parse_summary: dict[str, str] = {}
    failure_lines: list[str] = []

    resume_files = [f for f in sorted(state["resumes_dir"].iterdir()) if f.is_file()]
    if not resume_files:
        raise UserFacingPipelineMessage("No resume files found. Add at least one PDF or DOCX.",
                                        output_dir=state["run_output_dir"])

    processed = asyncio.run(process_resumes_async(resume_files, state["prompts"]["candidate"]))
    candidates, parse_summary = finalize_resume_ingestion(run_id, processed, repo, vs)

    for p in processed:
        if p.status != "ok" or p.profile is None:
            failure_lines.append(f"  • {p.candidate_id}: {p.status} — {(p.warning or p.status)[:240]}")

    (state["run_output_dir"] / "parse_summary.json").write_text(json.dumps(parse_summary, indent=2))

    if not candidates:
        raise UserFacingPipelineMessage(
            "None of the uploaded files could be ingested as resumes.\n\n"
            + ("\n".join(failure_lines) or "  • (no per-file details)"),
            output_dir=state["run_output_dir"],
        )

    write_trace_event(trace_path, TraceEvent(run_id=run_id, stage="ingest", message="Ingest complete"))
    return {**state, "candidates": candidates}


def _score_candidates_node(state: AgenticState) -> AgenticState:
    run_id = state["run_id"]
    repo = get_repository()
    trace_path = state["trace_path"]
    scorecards, score_failures = get_scoring_service().score(
        run_id, state["candidates"], state["rubric"],
        jd_text=state.get("jd_text"), judge_system_prompt=state["prompts"].get("judge", ""),
    )
    out_dir = state["run_output_dir"]
    if score_failures:
        fail_path = out_dir / "scoring_failures.json"
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
        repo.save_scorecard(run_id, sc.candidate_id, {
            "base_score": sc.base_score, "final_score": sc.final_score,
            "strengths": sc.strengths, "concerns": sc.concerns,
            "reasoning": sc.reasoning,
            "additional_skills": sc.additional_skills,
            "career_signals": sc.career_signals,
            "criterion_scores": [s.__dict__ for s in sc.criterion_scores],
        })
    return {**state, "scorecards": scorecards}


def _generate_artifacts_node(state: AgenticState) -> AgenticState:
    run_id, trace_path = state["run_id"], state["trace_path"]
    repo = get_repository()
    shortlisted = state["scorecards"][:CONFIG.pipeline.shortlist_size]

    shortlist_json, shortlist_md = write_shortlist(state["run_output_dir"], state["scorecards"])
    repo.save_artifact(run_id, "shortlist_json", str(shortlist_json))
    repo.save_artifact(run_id, "shortlist_md", str(shortlist_md))

    outreach_dir = state["run_output_dir"] / "outreach"
    interview_dir = state["run_output_dir"] / "interview_kits"
    outreach_dir.mkdir(exist_ok=True)
    interview_dir.mkdir(exist_ok=True)

    role_title = state.get("role_title", "the role")
    art_workers = max(1, min(len(shortlisted), CONFIG.pipeline.max_parallel_workers))

    def _one(sc: CandidateScoreCard):
        materialize_shortlist_candidate_artifacts(
            run_id=run_id,
            repository=repo,
            sc=sc,
            candidate=state["candidates"][sc.candidate_id],
            role_title=role_title,
            outreach_prompt=state["prompts"]["outreach"],
            interview_prompt=state["prompts"]["interview"],
            outreach_dir=outreach_dir,
            interview_dir=interview_dir,
        )

    with ThreadPoolExecutor(max_workers=art_workers) as pool:
        futs = [pool.submit(_one, sc) for sc in shortlisted]
        for fut in as_completed(futs):
            fut.result()

    write_trace_event(trace_path, TraceEvent(run_id=run_id, stage="finalize", message="Done"))
    return {**state, "shortlist_ids": [s.candidate_id for s in shortlisted]}
