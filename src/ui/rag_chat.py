"""RAG chat: retrieve chunks, Langfuse rag answer prompt, optional rewrite + judge."""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from src.config.settings import CONFIG
from src.core.logging import TraceEvent, build_logger, log_prompt_usage, write_trace_event
from src.retrieval.embedder import embed_texts
from src.retrieval.rag_retrieval import lexical_rerank_fuse, retrieve_results_for_scope
from src.ui.rag_llm_stages import is_vague_query, judge_answer_and_suggestions, rewrite_query_for_retrieval
from src.ui.rag_support import (
    compact_shortlist_hint,
    dedupe_preserve,
    evidence_digest_for_judge,
    format_executive_answer,
    format_simple_markdown,
    generate_from_evidence,
    heuristic_human_review,
    rag_pipeline_params,
)


def _rag_chat_blocked_reason() -> str | None:
    if st.session_state.get("pipeline_running"):
        return "Wait for the pipeline to finish before chatting."
    out = st.session_state.get("full_pipeline_output") or {}
    if not out:
        return "Upload a JD and resumes, then run the pipeline from the sidebar."
    rid = str(st.session_state.get("active_run_id") or out.get("run_id") or "").strip()
    if not rid:
        return "No active run id — run the pipeline again."
    return None


def render_chat(scope: str, vector_store, prompt_manager) -> None:
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(msg["content"])
            else:
                st.write(msg["content"])
    block = _rag_chat_blocked_reason()
    if block:
        st.caption(block)
    user_query = st.chat_input("Ask a question about candidates...", disabled=block is not None)
    if not user_query:
        return
    st.session_state.chat_messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
    with st.chat_message("assistant"):
        answer_text = _answer_question(user_query, scope, vector_store, prompt_manager)
        st.markdown(answer_text)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer_text})


def _answer_question(query: str, scope: str, vector_store, prompt_manager) -> str:
    logger = build_logger("streamlit_app")
    prereq = _rag_chat_blocked_reason()
    if prereq:
        return format_simple_markdown("Chat not ready", prereq)
    job_run_id = str(st.session_state.active_run_id or "").strip()
    if not job_run_id:
        return format_simple_markdown("Chat not ready", "No job run id — run the pipeline from the sidebar.")
    run_id = st.session_state.active_run_id or None
    trace_path = _active_trace_path()
    t0 = time.perf_counter()
    try:
        shortlist_rows = st.session_state.full_pipeline_output.get("shortlist_rows", [])
        pc = CONFIG.pipeline
        retrieval_query = query.strip()
        if bool(getattr(pc, "rag_enable_query_rewrite", True)) and is_vague_query(query):
            rw_name = CONFIG.prompts.rag_query_rewrite_prompt_name
            role_hint = compact_shortlist_hint(shortlist_rows)
            rw_prompt, rw_src = prompt_manager.get_prompt(
                rw_name,
                variables={"user_question": query.strip(), "role_hint": role_hint if role_hint.strip() else "(none)"},
            )
            log_prompt_usage(logger, prompt_name=rw_name, prompt_source=rw_src, prompt_text=rw_prompt, stage="rag_query_rewrite", run_id=run_id, trace_path=trace_path)
            if rewritten := rewrite_query_for_retrieval(rw_prompt):
                retrieval_query = rewritten

        query_embedding = embed_texts([retrieval_query], allow_fallback=False)[0]
        candidate_filters = _extract_candidate_filters(query, shortlist_rows)
        shortlisted_ids = _get_shortlisted_candidate_ids()
        if scope == "Ranked candidates only" and not shortlisted_ids:
            return format_simple_markdown(
                "No ranked candidates in scope",
                "The ranked list is empty. Choose **All indexed resumes** in the sidebar, or re-run the pipeline.",
            )
        max_dist, near_margin, ev_chars, max_chunks = rag_pipeline_params()
        pool_mult = float(getattr(pc, "rag_retrieval_pool_multiplier", 2.5))
        k_pool = max(5, int(pc.retrieval_top_k * pool_mult))
        results = retrieve_results_for_scope(
            vector_store=vector_store,
            job_run_id=job_run_id,
            query_embedding=query_embedding,
            k=k_pool,
            scope=scope,
            shortlisted_ids=shortlisted_ids,
            candidate_filters=candidate_filters,
            max_distance=max_dist,
            near_best_margin=near_margin,
        )
        lex_w = float(getattr(pc, "rag_lexical_weight", 0.32))
        results = lexical_rerank_fuse(results, retrieval_query, lex_w)
        if not results:
            _record_rag_turn(logger, run_id=run_id, trace_path=trace_path, query=query, scope=scope, retrieved=0, context_chunks=0, context_candidates=[], chunk_trace=[], duration_ms=_ms(t0), ok=True, detail="no_hits")
            return format_simple_markdown(
                "No matching resume text",
                "Nothing in the indexed resumes matched this question in the current scope. Try rephrasing or broadening the chat scope.",
            )

        answer_prompt, answer_prompt_source = prompt_manager.get_prompt(CONFIG.prompts.rag_answer_prompt_name)
        log_prompt_usage(logger, prompt_name=CONFIG.prompts.rag_answer_prompt_name, prompt_source=answer_prompt_source, prompt_text=answer_prompt, stage="rag_prompt", run_id=run_id, trace_path=trace_path)
        cap = min(max_chunks, len(results))
        context_candidates = sorted({c.candidate_id for c in results[:cap]})
        chunk_trace = [{"chunk_id": c.chunk_id, "candidate_id": c.candidate_id, "section": c.section, "distance": round(float(c.distance), 4)} for c in results[:cap]]
        answer_body, rationale, model_wants_review, model_review_note = generate_from_evidence(query, results, answer_prompt, prompt_manager)
        h_review, h_note = heuristic_human_review(answer_body, rationale)
        needs_review, review_note = model_wants_review or h_review, (model_review_note or h_note).strip()

        judge_verdict, follow_up_questions = None, []
        if bool(getattr(pc, "rag_enable_answer_judge", True)):
            digest = evidence_digest_for_judge(results[:cap], snip=ev_chars)[:12000]
            j_name = CONFIG.prompts.rag_answer_judge_prompt_name
            j_prompt, j_src = prompt_manager.get_prompt(
                j_name,
                variables={"user_question": query.strip(), "answer": answer_body, "rationale": rationale, "evidence_digest": digest},
            )
            log_prompt_usage(logger, prompt_name=j_name, prompt_source=j_src, prompt_text=j_prompt, stage="rag_answer_judge", run_id=run_id, trace_path=trace_path)
            judge = judge_answer_and_suggestions(j_prompt)
            judge_verdict = str(judge.get("verdict"))
            judge_explanation = str(judge.get("explanation", "")).strip()
            follow_up_questions = list(judge.get("follow_up_questions") or [])
            if judge.get("needs_human_review"):
                needs_review = True
            if judge_verdict in {"partial", "unsupported"}:
                needs_review = True
                if judge_explanation and judge_explanation not in review_note:
                    review_note = f"{review_note}\n\n**Automated check:** {judge_explanation}".strip() if review_note else f"**Automated check:** {judge_explanation}"

        _record_rag_turn(logger, run_id=run_id, trace_path=trace_path, query=query, scope=scope, retrieved=len(results), context_chunks=cap, context_candidates=context_candidates, chunk_trace=chunk_trace, duration_ms=_ms(t0), ok=True, detail="answer", judge_verdict=judge_verdict)
        return format_executive_answer(answer_body, rationale, needs_review, review_note, follow_up_questions=follow_up_questions)
    except Exception as exc:  # noqa: BLE001
        logger.exception("rag_chat_turn_failed | run_id=%s", run_id)
        _record_rag_turn(
            logger,
            run_id=run_id,
            trace_path=trace_path,
            query=query,
            scope=scope,
            retrieved=0,
            context_chunks=0,
            context_candidates=[],
            chunk_trace=[],
            duration_ms=_ms(t0),
            ok=False,
            detail="error",
            error=type(exc).__name__,
        )
        return format_simple_markdown("We couldn’t complete that answer", "Try again in a moment. If this repeats, check API keys and connectivity.")


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _active_trace_path() -> Path | None:
    output_dir = (st.session_state.full_pipeline_output or {}).get("output_dir")
    return Path(str(output_dir)) / "trace.jsonl" if output_dir else None


def _record_rag_turn(logger, **kw) -> None:
    preview = kw["query"].replace("\n", " ").strip()[:200]
    chunk_trace = kw["chunk_trace"]
    id_preview = ",".join(str(x["chunk_id"]) for x in chunk_trace[:10] if x.get("chunk_id") is not None)
    cc = kw["context_candidates"]
    logger.info(
        "rag_chat_turn | ok=%s detail=%s ms=%s retrieved=%s context_chunks=%s candidates=%s chunk_ids=%s judge=%s scope=%s preview=%r",
        kw["ok"], kw["detail"], kw["duration_ms"], kw["retrieved"], kw["context_chunks"], ",".join(cc) if cc else "-", id_preview or "-", kw.get("judge_verdict") or "-", kw["scope"], preview,
    )
    run_id, trace_path = kw.get("run_id"), kw.get("trace_path")
    if run_id and trace_path:
        meta = {
            "scope": kw["scope"],
            "retrieved_chunks": kw["retrieved"],
            "context_chunks": kw["context_chunks"],
            "context_candidate_ids": cc,
            "retrieval_chunk_trace": chunk_trace,
            "duration_ms": kw["duration_ms"],
            "ok": kw["ok"],
            "detail": kw["detail"],
            "query_preview": preview,
        }
        if kw.get("error"):
            meta["error"] = kw["error"]
        if j := kw.get("judge_verdict"):
            meta["judge_verdict"] = j
        write_trace_event(trace_path, TraceEvent(run_id=run_id, stage="rag_chat_turn", message="RAG chat turn", latency_ms=kw["duration_ms"], metadata=meta))


def _get_shortlisted_candidate_ids() -> list[str]:
    rows = st.session_state.full_pipeline_output.get("shortlist_rows", [])
    return dedupe_preserve([str(r.get("candidate_id", "")).strip() for r in rows if str(r.get("candidate_id", "")).strip()])


def _extract_candidate_filters(query: str, shortlist_rows: list[dict]) -> list[str]:
    q, matched = query.lower(), []
    for row in shortlist_rows:
        cid = str(row.get("candidate_id", "")).strip()
        if not cid:
            continue
        name = str(row.get("name", "")).strip()
        toks = [cid.lower(), name.lower()] if name else [cid.lower()]
        if any(n and n in q for n in toks):
            matched.append(cid)
    return dedupe_preserve(matched)
