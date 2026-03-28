from __future__ import annotations

import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.agentic_langgraph_runner import run_agentic_pipeline_with_langgraph
from src.config.settings import CONFIG
from src.core.container import get_prompt_provider, get_repository, get_vector_store
from src.core.errors import UserFacingPipelineMessage
from src.core.input_validation import validate_uploaded_input_types
from src.ui.rag_chat import render_chat
from src.ui.preflight_ui import (
    build_preflight_report,
    can_retry_failed,
    render_preflight_report,
    select_failed_resumes_for_retry,
)
from src.ui.shortlist_ui import render_run_summary
from src.ui.styles import inject_global_styles
from src.ui.upload_names import sanitize_upload_filename

UPLOAD_DIR = Path(".streamlit_uploads")


def main() -> None:
    st.set_page_config(page_title="Confido Hiring Copilot", layout="wide")
    inject_global_styles()
    st.title("Confido Hiring Copilot")
    st.caption("Upload JD + resumes, run agentic pipeline once, then ask follow-up questions.")

    _init_session_state()
    _ensure_openai_key()
    vector_store = get_vector_store()
    prompt_manager = get_prompt_provider()

    jd_file, resumes, scope = _sidebar_controls()
    upload_fingerprint = _compute_upload_fingerprint(jd_file, resumes)
    preflight = build_preflight_report(jd_file, resumes, validate_upload_types=_validate_upload_types)
    run_disabled = _is_run_disabled(
        jd_file,
        resumes,
        upload_fingerprint,
        has_preflight_blockers=bool(preflight["has_blockers"]),
    )
    render_preflight_report(preflight)

    if st.session_state.pipeline_running:
        st.info("Agentic pipeline in progress...")
    if _is_duplicate_upload(upload_fingerprint):
        st.caption("Same upload set already processed. Upload different files or reset session.")

    if st.sidebar.button("Run Agentic Pipeline", type="primary", disabled=run_disabled):
        _run_pipeline_with_status(jd_file, resumes, upload_fingerprint)

    if st.sidebar.button(
        "Retry Failed Files Only",
        disabled=not can_retry_failed(jd_file, resumes, st.session_state.full_pipeline_output),
    ):
        _retry_failed_files(jd_file, resumes)

    if st.sidebar.button("Reset Session"):
        _reset_session()
        st.rerun()

    render_run_summary(st.session_state.full_pipeline_output)
    render_chat(scope, vector_store, prompt_manager)


def _run_pipeline_with_status(jd_file, resumes, upload_fingerprint: str) -> None:
    st.session_state.pipeline_running = True
    try:
        with st.status("Running pipeline", expanded=True) as status:
            status.write("1/4 Parsing documents")
            status.write("2/4 Extracting candidate profiles")
            status.write("3/4 Scoring and ranking")
            status.write("4/4 Generating artifacts")
            try:
                output = _run_full_pipeline_from_uploads(jd_file, resumes)
            except UserFacingPipelineMessage as msg:
                st.session_state.full_pipeline_output = {}
                st.session_state.active_run_id = ""
                st.info("**Shortlist not built** — see below.")
                st.markdown(str(msg))
                if getattr(msg, "output_dir", None) is not None:
                    st.caption(f"Run folder: `{msg.output_dir}`")
                return
            status.update(label="Pipeline completed", state="complete")
        st.session_state.full_pipeline_output = output
        st.session_state.active_run_id = output["run_id"]
        st.session_state.last_full_pipeline_fingerprint = upload_fingerprint
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Pipeline completed for run `{output['run_id']}`.\n"
                    "Ask any question about shortlist fit, strengths, and concerns."
                ),
            }
        ]
    finally:
        st.session_state.pipeline_running = False


def _retry_failed_files(jd_file, resumes) -> None:
    failed_resumes = select_failed_resumes_for_retry(resumes, st.session_state.full_pipeline_output)
    if not failed_resumes:
        st.sidebar.info("No failed resumes from previous run were found in current uploads.")
        return

    st.session_state.pipeline_running = True
    try:
        with st.status("Retrying failed resumes", expanded=True) as status:
            status.write(f"Retrying {len(failed_resumes)} failed file(s)")
            output = _run_full_pipeline_from_uploads(jd_file, failed_resumes)
            status.update(label="Retry completed", state="complete")
        st.session_state.full_pipeline_output = output
        st.session_state.active_run_id = output["run_id"]
        st.session_state.last_full_pipeline_fingerprint = _compute_upload_fingerprint(jd_file, failed_resumes)
    except UserFacingPipelineMessage as msg:
        st.sidebar.warning(str(msg))
    finally:
        st.session_state.pipeline_running = False


def _sidebar_controls() -> tuple[object, list[object] | None, str]:
    with st.sidebar:
        st.subheader("Setup")
        st.write(f"Embedding model: `{CONFIG.models.embedding_model}`")
        st.write(f"DB: `{CONFIG.database.dsn}`")
        jd_file = st.file_uploader("Upload JD", type=["md", "pdf", "docx"], key="jd_upload")
        resumes = st.file_uploader(
            "Upload resumes",
            type=["pdf", "docx", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="resumes_upload",
        )
        scope = st.radio(
            "Answer scope",
            options=["Ranked candidates only", "All indexed resumes"],
            key="answer_scope",
        )
    return jd_file, resumes, scope


def _ensure_openai_key() -> None:
    import os

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is required for strict LLM + embedding mode.", icon="🚫")
        st.stop()


def _init_session_state() -> None:
    defaults = {
        "job_run_id": uuid.uuid4().hex[:12],
        "full_pipeline_output": {},
        "pipeline_running": False,
        "last_full_pipeline_fingerprint": "",
        "active_run_id": "",
        "chat_messages": [{"role": "assistant", "content": "Upload JD + resumes, then click Run Agentic Pipeline."}],
        "review_decisions": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_session() -> None:
    st.session_state.job_run_id = uuid.uuid4().hex[:12]
    st.session_state.full_pipeline_output = {}
    st.session_state.pipeline_running = False
    st.session_state.last_full_pipeline_fingerprint = ""
    st.session_state.active_run_id = ""
    st.session_state.chat_messages = [{"role": "assistant", "content": _assistant_reset_message()}]
    st.session_state.review_decisions = {}


def _assistant_reset_message() -> str:
    return "Session reset. Upload JD + resumes and run pipeline again."


def _is_run_disabled(jd_file, resumes, fingerprint: str, has_preflight_blockers: bool = False) -> bool:
    return (
        not jd_file
        or not resumes
        or has_preflight_blockers
        or st.session_state.pipeline_running
        or (
            fingerprint
            and fingerprint == st.session_state.last_full_pipeline_fingerprint
            and bool(st.session_state.full_pipeline_output)
        )
    )


def _is_duplicate_upload(fingerprint: str) -> bool:
    return bool(
        fingerprint
        and fingerprint == st.session_state.last_full_pipeline_fingerprint
        and st.session_state.full_pipeline_output
    )


def _run_full_pipeline_from_uploads(jd_file, uploaded_resumes) -> dict:
    session_id = st.session_state.job_run_id
    base_dir = UPLOAD_DIR / session_id / "agentic_pipeline_inputs"
    resumes_dir = base_dir / "resumes"
    outputs_dir = base_dir / "outputs"
    _validate_upload_types(jd_file, uploaded_resumes)
    _cleanup_previous_run_state(current_session_id=session_id)
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    resumes_dir.mkdir(parents=True, exist_ok=True)

    jd_target = base_dir / sanitize_upload_filename(jd_file.name)
    jd_target.write_bytes(jd_file.getbuffer())
    for file in uploaded_resumes:
        (resumes_dir / sanitize_upload_filename(file.name)).write_bytes(file.getbuffer())

    result = run_agentic_pipeline_with_langgraph(
        jd_path=jd_target,
        resumes_dir=resumes_dir,
        output_dir=outputs_dir,
    )
    out = result.output_dir
    return {
        "run_id": result.run_id,
        "output_dir": str(out),
        "shortlist_rows": _load_shortlist_rows(out),
        "rejected_files": _load_rejected_files(out),
        "scoring_failures": _load_scoring_failures(out),
    }


def _load_shortlist_rows(output_dir: Path) -> list[dict[str, object]]:
    path = output_dir / "shortlist.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return []
    return [
        {
            "candidate_id": item.get("candidate_id", ""),
            "name": item.get("name", ""),
            "total_score": item.get("final_score", item.get("total_score", 0.0)),
            "strengths": item.get("strengths", []),
            "additional_skills": item.get("additional_skills", []),
            "concerns": item.get("concerns", []),
            "reasoning": item.get("reasoning", ""),
            "career_signals": item.get("career_signals", []),
            "criterion_scores": item.get("criterion_scores", []),
        }
        for item in payload
    ]


def _load_scoring_failures(output_dir: Path) -> list[dict[str, str]]:
    path = output_dir / "scoring_failures.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        raw = data.get("failures", [])
        if not isinstance(raw, list):
            return []
        out: list[dict[str, str]] = []
        for item in raw:
            if isinstance(item, dict) and item.get("candidate_id"):
                out.append(
                    {
                        "candidate_id": str(item.get("candidate_id", "")),
                        "error": str(item.get("error", ""))[:2000],
                    }
                )
        return out
    except Exception:  # noqa: BLE001
        return []


def _load_rejected_files(output_dir: Path) -> list[tuple[str, str]]:
    path = output_dir / "parse_summary.json"
    if not path.exists():
        return []
    try:
        summary = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return []
    return [(cid, status) for cid, status in summary.items() if status != "ok"]


def _validate_upload_types(jd_file, uploaded_resumes) -> None:
    validate_uploaded_input_types(
        jd_filename=str(jd_file.name),
        resume_filenames=[str(file.name) for file in uploaded_resumes],
        error_factory=UserFacingPipelineMessage,
    )


def _cleanup_previous_run_state(current_session_id: str) -> None:
    try:
        get_vector_store().delete_all_data()
        get_repository().delete_all_data()
    except Exception:  # noqa: BLE001
        pass

    if not UPLOAD_DIR.exists():
        return
    for session_dir in UPLOAD_DIR.iterdir():
        if session_dir.is_dir() and session_dir.name != current_session_id:
            shutil.rmtree(session_dir, ignore_errors=True)


def _compute_upload_fingerprint(jd_file, uploaded_resumes) -> str:
    if not jd_file or not uploaded_resumes:
        return ""
    hasher = hashlib.sha256()
    hasher.update(jd_file.name.encode("utf-8"))
    hasher.update(jd_file.getbuffer())
    for file in sorted(uploaded_resumes, key=lambda item: item.name):
        hasher.update(file.name.encode("utf-8"))
        hasher.update(file.getbuffer())
    return hasher.hexdigest()


if __name__ == "__main__":
    main()
