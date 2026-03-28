#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.config.settings import CONFIG
from src.core.container import get_vector_store
from src.retrieval.embedder import embed_texts
from src.retrieval.rag_retrieval import lexical_rerank_fuse, retrieve_results_for_scope


def _load_cases(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def _evaluate_case(
    case: dict,
    *,
    job_run_id: str,
    scope: str,
    shortlisted_ids: list[str],
) -> tuple[bool, str]:
    q = str(case.get("query", "")).strip()
    if not q:
        return False, "empty query"
    top_k = int(case.get("top_k", CONFIG.pipeline.retrieval_top_k))
    pc = CONFIG.pipeline
    pool = max(5, int(top_k * float(getattr(pc, "rag_retrieval_pool_multiplier", 2.5))))
    max_dist = float(getattr(pc, "rag_max_distance", 0.62))
    near = float(getattr(pc, "rag_near_best_margin", 0.18))
    lex_w = float(getattr(pc, "rag_lexical_weight", 0.32))

    emb = embed_texts([q], allow_fallback=False)[0]
    vector_store = get_vector_store()
    chunks = retrieve_results_for_scope(
        vector_store=vector_store,
        job_run_id=job_run_id,
        query_embedding=emb,
        k=pool,
        scope=scope,
        shortlisted_ids=shortlisted_ids,
        candidate_filters=[],
        max_distance=max_dist,
        near_best_margin=near,
    )
    chunks = lexical_rerank_fuse(chunks, q, lex_w)
    window = chunks[: min(top_k, len(chunks))]
    if not window:
        return False, "no chunks retrieved"

    expect_any = case.get("expect_any")
    if isinstance(expect_any, list) and expect_any:
        blob = "\n".join(c.text.lower() for c in window)
        hits = [s for s in expect_any if str(s).lower() in blob]
        if not hits:
            ids = [c.chunk_id for c in window[:5]]
            return False, f"expect_any not found in top-{len(window)}; chunk_ids[:5]={ids}"

    return True, f"ok chunks={len(window)} ids={[c.chunk_id for c in window[:6]]}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval@k golden gate")
    parser.add_argument("--run-id", required=True, help="job_run_id with indexed resume_chunks")
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "eval" / "retrieval_golden.jsonl",
    )
    parser.add_argument(
        "--scope",
        default="All indexed resumes",
        choices=["Ranked candidates only", "All indexed resumes"],
    )
    parser.add_argument(
        "--shortlist-ids",
        default="",
        help="Comma-separated candidate_id values when using shortlist scope",
    )
    args = parser.parse_args()
    cases_path: Path = args.cases
    if not cases_path.is_file():
        print(f"Cases file not found: {cases_path}", file=sys.stderr)
        return 2

    shortlist = [x.strip() for x in args.shortlist_ids.split(",") if x.strip()]
    if args.scope == "Ranked candidates only" and not shortlist:
        print(
            "Warning: shortlist scope with empty --shortlist-ids falls back to global search in retrieve (no ids).",
            file=sys.stderr,
        )

    cases = _load_cases(cases_path)
    failed = 0
    for case in cases:
        cid = case.get("id", "?")
        ok, msg = _evaluate_case(
            case, job_run_id=args.run_id, scope=args.scope, shortlisted_ids=shortlist
        )
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {cid}: {msg}")
        if not ok:
            failed += 1

    if failed:
        print(f"\n{failed}/{len(cases)} failed", file=sys.stderr)
        return 1
    print(f"\nAll {len(cases)} cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
