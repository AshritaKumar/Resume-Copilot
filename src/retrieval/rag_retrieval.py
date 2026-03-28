"""Shared RAG retrieval: vector scope search + lightweight lexical fusion (cheap hybrid rerank)."""

from __future__ import annotations

import re

from src.retrieval.vector_store import RetrievedChunk


def retrieve_results_for_scope(
    vector_store,
    job_run_id: str,
    query_embedding: list[float],
    k: int,
    scope: str,
    shortlisted_ids: list[str],
    candidate_filters: list[str],
    max_distance: float,
    near_best_margin: float,
) -> list[RetrievedChunk]:
    if candidate_filters:
        targeted: list[RetrievedChunk] = []
        for candidate_id in candidate_filters:
            targeted.extend(
                vector_store.search_knn(
                    job_run_id=job_run_id,
                    query_embedding=query_embedding,
                    candidate_id=candidate_id,
                    k=max(k, 5),
                )
            )
        targeted.sort(key=lambda item: item.distance)
        filtered_targeted = _filter_low_relevance_results(
            targeted[:k], max_distance=max_distance, near_best_margin=near_best_margin
        )
        if filtered_targeted:
            return filtered_targeted

    if scope == "All indexed resumes":
        raw_results = vector_store.search_knn(job_run_id, query_embedding, k=k)
        filtered = _filter_low_relevance_results(raw_results, max_distance=max_distance, near_best_margin=near_best_margin)
        return filtered if filtered else _fallback_results(raw_results)

    if not shortlisted_ids:
        raw_results = vector_store.search_knn(job_run_id, query_embedding, k=k)
        filtered = _filter_low_relevance_results(raw_results, max_distance=max_distance, near_best_margin=near_best_margin)
        return filtered if filtered else _fallback_results(raw_results)

    merged: list[RetrievedChunk] = []
    for candidate_id in shortlisted_ids:
        merged.extend(
            vector_store.search_knn(
                job_run_id=job_run_id,
                query_embedding=query_embedding,
                candidate_id=candidate_id,
                k=k,
            )
        )
    merged.sort(key=lambda item: item.distance)
    topk = merged[:k]
    filtered = _filter_low_relevance_results(topk, max_distance=max_distance, near_best_margin=near_best_margin)
    return filtered if filtered else _fallback_results(topk)


def _filter_low_relevance_results(
    results: list[RetrievedChunk],
    max_distance: float,
    near_best_margin: float,
) -> list[RetrievedChunk]:
    if not results:
        return []
    best_distance = min(item.distance for item in results)
    filtered = [
        item
        for item in results
        if item.distance <= max_distance and item.distance <= (best_distance + near_best_margin)
    ]
    return filtered or []


def _fallback_results(results: list[RetrievedChunk], max_items: int = 3) -> list[RetrievedChunk]:
    if not results:
        return []
    ordered = sorted(results, key=lambda item: item.distance)
    return ordered[:max_items]


_STOP = frozenset(
    "the a an and or but in on for to of is are was were be been being with as at by from "
    "this that these those it its we you they he she who which what when where how why not no "
    "do does did can could should would any some all each every both".split()
)


def _query_terms(query: str, max_terms: int = 14) -> list[str]:
    raw = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if len(t) > 2 and t not in _STOP]
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def _vector_score(distance: float) -> float:
    return 1.0 / (1.0 + max(distance, 0.0))


def _lexical_coverage(text: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    low = text.lower()
    hits = sum(1 for t in terms if t in low)
    return hits / len(terms)


def lexical_rerank_fuse(chunks: list[RetrievedChunk], query: str, lexical_weight: float) -> list[RetrievedChunk]:
    """Re-order chunks using (1-w)*vector_similarity_proxy + w*term_overlap. Simple hybrid without BM25 deps."""
    if not chunks or lexical_weight <= 0:
        return chunks
    w = min(max(lexical_weight, 0.0), 0.95)
    terms = _query_terms(query)
    if not terms:
        return chunks
    scored: list[tuple[float, RetrievedChunk]] = []
    for c in chunks:
        combo = (1.0 - w) * _vector_score(c.distance) + w * _lexical_coverage(c.text, terms)
        scored.append((combo, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored]
