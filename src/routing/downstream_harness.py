"""End-to-end downstream-quality harness: router -> LLM -> judge pipeline.

The three callbacks (router_fn, llm_fn, judge_fn) are injected so the same
harness runs with real remote endpoints or with mocks in tests.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def run_downstream_eval(
    queries: list[dict],
    embeddings: np.ndarray,
    router_fn: Callable[[np.ndarray], int],
    llm_fn: Callable[[str, str], str],
    judge_fn: Callable[[str, str, str], int],
    domain_names: list[str],
) -> dict:
    """Run router -> LLM -> judge for each query and aggregate.

    Args:
        queries: list of dicts with keys 'question' (str), 'domain' (str),
            'domain_idx' (int). One per eval sample.
        embeddings: (N, D) array of pre-computed query embeddings, same order
            as `queries`.
        router_fn: emb (D,) -> int (chosen domain index).
        llm_fn: (question, routed_domain_name) -> generated answer.
        judge_fn: (question, answer, expected_domain_name) -> int score in [0, 5].
        domain_names: ordered list mapping domain_idx -> name.

    Returns:
        dict with keys 'per_query' (list of per-sample records), 'mean_score'
        (float), 'routing_accuracy' (float), 'mean_score_when_routed_correct',
        'mean_score_when_routed_wrong', 'n_queries'.
    """
    assert len(queries) == len(embeddings), "query-embedding length mismatch"
    per_query = []
    correct_scores: list[int] = []
    wrong_scores: list[int] = []

    for q, emb in zip(queries, embeddings):
        routed_idx = int(router_fn(emb))
        routed_name = domain_names[routed_idx]
        expected_name = q["domain"]
        answer = llm_fn(q["question"], routed_name)
        score = int(judge_fn(q["question"], answer, expected_name))
        is_correct_route = routed_idx == q["domain_idx"]
        per_query.append({
            "question": q["question"],
            "expected_domain": expected_name,
            "routed_domain": routed_name,
            "correct_route": is_correct_route,
            "answer": answer,
            "score": score,
        })
        (correct_scores if is_correct_route else wrong_scores).append(score)

    total = [r["score"] for r in per_query]
    mean = sum(total) / max(len(total), 1)
    routed_correct_n = sum(r["correct_route"] for r in per_query)
    routing_acc = routed_correct_n / max(len(per_query), 1)

    return {
        "per_query": per_query,
        "mean_score": mean,
        "routing_accuracy": routing_acc,
        "mean_score_when_routed_correct": (
            sum(correct_scores) / len(correct_scores) if correct_scores else 0.0
        ),
        "mean_score_when_routed_wrong": (
            sum(wrong_scores) / len(wrong_scores) if wrong_scores else 0.0
        ),
        "n_queries": len(per_query),
    }
