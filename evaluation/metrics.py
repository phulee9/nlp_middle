"""Information retrieval metrics for evaluating RAG retrieval quality."""
from __future__ import annotations

import math
from typing import Iterable, List, Set


def _ids(items: Iterable) -> List[str]:
    out = []
    for x in items:
        if isinstance(x, dict):
            out.append(str(x.get("id", "")))
        else:
            out.append(str(x))
    return [x for x in out if x]


def precision_at_k(retrieved: Iterable, relevant: Iterable[str], k: int) -> float:
    retrieved_ids = _ids(retrieved)[:k]
    relevant_set: Set[str] = set(relevant)
    if k <= 0:
        return 0.0
    return sum(1 for x in retrieved_ids if x in relevant_set) / k


def recall_at_k(retrieved: Iterable, relevant: Iterable[str], k: int) -> float:
    retrieved_ids = _ids(retrieved)[:k]
    relevant_set: Set[str] = set(relevant)
    if not relevant_set:
        return 0.0
    return sum(1 for x in retrieved_ids if x in relevant_set) / len(relevant_set)


def mrr(retrieved: Iterable, relevant: Iterable[str]) -> float:
    retrieved_ids = _ids(retrieved)
    relevant_set: Set[str] = set(relevant)
    for i, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Iterable, relevant: Iterable[str], k: int) -> float:
    retrieved_ids = _ids(retrieved)[:k]
    relevant_set: Set[str] = set(relevant)
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved_ids, start=1):
        rel = 1.0 if chunk_id in relevant_set else 0.0
        dcg += rel / math.log2(i + 1)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
