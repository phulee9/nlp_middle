import logging
import numpy as np
from typing import List, Dict

from rag.bm25 import BM25Retriever
from rag.embedding import EmbeddingRetriever

logger = logging.getLogger(__name__)


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:

    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-10:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


class HybridRetriever:

    def __init__(
        self,
        corpus: List[Dict],
        bm25: BM25Retriever,
        semantic: EmbeddingRetriever,
    ):
        self.corpus   = corpus
        self.bm25     = bm25
        self.semantic = semantic
        logger.info("HybridRetriever ready (%d chunks)", len(corpus))

    # ── Public search ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> List[Dict]:

        # ── Step 1: Get raw scores for ALL chunks ─────────────────────────
        bm25_scores   = self.bm25.get_all_scores(query)       # (N,)
        cosine_scores = self.semantic.get_all_scores(query)   # (N,)

        # ── Step 2: Normalise to [0, 1] ───────────────────────────────────
        bm25_norm   = _minmax_normalize(bm25_scores)
        cosine_norm = _minmax_normalize(cosine_scores)

        # ── Step 3: Weighted sum ──────────────────────────────────────────
        hybrid_scores = alpha * bm25_norm + (1 - alpha) * cosine_norm   # (N,)

        # ── Step 4: Rank and select top_k ─────────────────────────────────
        ranked_idx = np.argsort(hybrid_scores)[::-1][:top_k]

        results: List[Dict] = []
        for rank, idx in enumerate(ranked_idx, start=1):
            chunk = dict(self.corpus[idx])
            chunk["bm25_raw"]       = float(bm25_scores[idx])
            chunk["bm25_norm"]      = float(bm25_norm[idx])
            chunk["cosine_score"]   = float(cosine_scores[idx])
            chunk["cosine_norm"]    = float(cosine_norm[idx])
            chunk["hybrid_score"]   = float(hybrid_scores[idx])
            chunk["alpha"]          = alpha
            chunk["retrieval_rank"] = rank
            results.append(chunk)

        return results