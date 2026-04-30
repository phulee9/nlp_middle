"""
reranker.py — Cross-encoder reranking with BAAI/bge-reranker-base.

What is reranking?
  First-stage retrieval (BM25 / embedding) is fast but imprecise:
  it scores query and document independently (bi-encoder).

  A cross-encoder reads query AND document together in a single
  forward pass through a transformer, allowing full self-attention
  between them.  This is much more accurate but too slow to run
  against every document — so we only run it on the top_k candidates
  from the first stage.

Pipeline:
  Hybrid retrieval top_k * expand_factor  →  Reranker  →  top_k

Why bge-reranker-base?
  • State-of-the-art multilingual reranker from BAAI.
  • Small enough to run on CPU in reasonable time.
  • Pairs naturally with bge-m3 embeddings.
  Larger alternative: BAAI/bge-reranker-large (slower, more accurate).
"""

import logging
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranker using FlagEmbedding or sentence-transformers.

    Usage:
        reranker = Reranker()
        reranked = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        backend: str = "auto",      # "auto" | "flag" | "st"
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self._backend, self._model = self._load_model(model_name, backend, device)
        logger.info("Reranker loaded: %s (backend=%s)", model_name, self._backend)

    # ── Public API ───────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Rerank `candidates` using the cross-encoder.

        Args:
            query:      User question.
            candidates: List of chunk dicts from first-stage retrieval.
            top_k:      How many to return after reranking.

        Returns:
            top_k chunks sorted by reranker score (descending),
            each enriched with "reranker_score" and updated "retrieval_rank".
        """
        if not candidates:
            return []

        texts = [chunk["text"] for chunk in candidates]
        scores = self._score(query, texts)   # (len(candidates),)

        # Sort by reranker score descending
        ranked_idx = np.argsort(scores)[::-1][:top_k]

        results: List[Dict] = []
        for rank, idx in enumerate(ranked_idx, start=1):
            chunk = dict(candidates[idx])
            chunk["reranker_score"] = float(scores[idx])
            chunk["retrieval_rank"] = rank
            results.append(chunk)

        return results

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_model(self, model_name: str, backend: str, device):
        """Try FlagEmbedding FlagReranker first, fall back to sentence-transformers."""
        if backend in ("auto", "flag"):
            try:
                from FlagEmbedding import FlagReranker
                model = FlagReranker(model_name, use_fp16=True)
                return "flag", model
            except Exception as exc:
                if backend == "flag":
                    raise
                logger.warning("FlagEmbedding reranker failed (%s); trying sentence-transformers", exc)

        # sentence-transformers CrossEncoder fallback
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(model_name, device=device)
        return "st", model

    def _score(self, query: str, texts: List[str]) -> np.ndarray:
        """Compute reranker scores for each (query, text) pair."""
        pairs = [(query, t) for t in texts]

        if self._backend == "flag":
            # FlagReranker.compute_score returns a list of floats
            scores = self._model.compute_score(pairs, normalize=True)
        else:
            # CrossEncoder.predict returns ndarray
            scores = self._model.predict(pairs)

        return np.array(scores, dtype=np.float32)