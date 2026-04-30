"""
bm25.py — BM25 sparse retrieval.

What is BM25?
  BM25 (Best Match 25) is a probabilistic ranking function used in
  information retrieval since the 1990s.  It extends TF-IDF by:
    • Saturating term frequency (so one word repeated 100× doesn't
      dominate — controlled by parameter k1).
    • Normalising for document length (long docs are penalised —
      controlled by parameter b).

  Score for document D given query Q with terms q₁…qₙ:
      score(D, Q) = Σᵢ IDF(qᵢ) · (tf(qᵢ,D) · (k1+1)) /
                       (tf(qᵢ,D) + k1 · (1 − b + b · |D|/avgdl))

  Default params (BM25Okapi): k1=1.5, b=0.75.

Strengths:
  ✔ Very fast to build and query.
  ✔ Excellent for exact keyword / code / ID matching.
  ✔ No GPU needed.

Weaknesses:
  ✗ Bag-of-words: ignores word order and semantics.
  ✗ Cannot handle synonyms or paraphrases.
"""

import re
import logging
import numpy as np
from typing import List, Dict

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Tokeniser
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """
    Lowercase and split on non-alphanumeric characters.
    Production tip: add stopword removal + stemming for higher precision.
    """
    return [tok for tok in re.split(r"[^a-zA-Z0-9]+", text.lower()) if tok]


# ──────────────────────────────────────────────────────────────────────────────
# BM25Retriever
# ──────────────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    Thin wrapper around rank_bm25.BM25Okapi with a simple search interface.

    Usage:
        retriever = BM25Retriever(corpus)
        results = retriever.search("your query", top_k=5)
    """

    def __init__(self, corpus: List[Dict]):
        """
        Build the BM25 inverted index.

        Args:
            corpus: List of chunk dicts with at least a "text" key.
        """
        self.corpus = corpus

        # Tokenise every chunk — this list mirrors the corpus order
        self.tokenized_corpus = [_tokenize(chunk["text"]) for chunk in corpus]

        # BM25Okapi builds the inverted index here (O(N) time)
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info("BM25 index built over %d chunks", len(corpus))

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top_k most relevant chunks for `query`.

        Returns:
            List of chunk dicts enriched with:
                "bm25_score"     — raw BM25 score (higher is better)
                "retrieval_rank" — 1-indexed rank in this result list
        """
        query_tokens = _tokenize(query)

        # get_scores returns an ndarray of shape (N,) with BM25 scores
        scores: np.ndarray = self.bm25.get_scores(query_tokens)

        # Rank descending and take top_k
        ranked_idx = np.argsort(scores)[::-1][:top_k]

        results: List[Dict] = []
        for rank, idx in enumerate(ranked_idx, start=1):
            chunk = dict(self.corpus[idx])          # copy to avoid mutation
            chunk["bm25_score"]     = float(scores[idx])
            chunk["retrieval_rank"] = rank
            results.append(chunk)

        return results

    def get_all_scores(self, query: str) -> np.ndarray:
        """Return raw BM25 scores for ALL corpus chunks (needed for hybrid)."""
        return self.bm25.get_scores(_tokenize(query))