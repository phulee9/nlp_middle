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

    return [tok for tok in re.split(r"[^a-zA-Z0-9]+", text.lower()) if tok]


# ──────────────────────────────────────────────────────────────────────────────
# BM25Retriever
# ──────────────────────────────────────────────────────────────────────────────

class BM25Retriever:


    def __init__(self, corpus: List[Dict]):

        self.corpus = corpus

        # Tokenise every chunk — this list mirrors the corpus order
        self.tokenized_corpus = [_tokenize(chunk["text"]) for chunk in corpus]

        # BM25Okapi builds the inverted index here (O(N) time)
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info("BM25 index built over %d chunks", len(corpus))

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict]:

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
        return self.bm25.get_scores(_tokenize(query))