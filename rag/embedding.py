import logging
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingRetriever:

    def __init__(
        self,
        corpus: List[Dict],
        model_name: str = "all-MiniLM-L6-v2",
        precomputed_embeddings=None,
    ):

        self.corpus = corpus
        self.model_name = model_name

        # Load mô hình embedding
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

        # Encode tất cả chunk (hoặc dùng cache)
        if precomputed_embeddings is not None:
            logger.info("Using cached embeddings")
            self.embeddings = precomputed_embeddings
        else:
            logger.info("Encoding %d chunks...", len(corpus))
            texts = [chunk["text"] for chunk in corpus]
            self.embeddings = self._encode(texts)

        logger.info("Embedding matrix shape: %s", self.embeddings.shape)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:

        scores = self.get_all_scores(query)           # (N,)
        ranked_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_idx, start=1):
            chunk = dict(self.corpus[idx])
            chunk["cosine_score"] = float(scores[idx])
            chunk["retrieval_rank"] = rank
            results.append(chunk)

        return results

    def get_all_scores(self, query: str) -> np.ndarray:

        query_emb = self._encode([query])              # (1, D)
        # Vì đã L2-normalize, dot product = cosine similarity
        scores = (query_emb @ self.embeddings.T).squeeze(0)   # (N,)
        return scores

    def _encode(self, texts: List[str]) -> np.ndarray:

        vecs = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,   # tự động L2-normalize
            convert_to_numpy=True,
        ).astype(np.float32)
        return vecs