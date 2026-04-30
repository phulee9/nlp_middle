"""
embedding.py — Dense / semantic retrieval using BAAI/bge-m3.

What is dense retrieval?
  Each chunk of text is mapped to a fixed-length vector (embedding)
  using a transformer encoder model.  Similarity between a query and
  a chunk is measured by cosine distance of their vectors.

  Unlike BM25, dense retrieval captures *meaning* not just token
  overlap: "car" and "automobile" get similar embeddings even though
  they share no characters.

Why bge-m3?
  BAAI/bge-m3 is a state-of-the-art multilingual embedding model that:
    • Supports 100+ languages.
    • Achieves top performance on MTEB benchmark.
    • Produces 1024-dim embeddings.
    • Is available from Hugging Face with an MIT/research licence.
  Fallback: all-MiniLM-L6-v2 (384-dim, English, very fast).

Storage:
  Embeddings are stored as a numpy float32 matrix (N × D).
  For prototyping this is fine; at scale, use FAISS (included) for
  approximate nearest-neighbour search in sub-linear time.
"""

import logging
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# EmbeddingRetriever
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingRetriever:
    """
    Dense retrieval via sentence-transformers / FlagEmbedding.

    Supports two backends:
      • "flag"  — FlagEmbedding BGEM3FlagModel (bge-m3, recommended)
      • "st"    — sentence_transformers SentenceTransformer (any model)
    """

    def __init__(
        self,
        corpus: List[Dict],
        model_name: str = "BAAI/bge-m3",
        backend: str = "auto",         # "auto" | "flag" | "st"
        batch_size: int = 32,
        device: Optional[str] = None,  # None → auto-detect
        precomputed_embeddings=None,
    ):
        self.corpus = corpus
        self.model_name = model_name
        self.batch_size = batch_size

        # ── Load model ───────────────────────────────────────────────────────
        self._backend, self._model = self._load_model(model_name, backend, device)
        logger.info("Embedding backend: %s | model: %s", self._backend, model_name)

        # ── Encode all chunks ────────────────────────────────────────────────
        if precomputed_embeddings is not None:
            logger.info("Using cached embeddings")
            self.embeddings = precomputed_embeddings
        else:
            logger.info("Encoding %d chunks with %s …", len(corpus), model_name)
            texts = [chunk["text"] for chunk in corpus]
            self.embeddings = self._encode(texts)

        logger.info("Embedding matrix: %s", self.embeddings.shape)

    # ── Public search ────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Return top_k chunks ranked by cosine similarity to `query`.

        Returns:
            List of chunk dicts enriched with:
                "cosine_score"   — similarity in [0, 1] (L2-normalised)
                "retrieval_rank" — 1-indexed rank
        """         
        scores    = self.get_all_scores(query)          # (N,)

        ranked_idx = np.argsort(scores)[::-1][:top_k]

        results: List[Dict] = []
        for rank, idx in enumerate(ranked_idx, start=1):
            chunk = dict(self.corpus[idx])
            chunk["cosine_score"]   = float(scores[idx])
            chunk["retrieval_rank"] = rank
            results.append(chunk)

        return results

    def get_all_scores(self, query: str) -> np.ndarray:
        """Return cosine similarity scores for ALL corpus chunks."""
        query_emb = self._encode([query])               # (1, D)
        # Since embeddings are L2-normalised, dot product == cosine similarity
        scores = (query_emb @ self.embeddings.T).squeeze(0)   # (N,)
        return scores

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_model(self, model_name: str, backend: str, device):
        """Try FlagEmbedding first, fall back to sentence-transformers."""
        if backend in ("auto", "flag"):
            try:
                from FlagEmbedding import BGEM3FlagModel
                kwargs = {"use_fp16": True}
                if device:
                    kwargs["device"] = device
                model = BGEM3FlagModel(model_name, **kwargs)
                return "flag", model
            except Exception as exc:
                if backend == "flag":
                    raise
                logger.warning("FlagEmbedding not available (%s); using sentence-transformers", exc)

        # sentence-transformers fallback
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        return "st", model

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts → L2-normalised float32 matrix."""
        if self._backend == "flag":
            # BGEM3FlagModel returns a dict; "dense_vecs" is the standard embedding
            output = self._model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vecs = np.array(output["dense_vecs"], dtype=np.float32)
        else:
            vecs = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 50,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)

        # L2-normalise so dot product == cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        return vecs / norms