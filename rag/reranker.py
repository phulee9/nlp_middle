

import logging
from typing import List, Dict, Optional

import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
  

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
      
        logger.info("Loading reranker model: %s", model_name)
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)
        logger.info("Reranker ready: %s", model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        
        if not candidates:
            return []

        # Tạo list các cặp (query, text) để đưa vào cross-encoder
        pairs = [(query, chunk["text"]) for chunk in candidates]

        # Cross-encoder predict trả về array điểm số
        scores = self.model.predict(pairs)          # ndarray shape (N,)
        scores = np.array(scores, dtype=np.float32)

        # Sắp xếp theo điểm giảm dần, lấy top_k
        ranked_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_idx, start=1):
            chunk = dict(candidates[idx])           # copy để không làm thay đổi bản gốc
            chunk["reranker_score"] = float(scores[idx])
            chunk["retrieval_rank"] = rank
            results.append(chunk)

        logger.info(
            "Reranked %d candidates → top %d | best score: %.4f",
            len(candidates), top_k, float(scores[ranked_idx[0]]) if len(ranked_idx) else 0,
        )
        return results