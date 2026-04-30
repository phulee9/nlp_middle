"""
pipeline.py — Main orchestration for the Hybrid RAG system.

Supported modes:
  - bm25
  - embedding
  - hybrid
  - hybrid_rerank
  - full

The full pipeline is:
User query → Multi-query → Hybrid retrieval → Rerank → Context compression
→ Context reorder → Groq LLM answer generation.
"""

from __future__ import annotations
import pickle
import hashlib
import numpy as np
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from rag.pdf_loader import load_pdf
from rag.chunker import split_into_chunks
from rag.bm25 import BM25Retriever
from rag.embedding import EmbeddingRetriever
from rag.hybrid import HybridRetriever
from rag.reranker import Reranker
from rag.multi_query import generate_queries, multi_query_retrieve
from rag.context_processing import ContextCompressor, reorder_chunks, build_context
from rag.groq_llm import GroqLLM

load_dotenv()
logger = logging.getLogger(__name__)

ANSWER_SYSTEM_PROMPT = """
You are a careful PDF question-answering assistant.

Rules:
- Answer ONLY using the provided context.
- If multiple numeric values are present, you MUST:
  1. Extract all relevant values
  2. Perform simple calculations if needed (e.g., sum)
  3. Provide both individual values AND final total if applicable

- Be concise, factual, and cite passages as [Passage N].
- If the context does not contain the answer, say you do not know.
- Format answers using clear bullet points.
- Put calculations on separate lines.
- Do not write "To answer the question..."
- Do not double-count duplicated values if they refer to the same asset/category.
- If the same amount appears in multiple passages, count it only once.
- Preserve the unit exactly as written, e.g. "VND1,768 billion" must remain billion.
"""

def _file_fingerprint(pdf_path: str, embedding_model: str, chunk_size: int, chunk_overlap: int) -> str:
    path = Path(pdf_path)
    stat = path.stat()

    raw = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime}|{embedding_model}|{chunk_size}|{chunk_overlap}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_path(pdf_path: str, embedding_model: str, chunk_size: int, chunk_overlap: int) -> Path:
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)

    pdf_name = Path(pdf_path).stem.replace(" ", "_")
    fp = _file_fingerprint(pdf_path, embedding_model, chunk_size, chunk_overlap)

    return cache_dir / f"{pdf_name}_{fp}.pkl"

class RAGPipeline:
    """Build indexes for one PDF and answer questions using selectable RAG modes."""

    def __init__(
        self,
        pdf_path: str,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_model: Optional[str] = None,
        reranker_model: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.pdf_path = str(pdf_path)
        self.source_name = Path(pdf_path).name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.reranker_model = reranker_model or os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
        self.device = device or os.getenv("RAG_DEVICE") or None

        logger.info("Loading PDF and building indexes: %s", self.pdf_path)

        cache_file = _cache_path(
            self.pdf_path,
            self.embedding_model,
            self.chunk_size,
            self.chunk_overlap,
        )

        cached_embeddings = None

        if cache_file.exists():
            logger.info("Loading RAG cache from: %s", cache_file)
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)

            self.chunks = cache["chunks"]
            cached_embeddings = cache["embeddings"]

        else:
            logger.info("No cache found. OCR/chunk/embed will run once.")

            text = load_pdf(self.pdf_path)
            self.chunks = split_into_chunks(
                text,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                source=self.source_name,
            )

        if not self.chunks:
            raise ValueError("No chunks were created from the PDF.")

        self.llm = GroqLLM()
        self.bm25 = BM25Retriever(self.chunks)

        self.embedding = EmbeddingRetriever(
            self.chunks,
            model_name=self.embedding_model,
            backend="auto",
            device=self.device,
            precomputed_embeddings=cached_embeddings,
        )

        # Nếu chưa có cache thì lưu lại sau khi embedding xong
        if cached_embeddings is None:
            logger.info("Saving RAG cache to: %s", cache_file)
            with open(cache_file, "wb") as f:
                pickle.dump(
                    {
                        "chunks": self.chunks,
                        "embeddings": self.embedding.embeddings,
                        "embedding_model": self.embedding_model,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "source": self.source_name,
                    },
                    f,
                )

        # 3 dòng này PHẢI nằm ngoài if
        self.hybrid = HybridRetriever(self.chunks, self.bm25, self.embedding)
        self.reranker = Reranker(model_name=self.reranker_model, backend="st", device=self.device)
        self.compressor = ContextCompressor(self.llm)

    def answer(self, question: str, mode: str = "full", top_k: int = 5) -> Dict:
        """Run selected mode and return answer + contexts."""
        mode = (mode or "full").lower().strip()
        retrieved = self.retrieve(question, mode=mode, top_k=top_k)

        final_chunks = retrieved
        if mode == "full":
            final_chunks = self.compressor.compress(question, retrieved, max_chunks=min(3, top_k))
            final_chunks = reorder_chunks(final_chunks)

        context = build_context(final_chunks, max_chars=5000)
        user_message = f"Context:\n{context}\n\nQuestion: {question}"
        answer = self.llm.generate(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.2,
            max_tokens=1024,
        )

        return {
            "question": question,
            "mode": mode,
            "answer": answer,
            "contexts": self._public_chunks(final_chunks),
            "num_chunks_indexed": len(self.chunks),
            "source": self.source_name,
        }

    def retrieve(self, question: str, mode: str = "full", top_k: int = 5) -> List[Dict]:
        """Return retrieved chunks for a selected method."""
        mode = (mode or "full").lower().strip()

        if mode == "bm25":
            return self.bm25.search(question, top_k=top_k)

        if mode == "embedding":
            return self.embedding.search(question, top_k=top_k)

        if mode == "hybrid":
            return self.hybrid.search(question, top_k=top_k, alpha=0.5)

        if mode == "hybrid_rerank":
            candidates = self.hybrid.search(question, top_k=max(top_k * 3, 10), alpha=0.5)
            return self.reranker.rerank(question, candidates, top_k=top_k)

        if mode == "full":
            def llm_generate_fn(prompt: str, system: str) -> str:
                return self.llm.generate(system_prompt=system, user_message=prompt, temperature=0.0)

            queries = generate_queries(question, llm_generate_fn, n=3)

            def retrieve_fn(q: str) -> List[Dict]:
                return self.hybrid.search(q, top_k=max(top_k * 3, 10), alpha=0.5)

            candidates = multi_query_retrieve(queries, retrieve_fn, top_k=max(top_k * 3, 10))
            return self.reranker.rerank(question, candidates, top_k=top_k)

        raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def _public_chunks(chunks: List[Dict]) -> List[Dict]:
        keep = [
            "id", "source", "text", "char_start", "char_end", "retrieval_rank",
            "bm25_score", "bm25_raw", "bm25_norm", "cosine_score", "cosine_norm",
            "hybrid_score", "reranker_score", "alpha",
        ]
        return [{k: c[k] for k in keep if k in c} for c in chunks]

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--mode", default="full")

    args = parser.parse_args()

    pipeline = RAGPipeline(args.pdf)
    result = pipeline.answer(args.question, mode=args.mode)

    print(json.dumps(result, ensure_ascii=False, indent=2))
