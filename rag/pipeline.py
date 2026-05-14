

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from rag.bm25 import BM25Retriever
from rag.chunker import split_into_chunks
from rag.embedding import EmbeddingRetriever
from rag.groq_llm import GroqLLM
from rag.hybrid import HybridRetriever
from rag.pdf_loader import load_pdf
from rag.reranker import Reranker

load_dotenv()
logger = logging.getLogger(__name__)


# System prompt — hướng dẫn LLM cách trả lời
ANSWER_SYSTEM_PROMPT = """
## ROLE
## ROLE
You are a precise document analyst answering questions strictly from provided context passages.

---

## GROUNDING RULES  *(highest priority)*
1. Use ONLY information explicitly stated in the context passages.  
2. If the answer is not present, respond exactly: `"The provided context does not contain enough information to answer this question."`  
3. Never infer, assume, or use external knowledge.

---

## ANSWER STRUCTURE
4. Open with a **one-sentence direct answer** to the question.  
5. Follow up with **bullet points** for supporting details, additional context, or multi-part answers.  
6. If applicable, **quote** or **cite specific sections** or **passages** where the information was sourced from.

---

## CITATION RULES
7. Cite every factual claim inline as `[Passage N]`.  
8. If a single fact is supported by multiple passages, cite all: `[Passage 1, Passage 3]`.

---

## OUTPUT FORMAT
9. The first sentence must be **a direct, concise response** to the question, summarizing the key point.  
10. Use bullet points to expand on the answer or include further explanation.  
11. If any data or reference to a document is included, properly **cite passages**.  
12. Keep responses **concise and focused**. Avoid unnecessary elaboration.

---

## NUMERIC & CALCULATION RULES *(only when applicable)*
13. When numeric values are relevant:
   a. List individual values with their sources (e.g., `VND 1,200 billion [Passage 2]`).  
   b. Only perform arithmetic if explicitly requested (sum, average, etc.).  
   c. Show the calculation on its own line: `Total = VND 1,200 billion + VND 568 billion = VND 1,768 billion`.  
14. **Preserve units exactly** as written in the source (e.g. "billion", "percent", "triệu đồng").  
15. If the same figure appears in multiple passages, **mention it once** and cite the passages (e.g., `(same figure reported in [Passage 1] and [Passage 3])`).

---

## STRICT PROHIBITIONS
- Do NOT start with "To answer the question…" or any meta-commentary.  
- Do NOT repeat the question back to the user.  
- Do NOT fabricate figures, dates, or names not present in the context.  
- Do NOT round numbers unless the source already rounds them.  
- Avoid overly detailed explanations when they are not required for answering the question.

"""


# ──────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────────────────────────────────────

def _file_fingerprint(pdf_path: str, embedding_model: str, chunk_size: int, chunk_overlap: int) -> str:

    path = Path(pdf_path)
    stat = path.stat()
    raw = (
        f"{path.resolve()}|{stat.st_size}|{stat.st_mtime}"
        f"|{embedding_model}|{chunk_size}|{chunk_overlap}"
    )
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(pdf_path: str, embedding_model: str, chunk_size: int, chunk_overlap: int) -> Path:

    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    pdf_name = Path(pdf_path).stem.replace(" ", "_")
    fp = _file_fingerprint(pdf_path, embedding_model, chunk_size, chunk_overlap)
    return cache_dir / f"{pdf_name}_{fp}.pkl"


# ──────────────────────────────────────────────────────────────────────────────
# Context builder
# ──────────────────────────────────────────────────────────────────────────────

def build_context(chunks: List[Dict], max_chars: int = 4000) -> str:
    parts = []
    total = 0
    for i, chunk in enumerate(chunks, start=1):
        header = f"[Passage {i} | source: {chunk.get('source', 'unknown')}]"
        entry = f"{header}\n{chunk['text']}"
        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining > 50:
                parts.append(entry[:remaining] + "…")
            break
        parts.append(entry)
        total += len(entry) + 2   # +2 cho "\n\n" separator
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# RAGPipeline
# ──────────────────────────────────────────────────────────────────────────────

class RAGPipeline:

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
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self.reranker_model = reranker_model or os.getenv(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.device = device or os.getenv("RAG_DEVICE") or None

        logger.info("Initializing RAG pipeline for: %s", self.pdf_path)

        cached_bm25 = None

        # ── Bước 1: Load chunks + embeddings (từ cache nếu có) ───────────────
        cache_file = _cache_path(
            self.pdf_path, self.embedding_model, self.chunk_size, self.chunk_overlap
        )
        cached_embeddings = None

        if cache_file.exists():
            logger.info("Cache found → loading from: %s", cache_file)
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            self.chunks = cache["chunks"]
            cached_embeddings = cache["embeddings"]
            cached_bm25 = cache.get("bm25")
        else:
            logger.info("No cache → reading PDF and chunking...")
            text = load_pdf(self.pdf_path)
            self.chunks = split_into_chunks(
                text,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                source=self.source_name,
            )

        if not self.chunks:
            raise ValueError("No chunks were created from the PDF.")

        logger.info("Total chunks: %d", len(self.chunks))

        # ── Bước 2: Khởi tạo các thành phần ─────────────────────────────────

        # LLM để sinh câu trả lời cuối cùng
        self.llm = GroqLLM()

        # BM25 — sparse retrieval (inverted index, keyword-based)
        # Luôn build lại vì rất nhanh (< 1s)
        # self.bm25 = BM25Retriever(self.chunks)
        if cached_bm25 is not None:
            logger.info("Using cached BM25 index")
            self.bm25 = cached_bm25  # ← dùng lại, không build lại
        else:
            self.bm25 = BM25Retriever(self.chunks)

        # Embedding — dense retrieval (semantic vector search)
        # Dùng cache nếu có, tránh encode lại toàn bộ corpus
        self.embedding = EmbeddingRetriever(
            self.chunks,
            model_name=self.embedding_model,
            precomputed_embeddings=cached_embeddings,
        )

        # Hybrid — kết hợp BM25 + Embedding
        self.hybrid = HybridRetriever(self.chunks, self.bm25, self.embedding)

        # Reranker — cross-encoder để rerank candidates từ hybrid
        self.reranker = Reranker(model_name=self.reranker_model, device=self.device)

        # ── Bước 3: Lưu cache nếu chưa có ────────────────────────────────────
        # if cached_embeddings is None:
        #     logger.info("Saving cache to: %s", cache_file)
        #     with open(cache_file, "wb") as f:
        #         pickle.dump(
        #             {
        #                 "chunks": self.chunks,
        #                 "embeddings": self.embedding.embeddings,
        #                 "embedding_model": self.embedding_model,
        #                 "chunk_size": self.chunk_size,
        #                 "chunk_overlap": self.chunk_overlap,
        #                 "source": self.source_name,
        #             },
        #             f,
        #         )
        if cached_embeddings is None or cached_bm25 is None:
            logger.info("Saving cache to: %s", cache_file)
            with open(cache_file, "wb") as f:
                pickle.dump(
                    {
                        "chunks": self.chunks,
                        "embeddings": self.embedding.embeddings,
                        "bm25": self.bm25,        # ← thêm
                        "embedding_model": self.embedding_model,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "source": self.source_name,
                    },
                    f,
                )

    # ── Public API ────────────────────────────────────────────────────────────

    def answer(self, question: str, mode: str = "hybrid_rerank", top_k: int = 5) -> Dict:

        retrieved = self.retrieve(question, mode=mode, top_k=top_k)
        context = build_context(retrieved, max_chars=4000)

        user_message = f"Context:\n{context}\n\nQuestion: {question}"
        # answer = self.llm.generate(
        #     system_prompt=ANSWER_SYSTEM_PROMPT,
        #     user_message=user_message,
        #     temperature=0.2,
        #     max_tokens=1024,
        # )
        context_length = len(context)
        if context_length < 1000:
            max_tokens = 512     # context ngắn → câu hỏi đơn giản
        elif context_length < 2500:
            max_tokens = 768     # context trung bình
        else:
            max_tokens = 1024    # context dài, câu hỏi phức tạp

        answer = self.llm.generate(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.2,
            max_tokens=max_tokens,
        )

        return {
            "question": question,
            "mode": mode,
            "answer": answer,
            "contexts": self._public_chunks(retrieved),
            "num_chunks_indexed": len(self.chunks),
            "source": self.source_name,
        }

    def retrieve(self, question: str, mode: str = "hybrid_rerank", top_k: int = 5) -> List[Dict]:

        mode = (mode or "hybrid_rerank").lower().strip()

        if mode == "bm25":
            # Chỉ dùng BM25 — nhanh, tốt cho keyword/số liệu cụ thể
            return self.bm25.search(question, top_k=top_k)

        if mode == "embedding":
            # Chỉ dùng embedding — tốt cho câu hỏi ngữ nghĩa / paraphrase
            return self.embedding.search(question, top_k=top_k)

        if mode == "hybrid":
            # Kết hợp BM25 + embedding, không rerank
            return self.hybrid.search(question, top_k=top_k, alpha=0.5)

        # if mode == "hybrid_rerank":
        #     # Bước 1: Hybrid lấy nhiều candidates hơn cần thiết
        #     n_candidates = max(top_k * 3, 15)   # ví dụ top_k=5 → lấy 15 candidates
        #     candidates = self.hybrid.search(question, top_k=n_candidates, alpha=0.5)
        #     # Bước 2: Reranker chọn top_k tốt nhất từ candidates
        #     return self.reranker.rerank(question, candidates, top_k=top_k)
        if mode == "hybrid_rerank":
            corpus_size = len(self.chunks)

            if corpus_size < 100:
                n_candidates = max(top_k * 2, 8)     # corpus nhỏ, rerank ít thôi
            elif corpus_size < 500:
                n_candidates = max(top_k * 2, 10)    # mặc định hợp lý
            else:
                n_candidates = max(top_k * 3, 15)    # corpus lớn, cần nhiều candidates hơn

            logger.info(
                "hybrid_rerank: corpus=%d, n_candidates=%d, top_k=%d",
                corpus_size, n_candidates, top_k,
            )
            candidates = self.hybrid.search(question, top_k=n_candidates, alpha=0.5)
            return self.reranker.rerank(question, candidates, top_k=top_k)

        raise ValueError(
            f"Unknown mode: '{mode}'. Supported: bm25, embedding, hybrid, hybrid_rerank"
        )

    @staticmethod
    def _public_chunks(chunks: List[Dict]) -> List[Dict]:

        keep = [
            "id", "source", "text", "char_start", "char_end",
            "retrieval_rank",
            "bm25_score", "bm25_raw", "bm25_norm",
            "cosine_score", "cosine_norm",
            "hybrid_score", "alpha",
            "reranker_score",
        ]
        return [{k: c[k] for k in keep if k in c} for c in chunks]


# ── Chạy trực tiếp từ terminal ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Hybrid RAG Pipeline")
    parser.add_argument("--pdf", required=True, help="Đường dẫn file PDF")
    parser.add_argument("--question", required=True, help="Câu hỏi")
    parser.add_argument(
        "--mode", default="hybrid_rerank",
        choices=["bm25", "embedding", "hybrid", "hybrid_rerank"],
        help="Phương pháp truy xuất"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Số chunk lấy về")
    args = parser.parse_args()

    pipeline = RAGPipeline(args.pdf)
    result = pipeline.answer(args.question, mode=args.mode, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))