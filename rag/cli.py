"""CLI bridge used by Node.js backend and by developers."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag.pipeline import RAGPipeline

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid RAG CLI")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--mode", default="full", choices=["bm25", "embedding", "hybrid", "hybrid_rerank", "full"])
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(json.dumps({"error": f"PDF not found: {args.pdf}"}, ensure_ascii=False), file=sys.stdout)
        return 2

    try:
        pipeline = RAGPipeline(args.pdf)
        result = pipeline.answer(args.question, mode=args.mode, top_k=args.top_k)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        logging.exception("RAG CLI failed")
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stdout)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
