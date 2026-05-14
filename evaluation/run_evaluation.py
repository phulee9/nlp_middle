from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

from evaluation.metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k
from rag.pipeline import RAGPipeline

load_dotenv()

MODES = ["bm25", "embedding", "hybrid", "hybrid_rerank"]


def load_questions(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--questions", default="evaluation/questions.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args.questions)
    pipeline = RAGPipeline(args.pdf)

    rows = []
    answer_lines = ["# Sample Answers\n"]

    for q_idx, item in enumerate(questions, start=1):
        question = item["question"]
        relevant = item.get("relevant_chunks", [])
        answer_lines.append(f"\n## Q{q_idx}. {question}\n")

        for mode in MODES:
            result = pipeline.answer(question, mode=mode, top_k=args.top_k)
            contexts = result["contexts"]
            rows.append({
                "question_id": q_idx,
                "mode": mode,
                "precision_at_k": precision_at_k(contexts, relevant, args.top_k),
                "recall_at_k": recall_at_k(contexts, relevant, args.top_k),
                "mrr": mrr(contexts, relevant),
                "ndcg_at_k": ndcg_at_k(contexts, relevant, args.top_k),
                "answer": result["answer"],
            })
            answer_lines.append(f"### Mode: {mode}\n")
            answer_lines.append(result["answer"] + "\n")
            answer_lines.append("Retrieved chunks:\n")
            for c in contexts:
                score = c.get("reranker_score", c.get("hybrid_score", c.get("cosine_score", c.get("bm25_score", ""))))
                answer_lines.append(f"- `{c.get('id')}` score={score}\n")

    df = pd.DataFrame(rows)
    csv_path = output_dir / "evaluation_results.csv"
    md_path = output_dir / "sample_answers.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text("".join(answer_lines), encoding="utf-8")

    print(df[["question_id", "mode", "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"]])
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
