import json
import argparse
import math
from rag.pipeline import RAGPipeline


def normalize(text: str) -> str:
    return (text or "").lower().replace("\n", " ").strip()


def is_relevant(chunk_text: str, keywords: list[str]) -> bool:
    if not keywords:
        return False

    text = normalize(chunk_text)

    # Chỉ cần match 1 keyword/cụm thông tin là tính relevant
    return any(normalize(k) in text for k in keywords)


def precision_at_k(relevance, k):
    if not relevance:
        return 0.0
    return sum(relevance[:k]) / k


def recall_at_k(relevance, k):
    # Với dataset keyword-based, tính binary recall:
    # top-k có ít nhất một chunk đúng hay không.
    return 1.0 if sum(relevance[:k]) > 0 else 0.0


def mrr_score(relevance):
    for i, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevance, k):
    score = 0.0
    for i, rel in enumerate(relevance[:k], start=1):
        if rel:
            score += 1.0 / math.log2(i + 1)
    return score


def ndcg_at_k(relevance, k):
    dcg = dcg_at_k(relevance, k)
    ideal = sorted(relevance, reverse=True)
    idcg = dcg_at_k(ideal, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate(pdf_path, questions_path, modes, top_k):
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("\n==============================")
    print("BUILDING RAG PIPELINE")
    print("==============================")
    pipeline = RAGPipeline(pdf_path)

    all_rows = []

    for mode in modes:
        print("\n\n==============================")
        print(f"EVALUATING MODE: {mode}")
        print("==============================")

        mode_rows = []

        for item in questions:
            qid = item["id"]
            question = item["question"]
            keywords = item.get("relevant_keywords", [])

            # Out-of-scope question: bỏ qua retrieval metric
            if not keywords:
                print(f"\n{qid} | OUT-OF-SCOPE QUESTION")
                print(f"Question: {question}")
                print("Skipped retrieval metrics because relevant_keywords is empty.")
                continue

            retrieved = pipeline.retrieve(question, mode=mode, top_k=top_k)

            relevance = [
                1 if is_relevant(chunk.get("text", ""), keywords) else 0
                for chunk in retrieved
            ]

            p = precision_at_k(relevance, top_k)
            r = recall_at_k(relevance, top_k)
            mrr = mrr_score(relevance)
            ndcg = ndcg_at_k(relevance, top_k)

            row = {
                "mode": mode,
                "qid": qid,
                "precision": p,
                "recall": r,
                "mrr": mrr,
                "ndcg": ndcg,
            }

            mode_rows.append(row)
            all_rows.append(row)

            top_chunk_id = retrieved[0]["id"] if retrieved else "N/A"

            print(f"\n{qid}: {question}")
            print(f"Top chunk: {top_chunk_id}")
            print(f"Relevance list: {relevance}")
            print(f"Precision@{top_k}: {p:.3f}")
            print(f"Recall@{top_k}:    {r:.3f}")
            print(f"MRR:          {mrr:.3f}")
            print(f"nDCG@{top_k}:      {ndcg:.3f}")

        if mode_rows:
            avg_p = sum(x["precision"] for x in mode_rows) / len(mode_rows)
            avg_r = sum(x["recall"] for x in mode_rows) / len(mode_rows)
            avg_mrr = sum(x["mrr"] for x in mode_rows) / len(mode_rows)
            avg_ndcg = sum(x["ndcg"] for x in mode_rows) / len(mode_rows)

            print("\n------------------------------")
            print(f"SUMMARY FOR MODE: {mode}")
            print("------------------------------")
            print(f"Avg Precision@{top_k}: {avg_p:.3f}")
            print(f"Avg Recall@{top_k}:    {avg_r:.3f}")
            print(f"Avg MRR:          {avg_mrr:.3f}")
            print(f"Avg nDCG@{top_k}:      {avg_ndcg:.3f}")

    print("\n\n==============================")
    print("FINAL SUMMARY")
    print("==============================")

    for mode in modes:
        rows = [x for x in all_rows if x["mode"] == mode]
        if not rows:
            continue

        avg_p = sum(x["precision"] for x in rows) / len(rows)
        avg_r = sum(x["recall"] for x in rows) / len(rows)
        avg_mrr = sum(x["mrr"] for x in rows) / len(rows)
        avg_ndcg = sum(x["ndcg"] for x in rows) / len(rows)

        print(
            f"{mode:15s} | "
            f"P@{top_k}: {avg_p:.3f} | "
            f"R@{top_k}: {avg_r:.3f} | "
            f"MRR: {avg_mrr:.3f} | "
            f"nDCG@{top_k}: {avg_ndcg:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdf", required=True)
    parser.add_argument("--questions", default="rag/evaluation/questions.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["bm25", "embedding", "hybrid", "hybrid_rerank", "full"]
    )

    args = parser.parse_args()

    evaluate(
        pdf_path=args.pdf,
        questions_path=args.questions,
        modes=args.modes,
        top_k=args.top_k,
    )