"""
multi_query.py — Multi-query generation via LLM rewriting.

Why multi-query?
  A user's original question is just one way to phrase their intent.
  If the answer sits in a chunk that uses different vocabulary, a single
  query might miss it.

  By asking the LLM to rephrase the question into N alternative queries,
  we cast a wider retrieval net:
    • Paraphrases catch semantic variation.
    • Keyword-focused variants help BM25.
    • Decomposed sub-questions capture multi-hop answers.

Pipeline:
  User query
    ↓ LLM generates N alternative queries
    ↓ Retrieve for EACH query independently
    ↓ Union / deduplicate results by chunk ID
    ↓ Re-score and return top_k unique chunks
"""

import logging
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

# How many alternative queries to generate
DEFAULT_N_ALTERNATIVES = 3


# ──────────────────────────────────────────────────────────────────────────────
# Multi-query generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_queries(
    original_query: str,
    llm_generate_fn: Callable[[str, str], str],
    n: int = DEFAULT_N_ALTERNATIVES,
) -> List[str]:
    """
    Use the LLM to generate `n` alternative phrasings of `original_query`.

    Args:
        original_query:  The user's original question.
        llm_generate_fn: Callable(prompt, system) → str  (wraps GroqLLM.generate)
        n:               Number of alternatives to generate.

    Returns:
        List of query strings (including the original at position 0).
    """
    system = (
        "You are a query rewriting assistant. Your job is to rephrase a user's "
        "question into multiple alternative questions that preserve the original "
        "intent but use different vocabulary and structure. "
        "This helps retrieve diverse relevant passages from a document."
    )

    prompt = (
        f"Original question: {original_query}\n\n"
        f"Generate exactly {n} alternative versions of this question. "
        "Rules:\n"
        "1. Each alternative must capture the same information need.\n"
        "2. Vary vocabulary, structure, and specificity.\n"
        "3. One variant should be more keyword-focused (good for BM25).\n"
        "4. One variant should be more semantic/conceptual.\n"
        "5. Return ONLY the alternatives, one per line, no numbering or bullets.\n"
        "6. Do NOT repeat the original question."
    )

    try:
        raw_output = llm_generate_fn(prompt, system)
        alternatives = [
            line.strip()
            for line in raw_output.strip().split("\n")
            if line.strip() and len(line.strip()) > 10
        ][:n]   # keep at most n
    except Exception as exc:
        logger.warning("Multi-query generation failed: %s — using original only", exc)
        alternatives = []

    # Always include the original as the first query
    all_queries = [original_query] + alternatives
    logger.info("Generated %d queries (original + %d alternatives)",
                len(all_queries), len(alternatives))
    return all_queries


# ──────────────────────────────────────────────────────────────────────────────
# Multi-query retrieval
# ──────────────────────────────────────────────────────────────────────────────

def multi_query_retrieve(
    queries: List[str],
    retrieve_fn: Callable[[str], List[Dict]],
    top_k: int = 5,
) -> List[Dict]:
    """
    Run `retrieve_fn` for each query and merge results by deduplication.

    Deduplication strategy:
      • Track seen chunk IDs.
      • Keep the copy with the highest score (by first-seen rank if scores absent).
      • Re-rank merged list by hybrid_score > cosine_score > bm25_score.

    Args:
        queries:     List of queries (original + alternatives).
        retrieve_fn: A callable(query) → List[Dict] (returns enriched chunks).
        top_k:       Final number of unique chunks to return.

    Returns:
        top_k deduplicated, re-ranked chunk dicts.
    """
    seen: Dict[str, Dict] = {}   # chunk_id → best chunk so far

    for query in queries:
        try:
            results = retrieve_fn(query)
        except Exception as exc:
            logger.warning("Retrieval failed for query '%s': %s", query[:60], exc)
            continue

        for chunk in results:
            chunk_id = chunk["id"]
            if chunk_id not in seen:
                seen[chunk_id] = chunk
            else:
                # Keep the chunk with the higher score
                current_score  = _primary_score(seen[chunk_id])
                candidate_score = _primary_score(chunk)
                if candidate_score > current_score:
                    seen[chunk_id] = chunk

    # Sort all unique chunks by best available score
    merged = sorted(seen.values(), key=_primary_score, reverse=True)

    # Re-assign ranks
    for rank, chunk in enumerate(merged[:top_k], start=1):
        chunk["retrieval_rank"] = rank

    return merged[:top_k]


def _primary_score(chunk: Dict) -> float:
    """Extract the best available numeric score from a chunk dict."""
    for key in ("hybrid_score", "reranker_score", "cosine_score", "bm25_score", "bm25_norm"):
        if key in chunk:
            return float(chunk[key])
    return 0.0