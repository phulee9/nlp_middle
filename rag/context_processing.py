"""
context_processing.py
─────────────────────
Two post-retrieval processing steps:

1. Context Compression
   The retrieved chunks may contain sentences irrelevant to the query.
   We ask the LLM to extract only the truly relevant sentences.
   This reduces noise, saves tokens, and improves answer quality.

2. Context Reordering (Lost-in-the-Middle mitigation)
   Research shows LLMs best recall information at the START and END of context.
   Information in the MIDDLE is often "lost".
   → We reorder chunks so the MOST relevant ones appear at positions 1 and last,
     with less-relevant chunks in the middle.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Context compression
# ─────────────────────────────────────────────────────────────────────────────

COMPRESSION_SYSTEM = (
    "You are an expert at extracting relevant information from text. "
    "Given a question and a context passage, extract only the sentences that are "
    "directly relevant to answering the question. "
    "Return ONLY the extracted sentences, preserving their original wording. "
    "If no sentences are relevant, return: 'No relevant information found.'"
)

COMPRESSION_TEMPLATE = """Question: {question}

Context passage:
\"\"\"
{context}
\"\"\"

Extract only the sentences relevant to the question. Return them as-is."""


class ContextCompressor:
    """
    Compresses each retrieved chunk by keeping only query-relevant sentences.
    Uses the LLM as a filter.

    Note: This adds one LLM call per chunk.  Use judiciously (e.g. only for top-3).
    """

    def __init__(self, llm):
        self.llm = llm

    def compress(self, query: str, chunks: List[Dict], max_chunks: int = 3) -> List[Dict]:
        """
        Compress the first *max_chunks* chunks by extracting relevant sentences.

        Args:
            query:      user query
            chunks:     retrieved and ranked chunk dicts
            max_chunks: how many chunks to compress (others are passed through)

        Returns:
            List of chunk dicts with 'text' replaced by compressed version.
        """
        compressed = []
        for i, chunk in enumerate(chunks):
            if i >= max_chunks:
                # Pass through without compression
                compressed.append(chunk)
                continue

            prompt = COMPRESSION_TEMPLATE.format(
                question=query,
                context=chunk["text"]
            )

            try:
                new_text = self.llm.generate(
                    system_prompt=COMPRESSION_SYSTEM,
                    user_message=prompt,
                    temperature=0.0,
                    max_tokens=500,
                )
                c = dict(chunk)
                c["original_text"] = chunk["text"]
                c["text"] = new_text.strip() or chunk["text"]  # fallback to original
                compressed.append(c)
            except Exception as e:
                logger.warning(f"[Compress] LLM compression failed for chunk {i}: {e}")
                compressed.append(chunk)

        return compressed


# ─────────────────────────────────────────────────────────────────────────────
# Context reordering
# ─────────────────────────────────────────────────────────────────────────────

def reorder_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Lost-in-the-Middle reordering.

    Given chunks sorted by relevance (best first), rearrange them so:
      - Odd-indexed (most relevant)  → placed at the start
      - Even-indexed (less relevant) → placed at the end

    Example with 5 chunks ranked [1, 2, 3, 4, 5]:
      Result: [1, 3, 5, 4, 2]
              ↑ best at start        ↑ 2nd best at end

    The intuition: LLMs pay more attention to the first and last parts of context.
    """
    if len(chunks) <= 2:
        return chunks  # nothing to reorder

    # Split into two groups by alternating index
    start_group = chunks[0::2]   # indices 0, 2, 4, ... (more relevant)
    end_group   = chunks[1::2]   # indices 1, 3, 5, ... (less relevant, reversed)

    reordered = start_group + list(reversed(end_group))

    logger.debug(
        f"[Reorder] Chunks reordered: {[c.get('id','?') for c in chunks]} → "
        f"{[c.get('id','?') for c in reordered]}"
    )
    return reordered


# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────

def build_context(chunks: List[Dict], max_chars: int = 4000) -> str:
    """
    Concatenate chunk texts into a single context string for the LLM prompt.
    Truncates to max_chars to avoid exceeding the context window.

    Each chunk is labelled with its source and index for traceability.
    """
    parts = []
    total = 0
    for i, chunk in enumerate(chunks, start=1):
        header = f"[Passage {i} | source: {chunk.get('source', 'unknown')}]"
        body   = chunk["text"]
        entry  = f"{header}\n{body}"

        if total + len(entry) > max_chars:
            # Truncate the last chunk to fit within max_chars
            remaining = max_chars - total
            if remaining > 50:
                parts.append(entry[:remaining] + "…")
            break

        parts.append(entry)
        total += len(entry) + 2  # +2 for the \n\n separator

    return "\n\n".join(parts)