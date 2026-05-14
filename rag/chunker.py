import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def split_into_chunks(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    source: str = "unknown",
) -> List[Dict]:

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[Dict] = []
    step = chunk_size - chunk_overlap  # how far to advance each iteration
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        # Skip nearly-empty chunks (e.g. page-number-only pages)
        if len(chunk_text) > 30:
            chunks.append({
                "id":         f"{source}_chunk_{idx:04d}",
                "text":       chunk_text,
                "source":     source,
                "char_start": start,
                "char_end":   min(end, len(text)),
            })
            idx += 1

        start += step

    logger.info(
        "Chunked '%s': %d chunks (size=%d, overlap=%d)",
        source, len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def build_corpus(
    pdf_texts: Dict[str, str],     # {filename_stem: cleaned_text}
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Dict]:

    all_chunks: List[Dict] = []
    for name, text in pdf_texts.items():
        source = Path(name).stem if "." in name else name
        chunks = split_into_chunks(text, chunk_size, chunk_overlap, source)
        all_chunks.extend(chunks)
        logger.info("'%s': %d chunks", source, len(chunks))

    logger.info("Corpus total: %d chunks from %d document(s)",
                len(all_chunks), len(pdf_texts))
    return all_chunks