"""
pdf_loader.py — Extract and clean text from PDF files.

Strategy:
  1. Try pdfplumber first.
  2. Fall back to pypdf.
  3. If PDF is scanned/image-only, fall back to OCR with PyMuPDF + Tesseract.
"""

import re
import io
import logging
from pathlib import Path
from typing import List

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def load_pdf(pdf_path: str) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Loading PDF: %s", path.name)

    raw_text = _extract_text(str(path))
    clean = _clean_text(raw_text)

    if not clean.strip():
        raise ValueError(
            f"Could not extract any text from {path.name}. "
            "Even OCR did not find readable text."
        )

    logger.info("Extracted %d characters from %s", len(clean), path.name)
    return clean


def _extract_text(pdf_path: str) -> str:
    pages: List[str] = []

    # 1. Try pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"[Page {i + 1}]\n{text}")

        if pages:
            logger.info("pdfplumber extracted %d pages", len(pages))
            return "\n\n".join(pages)

        logger.warning("pdfplumber returned no text; trying pypdf")

    except Exception as exc:
        logger.warning("pdfplumber failed (%s); trying pypdf", exc)

    # 2. Try pypdf
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {i + 1}]\n{text}")

        if pages:
            logger.info("pypdf extracted %d pages", len(pages))
            return "\n\n".join(pages)

        logger.warning("pypdf returned no text; trying OCR")

    except Exception as exc:
        logger.warning("pypdf failed (%s); trying OCR", exc)

    # 3. OCR fallback
    return _extract_text_ocr(pdf_path)


def _extract_text_ocr(pdf_path: str) -> str:
    """
    OCR scanned/image-only PDF pages.

    Requires:
      pip install pymupdf pytesseract pillow

    Also requires Tesseract OCR installed on Windows.
    """
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image

        # Nếu Windows không tự nhận tesseract, mở comment dòng dưới:
        # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    except ImportError as exc:
        raise ImportError(
            "OCR dependencies missing. Please run:\n"
            "pip install pymupdf pytesseract pillow"
        ) from exc

    pages: List[str] = []

    doc = fitz.open(pdf_path)

    for i, page in enumerate(doc):
        logger.info("OCR page %d/%d", i + 1, len(doc))

        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # vie+eng: hỗ trợ tiếng Việt + tiếng Anh
        try:
            text = pytesseract.image_to_string(img, lang="vie+eng")
        except Exception:
            # fallback nếu máy chưa có language vie
            text = pytesseract.image_to_string(img, lang="eng")

        if text.strip():
            pages.append(f"[Page {i + 1}]\n{text}")

    return "\n\n".join(pages)


def _clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()