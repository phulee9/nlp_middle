# Hybrid RAG PDF Question Answering System

This project is a complete final-project implementation for an NLP course. It builds a PDF question-answering system using a Hybrid RAG architecture.



## Pipeline

```text
User Query
↓
PDF text extraction / OCR if needed
↓
Chunking with overlap
↓
Hybrid retrieval: BM25 + all-MiniLM-L6-v2 embedding
↓
Optional reranking: BAAI/bge-reranker-base
↓
Answer generation using Groq LLM
```

## Project Structure

```text
project-root/
├── backend/                 # Node.js Express backend
│   ├── server.js
│   ├── routes/
│   └── controllers/
├── frontend/                # React Vite frontend
│   ├── src/
│   └── components/
├── rag/                     # Python RAG core
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── bm25.py
│   ├── embedding.py
│   ├── hybrid.py
│   ├── reranker.py
│   ├── groq_llm.py
│   ├── pipeline.py
│   └── cli.py
├── evaluation/
│   ├── metrics.py
│   ├── questions.json
│   └── run_evaluation.py
├── outputs/
├── uploads/
├── .env.example
├── requirements.txt
└── README.md
```

## Features

- PDF upload
- PDF text extraction
- OCR support for scanned PDFs using Tesseract
- Chunking with overlap
- BM25-only retrieval
- Embedding-only retrieval using `sentence-transformers/all-MiniLM-L6-v2`
- Hybrid retrieval using BM25 + `all-MiniLM-L6-v2`
- Hybrid + reranker
- React chat UI
- Retrieved context display
- Evaluation metrics:
  - Precision@k
  - Recall@k
  - MRR
  - nDCG@k

## 1. Environment Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## 2. Install Tesseract OCR

This project supports OCR for scanned PDFs or image-based PDF pages.

`pytesseract` is only a Python wrapper. The Tesseract OCR engine must be installed separately on your machine.

### Windows

Download and install Tesseract OCR for Windows.

Recommended default install path:

```text
C:\Program Files\Tesseract-OCR
```

After installation, make sure this folder is added to the system `PATH`.

Verify installation:

```powershell
tesseract --version
```

If Python cannot find Tesseract, add this to your `.env` file:

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
```

### Ubuntu / WSL

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-eng
```

Optional Vietnamese OCR support:

```bash
sudo apt install -y tesseract-ocr-vie
```

Verify installation:

```bash
tesseract --version
```

### macOS

```bash
brew install tesseract
```

Verify installation:

```bash
tesseract --version
```

### Python OCR Package

Make sure `pytesseract` is included in `requirements.txt`:

```text
pytesseract
```

Or install manually:

```bash
pip install pytesseract
```

If the PDF is text-based, the system will extract text directly.

If the PDF is scanned or image-based, OCR will be used through Tesseract.

## 3. Environment Variables

Copy environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
copy .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=BAAI/bge-reranker-base
RAG_DEVICE=cpu

TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
```

Do not hard-code the Groq API key in source code.

## 4. Test Python RAG CLI

Put a PDF file inside `uploads/`, for example:

```text
uploads/sample.pdf
```

Run:

```bash
python -m rag.cli --pdf uploads/sample.pdf --question "What is this document about?" --mode hybrid_rerank --top-k 5
```

Available modes:

```text
bm25
embedding
hybrid
hybrid_rerank
```

## 5. Run Backend

```bash
cd backend
npm install
npm run dev
```

Backend runs at:

```text
http://localhost:3002
```

Health check:

```text
http://localhost:3002/health
```

## 6. Run Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:

```text
http://localhost:5173
```

## 7. How to Use the Web App

1. Upload a PDF.
2. Select retrieval mode:
   - BM25
   - Embedding
   - Hybrid
   - Hybrid + Rerank
3. Ask a question.
4. View the answer and retrieved chunks.

No database is used. If you reload the browser, chat history is reset.

## 8. Run Evaluation

Edit `evaluation/questions.json`:

```json
[
  {
    "question": "Your question here",
    "expected_answer": "Expected answer here",
    "relevant_chunks": ["sample.pdf_chunk_0001"]
  }
]
```

Run:

```bash
python -m evaluation.run_evaluation --pdf uploads/sample.pdf --questions evaluation/questions.json --top-k 5
```

Outputs:

```text
outputs/evaluation_results.csv
outputs/sample_answers.md
```

## 9. Notes for Limited GPU/CPU Machines

- `sentence-transformers/all-MiniLM-L6-v2` is lightweight and suitable for local CPU testing.
- It is faster than larger embedding models and works well for small to medium PDF question-answering projects.
- Reranking with `BAAI/bge-reranker-base` may still be slower on CPU because it uses a cross-encoder model.
- For low-resource machines, use:

```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=BAAI/bge-reranker-base
RAG_DEVICE=cpu
```

## 10. Important Implementation Notes

- BM25 handles exact keywords, IDs, codes, and technical terms well.
- `all-MiniLM-L6-v2` handles semantic similarity and paraphrases.
- Hybrid retrieval combines normalized BM25 scores and embedding cosine similarity scores.
- Reranking improves candidate ordering using a cross-encoder.
- This README version does not use LLM rewrite or multi-query generation.
- Tesseract OCR is used only when the PDF page does not contain extractable text or when OCR fallback is required.

## 11. Troubleshooting

### TesseractNotFoundError

If you see this error:

```text
TesseractNotFoundError: tesseract is not installed or it's not in your PATH
```

Check that Tesseract is installed:

```bash
tesseract --version
```

On Windows, make sure this path exists:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Then add the following lines to `.env`:

```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
```

### Missing OCR Language Data

If Tesseract cannot find language data, install the required language package.

Ubuntu / WSL:

```bash
sudo apt install -y tesseract-ocr-eng
```

For Vietnamese OCR:

```bash
sudo apt install -y tesseract-ocr-vie
```

### Slow First Run

The first run may be slow because embedding and reranker models need to be downloaded and loaded into memory.

After the first run, later queries are usually faster if the same process is still running.

### Embedding Model Download

The first time you run the project, `sentence-transformers/all-MiniLM-L6-v2` will be downloaded automatically.

If the model download fails, check your internet connection and run the command again.

## 12. Example Commands

Run BM25 only:

```bash
python -m rag.cli --pdf uploads/sample.pdf --question "What is this document about?" --mode bm25 --top-k 5
```

Run embedding only with `all-MiniLM-L6-v2`:

```bash
python -m rag.cli --pdf uploads/sample.pdf --question "What is this document about?" --mode embedding --top-k 5
```

Run hybrid retrieval using BM25 + `all-MiniLM-L6-v2`:

```bash
python -m rag.cli --pdf uploads/sample.pdf --question "What is this document about?" --mode hybrid --top-k 5
```

Run hybrid retrieval with reranker:

```bash
python -m rag.cli --pdf uploads/sample.pdf --question "What is this document about?" --mode hybrid_rerank --top-k 5
```

Run evaluation:

```bash
python -m evaluation.run_evaluation --pdf uploads/sample.pdf --questions evaluation/questions.json --top-k 5
```
