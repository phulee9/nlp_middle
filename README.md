# Hybrid RAG PDF Question Answering System

This project is a complete final-project implementation for an NLP course. It builds a PDF question-answering system using a Hybrid RAG architecture.

## Pipeline

```text
User Query
↓
Multi-query generation using LLM rewrite
↓
Hybrid retrieval: BM25 + BAAI/bge-m3 embedding
↓
Reranking: BAAI/bge-reranker
↓
Context compression using LLM
↓
Context reorder to reduce lost-in-the-middle
↓
Final answer generation using Groq LLM
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
│   ├── multi_query.py
│   ├── context_processing.py
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
- Chunking with overlap
- BM25-only retrieval
- Embedding-only retrieval using `BAAI/bge-m3`
- Hybrid retrieval
- Hybrid + reranker
- Full RAG pipeline
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
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-base
RAG_DEVICE=cpu
```

Do not hard-code the Groq API key in source code.

## 2. Test Python RAG CLI

Put a PDF file inside `uploads/`, for example:

```text
uploads/sample.pdf
```

Run:

```bash
python -m rag.cli --pdf uploads/sample.pdf --question "What is this document about?" --mode full --top-k 5
```

Available modes:

```text
bm25
embedding
hybrid
hybrid_rerank
full
```

## 3. Run Backend

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

## 4. Run Frontend

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

## 5. How to Use the Web App

1. Upload a PDF.
2. Select retrieval mode:
   - BM25
   - Embedding
   - Hybrid
   - Hybrid + Rerank
   - Full Pipeline
3. Ask a question.
4. View the answer and retrieved chunks.

No database is used. If you reload the browser, chat history is reset.

## 6. Run Evaluation

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

## 7. Notes for Limited GPU/CPU Machines

- `BAAI/bge-m3` can be heavy on CPU.
- For faster local testing, temporarily set:

```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=BAAI/bge-reranker-base
RAG_DEVICE=cpu
```

But for the final project requirement, `BAAI/bge-m3` is the target embedding model.

## 8. Important Implementation Notes

- BM25 handles exact keywords, IDs, codes, and technical terms well.
- Embedding retrieval handles semantic similarity and paraphrases.
- Hybrid retrieval combines normalized BM25 and cosine scores.
- Reranking improves candidate ordering using a cross-encoder.
- Context compression reduces irrelevant text before answer generation.
- Reordering puts important context near the beginning and end to reduce lost-in-the-middle.
