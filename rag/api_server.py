import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.pipeline import RAGPipeline

load_dotenv()

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# Sửa đúng path tessdata của bạn
os.environ["TESSDATA_PREFIX"] = r"C:\Users\train\miniconda3\envs\nlp_env\share\tessdata"

app = FastAPI(title="Hybrid RAG Python API")

# cache pipeline trong RAM
PIPELINES: Dict[str, RAGPipeline] = {}


class IndexRequest(BaseModel):
    filename: str
    pdf_path: str


class AskRequest(BaseModel):
    filename: str
    question: str
    mode: str = "hybrid"
    top_k: int = 5


@app.get("/health")
def health():
    return {
        "success": True,
        "message": "Python RAG API is running",
        "loaded_files": list(PIPELINES.keys()),
    }


@app.post("/index")
def index_pdf(req: IndexRequest):
    pdf_path = Path(req.pdf_path)

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {req.pdf_path}")

    try:
        # Load OCR/chunk/embed/index đúng 1 lần
        pipeline = RAGPipeline(str(pdf_path))
        PIPELINES[req.filename] = pipeline

        return {
            "success": True,
            "message": "PDF indexed successfully",
            "filename": req.filename,
            "num_chunks_indexed": len(pipeline.chunks),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask(req: AskRequest):
    pipeline = PIPELINES.get(req.filename)

    if pipeline is None:
        raise HTTPException(
            status_code=400,
            detail="PDF is not indexed yet. Please upload/index the PDF first.",
        )

    try:
        result = pipeline.answer(
            question=req.question,
            mode=req.mode,
            top_k=req.top_k,
        )

        return {
            "success": True,
            "data": result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))