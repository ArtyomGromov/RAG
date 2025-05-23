# main.py

from fastapi import FastAPI, UploadFile, File, Form
import shutil
from qdrant_client import QdrantClient
from rag import DenseQdrantRAG 

HOST = "127.0.0.1"
PORT = 6333

qdrant = QdrantClient(host=HOST, port=PORT, timeout=30)
COLLECTION = "pdfminer_all-MiniLM-L6-v2"
rag = DenseQdrantRAG(qdrant, model_name="all-MiniLM-L6-v2", use_gpu=False)

app = FastAPI()

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    import tempfile
    import os
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, file.filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    rag.add_pdf(tmp_path, COLLECTION)
    return {"status": "ok"}

@app.post("/ask")
async def ask(query: str = Form(...), top_k: int = 2):
    result = rag.generate_answer(query, COLLECTION, top_k=top_k)
    return result
