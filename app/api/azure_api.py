from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pypdf import PdfReader
import os

from app.services.storage_service import AzureStorageService
from app.services.ingestion_azure_service import process_single_pdf

from pydantic import BaseModel
from typing import Optional, List
from app.services.retrieval_service import hybrid_search
from app.services.chat_service import chat, clear_history

from app.services.embedding_service import generate_embedding

from app.services.db_service import (
    create_document,
    update_status,
    update_page_count
)

from app.core.config import settings

app = FastAPI()
storage = AzureStorageService()
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = "default_session"

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

# ========================
# 📤 UPLOAD + OCR
# ========================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    file_ext = file.filename.split(".")[-1].lower()

    if file_ext == "pdf":
        file_type = "pdf"
    elif file_ext in ["png", "jpg", "jpeg"]:
        file_type = "image"
    elif file_ext == "csv":
        file_type = "csv"
    else:
        raise ValueError("Unsupported file type")

    blob_name = storage.upload_file(temp_path, file_type=file_type)

    doc_id, exists = create_document(
    file_name=file.filename,
    blob_path=blob_name,
    container=settings.AZURE_STORAGE_CONTAINER,
    file_type=file_type 
)
    if exists:
        return {
            "message": "Document already processed",
            "document_id": doc_id
        }

    update_status(doc_id, "processing")

    try:
        docs = process_single_pdf(temp_path, blob_name, doc_id, file_type)

        reader = PdfReader(temp_path)
        page_count = len(reader.pages)
        update_page_count(doc_id, page_count)
        update_status(doc_id, "completed")

    except Exception as e:
        update_status(doc_id, "failed")
        raise

    return {
        "message": "uploaded + processed",
        "file": blob_name,
        "pages": len(docs)
    }


# ========================
# 📥 LIST PDFs
# ========================
@app.get("/pdfs")
def list_pdfs():
    return storage.list_pdfs()


# ========================
# 📥 DOWNLOAD + OCR
# ========================
@app.post("/process/{file_name}")
def process_pdf(file_name: str):
    local_path = f"/tmp/{file_name}"

    storage.download_file(file_name, local_path)   # ✅ FIX

    doc_id = create_document(
        file_name=file_name,
        blob_path=file_name,
        container=settings.AZURE_STORAGE_CONTAINER
    )

    update_status(doc_id, "processing")

    try:
        docs = process_single_pdf(local_path, file_name, doc_id)
        reader = PdfReader(temp_path)
        page_count = len(reader.pages)
        update_page_count(doc_id, page_count)
        update_status(doc_id, "completed")

    except Exception as e:
        update_status(doc_id, "failed")
        raise

    return {
        "message": "processed",
        "pages": len(docs)
    }


@app.get("/download/{blob_path:path}")
def download_pdf(blob_path: str):
    blob_client = storage.container_client.get_blob_client(blob_path)

    stream = blob_client.download_blob()
    props = blob_client.get_blob_properties()

    return StreamingResponse(
        stream.chunks(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={blob_path.split('/')[-1]}",
            "Content-Length": str(props.size)
        }
    )

# ========================
# 🔍 SEARCH (Member 2)
# ========================
@app.post("/search")
async def search_documents(request: SearchRequest):
    """
    Exposes the Hybrid Search (BM25 + HNSW).
    """
    results = hybrid_search(query=request.query, top_k=request.top_k)
    return {"results": results}

# ========================
# 💬 CHAT (Member 2)
# ========================
@app.post("/chat")
async def chat_with_docs(request: ChatRequest):
    """
    RAG Pipeline: Retrieves context and generates answer using Llama.
    """
    try:
        response = chat(
            user_message=request.message, 
            conversation_id=request.conversation_id
        )
        return response
    except Exception as e:
        # Useful for debugging if Ollama isn't running
        return {"error": str(e), "detail": "Check if Llama/Ollama is running locally"}

@app.delete("/chat/history/{conversation_id}")
async def reset_chat(conversation_id: str):
    clear_history(conversation_id)
    return {"message": f"History for {conversation_id} cleared."}