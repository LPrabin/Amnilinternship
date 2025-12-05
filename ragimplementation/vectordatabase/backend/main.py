from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import shutil
import os
from .models import NotebookCreate, NotebookResponse, QueryRequest, QueryResponse, ResourceResponse
from .rag_engine import RAGService

app = FastAPI(title="Notebook API")
rag_service = RAGService()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/notebooks", response_model=List[str])
def list_notebooks():
    return rag_service.list_notebooks()

@app.post("/notebooks", response_model=NotebookResponse)
def create_notebook(notebook: NotebookCreate):
    rag_service.create_notebook(notebook.name)
    return NotebookResponse(name=notebook.name, resource_count=0)

@app.delete("/notebooks/{name}")
def delete_notebook(name: str):
    rag_service.delete_notebook(name)
    return {"status": "deleted"}

@app.post("/notebooks/{name}/resources")
def add_resource(name: str, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        chunks = rag_service.add_document(name, file_path, file.filename)

    finally:
        
        if os.path.exists(file_path):
            os.remove(file_path)
            
    return {"status": "added", "filename": file.filename}

@app.get("/notebooks/{name}/resources", response_model=List[str])
def list_resources(name: str):
    return rag_service.list_resources(name)

@app.delete("/notebooks/{name}/resources/{resource_name}")
def delete_resource(name: str, resource_name: str):
    rag_service.delete_resource(name, resource_name)
    return {"status": "deleted"}

@app.post("/notebooks/{name}/query", response_model=QueryResponse)
def query_notebook(name: str, request: QueryRequest):
    return rag_service.query_notebook(name, request.query)
