from pydantic import BaseModel
from typing import List, Optional

class NotebookCreate(BaseModel):
    name: str

class NotebookResponse(BaseModel):
    name: str
    resource_count: int

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    top3docs: List[str]

class ResourceResponse(BaseModel):
    id: str
    name: str
    type: str
