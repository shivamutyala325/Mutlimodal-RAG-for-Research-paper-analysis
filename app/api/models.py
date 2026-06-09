from typing import List, Optional
from pydantic import BaseModel


class IngestResponse(BaseModel):
    paper_id: str
    status: str   # queued | processing | already_indexed
    message: str


class StatusResponse(BaseModel):
    paper_id: str
    status: str   # queued | processing | completed | failed
    message: str


class RetrieveRequest(BaseModel):
    query: str
    paper_id: Optional[str] = None   # scope to a specific document; None = all papers


class ChunkResult(BaseModel):
    score: float
    chunk_type: str   # text | image | table
    content: str
    metadata: dict
    paper_id: str


class RetrieveResponse(BaseModel):
    query: str
    results: List[ChunkResult]
    total: int


class PaperInfo(BaseModel):
    paper_id: str
    has_original: bool
    has_markdown: bool


class PapersResponse(BaseModel):
    papers: List[PaperInfo]
    total: int


class ImageUrlResponse(BaseModel):
    object_key: str
    url: str
    expires_in_seconds: int
