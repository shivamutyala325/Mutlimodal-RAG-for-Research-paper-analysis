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
    filename: Optional[str] = None
    has_original: bool
    has_markdown: bool


class PapersResponse(BaseModel):
    papers: List[PaperInfo]
    total: int


class ImageUrlResponse(BaseModel):
    object_key: str
    url: str
    expires_in_seconds: int


# ── Chat models ───────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    paper_id: Optional[str] = None   # scope retrieval to a specific paper; None = all papers


class ContextChunk(BaseModel):
    chunk_type: str   # text | image | table
    content: str
    metadata: dict
    paper_id: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    retrieved: bool              # whether retrieval was triggered this turn
    context: List[ContextChunk]  # chunks used as context (empty if no retrieval)


class MessageRecord(BaseModel):
    role: str     # human | ai
    content: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[MessageRecord]
    total_turns: int
