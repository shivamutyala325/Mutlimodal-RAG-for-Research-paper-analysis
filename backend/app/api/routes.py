import tempfile
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from langchain_core.messages import HumanMessage, AIMessage

from app.api.models import (
    ChunkResult,
    ContextChunk,
    ChatRequest,
    ChatResponse,
    ChatHistoryResponse,
    MessageRecord,
    ImageUrlResponse,
    IngestResponse,
    PaperInfo,
    PapersResponse,
    RetrieveRequest,
    RetrieveResponse,
    StatusResponse,
)
from app.ingestion.parser import compute_paper_id
from app.pipeline import run_ingestion
from app.storage.minio_store import MinioStore
from app.storage.retriver import Retriever
from app.generation.graph import Chatbot
from app.utils.logger import get_logger

# One shared graph instance for all sessions — MemorySaver holds state per thread_id
_chatgraph = Chatbot().get_chatgraph()

logger = get_logger("routes")
router = APIRouter()

# In-memory job tracker — replaced by Redis in Phase 4
# Survives for the lifetime of the server process only
_jobs: Dict[str, dict] = {}


def _pipeline_task(file_path: str, paper_id: str, original_filename: str) -> None:
    """Runs in a background thread via FastAPI BackgroundTasks."""
    _jobs[paper_id] = {"status": "processing", "message": "Pipeline is running"}
    try:
        minio_store = MinioStore()
        run_ingestion(file_path, paper_id, minio_store, original_filename=original_filename)
        _jobs[paper_id] = {"status": "completed", "message": "Ingestion complete"}
        logger.info(f"Pipeline completed for {paper_id}")
    except Exception as e:
        _jobs[paper_id] = {"status": "failed", "message": str(e)}
        logger.error(f"Pipeline failed for {paper_id}: {e}")
    finally:
        Path(file_path).unlink(missing_ok=True)


# ── POST /ingest ──────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=202,
    summary="Ingest a PDF into the knowledge store",
)
async def ingest(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    content = await file.read()

    # Save to a named temp file — pipeline needs a real file path
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(content)
    tmp.flush()
    tmp.close()
    tmp_path = tmp.name

    paper_id = compute_paper_id(tmp_path)

    # Already fully ingested
    minio_store = MinioStore()
    if minio_store.object_exists(f"{paper_id}/original.pdf"):
        Path(tmp_path).unlink(missing_ok=True)
        return IngestResponse(
            paper_id=paper_id,
            status="already_indexed",
            message="This document is already in the knowledge store",
        )

    # Already queued or running in this server session
    if _jobs.get(paper_id, {}).get("status") in ("queued", "processing"):
        Path(tmp_path).unlink(missing_ok=True)
        return IngestResponse(
            paper_id=paper_id,
            status=_jobs[paper_id]["status"],
            message="This document is already being processed",
        )

    _jobs[paper_id] = {"status": "queued", "message": "Queued for ingestion"}
    background_tasks.add_task(_pipeline_task, tmp_path, paper_id, file.filename)

    return IngestResponse(
        paper_id=paper_id,
        status="queued",
        message="Document accepted and queued for ingestion",
    )


# ── GET /ingest/{paper_id}/status ────────────────────────────────────────────

@router.get(
    "/ingest/{paper_id}/status",
    response_model=StatusResponse,
    summary="Check ingestion status for a paper",
)
def get_status(paper_id: str):
    if paper_id in _jobs:
        return StatusResponse(
            paper_id=paper_id,
            status=_jobs[paper_id]["status"],
            message=_jobs[paper_id]["message"],
        )

    # Not in memory — check MinIO as fallback (e.g. after server restart)
    minio_store = MinioStore()
    if minio_store.object_exists(f"{paper_id}/original.pdf"):
        return StatusResponse(
            paper_id=paper_id,
            status="completed",
            message="Document is in the knowledge store",
        )

    raise HTTPException(status_code=404, detail=f"No record found for paper_id: {paper_id}")


# ── POST /retrieve ────────────────────────────────────────────────────────────

@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieve relevant chunks from the knowledge store",
)
def retrieve(request: RetrieveRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    retriever = Retriever()
    raw_results = retriever.retrieve(request.query, paper_id=request.paper_id)

    results = [
        ChunkResult(
            score=r["score"],
            chunk_type=r["chunk"]["chunk_type"],
            content=r["chunk"]["content"],
            metadata=r["chunk"]["metadata"],
            paper_id=r["chunk"]["paper_id"],
        )
        for r in raw_results
    ]

    return RetrieveResponse(
        query=request.query,
        results=results,
        total=len(results),
    )


# ── GET /papers ───────────────────────────────────────────────────────────────

@router.get(
    "/papers",
    response_model=PapersResponse,
    summary="List all documents in the knowledge store",
)
def list_papers():
    minio_store = MinioStore()
    paper_ids = minio_store.list_papers()

    papers = []
    for pid in paper_ids:
        meta = minio_store.get_json(f"{pid}/document_metadata.json")
        papers.append(PaperInfo(
            paper_id=pid,
            filename=meta.get("original_filename"),
            has_original=minio_store.object_exists(f"{pid}/original.pdf"),
            has_markdown=minio_store.object_exists(f"{pid}/markdown/document.md"),
        ))

    return PapersResponse(papers=papers, total=len(papers))


# ── GET /papers/{paper_id}/images/{filename} ─────────────────────────────────

@router.get(
    "/papers/{paper_id}/images/{filename}",
    response_model=ImageUrlResponse,
    summary="Get a presigned URL for an extracted figure",
)
def get_image_url(paper_id: str, filename: str):
    object_key = f"{paper_id}/images/{filename}"
    minio_store = MinioStore()

    if not minio_store.object_exists(object_key):
        raise HTTPException(status_code=404, detail=f"Image not found: {object_key}")

    url = minio_store.get_url(object_key, expires_seconds=3600)

    return ImageUrlResponse(
        object_key=object_key,
        url=url,
        expires_in_seconds=3600,
    )


# ── POST /chat/{session_id} ───────────────────────────────────────────────────

@router.post(
    "/chat/{session_id}",
    response_model=ChatResponse,
    summary="Send a message and get a response from the knowledge base",
)
def chat(session_id: str, request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty")

    config = {"configurable": {"thread_id": session_id}}

    result = _chatgraph.invoke(
        {
            "messages": [HumanMessage(content=request.message)],
            "paper_id": request.paper_id,
        },
        config=config,
    )

    ai_response = result["messages"][-1].content
    retrieved   = result.get("retrieve", False)
    raw_context = result.get("context", [])

    context = [
        ContextChunk(
            chunk_type=c.get("chunk_type", "text"),
            content=c.get("content", ""),
            metadata=c.get("metadata", {}),
            paper_id=c.get("paper_id", ""),
        )
        for c in raw_context
    ]

    return ChatResponse(
        session_id=session_id,
        response=ai_response,
        retrieved=retrieved,
        context=context,
    )


# ── GET /chat/{session_id}/history ────────────────────────────────────────────

@router.get(
    "/chat/{session_id}/history",
    response_model=ChatHistoryResponse,
    summary="Get the full conversation history for a session",
)
def get_chat_history(session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    state  = _chatgraph.get_state(config)

    if not state or not state.values:
        raise HTTPException(status_code=404, detail=f"No session found for id: {session_id}")

    raw_messages = state.values.get("messages", [])

    messages = []
    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            messages.append(MessageRecord(role="human", content=msg.content))
        elif isinstance(msg, AIMessage):
            messages.append(MessageRecord(role="ai", content=msg.content))

    # Count turns: one turn = one human + one AI exchange
    turns = sum(1 for m in messages if m.role == "human")

    return ChatHistoryResponse(
        session_id=session_id,
        messages=messages,
        total_turns=turns,
    )


# ── DELETE /chat/{session_id} ─────────────────────────────────────────────────

@router.delete(
    "/chat/{session_id}",
    summary="Clear the conversation history for a session",
)
def clear_session(session_id: str):
    storage = _chatgraph.checkpointer.storage
    keys_to_delete = [k for k in storage if session_id in str(k)]

    if not keys_to_delete:
        raise HTTPException(status_code=404, detail=f"No session found for id: {session_id}")

    for k in keys_to_delete:
        del storage[k]

    logger.info(f"Cleared session {session_id}")
    return {"session_id": session_id, "status": "cleared"}
