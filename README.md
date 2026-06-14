# Research Paper Analyzer

A multimodal RAG (Retrieval-Augmented Generation) system for ingesting and querying research papers. Upload PDFs and chat with their content — text, tables, and figures are all indexed and retrievable.

## Architecture

```
frontend (Streamlit)  →  backend (FastAPI)  →  Qdrant (vector store)
                                            →  MinIO  (object store)
                                            →  Groq   (LLM inference)
```

**Ingestion pipeline** — runs in the background after upload:
1. **Parser** — Docling extracts text, tables, and images from the PDF
2. **Processors** — image descriptions, table summaries, and text chunking run concurrently
3. **Embedder** — FastEmbed generates embeddings for all chunks
4. **Storage** — chunks stored in Qdrant; original PDF, markdown, and metadata in MinIO

**Generation** — LangGraph chatbot with a route → retrieve → chat flow:
- Router decides whether to retrieve context based on the question
- Retriever pulls relevant chunks from Qdrant (scoped to a paper or across all papers)
- Chat node calls Groq (`llama-3.3-70b-versatile`) with context

## Services

| Service  | Port | Purpose |
|----------|------|---------|
| Frontend | 8501 | Streamlit UI |
| Backend  | 8000 | FastAPI REST API |
| Qdrant   | 6333 | Vector database |
| MinIO    | 9000 | Object storage |
| MinIO UI | 9001 | MinIO web console |

## Quickstart

### Prerequisites
- Docker and Docker Compose
- Groq API key

### 1. Configure environment

Create a `.env` file in the project root:

```env
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=your_minio_password
GROQ_API_KEY=your_groq_api_key
```

### 2. Start the stack

```bash
docker compose up -d
```

Open the UI at `http://localhost:8501`.

### Local development (without Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
python server.py

# Frontend (separate terminal)
cd frontend
pip install -r requirements.txt
streamlit run Home.py
```

When running locally, set `MINIO_ENDPOINT=localhost:9000` and `QDRANT_HOST=localhost` in your `.env`. The docker-compose overrides these with internal service names automatically.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/ingest` | Upload a PDF for ingestion |
| `GET` | `/api/v1/ingest/{paper_id}/status` | Check ingestion status |
| `GET` | `/api/v1/papers` | List all indexed papers |
| `POST` | `/api/v1/retrieve` | Retrieve relevant chunks by query |
| `POST` | `/api/v1/chat/{session_id}` | Send a chat message |
| `GET` | `/api/v1/chat/{session_id}/history` | Get conversation history |
| `DELETE` | `/api/v1/chat/{session_id}` | Clear a chat session |

Full interactive docs available at `http://localhost:8000/docs`.

## Deployment (VM / EC2)

Build and push images:

```bash
docker build -t your-repo/multimodal_rag-backend:latest ./backend
docker build -t your-repo/multimodal_rag-frontend:latest ./frontend
docker push your-repo/multimodal_rag-backend:latest
docker push your-repo/multimodal_rag-frontend:latest
```

On the VM, create a `docker-compose.yml` using the pre-built images (see `docker-compose.yml` for the full example). The VM `.env` only needs credentials — `MINIO_ENDPOINT` and `QDRANT_HOST` are handled by Docker Compose service name resolution.

**Disk space note:** Model downloads (Docling, FastEmbed) require ~2–3 GB. Use an instance with at least 20 GB of disk. On AWS, expand the EBS volume via EC2 → Volumes → Modify if needed.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for LLM inference |
| `MINIO_ROOT_USER` | Yes | MinIO admin username |
| `MINIO_ROOT_PASSWORD` | Yes | MinIO admin password |
| `MINIO_ENDPOINT` | No | MinIO host:port (default: `localhost:9000`) |
| `QDRANT_HOST` | No | Qdrant hostname (default: `localhost`) |
| `CHAT_MODEL` | No | Groq model ID (default: `llama-3.3-70b-versatile`) |
