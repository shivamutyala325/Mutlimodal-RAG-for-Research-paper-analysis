from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Multimodal RAG Knowledge Base",
    description=(
        "Ingest PDF documents into a multimodal vector knowledge store "
        "and retrieve grounded knowledge across text, image, and table modalities."
    ),
    version="1.0.0",
)

app.include_router(router, prefix="/api/v1")
