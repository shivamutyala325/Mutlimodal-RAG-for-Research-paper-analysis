import shutil
from dataclasses import asdict

from app.ingestion.parser import PDFParser
from app.ingestion.image_processor import ImageProcessor
from app.ingestion.text_chunker import TextChunker
from app.ingestion.chunk_manager import ChunkManager
from app.ingestion.table_processor import TableProcessor
from app.ingestion.embedder import Embedder
from app.storage.vector_store import VectorStore
from app.storage.minio_store import MinioStore
from app.utils.logger import get_logger

logger = get_logger("pipeline")


def run_ingestion(file_path: str, paper_id: str, minio_store: MinioStore) -> None:
    parser = PDFParser()
    result = parser.parse(file_path)
    logger.info(f"Parsed {file_path}")

    image_processor = ImageProcessor()
    image_chunks = image_processor.process(
        paper_id=result["paper_id"],
        doc=result["document"],
        images_dir=result["images_dir"],
        minio_store=minio_store,
    )
    logger.info(f"{len(image_chunks)} image chunks created")

    table_processor = TableProcessor()
    table_chunks = table_processor.process(result["paper_id"], result["document"])
    logger.info(f"{len(table_chunks)} table chunks created")

    text_chunker = TextChunker()
    text_chunks = text_chunker.process(result["paper_id"], result["document"])
    logger.info(f"{len(text_chunks)} text chunks created")

    manager = ChunkManager()
    all_chunks = manager.combine(text_chunks, image_chunks, table_chunks)

    embedder = Embedder()
    all_embeddings = embedder.embed_chunks([asdict(c) for c in all_chunks])
    logger.info(f"Generated embeddings for {len(all_embeddings)} chunks")

    vector_store = VectorStore()
    vector_store.upsert(all_embeddings)
    logger.info("Embeddings stored in Qdrant")

    minio_store.upload_file(
        f"{paper_id}/markdown/document.md",
        str(result["markdown_path"]),
        content_type="text/markdown",
    )
    minio_store.upload_file(
        f"{paper_id}/original.pdf",
        file_path,
        content_type="application/pdf",
    )
    logger.info(f"Paper {paper_id} committed to MinIO")

    shutil.rmtree(result["paper_dir"])
    logger.info(f"Cleaned up local temp dir {result['paper_dir']}")
