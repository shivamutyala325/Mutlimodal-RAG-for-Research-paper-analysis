import shutil
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

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


def run_ingestion(file_path: str, paper_id: str, minio_store: MinioStore, original_filename: str = "") -> None:
    parser = PDFParser()
    result = parser.parse(file_path, original_filename=original_filename)
    logger.info(f"Parsed {file_path}")

    image_processor = ImageProcessor()
    table_processor = TableProcessor()
    text_chunker = TextChunker()

    # Image descriptions, table summaries, and text chunking are fully independent
    # — run all three concurrently to eliminate sequential Groq API wait time
    with ThreadPoolExecutor(max_workers=3) as executor:
        image_future = executor.submit(
            image_processor.process,
            result["paper_id"], result["document"], result["images_dir"], minio_store
        )
        table_future = executor.submit(
            table_processor.process,
            result["paper_id"], result["document"]
        )
        text_future = executor.submit(
            text_chunker.process,
            result["paper_id"], result["document"]
        )
        wait([image_future, table_future, text_future], return_when=ALL_COMPLETED)

    image_chunks = image_future.result()
    table_chunks = table_future.result()
    text_chunks  = text_future.result()

    logger.info(f"{len(image_chunks)} image | {len(table_chunks)} table | {len(text_chunks)} text chunks")

    manager = ChunkManager()
    all_chunks = manager.combine(text_chunks, image_chunks, table_chunks)

    embedder = Embedder()
    all_embeddings = embedder.embed_chunks([asdict(c) for c in all_chunks])
    logger.info(f"Generated embeddings for {len(all_embeddings)} chunks")

    vector_store = VectorStore()
    vector_store.upsert(all_embeddings)
    logger.info("Embeddings stored in Qdrant")

    minio_store.upload_file(
        f"{paper_id}/document_metadata.json",
        str(result["metadata_path"]),
        content_type="application/json",
    )
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
