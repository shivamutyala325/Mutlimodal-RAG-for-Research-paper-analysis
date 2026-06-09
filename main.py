import shutil
from dataclasses import asdict
from app.ingestion.parser import PDFParser, compute_paper_id
from app.ingestion.image_processor import ImageProcessor
from app.ingestion.text_chunker import TextChunker
from app.ingestion.chunk_manager import ChunkManager
from app.ingestion.table_processor import TableProcessor
from app.ingestion.embedder import Embedder
from app.storage.vector_store import VectorStore
from app.storage.minio_store import MinioStore
from app.generation.generator import Chat
from app.utils.logger import get_logger


logger = get_logger("main")

file_path = "data/document2.pdf"

#--------------------------Early dedup guard------------------

minio_store = MinioStore()
paper_id = compute_paper_id(file_path)

if minio_store.object_exists(f"{paper_id}/original.pdf"):
    logger.info(f"Paper {paper_id} already indexed — skipping all processing")
else:
    #-----------------------Parsing---------------------------
    parser = PDFParser()
    result = parser.parse(file_path)
    logger.info(f"parsed {file_path}")

    #--------------------image processing---------------------
    image_processor = ImageProcessor()
    image_chunks = image_processor.process(
        paper_id=result["paper_id"],
        doc=result["document"],
        images_dir=result["images_dir"],
        minio_store=minio_store,
    )
    logger.info(f"Generated {len(image_chunks)} image chunks")
      

    #-----------------------Table processing------------------
    table_processor = TableProcessor()
    table_chunks = table_processor.process(
        result["paper_id"],
        result["document"]
    )
    logger.info(f"In total {len(table_chunks)} table chunks are created")



    #------------------------Text processing------------------
    text_chunker = TextChunker()
    text_chunks = text_chunker.process(
        result["paper_id"],
        result["document"]
    )
    logger.info(f"In total {len(text_chunks)} text chunks are created")

    manager = ChunkManager()
    all_chunks = manager.combine(text_chunks, image_chunks, table_chunks)

    chunk_file = result["paper_dir"] / "all_chunks.json"
    manager.save_chunks(all_chunks, chunk_file)
    logger.info(f"all chunks are saved to {chunk_file}")

    #--------------------------Embeddings---------------------
    embedder = Embedder()
    all_chunks_dicts = [asdict(c) for c in all_chunks]
    all_embeddings = embedder.embed_chunks(all_chunks_dicts)
    logger.info(f"Generated embeddings for {len(all_embeddings)} chunks")

    #--------------------------Vector Store-------------------
    vector_store = VectorStore()
    vector_store.upsert(all_embeddings)
    logger.info("All embeddings stored in Qdrant")

    #--------------------------MinIO commit-------------------
    minio_store.upload_file(
        f"{paper_id}/markdown/document.md",
        str(result["markdown_path"]),
        content_type="text/markdown",
    )
    # Upload PDF last — acts as the completion marker for dedup
    minio_store.upload_file(
        f"{paper_id}/original.pdf",
        file_path,
        content_type="application/pdf",
    )
    logger.info(f"Paper {paper_id} committed to MinIO")

    #--------------------------Cleanup------------------------
    shutil.rmtree(result["paper_dir"])
    logger.info(f"Removed local temp directory {result['paper_dir']}")

logger.info("main.py is done")


#-----------------------try chat-----------------------------


Chat()