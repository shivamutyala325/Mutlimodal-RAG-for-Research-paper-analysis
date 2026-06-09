from dotenv import load_dotenv
load_dotenv()

from app.ingestion.parser import compute_paper_id
from app.pipeline import run_ingestion
from app.storage.minio_store import MinioStore
from app.generation.generator import Chat
from app.utils.logger import get_logger

logger = get_logger("main")

file_path = "data/document2.pdf"

minio_store = MinioStore()
paper_id = compute_paper_id(file_path)

if minio_store.object_exists(f"{paper_id}/original.pdf"):
    logger.info(f"Paper {paper_id} already indexed — skipping")
else:
    run_ingestion(file_path, paper_id, minio_store)

logger.info("Ingestion done")

Chat()