import uuid
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.models.schemas import ChunkRecord, ChunkType
from app.utils.logger import get_logger

logger = get_logger("table_processor")
TABULAR_PROCESSING_MODEL = os.getenv("TABULAR_PROCESSING_MODEL", "llama-3.3-70b-versatile")

_MAX_CONCURRENT = 5


class TableProcessor:

    def __init__(self, model_name=TABULAR_PROCESSING_MODEL):
        self.model_name = model_name

    def _system_prompt(self):
        return (
            "You are analyzing a table extracted from a research paper. Provide: "
            "1. What the table is about. "
            "2. Important metrics and columns. "
            "3. Best-performing entries. "
            "4. Important trends and observations. "
            "5. Information useful for semantic retrieval. "
            "Keep the summary concise but informative."
        )

    def generate_summary(self, table_markdown):
        model = ChatGroq(model=self.model_name)
        response = model.invoke([
            SystemMessage(content=self._system_prompt()),
            HumanMessage(content=table_markdown)
        ])
        return response.content

    def extract_table_markdown(self, table):
        try:
            return table.export_to_markdown()
        except Exception:
            return str(table)

    def _process_single(self, idx, table, paper_id):
        table_markdown = self.extract_table_markdown(table)
        summary = self.generate_summary(table_markdown)
        content = f"Table Summary:\n{summary}\n\nRaw Table:\n{table_markdown}"
        logger.info(f"Table chunk {idx} generated successfully")
        return ChunkRecord(
            id=str(uuid.uuid4()),
            paper_id=paper_id,
            chunk_type=ChunkType.TABLE,
            content=content,
            metadata={"table_index": idx}
        )

    def process(self, paper_id, doc):
        if not doc.tables:
            return []

        with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT) as executor:
            futures = {
                executor.submit(self._process_single, idx, table, paper_id): idx
                for idx, table in enumerate(doc.tables)
            }
            chunks = [f.result() for f in as_completed(futures)]

        logger.info(f"{len(chunks)} table chunks created")
        return chunks
