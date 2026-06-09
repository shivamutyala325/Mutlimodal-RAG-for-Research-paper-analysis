import uuid
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from app.models.schemas import ChunkRecord, ChunkType
from app.utils.logger import get_logger
import os

logger = get_logger("table_processor")
TABULAR_PROCESSING_MODEL = os.getenv("TABULAR_PROCESSING_MODEL")
class TableProcessor:
    
    def __init__(self, model_name=TABULAR_PROCESSING_MODEL):
        self.model_name = model_name
        self.counter = 0 # counter for tracking the number of table chunks generated

    def _system_prompt(self):
        return """
            You are analyzing a table extracted from a research paper.

            Provide:
            1. What the table is about.
            2. Important metrics and columns.
            3. Best-performing entries.
            4. Important trends and observations.
            5. Information useful for semantic retrieval.

            Keep the summary concise but informative.
            """

    def generate_summary(self, table_markdown):

        table_summary_llm = ChatOllama(model = self.model_name)

        response = table_summary_llm.invoke(
            [
                SystemMessage(content=self._system_prompt()),
                HumanMessage(content = table_markdown)

            ]
        )

        return response.content

    def extract_table_markdown(self, table):
        try:
            return table.export_to_markdown()
        except Exception:
            return str(table)

    def process(self, paper_id, doc):
        table_chunks = []

        for idx, table in enumerate(doc.tables):
            self.counter+=1
            table_markdown = self.extract_table_markdown(table)
            summary = self.generate_summary(table_markdown)

            content = f"""
                    Table Summary:
                    {summary}

                    Raw Table:
                    {table_markdown}
                    """

            chunk = ChunkRecord(
                id=str(uuid.uuid4()),
                paper_id=paper_id,
                chunk_type=ChunkType.TABLE,
                content=content,
                metadata={
                    "table_index": idx
                }
            )
            table_chunks.append(chunk)
            logger.info(f"{self.counter}th table chunks is generated sucessfully")

        return table_chunks