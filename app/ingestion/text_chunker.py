import uuid
from docling.chunking import HybridChunker
from app.models.schemas import ChunkRecord,ChunkType
from app.utils.logger import get_logger

logger = get_logger("text_chunker")


class TextChunker:
    def __init__(self):
        self.chunker = HybridChunker()
        self.counter = 0  # counter for tracing the number of chunks

    def build_metadata(self, chunk):
        metadata = {}
        metadata["headings"] = (chunk.meta.headings or [])
        metadata["doc_items"] = [item.self_ref for item in chunk.meta.doc_items]

        
        return metadata

    def process(self, paper_id, doc):
        text_chunks = []

        for chunk in self.chunker.chunk(doc):
            self.counter+=1
            text = self.chunker.contextualize(chunk)
            metadata = self.build_metadata(chunk)
            record = ChunkRecord(
                id=str(uuid.uuid4()),
                paper_id=paper_id,
                chunk_type=ChunkType.TEXT,
                content=text,
                metadata=metadata
            )
            logger.info(f"{self.counter}th text_chunk is created")
            text_chunks.append(record)

        return text_chunks