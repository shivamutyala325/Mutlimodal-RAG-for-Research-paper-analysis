import os
from fastembed import TextEmbedding
from app.utils.logger import get_logger

logger = get_logger("embedder")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


class Embedder:
    def __init__(self, batch_size=96):
        self.model_name = EMBEDDING_MODEL
        self.batch_size = batch_size
        self.batch_number = 0

    def _batch(self, items):
        for i in range(0, len(items), self.batch_size):
            self.batch_number += 1
            yield items[i:i + self.batch_size]

    def embed_texts(self, texts):
        model = TextEmbedding(model_name=self.model_name)
        embeddings = []
        for batch in self._batch(texts):
            embeddings.extend([emb.tolist() for emb in model.embed(batch)])
            logger.info(f"Batch {self.batch_number} embedded ({len(batch)} texts)")
        return embeddings

    def embed_chunks(self, chunks):
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embed_texts(texts)
        return [{"chunk": chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]