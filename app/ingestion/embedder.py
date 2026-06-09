from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv
from app.utils.logger import get_logger


logger = get_logger("embedder")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


class Embedder:
    def __init__(self, batch_size=16):
        self.model_name = EMBEDDING_MODEL
        self.batch_size = batch_size
        self.batch_number = 0 # counter for tracking number of batches

    def _batch(self, items):
        for i in range(0, len(items), self.batch_size):
            self.batch_number+=1
            yield items[i:i+self.batch_size]

    def embed_texts(self,texts):
        model = OllamaEmbeddings(model = self.model_name)
        embeddings = []
        for batch in self._batch(texts):
            response = model.embed_documents(batch)

            embeddings.extend(response)
        logger.info(f"embeddings for {self.batch_number}th batch is done")
        return embeddings

    def embed_chunks(self, chunks):
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        embedded_records = []

        for chunk, embedding in zip(chunks, embeddings):
            embedded_records.append(
                {
                    "chunk": chunk,
                    "embedding": embedding
                }
            )

        return embedded_records
    


