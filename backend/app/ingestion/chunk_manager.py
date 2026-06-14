import json
from dataclasses import asdict
from app.utils.logger import get_logger

logger = get_logger("chunk_manager")
class ChunkManager:

    def combine(self, text_chunks, image_chunks, table_chunks):        
        return [*text_chunks, *image_chunks, *table_chunks]
    

    def save_chunks(self, chunks, output_path):
        data = [asdict(chunk) for chunk in chunks]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data,f,indent=2, ensure_ascii=False)
        logger.info(f"all the combined chunks are stored in {output_path} ")