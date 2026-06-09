import uuid
from pathlib import Path
import base64
import os
from dotenv import load_dotenv
from app.utils.logger import get_logger
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()
IMAGE_PROCESSING_MODEL = os.getenv("IMAGE_PROCESSING_MODEL")
logger = get_logger("image_processor")

from app.models.schemas import (
    ChunkRecord,
    ChunkType
)

class ImageProcessor:

    def __init__(self, model_name: str = IMAGE_PROCESSING_MODEL):
        self.model_name = model_name


    def _system_prompt(self):
        return """
    You are analyzing a figure extracted from a research paper.
    Describe:
    1. What this figure represents.
    2. The important components.
    3. Labels, axes, legends, and annotations.
    4. Relationships between components.
    5. Key technical concepts shown.
    6. Insights that would help answer questions about this figure.
    Write a detailed description optimized for semantic retrieval in a RAG system.
    """
    
    def image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    
    def generate_description(self, image_path):

        image_description_model = ChatOllama(model=self.model_name)
        image_base64 = self.image_to_base64(image_path)

        response = image_description_model.invoke(
            [
                SystemMessage(content=self._system_prompt()),
                HumanMessage(content=[
                    {"type": "text", "text": "Describe this figure from a research paper."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ])
            ]
        )

        return response.content

    

    def build_picture_map(self, doc, images_dir: Path):
        picture_map = {}

        image_files = sorted( images_dir.glob("*.png"))

        for pic, image_file in zip(doc.pictures, image_files):
            picture_map[pic.self_ref] = image_file

        logger.info(f"{len(picture_map)} pictures are mapped with references sucessfully")

        return picture_map
    
    def extract_caption(self, picture, doc) -> str:
        captions = []
        for cap in getattr(picture, "captions", []):
            cref = getattr(cap, "cref", "")
            if "/texts/" in cref:
                try:
                    idx = int(cref.split("/texts/")[-1])
                    captions.append(doc.texts[idx].text)
                except (IndexError, AttributeError, ValueError):
                    pass
        return " ".join(captions)

    def process(self, paper_id: str, doc, images_dir: Path, minio_store=None):
        image_chunks = []

        picture_map = self.build_picture_map(doc, images_dir)
        c = 0

        for picture in doc.pictures:
            c += 1
            image_path = picture_map.get(picture.self_ref)

            if image_path is None:
                logger.info(f"no image path found :{image_path}")
                continue

            description = self.generate_description(str(image_path))
            logger.info(f"descriptions for {c}th image is generated")

            caption = self.extract_caption(picture, doc)
            logger.info(f"caption for {c}th image is extracted")

            object_key = f"{paper_id}/images/{image_path.name}"
            if minio_store:
                minio_store.upload_file(object_key, str(image_path), content_type="image/png")
                logger.info(f"image {c} uploaded to MinIO at {object_key}")

            chunk = ChunkRecord(
                id=str(uuid.uuid4()),
                paper_id=paper_id,
                chunk_type=ChunkType.IMAGE,
                content=description,
                metadata={
                    "image_path": object_key if minio_store else str(image_path),
                    "caption": caption,
                    "picture_ref": picture.self_ref,
                }
            )

            logger.info(f"{c}th chunk is generated")
            image_chunks.append(chunk)

        return image_chunks