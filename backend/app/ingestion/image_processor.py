import uuid
import base64
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.utils.logger import get_logger
from app.models.schemas import ChunkRecord, ChunkType

IMAGE_PROCESSING_MODEL = os.getenv("IMAGE_PROCESSING_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
logger = get_logger("image_processor")

# Cap concurrent Groq vision calls to avoid rate limit errors
_MAX_CONCURRENT = 5


class ImageProcessor:

    def __init__(self, model_name: str = IMAGE_PROCESSING_MODEL):
        self.model_name = model_name

    def _system_prompt(self):
        return (
            "You are analyzing a figure extracted from a research paper. Describe: "
            "1. What this figure represents. "
            "2. The important components. "
            "3. Labels, axes, legends, and annotations. "
            "4. Relationships between components. "
            "5. Key technical concepts shown. "
            "6. Insights that would help answer questions about this figure. "
            "Write a detailed description optimized for semantic retrieval in a RAG system."
        )

    def image_to_base64(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate_description(self, image_path):
        model = ChatGroq(model=self.model_name)
        image_base64 = self.image_to_base64(image_path)
        response = model.invoke([
            SystemMessage(content=self._system_prompt()),
            HumanMessage(content=[
                {"type": "text", "text": "Describe this figure from a research paper."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ])
        ])
        return response.content

    def build_picture_map(self, doc, images_dir: Path):
        picture_map = {}
        for pic, image_file in zip(doc.pictures, sorted(images_dir.glob("*.png"))):
            picture_map[pic.self_ref] = image_file
        logger.info(f"{len(picture_map)} pictures mapped successfully")
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

    def _process_single(self, c, picture, picture_map, paper_id, doc, minio_store):
        image_path = picture_map.get(picture.self_ref)
        if image_path is None:
            logger.warning(f"No image file found for picture {c}")
            return None

        description = self.generate_description(str(image_path))
        caption = self.extract_caption(picture, doc)

        object_key = f"{paper_id}/images/{image_path.name}"
        if minio_store:
            minio_store.upload_file(object_key, str(image_path), content_type="image/png")

        logger.info(f"Image chunk {c} created")
        return ChunkRecord(
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

    def process(self, paper_id: str, doc, images_dir: Path, minio_store=None):
        picture_map = self.build_picture_map(doc, images_dir)

        if not doc.pictures:
            return []

        with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT) as executor:
            futures = {
                executor.submit(self._process_single, c, picture, picture_map, paper_id, doc, minio_store): c
                for c, picture in enumerate(doc.pictures, start=1)
            }
            chunks = [f.result() for f in as_completed(futures) if f.result() is not None]

        logger.info(f"{len(chunks)} image chunks created")
        return chunks
