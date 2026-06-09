import hashlib
import json
import uuid
from pathlib import Path
from app.utils.logger import get_logger
from docling.document_converter import DocumentConverter,PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode

logger = get_logger("parser")


def compute_paper_id(pdf_path: str) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, h.hexdigest()))


class PDFParser:
    def __init__(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 2.0
        pipeline_options.do_formula_enrichment = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def parse(self, pdf_path: str, storage_root: str = "storage"):

        paper_id = compute_paper_id(pdf_path)
        paper_dir = (Path(storage_root)/ "papers"/ paper_id)

        markdown_dir = paper_dir / "markdown"
        images_dir = paper_dir / "images"

        markdown_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        images_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        result = self.converter.convert(pdf_path)

        doc = result.document

        markdown_path = (markdown_dir/"document.md")

        doc.save_as_markdown(
            markdown_path,
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=Path("../images")
        )

        metadata = {
            "paper_id": paper_id,
            "source_file": pdf_path,
            "num_tables": len(doc.tables),
            "num_images": len(doc.pictures),
            "num_text_items": len(doc.texts)
        }

        metadata_path = (paper_dir/"document_metadata.json")

        with open(metadata_path, "w",encoding="utf-8") as f:
            json.dump(metadata, f,indent=4)

        logger.info(f" metadata for {pdf_path} is saved at {metadata_path}")
        return {
            "paper_id": paper_id,
            "document": doc,
            "paper_dir": paper_dir,
            "markdown_path": markdown_path,
            "images_dir": images_dir,
            "metadata_path": metadata_path
        }