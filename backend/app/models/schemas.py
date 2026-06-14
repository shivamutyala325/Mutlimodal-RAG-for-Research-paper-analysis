from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChunkType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class ChunkRecord:
    id: str
    paper_id: str

    chunk_type: ChunkType

    content: str

    metadata: dict[str, Any] = field(default_factory=dict)


