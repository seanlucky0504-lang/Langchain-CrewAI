from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import pytesseract
from langchain.docstore.document import Document
from pdfminer.high_level import extract_text
from PIL import Image


@dataclass
class ParsedModal:
    """Structured representation of multi-modal content."""

    content: str
    modality: str
    metadata: Dict[str, Any]


def parse_pdf(content: bytes, source: str) -> ParsedModal:
    text = extract_text(io.BytesIO(content))
    return ParsedModal(content=text, modality="pdf", metadata={"source": source})


def parse_table(content: bytes, source: str) -> ParsedModal:
    df = pd.read_excel(io.BytesIO(content))
    return ParsedModal(content=df.to_markdown(index=False), modality="table", metadata={"source": source})


def parse_image(content: bytes, source: str) -> ParsedModal:
    with Image.open(io.BytesIO(content)) as img:
        ocr_text = pytesseract.image_to_string(img)
    return ParsedModal(content=ocr_text, modality="image", metadata={"source": source, "height": img.height, "width": img.width})


def parse_text(content: str, source: str) -> ParsedModal:
    return ParsedModal(content=content, modality="text", metadata={"source": source})


def to_documents(parsed: List[ParsedModal]) -> List[Document]:
    return [Document(page_content=item.content, metadata={"modality": item.modality, **item.metadata}) for item in parsed]
