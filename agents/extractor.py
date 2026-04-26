"""
Extractor agent for trade documents.

Accepts a PDF or image, sends rendered pages to a vision-capable LLM, and returns
structured field-level extractions with confidence scores.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import litellm  # type: ignore[import-not-found]
from pydantic import BaseModel, Field, ValidationError


SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
DEFAULT_MODEL = os.getenv("LITELLM_VISION_MODEL", os.getenv("OPENAI_VISION_MODEL", "gpt-4o"))


class ExtractedField(BaseModel):
    """A single extracted document field."""

    value: str | None = Field(
        default=None,
        description="Extracted value, or null when the field is not present.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence from 0.0 to 1.0.",
    )
    source_snippet: str | None = Field(
        default=None,
        description="Exact visible text or nearby evidence used for this extraction.",
    )


class TradeDocumentExtraction(BaseModel):
    """Structured extraction schema required by the assignment."""

    consignee_name: ExtractedField
    hs_code: ExtractedField
    port_of_loading: ExtractedField
    port_of_discharge: ExtractedField
    incoterms: ExtractedField
    description_of_goods: ExtractedField
    gross_weight: ExtractedField
    invoice_number: ExtractedField


EXTRACTION_PROMPT = """
You are an extractor for international trade and logistics documents.

Extract the required fields from the provided document image(s). Return JSON only.
Each field must be an object with:
- value: the extracted value as a string, or null if absent
- confidence: a number between 0 and 1
- source_snippet: exact visible text or nearby evidence from the document

Required JSON keys:
- consignee_name
- hs_code
- port_of_loading
- port_of_discharge
- incoterms
- description_of_goods
- gross_weight
- invoice_number

Guidance:
- Prefer exact text from the document over normalized guesses.
- Use common synonyms: consignee/receiver/buyer, shipper/consignor, POL/loading port,
  POD/discharge port, HS/HTS/tariff code, incoterm/trade term.
- If a field is missing, set value to null and confidence to 0.0.
- If the field is ambiguous or inferred, lower the confidence and explain the evidence
  in source_snippet.
"""


def _encode_file_as_data_url(path: Path, mime_type: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _render_pdf_pages(path: Path, max_pages: int) -> list[str]:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PDF extraction requires PyMuPDF. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    data_urls: list[str] = []
    with fitz.open(path) as doc:
        for page in doc[:max_pages]:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            encoded = base64.b64encode(pixmap.tobytes("png")).decode("utf-8")
            data_urls.append(f"data:image/png;base64,{encoded}")

    return data_urls


def _document_to_image_urls(input_path: str, max_pages: int) -> list[str]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {input_path}")

    mime_type, _ = mimetypes.guess_type(path)
    if mime_type == "application/pdf":
        return _render_pdf_pages(path, max_pages=max_pages)

    if mime_type in SUPPORTED_IMAGE_MIME_TYPES:
        return [_encode_file_as_data_url(path, mime_type)]

    raise ValueError(
        f"Unsupported document type {mime_type or 'unknown'} for {input_path}. "
        "Use PDF, JPG, PNG, or WEBP."
    )


def _parse_model_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Extractor model returned invalid JSON: {content}") from exc


class ExtractorAgent:
    """Vision-LLM backed document extractor."""

    def __init__(self, model: str = DEFAULT_MODEL, completion_fn: Any | None = None) -> None:
        self.model = model
        self.completion_fn = completion_fn or litellm.completion
        self._configure_langfuse_callbacks()

    def _configure_langfuse_callbacks(self) -> None:
        if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
            return

        callbacks = list(getattr(litellm, "success_callback", []) or [])
        failure_callbacks = list(getattr(litellm, "failure_callback", []) or [])
        if "langfuse" not in callbacks:
            callbacks.append("langfuse")
        if "langfuse" not in failure_callbacks:
            failure_callbacks.append("langfuse")
        litellm.success_callback = callbacks
        litellm.failure_callback = failure_callbacks

    def extract(self, input_path: str, max_pages: int = 4) -> TradeDocumentExtraction:
        image_urls = _document_to_image_urls(input_path, max_pages=max_pages)
        message_content: list[dict[str, Any]] = [{"type": "text", "text": EXTRACTION_PROMPT}]
        message_content.extend(
            {"type": "image_url", "image_url": {"url": image_url}}
            for image_url in image_urls
        )

        response = self.completion_fn(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": message_content,
                }
            ],
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Extractor model returned an empty response")

        data = _parse_model_json(content)
        try:
            return TradeDocumentExtraction.model_validate(data)
        except AttributeError:
            return TradeDocumentExtraction.parse_obj(data)
        except ValidationError:
            raise


def _to_plain_dict(model: BaseModel) -> dict[str, Any]:
    try:
        return model.model_dump()
    except AttributeError:
        return model.dict()


def run(input_path: str) -> dict[str, Any]:
    """Extract required trade-document fields as structured JSON."""

    extraction = ExtractorAgent().extract(input_path)
    return _to_plain_dict(extraction)

