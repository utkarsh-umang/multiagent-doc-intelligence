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
from pydantic import BaseModel, Field


SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
DEFAULT_MODEL = os.getenv(
    "EXTRACTOR_MODEL",
    os.getenv("LITELLM_VISION_MODEL", os.getenv("OPENAI_VISION_MODEL", "gpt-4o")),
)
DEFAULT_RETRY_MODEL = os.getenv(
    "EXTRACTOR_RETRY_MODEL",
    os.getenv("LITELLM_VISION_RETRY_MODEL", DEFAULT_MODEL),
)
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_FIELD_NAMES = [
    "consignee_name",
    "hs_code",
    "port_of_loading",
    "port_of_discharge",
    "incoterms",
    "description_of_goods",
    "gross_weight",
    "invoice_number",
]


class FieldSchema(BaseModel):
    """One field the extractor should find in the document."""

    name: str
    description: str | None = None
    required: bool = True
    aliases: list[str] = Field(default_factory=list)


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


class ExtractionMetadata(BaseModel):
    """Auditable metadata for extraction and retry behavior."""

    model: str
    retry_model: str
    confidence_threshold: float
    field_schema: list[FieldSchema]
    initial_low_confidence_fields: list[str] = Field(default_factory=list)
    retried_fields: list[str] = Field(default_factory=list)
    persistent_low_confidence_fields: list[str] = Field(default_factory=list)
    attempts_by_field: dict[str, int] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Final extraction output plus metadata for graph state."""

    fields: dict[str, ExtractedField]
    metadata: ExtractionMetadata


EXTRACTION_PROMPT_TEMPLATE = """
You are an extractor for international trade and logistics documents.

Extract the required fields from the provided document image(s). Return JSON only.
Each field must be an object with:
- value: the extracted value as a string, or null if absent
- confidence: a number between 0 and 1
- source_snippet: exact visible text or nearby evidence from the document

Required JSON keys:
{field_list}

Guidance:
- Prefer exact text from the document over normalized guesses.
- Use common synonyms: consignee/receiver/buyer, shipper/consignor, POL/loading port,
  POD/discharge port, HS/HTS/tariff code, incoterm/trade term.
- If a field is missing, set value to null and confidence to 0.0.
- If the field is ambiguous or inferred, lower the confidence and explain the evidence
  in source_snippet.
- Do not return high confidence unless the value is grounded by source_snippet.
"""

RETRY_PROMPT_TEMPLATE = """
You are retrying only low-confidence fields from an international trade document.

Return JSON only for these fields:
{field_list}

For each field, return:
- value: the extracted value as a string, or null if absent
- confidence: a number between 0 and 1
- source_snippet: exact visible text or nearby evidence from the document

Use the document image(s) as the only source of truth. If the value is not clearly
grounded in visible text, keep confidence below {confidence_threshold}.
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


def _validate_model(model_class: type[BaseModel], data: dict[str, Any]) -> BaseModel:
    try:
        return model_class.model_validate(data)
    except AttributeError:
        return model_class.parse_obj(data)


def _default_field_schema() -> list[FieldSchema]:
    return [FieldSchema(name=field_name) for field_name in DEFAULT_FIELD_NAMES]


def _coerce_field_schema(field_schema: list[dict[str, Any]] | None) -> list[FieldSchema]:
    if not field_schema:
        return _default_field_schema()

    fields: list[FieldSchema] = []
    for payload in field_schema:
        field = _validate_model(FieldSchema, payload)
        if not isinstance(field, FieldSchema):
            raise TypeError(f"Invalid field schema payload: {payload}")
        fields.append(field)
    return fields


def _field_list_for_prompt(field_schema: list[FieldSchema]) -> str:
    lines: list[str] = []
    for field in field_schema:
        details = []
        if field.description:
            details.append(field.description)
        if field.aliases:
            details.append(f"aliases: {', '.join(field.aliases)}")
        suffix = f" ({'; '.join(details)})" if details else ""
        lines.append(f"- {field.name}{suffix}")
    return "\n".join(lines)


def _build_extraction_prompt(field_schema: list[FieldSchema]) -> str:
    return EXTRACTION_PROMPT_TEMPLATE.format(field_list=_field_list_for_prompt(field_schema))


def _build_retry_prompt(field_schema: list[FieldSchema], confidence_threshold: float) -> str:
    return RETRY_PROMPT_TEMPLATE.format(
        field_list=_field_list_for_prompt(field_schema),
        confidence_threshold=f"{confidence_threshold:.2f}",
    )


def _missing_field() -> ExtractedField:
    return ExtractedField(value=None, confidence=0.0, source_snippet=None)


def _ground_field(field: ExtractedField, confidence_threshold: float) -> ExtractedField:
    if field.value and field.confidence >= confidence_threshold and not field.source_snippet:
        return ExtractedField(
            value=field.value,
            confidence=max(0.0, confidence_threshold - 0.01),
            source_snippet=None,
        )
    return field


class ExtractorAgent:
    """Vision-LLM backed document extractor."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        retry_model: str = DEFAULT_RETRY_MODEL,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        completion_fn: Any | None = None,
    ) -> None:
        self.model = model
        self.retry_model = retry_model
        self.confidence_threshold = confidence_threshold
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

    def _call_model(
        self,
        *,
        model: str,
        prompt: str,
        image_urls: list[str],
    ) -> dict[str, Any]:
        message_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        message_content.extend(
            {"type": "image_url", "image_url": {"url": image_url}}
            for image_url in image_urls
        )

        response = self.completion_fn(
            model=model,
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
        return _parse_model_json(content)

    def _parse_fields(
        self,
        data: dict[str, Any],
        field_schema: list[FieldSchema],
    ) -> dict[str, ExtractedField]:
        fields: dict[str, ExtractedField] = {}
        for field in field_schema:
            raw_value = data.get(field.name)
            if raw_value is None:
                fields[field.name] = _missing_field()
                continue
            parsed = _validate_model(ExtractedField, raw_value)
            if not isinstance(parsed, ExtractedField):
                raise TypeError(f"Invalid extracted field payload for {field.name}")
            fields[field.name] = _ground_field(parsed, self.confidence_threshold)
        return fields

    def extract(
        self,
        input_path: str,
        *,
        field_schema: list[dict[str, Any]] | None = None,
        max_pages: int = 4,
    ) -> ExtractionResult:
        image_urls = _document_to_image_urls(input_path, max_pages=max_pages)
        schema = _coerce_field_schema(field_schema)
        prompt = _build_extraction_prompt(schema)
        first_pass = self._parse_fields(
            self._call_model(model=self.model, prompt=prompt, image_urls=image_urls),
            schema,
        )

        low_confidence = [
            field.name
            for field in schema
            if first_pass[field.name].confidence < self.confidence_threshold
        ]
        final_fields = dict(first_pass)
        attempts_by_field = {field.name: 1 for field in schema}

        if low_confidence:
            retry_schema = [field for field in schema if field.name in low_confidence]
            retry_prompt = _build_retry_prompt(retry_schema, self.confidence_threshold)
            retry_pass = self._parse_fields(
                self._call_model(
                    model=self.retry_model,
                    prompt=retry_prompt,
                    image_urls=image_urls,
                ),
                retry_schema,
            )
            for field_name, retry_field in retry_pass.items():
                attempts_by_field[field_name] = 2
                if retry_field.confidence >= final_fields[field_name].confidence:
                    final_fields[field_name] = retry_field

        persistent_low_confidence = [
            field_name
            for field_name, field in final_fields.items()
            if field.confidence < self.confidence_threshold
        ]

        return ExtractionResult(
            fields=final_fields,
            metadata=ExtractionMetadata(
                model=self.model,
                retry_model=self.retry_model,
                confidence_threshold=self.confidence_threshold,
                field_schema=schema,
                initial_low_confidence_fields=low_confidence,
                retried_fields=low_confidence,
                persistent_low_confidence_fields=persistent_low_confidence,
                attempts_by_field=attempts_by_field,
            ),
        )


def _to_plain_dict(model: BaseModel) -> dict[str, Any]:
    try:
        return model.model_dump()
    except AttributeError:
        return model.dict()


def run(
    input_path: str,
    field_schema: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract required trade-document fields as structured JSON."""

    extraction = ExtractorAgent().extract(input_path, field_schema=field_schema)
    return _to_plain_dict(extraction)

