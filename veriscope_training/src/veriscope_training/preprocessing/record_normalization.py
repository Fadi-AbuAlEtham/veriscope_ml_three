from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from veriscope_training.config import DatasetSourceConfig
from veriscope_training.datasets.base import DatasetRecord
from veriscope_training.preprocessing.html_extraction import extract_html_content
from veriscope_training.preprocessing.label_mapping import normalize_record_label
from veriscope_training.preprocessing.text_processing import TextNormalizationConfig, normalize_text
from veriscope_training.preprocessing.url_normalization import normalize_url
from veriscope_training.utils.hashing import sha256_text


@dataclass
class ProcessedRecord:
    sample_id: str
    source_dataset: str
    source_original_id: str | None
    source_split: str | None
    source_label: str | int | None
    normalized_label: int | None
    label_name: str
    label_mapping_reason: str
    is_auxiliary: bool
    is_supervised: bool
    language: str | None
    timestamp: str | None
    original_url: str | None
    normalized_url: str | None
    raw_html: str | None
    extracted_text: str | None
    normalized_text: str | None
    tabular_features: dict[str, Any] | None
    modality_flags: dict[str, bool]
    url_features: dict[str, Any] = field(default_factory=dict)
    html_features: dict[str, Any] = field(default_factory=dict)
    text_features: dict[str, Any] = field(default_factory=dict)
    original_metadata: dict[str, Any] = field(default_factory=dict)
    processing_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def view_names(self) -> tuple[str, ...]:
        views: list[str] = []
        if self.modality_flags.get("has_normalized_url"):
            views.append("unified_url_dataset")
        if not self.is_auxiliary and (
            self.modality_flags.get("has_raw_html")
            or self.modality_flags.get("has_extracted_text")
            or self.modality_flags.get("has_normalized_text")
        ):
            views.append("unified_webpage_dataset")
        if self.modality_flags.get("has_tabular_features"):
            views.append("unified_tabular_dataset")
        if self.is_auxiliary and (self.modality_flags.get("has_extracted_text") or self.modality_flags.get("has_normalized_text")):
            views.append("unified_auxiliary_text_dataset")
        return tuple(views)


def normalize_dataset_record(
    record: DatasetRecord,
    *,
    source_config: DatasetSourceConfig,
    text_config: TextNormalizationConfig | None = None,
    preprocessing_config: dict[str, Any] | None = None,
) -> ProcessedRecord:
    preprocessing_config = preprocessing_config or {}
    url_config = preprocessing_config.get("url", {})
    html_config = preprocessing_config.get("html", {})
    storage_config = preprocessing_config.get("storage", {})
    text_config = text_config or TextNormalizationConfig()

    label_result = normalize_record_label(
        source_dataset=record.source,
        source_label=record.original_label,
        adapter_normalized_label=record.normalized_label,
        metadata=record.metadata,
    )
    url_result = normalize_url(
        record.url,
        suspicious_keywords=tuple(url_config.get("suspicious_keywords", ()) or ()),
    )
    html_result = extract_html_content(
        record.html,
        base_url=url_result.normalized_url or record.url,
        max_action_texts=int(html_config.get("max_action_texts", 20)),
    )

    text_origin = "adapter_text" if record.text else "html_visible_text"
    extracted_text = record.text or html_result.visible_text
    text_result = normalize_text(extracted_text, config=text_config)

    modality_flags = {
        "has_original_url": bool(url_result.original_url),
        "has_normalized_url": bool(url_result.normalized_url),
        "has_raw_html": bool(record.html),
        "has_extracted_text": bool(extracted_text),
        "has_normalized_text": bool(text_result.normalized_text),
        "has_tabular_features": bool(record.tabular_features),
        "is_auxiliary": label_result.is_auxiliary,
    }
    processing_metadata = {
        "source_modalities": record.modalities_present(),
        "text_origin": text_origin if extracted_text else None,
        "url_warnings": url_result.warnings,
        "html_warnings": html_result.warnings,
        "text_warnings": text_result.warnings,
        "raw_html_hash": sha256_text(record.html),
        "normalized_text_hash": sha256_text(text_result.normalized_text),
        "text_prefix_hash": sha256_text((text_result.normalized_text or "")[:512]),
        "label_strategy": source_config.label_strategy,
    }

    keep_raw_html = bool(storage_config.get("keep_raw_html", False))
    keep_extracted_text = bool(storage_config.get("keep_extracted_text", True))
    keep_normalized_text = bool(storage_config.get("keep_normalized_text", True))

    stored_raw_html = record.html if keep_raw_html else None
    stored_extracted_text = extracted_text if keep_extracted_text else None
    stored_normalized_text = text_result.normalized_text if keep_normalized_text else None
    processing_metadata["storage_policy"] = {
        "keep_raw_html": keep_raw_html,
        "keep_extracted_text": keep_extracted_text,
        "keep_normalized_text": keep_normalized_text,
    }

    return ProcessedRecord(
        sample_id=record.sample_id or "",
        source_dataset=record.source,
        source_original_id=record.original_id,
        source_split=record.split,
        source_label=record.original_label,
        normalized_label=label_result.normalized_label,
        label_name=label_result.label_name,
        label_mapping_reason=label_result.mapping_reason,
        is_auxiliary=label_result.is_auxiliary,
        is_supervised=label_result.is_supervised,
        language=record.language,
        timestamp=record.timestamp,
        original_url=url_result.original_url,
        normalized_url=url_result.normalized_url,
        raw_html=stored_raw_html,
        extracted_text=stored_extracted_text,
        normalized_text=stored_normalized_text,
        tabular_features=record.tabular_features,
        modality_flags=modality_flags,
        url_features=url_result.features,
        html_features=html_result.features,
        text_features=text_result.features,
        original_metadata=record.metadata,
        processing_metadata=processing_metadata,
    )


def validate_processed_record(record: ProcessedRecord) -> list[str]:
    warnings: list[str] = []
    if not record.sample_id:
        warnings.append("missing_sample_id")
    if not record.source_dataset:
        warnings.append("missing_source_dataset")
    if record.normalized_label not in (0, 1, None):
        warnings.append("invalid_normalized_label")
    if record.is_auxiliary and record.normalized_label is not None:
        warnings.append("auxiliary_record_should_not_have_binary_label")
    if record.modality_flags.get("has_tabular_features") and (
        record.extracted_text or record.raw_html
    ):
        warnings.append("tabular_record_contains_webpage_fields")
    if not any(record.modality_flags.values()):
        warnings.append("record_has_no_modalities")
    if record.modality_flags.get("has_normalized_url") and not record.url_features.get("hostname"):
        warnings.append("normalized_url_without_hostname")
    return warnings
