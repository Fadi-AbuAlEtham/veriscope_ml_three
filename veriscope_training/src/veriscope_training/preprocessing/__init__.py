"""Preprocessing and canonical record normalization for VeriScope datasets."""

from veriscope_training.preprocessing.deduplication import ProcessedDeduplicator
from veriscope_training.preprocessing.html_extraction import extract_html_content
from veriscope_training.preprocessing.label_mapping import normalize_record_label
from veriscope_training.preprocessing.record_normalization import ProcessedRecord, normalize_dataset_record
from veriscope_training.preprocessing.text_processing import normalize_text
from veriscope_training.preprocessing.url_normalization import normalize_url

__all__ = [
    "ProcessedDeduplicator",
    "ProcessedRecord",
    "extract_html_content",
    "normalize_dataset_record",
    "normalize_record_label",
    "normalize_text",
    "normalize_url",
]
