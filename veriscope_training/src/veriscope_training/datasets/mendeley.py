from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from veriscope_training.datasets.base import DatasetAdapter, DatasetRecord
from veriscope_training.datasets.loaders import (
    HTML_FIELD_CANDIDATES,
    ID_FIELD_CANDIDATES,
    LABEL_FIELD_CANDIDATES,
    LANGUAGE_FIELD_CANDIDATES,
    SPLIT_FIELD_CANDIDATES,
    TEXT_FIELD_CANDIDATES,
    TIMESTAMP_FIELD_CANDIDATES,
    URL_FIELD_CANDIDATES,
    clean_text,
    exclude_fields,
    infer_label_from_path,
    infer_split_from_path,
    iter_structured_rows,
    looks_like_html,
    normalize_binary_label,
    open_text,
    pick_first,
    primary_suffix,
)
from veriscope_training.datasets.registry import register_dataset


@register_dataset("mendeley")
class MendeleyAdapter(DatasetAdapter):
    supported_formats = ("csv", "tsv", "jsonl", "json", "parquet", "html", "htm", "txt")
    source_type = "local_snapshot"

    def validation_report(self) -> dict[str, Any]:
        payload = super().validation_report()
        supported_paths = self._supported_paths()
        payload["supported_file_count"] = len(supported_paths)
        if not supported_paths:
            payload["issues"].append(
                "Expected structured snapshots (.csv/.jsonl/.parquet/...) or labeled raw .html/.txt files."
            )
        return payload

    def iterate_records(self) -> Iterator[DatasetRecord]:
        supported_paths = self._supported_paths()
        if not supported_paths:
            raise FileNotFoundError(
                f"Mendeley adapter could not find supported snapshots under {self.snapshot_root}."
            )
        for path in supported_paths:
            suffix = primary_suffix(path)
            if suffix in {".html", ".htm", ".txt"}:
                record = self._record_from_raw_file(path)
                if record is not None:
                    yield record
            else:
                split_hint = infer_split_from_path(path)
                for row_index, row in enumerate(iter_structured_rows(path), start=1):
                    record = self._record_from_mapping(row, path=path, row_index=row_index, split_hint=split_hint)
                    if record is not None:
                        yield record

    def _supported_paths(self) -> list[Path]:
        return [
            path
            for path in self.snapshot_paths()
            if primary_suffix(path) in {".csv", ".tsv", ".jsonl", ".json", ".parquet", ".html", ".htm", ".txt"}
        ]

    def _record_from_mapping(
        self,
        row: dict[str, Any],
        *,
        path: Path,
        row_index: int,
        split_hint: str | None,
    ) -> DatasetRecord | None:
        url = clean_text(pick_first(row, URL_FIELD_CANDIDATES))
        html = clean_text(pick_first(row, HTML_FIELD_CANDIDATES))
        text = clean_text(pick_first(row, TEXT_FIELD_CANDIDATES))
        if html is None and text and looks_like_html(text):
            html = text
            text = None
        original_label = pick_first(row, LABEL_FIELD_CANDIDATES)
        normalized_label = self._normalize_label(original_label)
        if url is None and html is None and text is None:
            return None
        metadata = exclude_fields(
            row,
            excluded=(
                *URL_FIELD_CANDIDATES,
                *HTML_FIELD_CANDIDATES,
                *TEXT_FIELD_CANDIDATES,
                *LABEL_FIELD_CANDIDATES,
                *ID_FIELD_CANDIDATES,
                *SPLIT_FIELD_CANDIDATES,
                *TIMESTAMP_FIELD_CANDIDATES,
                *LANGUAGE_FIELD_CANDIDATES,
            ),
        )
        metadata["source_path"] = str(path)
        metadata["source_format"] = primary_suffix(path).lstrip(".")
        metadata["label_mapping_status"] = "mapped" if normalized_label is not None else "unmapped_or_uncertain"
        return DatasetRecord(
            source=self.name,
            original_id=clean_text(pick_first(row, ID_FIELD_CANDIDATES)) or f"{path.name}:{row_index}",
            original_label=clean_text(original_label),
            normalized_label=normalized_label,
            url=url,
            html=html,
            text=text if text != html else None,
            timestamp=clean_text(pick_first(row, TIMESTAMP_FIELD_CANDIDATES)),
            language=clean_text(pick_first(row, LANGUAGE_FIELD_CANDIDATES)),
            split=clean_text(pick_first(row, SPLIT_FIELD_CANDIDATES)) or split_hint,
            metadata=metadata,
        )

    def _record_from_raw_file(self, path: Path) -> DatasetRecord | None:
        suffix = primary_suffix(path)
        with open_text(path) as handle:
            content = handle.read()
        payload = clean_text(content)
        if payload is None:
            return None
        html = payload if suffix in {".html", ".htm"} or looks_like_html(payload) else None
        text = payload if html is None else None
        label = infer_label_from_path(path)
        normalized_label = normalize_binary_label(
            label,
            phishing_values={"phishing"},
            benign_values={"benign"},
            uncertain_values={"suspicious"},
        )
        url = self._read_companion_url(path)
        metadata = {
            "source_path": str(path),
            "source_format": suffix.lstrip("."),
            "raw_file_ingestion": True,
            "label_inferred_from_path": label,
            "label_mapping_status": "mapped" if normalized_label is not None else "unmapped_or_missing",
        }
        return DatasetRecord(
            source=self.name,
            original_id=str(path.relative_to(self.snapshot_root)),
            original_label=label,
            normalized_label=normalized_label,
            url=url,
            html=html,
            text=text,
            split=infer_split_from_path(path),
            metadata=metadata,
        )

    def _read_companion_url(self, path: Path) -> str | None:
        for suffix in (".url", ".link"):
            companion = path.with_suffix(suffix)
            if companion.exists():
                with open_text(companion) as handle:
                    value = clean_text(handle.read())
                if value:
                    return value.splitlines()[0]
        return None

    def _normalize_label(self, value: Any) -> int | None:
        label_map = self.source_config.extra.get("label_map", {})
        if label_map:
            phishing_values = set(label_map.get("phishing", []))
            benign_values = set(label_map.get("benign", []))
            uncertain_values = set(label_map.get("uncertain", []))
        else:
            phishing_values = {"phish", "phishing", "malicious", "fraudulent"}
            benign_values = {"benign", "legitimate", "safe", "legit"}
            uncertain_values = {"suspicious", "unknown", "uncertain"}
        return normalize_binary_label(
            value,
            phishing_values=phishing_values,
            benign_values=benign_values,
            uncertain_values=uncertain_values,
        )
