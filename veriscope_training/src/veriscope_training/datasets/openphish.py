from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from veriscope_training.datasets.base import DatasetAdapter, DatasetRecord
from veriscope_training.datasets.loaders import (
    ID_FIELD_CANDIDATES,
    TIMESTAMP_FIELD_CANDIDATES,
    URL_FIELD_CANDIDATES,
    clean_text,
    exclude_fields,
    extract_timestamp_candidate,
    extract_url_candidate,
    iter_structured_rows,
    iter_text_lines,
    pick_first,
    primary_suffix,
)
from veriscope_training.datasets.registry import register_dataset


@register_dataset("openphish")
class OpenPhishAdapter(DatasetAdapter):
    supported_formats = ("txt", "csv", "tsv", "jsonl", "json", "parquet")
    source_type = "local_snapshot"

    def validation_report(self) -> dict[str, Any]:
        payload = super().validation_report()
        supported_paths = self._supported_paths()
        payload["supported_file_count"] = len(supported_paths)
        if not supported_paths:
            payload["issues"].append(
                "Expected one or more .txt/.csv/.tsv/.jsonl/.json/.parquet feed snapshots under the snapshot path."
            )
        return payload

    def iterate_records(self) -> Iterator[DatasetRecord]:
        supported_paths = self._supported_paths()
        if not supported_paths:
            raise FileNotFoundError(
                "OpenPhish adapter could not find supported local snapshots. "
                f"Place URL feed files under {self.snapshot_root}."
            )
        for path in supported_paths:
            suffix = primary_suffix(path)
            if suffix == ".txt":
                yield from self._iterate_text_feed(path)
            else:
                yield from self._iterate_structured_feed(path)

    def _supported_paths(self) -> list[Path]:
        supported = {".txt", ".csv", ".tsv", ".jsonl", ".json", ".parquet"}
        return [path for path in self.snapshot_paths() if primary_suffix(path) in supported]

    def _iterate_text_feed(self, path: Path) -> Iterator[DatasetRecord]:
        for line_number, line in iter_text_lines(path):
            url = extract_url_candidate(line)
            if url is None:
                continue
            timestamp = extract_timestamp_candidate(line.replace(url, "").strip())
            metadata = {
                "source_path": str(path),
                "source_format": "txt",
                "raw_line": line if line != url else None,
                "phishing_only_source": True,
            }
            yield DatasetRecord(
                source=self.name,
                original_id=f"{path.name}:{line_number}",
                original_label="phishing",
                normalized_label=1,
                url=url,
                timestamp=timestamp,
                metadata={key: value for key, value in metadata.items() if value is not None},
            )

    def _iterate_structured_feed(self, path: Path) -> Iterator[DatasetRecord]:
        for row_index, row in enumerate(iter_structured_rows(path), start=1):
            url = clean_text(pick_first(row, URL_FIELD_CANDIDATES))
            if url is None:
                continue
            timestamp = clean_text(pick_first(row, TIMESTAMP_FIELD_CANDIDATES))
            original_id = clean_text(pick_first(row, ID_FIELD_CANDIDATES)) or f"{path.name}:{row_index}"
            metadata = exclude_fields(
                row,
                excluded=(*URL_FIELD_CANDIDATES, *TIMESTAMP_FIELD_CANDIDATES, *ID_FIELD_CANDIDATES),
            )
            metadata["source_path"] = str(path)
            metadata["source_format"] = primary_suffix(path).lstrip(".")
            metadata["phishing_only_source"] = True
            yield DatasetRecord(
                source=self.name,
                original_id=original_id,
                original_label="phishing",
                normalized_label=1,
                url=url,
                timestamp=timestamp,
                metadata=metadata,
            )
