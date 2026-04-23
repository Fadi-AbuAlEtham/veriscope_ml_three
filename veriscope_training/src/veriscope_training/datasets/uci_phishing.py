from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from veriscope_training.datasets.arff import parse_arff
from veriscope_training.datasets.base import DatasetAdapter, DatasetRecord
from veriscope_training.datasets.loaders import (
    LABEL_FIELD_CANDIDATES,
    clean_scalar,
    clean_text,
    normalize_binary_label,
    pick_first,
    primary_suffix,
)
from veriscope_training.datasets.registry import register_dataset


@register_dataset("uci_phishing")
class UCIPhishingAdapter(DatasetAdapter):
    supported_formats = ("arff", "csv", "tsv", "jsonl", "json", "parquet")
    source_type = "local_snapshot"

    def validation_report(self) -> dict[str, Any]:
        payload = super().validation_report()
        supported_paths = self._supported_paths()
        payload["supported_file_count"] = len(supported_paths)
        if not supported_paths:
            payload["issues"].append(
                "Expected the UCI phishing websites dataset as .arff, .csv, .tsv, .jsonl, .json, or .parquet."
            )
        payload["label_mapping"] = {
            "-1": 1,
            "1": 0,
            "0": None,
            "phishy/phishing": 1,
            "legitimate/benign": 0,
            "suspicious": None,
        }
        return payload

    def iterate_records(self) -> Iterator[DatasetRecord]:
        supported_paths = self._supported_paths()
        if not supported_paths:
            raise FileNotFoundError(
                f"UCI phishing adapter could not find supported snapshots under {self.snapshot_root}."
            )
        for path in supported_paths:
            suffix = primary_suffix(path)
            if suffix == ".arff":
                _attributes, rows = parse_arff(path)
                for row_index, row in enumerate(rows, start=1):
                    record = self._record_from_mapping(row, path=path, row_index=row_index)
                    if record is not None:
                        yield record
            else:
                from veriscope_training.datasets.loaders import iter_structured_rows

                for row_index, row in enumerate(iter_structured_rows(path), start=1):
                    record = self._record_from_mapping(row, path=path, row_index=row_index)
                    if record is not None:
                        yield record

    def _supported_paths(self) -> list[Path]:
        return [
            path
            for path in self.snapshot_paths()
            if primary_suffix(path) in {".arff", ".csv", ".tsv", ".jsonl", ".json", ".parquet"}
        ]

    def _record_from_mapping(
        self,
        row: dict[str, Any],
        *,
        path: Path,
        row_index: int,
    ) -> DatasetRecord | None:
        label_field = self._detect_label_field(row)
        original_label = row.get(label_field) if label_field else None
        normalized_label = normalize_binary_label(
            original_label,
            phishing_values={"-1", "phish", "phishing", "phishy"},
            benign_values={"1", "legitimate", "benign", "legit"},
            uncertain_values={"0", "suspicious", "unknown"},
        )
        feature_dict = {
            key: clean_scalar(value)
            for key, value in row.items()
            if key != label_field
        }
        if not feature_dict:
            return None
        metadata = {
            "source_path": str(path),
            "source_format": primary_suffix(path).lstrip("."),
            "label_column": label_field,
            "feature_count": len(feature_dict),
            "modality_family": "tabular",
            "label_mapping_status": "mapped" if normalized_label is not None else "unmapped_or_uncertain",
        }
        return DatasetRecord(
            source=self.name,
            original_id=f"{path.name}:{row_index}",
            original_label=clean_text(original_label),
            normalized_label=normalized_label,
            tabular_features=feature_dict,
            metadata=metadata,
        )

    def _detect_label_field(self, row: dict[str, Any]) -> str | None:
        lowered = {str(key).lower(): key for key in row}
        for candidate in LABEL_FIELD_CANDIDATES:
            key = lowered.get(candidate.lower())
            if key is not None:
                return key
        return list(row)[-1] if row else None
