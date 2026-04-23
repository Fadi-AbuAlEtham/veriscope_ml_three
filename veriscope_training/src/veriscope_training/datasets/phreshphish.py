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
    infer_split_from_path,
    iter_structured_rows,
    normalize_binary_label,
    pick_first,
    primary_suffix,
)
from veriscope_training.datasets.registry import register_dataset


@register_dataset("phreshphish")
class PhreshPhishAdapter(DatasetAdapter):
    supported_formats = ("huggingface", "parquet", "jsonl", "json", "csv", "tsv")
    source_type = "hybrid_remote_or_snapshot"

    def adapter_metadata(self) -> dict[str, Any]:
        payload = super().adapter_metadata()
        payload["huggingface"] = {
            "enabled": bool(self.source_config.extra.get("huggingface", {}).get("enabled", True)),
            "dataset_id": self.source_config.extra.get("huggingface", {}).get(
                "dataset_id", "phreshphish/phreshphish"
            ),
            "splits": self.source_config.extra.get("huggingface", {}).get("splits", ["train", "test"]),
            "streaming": bool(self.source_config.extra.get("huggingface", {}).get("streaming", True)),
        }
        return payload

    def validate_snapshot_access(self) -> list[str]:
        issues = super().validate_snapshot_access()
        hf_enabled = bool(self.source_config.extra.get("huggingface", {}).get("enabled", True))
        if hf_enabled and issues:
            return []
        return issues

    def validation_report(self) -> dict[str, Any]:
        payload = super().validation_report()
        local_paths = self._structured_snapshot_paths()
        hf_enabled = bool(self.source_config.extra.get("huggingface", {}).get("enabled", True))
        payload["local_structured_file_count"] = len(local_paths)
        payload["uses_huggingface_fallback"] = bool(hf_enabled and not local_paths)
        payload["issues"] = self.validate_snapshot_access()
        if not local_paths and not hf_enabled:
            payload["issues"].append("No local snapshot files found and Hugging Face fallback is disabled.")
        return payload

    def iterate_records(self) -> Iterator[DatasetRecord]:
        local_paths = self._structured_snapshot_paths()
        if local_paths:
            for path in local_paths:
                split_hint = infer_split_from_path(path)
                for row_index, row in enumerate(iter_structured_rows(path), start=1):
                    record = self._record_from_mapping(row, row_index=row_index, path=path, split_hint=split_hint)
                    if record is not None:
                        yield record
            return
        yield from self._iterate_huggingface_records()

    def _structured_snapshot_paths(self) -> list[Path]:
        return [
            path
            for path in self.snapshot_paths()
            if primary_suffix(path) in {".parquet", ".jsonl", ".json", ".csv", ".tsv"}
        ]

    def _iterate_huggingface_records(self) -> Iterator[DatasetRecord]:
        hf_config = self.source_config.extra.get("huggingface", {})
        if not hf_config.get("enabled", True):
            issues = super().validate_snapshot_access()
            joined = "; ".join(issues) or "No local data available."
            raise FileNotFoundError(
                f"PhreshPhish requires either local snapshots or Hugging Face access. {joined}"
            )
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Optional dependency 'datasets' is required for Hugging Face PhreshPhish loading. "
                "Install veriscope-training[transformers] or place a local snapshot under "
                f"{self.snapshot_root}."
            ) from exc

        dataset_id = hf_config.get("dataset_id", "phreshphish/phreshphish")
        splits = hf_config.get("splits", ["train", "test"])
        streaming = bool(hf_config.get("streaming", True))
        use_auth_token = hf_config.get("use_auth_token")

        try:
            for split_name in splits:
                dataset_iterable = load_dataset(
                    dataset_id,
                    split=split_name,
                    streaming=streaming,
                    token=use_auth_token,
                )
                for row_index, row in enumerate(dataset_iterable, start=1):
                    record = self._record_from_mapping(
                        dict(row),
                        row_index=row_index,
                        path=Path(f"hf://{dataset_id}/{split_name}"),
                        split_hint=split_name,
                    )
                    if record is not None:
                        yield record
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load PhreshPhish from Hugging Face dataset '{dataset_id}'. "
                f"Provide local snapshots under {self.snapshot_root} or enable network access. "
                f"Underlying error: {exc}"
            ) from exc

    def _record_from_mapping(
        self,
        row: dict[str, Any],
        *,
        row_index: int,
        path: Path,
        split_hint: str | None,
    ) -> DatasetRecord | None:
        url = clean_text(pick_first(row, URL_FIELD_CANDIDATES))
        html = clean_text(pick_first(row, HTML_FIELD_CANDIDATES))
        text = clean_text(pick_first(row, TEXT_FIELD_CANDIDATES))
        original_label = pick_first(row, LABEL_FIELD_CANDIDATES)
        normalized_label = normalize_binary_label(
            original_label,
            phishing_values={"phish", "phishing", "1", "true", "malicious"},
            benign_values={"benign", "legitimate", "0", "false"},
        )
        if url is None and html is None and text is None:
            return None

        original_id = clean_text(pick_first(row, ID_FIELD_CANDIDATES)) or f"{path.name}:{row_index}"
        split = clean_text(pick_first(row, SPLIT_FIELD_CANDIDATES)) or split_hint
        timestamp = clean_text(pick_first(row, TIMESTAMP_FIELD_CANDIDATES))
        language = clean_text(pick_first(row, LANGUAGE_FIELD_CANDIDATES))
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
        metadata["modality_family"] = "webpage"
        metadata["label_mapping_status"] = "mapped" if normalized_label is not None else "unmapped"

        return DatasetRecord(
            source=self.name,
            original_id=original_id,
            original_label=clean_text(original_label),
            normalized_label=normalized_label,
            url=url,
            html=html,
            text=text if text and text != html else None,
            timestamp=timestamp,
            language=language,
            split=split,
            metadata=metadata,
        )
