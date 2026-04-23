from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from veriscope_training.datasets.base import DatasetAdapter, DatasetRecord
from veriscope_training.datasets.loaders import (
    ID_FIELD_CANDIDATES,
    LANGUAGE_FIELD_CANDIDATES,
    SPLIT_FIELD_CANDIDATES,
    TEXT_FIELD_CANDIDATES,
    clean_text,
    exclude_fields,
    iter_structured_rows,
    iter_text_lines,
    pick_first,
    primary_suffix,
)
from veriscope_training.datasets.registry import register_dataset


@register_dataset("oscar_aux")
class OscarAuxAdapter(DatasetAdapter):
    supported_formats = ("huggingface", "jsonl", "json", "parquet", "csv", "tsv", "txt")
    source_type = "hybrid_remote_or_snapshot"

    def adapter_metadata(self) -> dict[str, Any]:
        payload = super().adapter_metadata()
        payload["huggingface"] = {
            "enabled": bool(self.source_config.extra.get("huggingface", {}).get("enabled", False)),
            "dataset_id": self.source_config.extra.get("huggingface", {}).get("dataset_id", "oscar-corpus/oscar"),
            "config_name": self.source_config.extra.get("huggingface", {}).get("config_name"),
            "split": self.source_config.extra.get("huggingface", {}).get("split", "train"),
            "streaming": bool(self.source_config.extra.get("huggingface", {}).get("streaming", True)),
        }
        return payload

    def validate_snapshot_access(self) -> list[str]:
        issues = super().validate_snapshot_access()
        hf_enabled = bool(self.source_config.extra.get("huggingface", {}).get("enabled", False))
        if hf_enabled and issues:
            return []
        return issues

    def validation_report(self) -> dict[str, Any]:
        payload = super().validation_report()
        payload["local_structured_file_count"] = len(self._supported_paths())
        payload["uses_huggingface_fallback"] = bool(
            self.source_config.extra.get("huggingface", {}).get("enabled", False) and not self._supported_paths()
        )
        return payload

    def iterate_records(self) -> Iterator[DatasetRecord]:
        supported_paths = self._supported_paths()
        if supported_paths:
            for path in supported_paths:
                suffix = primary_suffix(path)
                if suffix == ".txt":
                    yield from self._iterate_txt(path)
                else:
                    yield from self._iterate_structured(path)
            return
        yield from self._iterate_huggingface()

    def _supported_paths(self) -> list[Path]:
        return [
            path
            for path in self.snapshot_paths()
            if primary_suffix(path) in {".jsonl", ".json", ".parquet", ".csv", ".tsv", ".txt"}
        ]

    def _iterate_txt(self, path: Path) -> Iterator[DatasetRecord]:
        language = self._infer_language_from_config()
        for line_number, line in iter_text_lines(path):
            yield DatasetRecord(
                source=self.name,
                original_id=f"{path.name}:{line_number}",
                original_label=None,
                normalized_label=None,
                text=line,
                language=language,
                metadata={
                    "source_path": str(path),
                    "source_format": "txt",
                    "auxiliary_corpus": True,
                    "supervision": "unlabeled",
                },
            )

    def _iterate_structured(self, path: Path) -> Iterator[DatasetRecord]:
        for row_index, row in enumerate(iter_structured_rows(path), start=1):
            text = clean_text(pick_first(row, TEXT_FIELD_CANDIDATES))
            if text is None:
                continue
            language = clean_text(pick_first(row, LANGUAGE_FIELD_CANDIDATES)) or self._infer_language_from_config()
            metadata = exclude_fields(
                row,
                excluded=(*TEXT_FIELD_CANDIDATES, *LANGUAGE_FIELD_CANDIDATES, *ID_FIELD_CANDIDATES, *SPLIT_FIELD_CANDIDATES),
            )
            metadata["source_path"] = str(path)
            metadata["source_format"] = primary_suffix(path).lstrip(".")
            metadata["auxiliary_corpus"] = True
            metadata["supervision"] = "unlabeled"
            yield DatasetRecord(
                source=self.name,
                original_id=clean_text(pick_first(row, ID_FIELD_CANDIDATES)) or f"{path.name}:{row_index}",
                original_label=None,
                normalized_label=None,
                text=text,
                language=language,
                split=clean_text(pick_first(row, SPLIT_FIELD_CANDIDATES)),
                metadata=metadata,
            )

    def _iterate_huggingface(self) -> Iterator[DatasetRecord]:
        hf_config = self.source_config.extra.get("huggingface", {})
        if not hf_config.get("enabled", False):
            issues = super().validate_snapshot_access()
            joined = "; ".join(issues) or "No local OSCAR auxiliary snapshot available."
            raise FileNotFoundError(
                f"OSCAR auxiliary adapter requires either local snapshots or enabled Hugging Face access. {joined}"
            )
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Optional dependency 'datasets' is required for Hugging Face OSCAR loading. "
                "Install veriscope-training[transformers] or provide local OSCAR snapshots."
            ) from exc

        dataset_id = hf_config.get("dataset_id", "oscar-corpus/oscar")
        config_name = hf_config.get("config_name")
        split_name = hf_config.get("split", "train")
        streaming = bool(hf_config.get("streaming", True))
        try:
            dataset_iterable = load_dataset(
                dataset_id,
                name=config_name,
                split=split_name,
                streaming=streaming,
                token=hf_config.get("use_auth_token"),
            )
            inferred_language = self._infer_language_from_config()
            for row_index, row in enumerate(dataset_iterable, start=1):
                mapping = dict(row)
                text = clean_text(pick_first(mapping, TEXT_FIELD_CANDIDATES)) or clean_text(mapping.get("text"))
                if text is None:
                    continue
                language = clean_text(pick_first(mapping, LANGUAGE_FIELD_CANDIDATES)) or inferred_language
                metadata = exclude_fields(
                    mapping,
                    excluded=(*TEXT_FIELD_CANDIDATES, *LANGUAGE_FIELD_CANDIDATES, *ID_FIELD_CANDIDATES),
                )
                metadata["source_path"] = f"hf://{dataset_id}/{config_name or 'default'}/{split_name}"
                metadata["source_format"] = "huggingface"
                metadata["auxiliary_corpus"] = True
                metadata["supervision"] = "unlabeled"
                yield DatasetRecord(
                    source=self.name,
                    original_id=clean_text(pick_first(mapping, ID_FIELD_CANDIDATES)) or f"{split_name}:{row_index}",
                    original_label=None,
                    normalized_label=None,
                    text=text,
                    language=language,
                    split=split_name,
                    metadata=metadata,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load OSCAR auxiliary data from Hugging Face dataset '{dataset_id}'. "
                f"Provide local snapshots under {self.snapshot_root} or enable network access. "
                f"Underlying error: {exc}"
            ) from exc

    def _infer_language_from_config(self) -> str | None:
        language = clean_text(self.source_config.extra.get("language"))
        if language:
            return language
        config_name = clean_text(self.source_config.extra.get("huggingface", {}).get("config_name"))
        if config_name and "_" in config_name:
            candidate = config_name.split("_")[-1]
            if candidate.isalpha():
                return candidate
        return None
