from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


STRING_FALLBACK_COLUMNS = {
    "sample_id",
    "source_dataset",
    "source_original_id",
    "source_split",
    "source_label",
    "label_name",
    "label_mapping_reason",
    "language",
    "timestamp",
    "original_url",
    "normalized_url",
    "raw_html",
    "extracted_text",
    "normalized_text",
    "tabular_features",
    "modality_flags",
    "url_features",
    "html_features",
    "text_features",
    "original_metadata",
    "processing_metadata",
}
BOOL_COLUMNS = {"is_auxiliary", "is_supervised"}
INT_COLUMNS = {"normalized_label"}
JSON_OBJECT_COLUMNS = {
    "tabular_features",
    "modality_flags",
    "url_features",
    "html_features",
    "text_features",
    "original_metadata",
    "processing_metadata",
}


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}, found {type(payload).__name__}.")
    return payload


def write_yaml(path: str | Path, payload: dict[str, Any]) -> Path:
    target = ensure_parent_dir(path)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    target = ensure_parent_dir(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True)
        handle.write("\n")
    return target


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object lines in {path}.")
            yield payload


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    target = ensure_parent_dir(path)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    return target


def _to_parquet_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def read_parquet_rows(path: str | Path, *, batch_size: int = 2048) -> Iterator[dict[str, Any]]:
    parquet_file = pq.ParquetFile(Path(path))
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        for row in table.to_pylist():
            yield _from_parquet_safe_row(row)


def read_records_file(path: str | Path, *, parquet_batch_size: int = 2048) -> Iterator[dict[str, Any]]:
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".jsonl":
        yield from read_jsonl(resolved)
        return
    if suffix == ".parquet":
        yield from read_parquet_rows(resolved, batch_size=parquet_batch_size)
        return
    raise ValueError(f"Unsupported record file format for {resolved}. Expected .jsonl or .parquet.")


def available_disk_space_bytes(path: str | Path) -> int:
    usage = shutil.disk_usage(Path(path))
    return int(usage.free)


def _from_parquet_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in row.items():
        if key in JSON_OBJECT_COLUMNS and isinstance(value, str) and value and value[0] in "{[":
            try:
                decoded[key] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        decoded[key] = value
    return decoded


class StructuredDatasetWriter:
    """Stream JSONL and optionally Parquet artifacts without keeping full datasets in memory."""

    def __init__(
        self,
        base_path: str | Path,
        *,
        output_format: str = "parquet",
        parquet_batch_size: int = 1000,
        parquet_compression: str = "zstd",
    ) -> None:
        self.base_path = Path(base_path)
        self.output_format = output_format
        self.parquet_batch_size = parquet_batch_size
        self.parquet_compression = parquet_compression
        self.count = 0
        self._jsonl_handle = None
        self._parquet_rows: list[dict[str, Any]] = []
        self._parquet_writer: pq.ParquetWriter | None = None
        self._parquet_path: Path | None = None
        self._parquet_tmp_path: Path | None = None
        self._parquet_schema: pa.Schema | None = None

        if output_format in {"jsonl", "both"}:
            jsonl_path = ensure_parent_dir(self.base_path.with_suffix(".jsonl"))
            self._jsonl_handle = jsonl_path.open("w", encoding="utf-8")
        if output_format in {"parquet", "both"}:
            self._parquet_path = self.base_path.with_suffix(".parquet")
            self._parquet_tmp_path = self.base_path.with_suffix(".parquet.tmp")
            ensure_parent_dir(self._parquet_path)
            if self._parquet_tmp_path.exists():
                self._parquet_tmp_path.unlink()

    @property
    def output_paths(self) -> dict[str, str]:
        payload: dict[str, str] = {}
        if self.output_format in {"jsonl", "both"}:
            payload["jsonl"] = str(self.base_path.with_suffix(".jsonl"))
        if self.output_format in {"parquet", "both"}:
            payload["parquet"] = str(self.base_path.with_suffix(".parquet"))
        return payload

    def write(self, row: dict[str, Any]) -> None:
        self.count += 1
        if self._jsonl_handle is not None:
            self._jsonl_handle.write(json.dumps(row, sort_keys=True))
            self._jsonl_handle.write("\n")

        if self.output_format in {"parquet", "both"}:
            self._parquet_rows.append({key: _to_parquet_safe(value) for key, value in row.items()})
            if len(self._parquet_rows) >= self.parquet_batch_size:
                self._flush_parquet()

    def _flush_parquet(self) -> None:
        if not self._parquet_rows:
            return
        schema = self._parquet_schema or _schema_for_rows(self._parquet_rows)
        table = pa.Table.from_pylist(self._parquet_rows, schema=schema)
        if self._parquet_writer is None:
            if self._parquet_tmp_path is None:
                raise RuntimeError("Parquet output path was not initialized.")
            self._parquet_schema = table.schema
            self._parquet_writer = pq.ParquetWriter(
                self._parquet_tmp_path,
                table.schema,
                compression=self.parquet_compression,
                use_dictionary=True,
            )
        self._parquet_writer.write_table(table)
        self._parquet_rows.clear()

    def close(self) -> None:
        if self._jsonl_handle is not None:
            self._jsonl_handle.close()
        if self.output_format in {"parquet", "both"}:
            self._flush_parquet()
            if self._parquet_writer is not None:
                self._parquet_writer.close()
            if self._parquet_tmp_path is not None and self._parquet_path is not None and self._parquet_tmp_path.exists():
                self._parquet_tmp_path.replace(self._parquet_path)


def _schema_for_rows(rows: list[dict[str, Any]]) -> pa.Schema:
    keys = sorted({key for row in rows for key in row})
    fields = [pa.field(key, _arrow_type_for_column(key, rows), nullable=True) for key in keys]
    return pa.schema(fields)


def _arrow_type_for_column(column_name: str, rows: list[dict[str, Any]]) -> pa.DataType:
    if column_name in BOOL_COLUMNS:
        return pa.bool_()
    if column_name in INT_COLUMNS:
        return pa.int64()
    if column_name in STRING_FALLBACK_COLUMNS:
        return pa.string()

    observed_type: type[Any] | None = None
    for row in rows:
        value = row.get(column_name)
        if value is None:
            continue
        candidate_type = type(value)
        if observed_type is None:
            observed_type = candidate_type
            continue
        if observed_type is not candidate_type:
            return pa.string()

    if observed_type is bool:
        return pa.bool_()
    if observed_type is int:
        return pa.int64()
    if observed_type is float:
        return pa.float64()
    return pa.string()
