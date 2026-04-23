from __future__ import annotations

import bz2
import csv
import gzip
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator
from xml.etree.ElementTree import iterparse

import pandas as pd
import pyarrow.parquet as pq


NULL_LIKE = {"", "na", "n/a", "nan", "none", "null"}
PHISHING_LABELS = {"phish", "phishing", "malicious", "fraud", "fraudulent", "bad"}
BENIGN_LABELS = {"benign", "legitimate", "legit", "safe", "good", "ham"}
UNCERTAIN_LABELS = {"suspicious", "unknown", "uncertain", "other", "unlabeled", "auxiliary"}
URL_FIELD_CANDIDATES = (
    "url",
    "uri",
    "link",
    "source_url",
    "landing_url",
    "page_url",
    "phish_url",
    "website",
    "domain",
)
HTML_FIELD_CANDIDATES = ("html", "raw_html", "page_html", "source_html", "content_html", "dom")
TEXT_FIELD_CANDIDATES = ("text", "visible_text", "content", "body", "page_text", "document", "message")
LABEL_FIELD_CANDIDATES = ("label", "class", "result", "category", "status")
ID_FIELD_CANDIDATES = ("id", "sample_id", "row_id", "sha256", "uuid", "phish_id")
SPLIT_FIELD_CANDIDATES = ("split", "fold", "partition", "subset")
TIMESTAMP_FIELD_CANDIDATES = (
    "timestamp",
    "date",
    "created_at",
    "submission_time",
    "verification_time",
    "first_seen",
    "last_seen",
    "ingested_at",
    "crawl_time",
)
LANGUAGE_FIELD_CANDIDATES = ("language", "lang", "page_language")
COMPRESSION_SUFFIXES = {".gz", ".bz2"}
URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
ISO_DATE_PREFIX = re.compile(r"\d{4}-\d{2}-\d{2}")


def is_null_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    if isinstance(value, str):
        return value.strip().lower() in NULL_LIKE
    return False


def clean_scalar(value: Any) -> Any:
    if is_null_like(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def clean_text(value: Any) -> str | None:
    cleaned = clean_scalar(value)
    if cleaned is None:
        return None
    return str(cleaned)


def looks_like_html(value: Any) -> bool:
    text = clean_text(value)
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in ("<html", "<body", "<head", "<form", "<script", "<div"))


def primary_suffix(path: str | Path) -> str:
    resolved = Path(path)
    suffixes = resolved.suffixes
    if not suffixes:
        return ""
    if suffixes[-1].lower() in COMPRESSION_SUFFIXES and len(suffixes) >= 2:
        return suffixes[-2].lower()
    return suffixes[-1].lower()


@contextmanager
def open_text(path: str | Path):
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".gz":
        with gzip.open(resolved, "rt", encoding="utf-8", errors="ignore", newline="") as handle:
            yield handle
        return
    if suffix == ".bz2":
        with bz2.open(resolved, "rt", encoding="utf-8", errors="ignore", newline="") as handle:
            yield handle
        return
    with resolved.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        yield handle


@contextmanager
def open_binary(path: str | Path):
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".gz":
        with gzip.open(resolved, "rb") as handle:
            yield handle
        return
    if suffix == ".bz2":
        with bz2.open(resolved, "rb") as handle:
            yield handle
        return
    with resolved.open("rb") as handle:
        yield handle


def sniff_delimiter(path: str | Path, default: str = ",") -> str:
    with open_text(path) as handle:
        sample = handle.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        return default
    return dialect.delimiter


def infer_split_from_path(path: str | Path) -> str | None:
    name = Path(path).name.lower()
    if "train" in name:
        return "train"
    if "valid" in name or "dev" in name or "val" in name:
        return "validation"
    if "test" in name:
        return "test"
    return None


def infer_label_from_path(path: str | Path) -> str | None:
    parts = [segment.lower() for segment in Path(path).parts]
    phishing_markers = {"phish", "phishing", "malicious", "bad"}
    benign_markers = {"benign", "legitimate", "legit", "safe", "good"}
    uncertain_markers = {"suspicious", "uncertain", "unknown"}
    if any(part in phishing_markers for part in parts):
        return "phishing"
    if any(part in benign_markers for part in parts):
        return "benign"
    if any(part in uncertain_markers for part in parts):
        return "suspicious"
    return None


def normalize_binary_label(
    value: Any,
    *,
    phishing_values: Iterable[Any] | None = None,
    benign_values: Iterable[Any] | None = None,
    uncertain_values: Iterable[Any] | None = None,
) -> int | None:
    if value is None:
        return None
    phishing_set = {str(item).strip().lower() for item in (phishing_values or PHISHING_LABELS)}
    benign_set = {str(item).strip().lower() for item in (benign_values or BENIGN_LABELS)}
    uncertain_set = {str(item).strip().lower() for item in (uncertain_values or UNCERTAIN_LABELS)}

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = int(value)
        if numeric in (1,):
            if "1" in phishing_set:
                return 1
            if "1" in benign_set:
                return 0
            return None
        if numeric in (-1,):
            if "-1" in phishing_set:
                return 1
            if "-1" in benign_set:
                return 0
            return None
        if numeric in (0,):
            if "0" in phishing_set:
                return 1
            if "0" in benign_set:
                return 0
            return None
    normalized = str(value).strip().lower()
    if normalized in phishing_set:
        return 1
    if normalized in benign_set:
        return 0
    if normalized in uncertain_set:
        return None
    return None


def pick_first(mapping: dict[str, Any], candidates: Iterable[str]) -> Any:
    lowered = {str(key).lower(): key for key in mapping}
    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is not None:
            return mapping[key]
    return None


def exclude_fields(mapping: dict[str, Any], excluded: Iterable[str]) -> dict[str, Any]:
    excluded_lower = {field.lower() for field in excluded}
    return {
        key: value
        for key, value in mapping.items()
        if str(key).lower() not in excluded_lower and not is_null_like(value)
    }


def iter_csv_dict_rows(path: str | Path, delimiter: str | None = None, chunksize: int = 5000) -> Iterator[dict[str, Any]]:
    sep = delimiter or sniff_delimiter(path)
    reader = pd.read_csv(
        Path(path),
        sep=sep,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        chunksize=chunksize,
        compression="infer",
    )
    for chunk in reader:
        for row in chunk.to_dict(orient="records"):
            yield row


def iter_jsonl_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def iter_json_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        for key in ("data", "records", "items", "entries", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        if all(isinstance(value, dict) for value in payload.values()):
            for value in payload.values():
                yield value
            return
        yield payload


def iter_parquet_rows(path: str | Path, batch_size: int = 2048) -> Iterator[dict[str, Any]]:
    parquet_file = pq.ParquetFile(Path(path))
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            if isinstance(row, dict):
                yield row


def iter_text_lines(path: str | Path, skip_comments: bool = True) -> Iterator[tuple[int, str]]:
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if skip_comments and stripped.startswith("#"):
                continue
            yield line_number, stripped


def extract_url_candidate(value: str) -> str | None:
    if not value:
        return None
    match = URL_PATTERN.search(value)
    if match:
        return match.group(0).rstrip(",;")
    tokens = value.split()
    for token in tokens:
        if token.startswith(("http://", "https://")):
            return token.rstrip(",;")
    return None


def extract_timestamp_candidate(value: str) -> str | None:
    if not value:
        return None
    if ISO_DATE_PREFIX.search(value):
        return value.strip()
    return None


def iter_structured_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    suffix = primary_suffix(path)
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else None
        yield from iter_csv_dict_rows(path, delimiter=delimiter)
        return
    if suffix == ".jsonl":
        yield from iter_jsonl_rows(path)
        return
    if suffix == ".json":
        yield from iter_json_rows(path)
        return
    if suffix == ".parquet":
        yield from iter_parquet_rows(path)
        return
    raise ValueError(f"Unsupported structured format for {path}.")


def iter_xml_elements(path: str | Path, tag: str) -> Iterator[Any]:
    with open_binary(path) as handle:
        for _event, element in iterparse(handle, events=("end",)):
            if element.tag == tag:
                yield element
                element.clear()
