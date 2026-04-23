from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class DatasetManifest:
    source: str
    description: str
    access: str
    snapshot_path: str
    snapshot_glob: str | None
    modalities: list[str]
    record_count: int | None = None
    adapter_class: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessedDatasetManifest:
    view_name: str
    output_paths: dict[str, str]
    record_count: int
    sources: list[str]
    source_counts: dict[str, int]
    label_counts: dict[str, int]
    modality_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    dedupe_summary: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BuildDatasetSummary:
    selected_sources: list[str]
    selected_views: list[str]
    output_paths: dict[str, dict[str, str]]
    source_summaries: dict[str, Any]
    dedupe_report_path: str | None
    duplicate_events_path: str | None
    total_raw_records: int
    total_processed_records: int
    total_kept_records: int
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
