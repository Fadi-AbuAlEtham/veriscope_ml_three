from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

from veriscope_training.config import AppConfig, DatasetSourceConfig
from veriscope_training.datasets.manifest import DatasetManifest
from veriscope_training.utils.hashing import make_sample_id


RecordMetadata = dict[str, Any]


@dataclass
class DatasetRecord:
    source: str
    original_id: str | None = None
    original_label: str | int | None = None
    normalized_label: int | None = None
    url: str | None = None
    html: str | None = None
    text: str | None = None
    tabular_features: dict[str, Any] | None = None
    timestamp: str | None = None
    language: str | None = None
    split: str | None = None
    sample_id: str | None = None
    feature_flags: dict[str, bool] = field(default_factory=dict)
    metadata: RecordMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_id is None:
            self.sample_id = make_sample_id(
                source=self.source,
                original_id=self.original_id,
                url=self.url,
                html=self.html,
                text=self.text,
            )
        if not self.feature_flags:
            self.feature_flags = {
                "has_url": bool(self.url),
                "has_html": bool(self.html),
                "has_text": bool(self.text),
                "has_tabular_features": bool(self.tabular_features),
                "has_timestamp": bool(self.timestamp),
                "has_language": bool(self.language),
            }

    def modalities_present(self) -> tuple[str, ...]:
        modalities: list[str] = []
        if self.url:
            modalities.append("url")
        if self.html:
            modalities.append("html")
        if self.text:
            modalities.append("text")
        if self.tabular_features:
            modalities.append("tabular")
        return tuple(modalities)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetRecord":
        return cls(**payload)


class DatasetAdapter(ABC):
    """Abstract dataset adapter for phishing-related sources."""

    dataset_name: str = ""
    supported_formats: tuple[str, ...] = ()
    source_type: str = "local_snapshot"

    def __init__(self, source_config: DatasetSourceConfig, app_config: AppConfig) -> None:
        if not self.dataset_name:
            self.dataset_name = source_config.name
        self.source_config = source_config
        self.app_config = app_config
        self.paths = app_config.paths

    @property
    def name(self) -> str:
        return self.dataset_name

    @property
    def snapshot_root(self) -> Path:
        configured = self.source_config.snapshot_path or f"data/raw/{self.name}"
        path = self.paths.resolve(configured)
        if path is None:
            raise ValueError(f"Could not resolve snapshot path for dataset '{self.name}'.")
        return path

    def snapshot_paths(self) -> list[Path]:
        root = self.snapshot_root
        if not root.exists():
            return []
        if root.is_file():
            return [root]
        pattern = self.source_config.snapshot_glob or "**/*"
        return sorted(path for path in root.glob(pattern) if path.is_file())

    def validate_snapshot_access(self) -> list[str]:
        issues: list[str] = []
        root = self.snapshot_root
        if not root.exists():
            issues.append(f"Snapshot path does not exist: {root}")
        elif root.is_dir() and not any(root.iterdir()):
            issues.append(f"Snapshot directory is empty: {root}")
        return issues

    def build_manifest(self, record_count: int | None = None, notes: list[str] | None = None) -> DatasetManifest:
        return DatasetManifest(
            source=self.name,
            description=self.source_config.description,
            access=self.source_config.access,
            snapshot_path=str(self.snapshot_root),
            snapshot_glob=self.source_config.snapshot_glob,
            modalities=list(self.source_config.modalities),
            record_count=record_count,
            adapter_class=f"{self.__class__.__module__}.{self.__class__.__name__}",
            notes=list(self.source_config.notes) + (notes or []),
        )

    def adapter_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "adapter_class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "description": self.source_config.description,
            "source_type": self.source_type,
            "supported_formats": list(self.supported_formats),
            "access": self.source_config.access,
            "modalities": list(self.source_config.modalities),
            "intended_use": list(self.source_config.intended_use),
            "snapshot_root": str(self.snapshot_root),
            "snapshot_glob": self.source_config.snapshot_glob,
            "notes": list(self.source_config.notes),
            "extra": self.source_config.extra,
        }

    def validation_report(self) -> dict[str, Any]:
        snapshot_paths = self.snapshot_paths()
        issues = self.validate_snapshot_access()
        return {
            "name": self.name,
            "adapter_class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "source_type": self.source_type,
            "supported_formats": list(self.supported_formats),
            "snapshot_root": str(self.snapshot_root),
            "snapshot_path_exists": self.snapshot_root.exists(),
            "snapshot_file_count": len(snapshot_paths),
            "sample_files": [str(path) for path in snapshot_paths[:5]],
            "issues": issues,
        }

    def preview_records(self, limit: int = 3) -> list[DatasetRecord]:
        return list(islice(self.iterate_records(), limit))

    @abstractmethod
    def iterate_records(self) -> Iterator[DatasetRecord]:
        """Yield dataset records lazily from local snapshots."""
