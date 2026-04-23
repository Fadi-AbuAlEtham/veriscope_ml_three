from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from veriscope_training.config import AppConfig, DatasetSourceConfig
from veriscope_training.utils.io import ensure_parent_dir, write_json


class AcquisitionError(RuntimeError):
    pass


class ManualActionRequired(AcquisitionError):
    pass


class CredentialRequired(AcquisitionError):
    pass


@dataclass
class FetchResult:
    dataset_name: str
    fetch_mode_used: str
    output_dir: str
    source_reference: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    files_written: list[str] = field(default_factory=list)
    validation_status: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetFetcher(ABC):
    dataset_name: str = ""

    def __init__(self, source_config: DatasetSourceConfig, app_config: AppConfig) -> None:
        self.source_config = source_config
        self.app_config = app_config
        self.paths = app_config.paths
        if not self.dataset_name:
            self.dataset_name = source_config.name

    @property
    def fetch_config(self) -> dict[str, Any]:
        return (self.source_config.extra.get("fetch") or {}).copy()

    @property
    def output_dir(self) -> Path:
        configured = self.fetch_config.get("output_dir") or self.source_config.snapshot_path or f"data/raw/{self.dataset_name}"
        resolved = self.paths.resolve(configured)
        if resolved is None:
            raise ValueError(f"Could not resolve fetch output directory for '{self.dataset_name}'.")
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def fetch_mode(self) -> str:
        return str(self.fetch_config.get("fetch_mode", "manual_snapshot"))

    def resumable(self) -> bool:
        return bool(self.fetch_config.get("resumable", True))

    def instructions_path(self) -> Path:
        return self.output_dir / "MANUAL_SERVER_SNAPSHOT.md"

    def metadata_path(self) -> Path:
        return self.output_dir / "_fetch_manifest.json"

    def existing_files(self) -> list[Path]:
        return sorted(path for path in self.output_dir.rglob("*") if path.is_file() and not path.name.startswith("_fetch_"))

    def env(self, name: str | None) -> str | None:
        if not name:
            return None
        return os.environ.get(name)

    @abstractmethod
    def fetch(self, *, force: bool = False) -> FetchResult:
        raise NotImplementedError

    def write_fetch_metadata(self, result: FetchResult) -> str:
        return str(write_json(self.metadata_path(), result.to_dict()))

    def write_manual_instructions(self, text: str) -> str:
        path = ensure_parent_dir(self.instructions_path())
        path.write_text(text, encoding="utf-8")
        return str(path)

    def download_to_path(
        self,
        *,
        url: str,
        destination: str | Path,
        headers: dict[str, str] | None = None,
        auth: tuple[str, str] | None = None,
        token: str | None = None,
        timeout: tuple[int, int] = (20, 300),
        force: bool = False,
    ) -> str:
        destination_path = Path(destination)
        ensure_parent_dir(destination_path)
        part_path = destination_path.with_suffix(destination_path.suffix + ".part")
        request_headers = dict(headers or {})
        if token:
            request_headers["Authorization"] = f"Bearer {token}"

        resume_from = 0
        mode = "wb"
        if self.resumable() and not force and part_path.exists():
            resume_from = part_path.stat().st_size
            if resume_from > 0:
                request_headers["Range"] = f"bytes={resume_from}-"
                mode = "ab"

        with requests.get(url, stream=True, headers=request_headers, auth=auth, timeout=timeout) as response:
            if response.status_code == 416 and part_path.exists():
                part_path.replace(destination_path)
                return str(destination_path)
            response.raise_for_status()
            if response.status_code != 206 and mode == "ab":
                mode = "wb"
            with part_path.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        part_path.replace(destination_path)
        return str(destination_path)
