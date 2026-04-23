from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def discover_project_root() -> Path:
    """Resolve the project root, allowing an explicit environment override."""
    env_root = os.environ.get("VERISCOPE_TRAINING_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").exists() and (parent / "src").exists():
            return parent
    return current.parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    src: Path
    config_dir: Path
    data_root: Path
    raw_data: Path
    interim_data: Path
    processed_data: Path
    manifest_data: Path
    outputs: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> "ProjectPaths":
        project_root = (root or discover_project_root()).resolve()
        data_root = project_root / "data"
        return cls(
            root=project_root,
            src=project_root / "src",
            config_dir=project_root / "configs",
            data_root=data_root,
            raw_data=data_root / "raw",
            interim_data=data_root / "interim",
            processed_data=data_root / "processed",
            manifest_data=data_root / "manifests",
            outputs=project_root / "outputs",
        )

    def ensure_directories(self) -> None:
        for path in (
            self.data_root,
            self.raw_data,
            self.interim_data,
            self.processed_data,
            self.manifest_data,
            self.outputs,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_or_absolute: str | Path | None) -> Path | None:
        if relative_or_absolute is None:
            return None
        path = Path(relative_or_absolute).expanduser()
        if path.is_absolute():
            return path
        return (self.root / path).resolve()
