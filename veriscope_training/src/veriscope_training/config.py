from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from veriscope_training.paths import ProjectPaths
from veriscope_training.utils.io import read_yaml


ConfigMapping = dict[str, Any]


@dataclass(frozen=True)
class DatasetSourceConfig:
    name: str
    enabled: bool
    description: str
    access: str = "manual_snapshot"
    snapshot_path: str | None = None
    snapshot_glob: str | None = None
    modalities: tuple[str, ...] = ()
    intended_use: tuple[str, ...] = ()
    label_strategy: str = "binary"
    notes: tuple[str, ...] = ()
    extra: ConfigMapping = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, name: str, mapping: ConfigMapping) -> "DatasetSourceConfig":
        reserved_keys = {
            "enabled",
            "description",
            "access",
            "snapshot_path",
            "snapshot_glob",
            "modalities",
            "intended_use",
            "label_strategy",
            "notes",
        }
        extra = {key: value for key, value in mapping.items() if key not in reserved_keys}
        return cls(
            name=name,
            enabled=bool(mapping.get("enabled", True)),
            description=str(mapping.get("description", "")).strip(),
            access=str(mapping.get("access", "manual_snapshot")),
            snapshot_path=mapping.get("snapshot_path"),
            snapshot_glob=mapping.get("snapshot_glob"),
            modalities=tuple(mapping.get("modalities", [])),
            intended_use=tuple(mapping.get("intended_use", [])),
            label_strategy=str(mapping.get("label_strategy", "binary")),
            notes=tuple(mapping.get("notes", [])),
            extra=extra,
        )


@dataclass(frozen=True)
class AppConfig:
    paths: ProjectPaths
    datasets_config: ConfigMapping
    experiments_config: ConfigMapping
    models_config: ConfigMapping
    sources: dict[str, DatasetSourceConfig]

    @classmethod
    def load(cls, root: str | Path | None = None) -> "AppConfig":
        paths = ProjectPaths.from_root(Path(root).resolve() if root is not None else None)
        datasets_config = read_yaml(paths.config_dir / "datasets.yaml")
        experiments_config = read_yaml(paths.config_dir / "experiments.yaml")
        models_config = read_yaml(paths.config_dir / "models.yaml")

        source_map = datasets_config.get("sources", {})
        sources = {
            name: DatasetSourceConfig.from_mapping(name, mapping)
            for name, mapping in source_map.items()
        }
        return cls(
            paths=paths,
            datasets_config=datasets_config,
            experiments_config=experiments_config,
            models_config=models_config,
            sources=sources,
        )

    def enabled_sources(self) -> dict[str, DatasetSourceConfig]:
        return {name: source for name, source in self.sources.items() if source.enabled}

    def get_source(self, name: str) -> DatasetSourceConfig:
        try:
            return self.sources[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.sources))
            raise KeyError(f"Unknown dataset source '{name}'. Available sources: {available}") from exc

    def snapshot_path_for(self, source_name: str) -> Path:
        source = self.get_source(source_name)
        configured_path = source.snapshot_path or f"data/raw/{source.name}"
        return self.paths.resolve(configured_path)  # type: ignore[return-value]

    def raw_section(self, section: str) -> ConfigMapping:
        if section == "datasets":
            return self.datasets_config
        if section == "experiments":
            return self.experiments_config
        if section == "models":
            return self.models_config
        raise KeyError(f"Unknown config section '{section}'.")
