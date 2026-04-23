from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any, Iterable

import joblib

from veriscope_training.utils.io import ensure_parent_dir, write_json


def save_joblib(path: str | Path, payload: Any, compress: int = 3) -> Path:
    target = ensure_parent_dir(path)
    joblib.dump(payload, target, compress=compress)
    return target


def load_joblib(path: str | Path) -> Any:
    return joblib.load(Path(path))


def installed_versions(packages: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def save_versions_snapshot(path: str | Path, packages: Iterable[str]) -> Path:
    return write_json(path, installed_versions(packages))


def save_json_snapshot(path: str | Path, payload: Any) -> Path:
    return write_json(path, payload)


def save_text_artifact(path: str | Path, text: str) -> Path:
    target = ensure_parent_dir(path)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(text)
    return target
