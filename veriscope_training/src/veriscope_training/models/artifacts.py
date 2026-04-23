from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from veriscope_training.config import AppConfig
from veriscope_training.utils.io import ensure_parent_dir, read_records_file, write_json, write_jsonl
from veriscope_training.utils.serialization import save_joblib, save_versions_snapshot


@dataclass
class TrainingRunContext:
    track: str
    model_name: str
    run_name: str
    run_dir: Path
    artifact_dir: Path
    reports_dir: Path
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_dir"] = str(self.run_dir)
        payload["artifact_dir"] = str(self.artifact_dir)
        payload["reports_dir"] = str(self.reports_dir)
        return payload


def create_training_run_context(
    config: AppConfig,
    *,
    track: str,
    model_name: str,
    run_name: str | None = None,
) -> TrainingRunContext:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name or timestamp
    training_root = config.paths.resolve(
        config.experiments_config.get("artifacts", {}).get("training_root", "outputs/training")
    )
    if training_root is None:
        raise ValueError("Training root could not be resolved.")
    run_dir = training_root / track / model_name / name
    artifact_dir = run_dir / "artifacts"
    reports_dir = run_dir / "reports"
    split_dir = run_dir / "splits"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    return TrainingRunContext(
        track=track,
        model_name=model_name,
        run_name=name,
        run_dir=run_dir,
        artifact_dir=artifact_dir,
        reports_dir=reports_dir,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def load_processed_view_records(config: AppConfig, view_name: str) -> list[dict[str, Any]]:
    base = config.paths.processed_data / view_name
    parquet_path = base.with_suffix(".parquet")
    jsonl_path = base.with_suffix(".jsonl")
    if parquet_path.exists():
        return list(read_records_file(parquet_path))
    if jsonl_path.exists():
        return list(read_records_file(jsonl_path))
    raise FileNotFoundError(
        f"Processed dataset for view '{view_name}' not found at {parquet_path} or {jsonl_path}. "
        "Run build-dataset first."
    )


def save_run_metadata(context: TrainingRunContext, *, payload: dict[str, Any]) -> str:
    path = context.run_dir / "run_summary.json"
    write_json(path, payload)
    return str(path)


def save_predictions(context: TrainingRunContext, split_name: str, rows: list[dict[str, Any]]) -> str:
    path = context.reports_dir / f"{split_name}_predictions.jsonl"
    write_jsonl(path, rows)
    return str(path)


def save_label_metadata(context: TrainingRunContext, payload: dict[str, Any]) -> str:
    path = context.artifact_dir / "label_metadata.json"
    write_json(path, payload)
    return str(path)


def save_config_snapshot(context: TrainingRunContext, payload: dict[str, Any]) -> str:
    path = context.run_dir / "config_snapshot.json"
    write_json(path, payload)
    return str(path)


def save_metrics(context: TrainingRunContext, payload: dict[str, Any]) -> str:
    path = context.reports_dir / "metrics.json"
    write_json(path, payload)
    return str(path)


def save_model_bundle(context: TrainingRunContext, name: str, payload: Any) -> str:
    path = context.artifact_dir / f"{name}.joblib"
    save_joblib(path, payload)
    return str(path)


def save_package_versions(context: TrainingRunContext, packages: list[str]) -> str:
    path = context.run_dir / "package_versions.json"
    save_versions_snapshot(path, packages)
    return str(path)


def load_training_run(path_or_dir: str | Path) -> dict[str, Any]:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Training run summary not found at {path}.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
