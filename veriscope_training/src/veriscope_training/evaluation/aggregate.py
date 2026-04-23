from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from veriscope_training.models.artifacts import load_training_run
from veriscope_training.utils.io import ensure_parent_dir, read_json, write_json


TRACK_VIEW_MAP = {
    "url": "unified_url_dataset",
    "webpage": "unified_webpage_dataset",
    "webpage_transformer": "unified_webpage_dataset",
    "tabular": "unified_tabular_dataset",
}


def discover_training_run_dirs(root: str | Path) -> list[Path]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(path.parent for path in base.rglob("run_summary.json"))


def load_run_bundle(run_dir_or_summary: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir_or_summary)
    summary = load_training_run(run_dir)
    if run_dir.is_file():
        run_dir = run_dir.parent

    metrics_path = _maybe_path(summary.get("artifact_paths", {}).get("metrics"))
    metrics = read_json(metrics_path) if metrics_path and metrics_path.exists() else summary.get("metrics", {})
    config_snapshot_path = run_dir / "config_snapshot.json"
    config_snapshot = read_json(config_snapshot_path) if config_snapshot_path.exists() else {}
    package_versions_path = run_dir / "package_versions.json"
    package_versions = read_json(package_versions_path) if package_versions_path.exists() else {}

    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "metrics": metrics,
        "config_snapshot": config_snapshot,
        "package_versions": package_versions,
    }


def aggregate_run_bundles(
    bundles: Iterable[dict[str, Any]],
    *,
    split_name: str = "test",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        row = bundle_to_comparison_row(bundle, split_name=split_name)
        if row is not None:
            rows.append(row)
    rows.sort(
        key=lambda item: (
            item.get("track_group") or "",
            -(item.get("f1") or -1.0),
            -(item.get("pr_auc") or -1.0),
        )
    )
    return {
        "split_name": split_name,
        "row_count": len(rows),
        "rows": rows,
        "tracks": sorted({row["track_group"] for row in rows}),
    }


def bundle_to_comparison_row(bundle: dict[str, Any], *, split_name: str = "test") -> dict[str, Any] | None:
    summary = bundle["summary"]
    metrics_by_split = bundle.get("metrics") or summary.get("metrics") or {}
    split_metrics = metrics_by_split.get(split_name)
    if not isinstance(split_metrics, dict):
        return None

    run_dir = Path(bundle["run_dir"])
    summary_track = summary.get("track")
    run_name = run_dir.name
    model_name = summary.get("model_name")
    track_group = _track_group(summary_track)
    artifact_paths = summary.get("artifact_paths", {})
    predictions_path = _maybe_path(
        artifact_paths.get(f"{split_name}_predictions") or artifact_paths.get("test_predictions")
    )
    split_manifest_path = _maybe_path(artifact_paths.get("split_manifest"))
    artifact_size_bytes = _directory_size(run_dir / "artifacts")

    config_snapshot = bundle.get("config_snapshot") or {}
    split_payload = config_snapshot.get("split") or {}
    source_metrics = {
        "run_id": f"{summary_track}:{model_name}:{run_name}",
        "run_name": run_name,
        "run_dir": str(run_dir),
        "timestamp": _file_timestamp(run_dir / "run_summary.json"),
        "track": summary_track,
        "track_group": track_group,
        "model_name": model_name,
        "split_strategy": split_payload.get("strategy") or metrics_by_split.get("split_strategy"),
        "dataset_view": TRACK_VIEW_MAP.get(summary_track),
        "training_mode": summary.get("training_mode") or config_snapshot.get("model_config", {}).get("training_mode"),
        "server_side_inference_only": bool(
            summary.get("server_side_inference_only") or config_snapshot.get("server_side_inference_only")
        ),
        "artifact_paths": artifact_paths,
        "metrics_path": str(_maybe_path(artifact_paths.get("metrics")) or ""),
        "predictions_path": str(predictions_path or ""),
        "split_manifest_path": str(split_manifest_path or ""),
        "dependency_snapshot_path": str(run_dir / "package_versions.json"),
        "artifact_size_bytes": artifact_size_bytes,
        "artifact_size_mb": round(artifact_size_bytes / (1024 * 1024), 6),
        "sample_count": split_metrics.get("sample_count"),
        "accuracy": split_metrics.get("accuracy"),
        "precision": split_metrics.get("precision"),
        "recall": split_metrics.get("recall"),
        "f1": split_metrics.get("f1"),
        "pr_auc": split_metrics.get("pr_auc"),
        "roc_auc": split_metrics.get("roc_auc"),
        "false_positive_rate": split_metrics.get("false_positive_rate"),
        "false_negative_rate": split_metrics.get("false_negative_rate"),
        "tn": (split_metrics.get("confusion_matrix") or {}).get("tn"),
        "fp": (split_metrics.get("confusion_matrix") or {}).get("fp"),
        "fn": (split_metrics.get("confusion_matrix") or {}).get("fn"),
        "tp": (split_metrics.get("confusion_matrix") or {}).get("tp"),
        "class_distribution": metrics_by_split.get("class_distribution"),
        "excluded_counts": summary.get("excluded_counts") or metrics_by_split.get("excluded_counts"),
        "text_field_priority": metrics_by_split.get("text_field_priority") or config_snapshot.get("text_field_priority"),
        "input_fields": config_snapshot.get("input_fields"),
        "feature_names_used": config_snapshot.get("feature_names_used"),
        "inference_timing_ms": None,
        "package_versions": bundle.get("package_versions") or {},
    }
    return source_metrics


def save_aggregate_payload(
    payload: dict[str, Any],
    *,
    output_dir: str | Path,
    basename: str = "comparison",
) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = write_json(target / f"{basename}.json", payload)
    csv_path = _write_comparison_csv(target / f"{basename}.csv", payload.get("rows", []))
    return {"json": str(json_path), "csv": str(csv_path)}


def _write_comparison_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = ensure_parent_dir(path)
    fieldnames = [
        "run_id",
        "track",
        "track_group",
        "model_name",
        "run_name",
        "split_strategy",
        "dataset_view",
        "training_mode",
        "artifact_size_mb",
        "sample_count",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "false_positive_rate",
        "false_negative_rate",
        "predictions_path",
        "run_dir",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flattened = {
                key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (dict, list)) else row.get(key)
                for key in fieldnames
            }
            writer.writerow(flattened)
    return target


def _track_group(track: str | None) -> str:
    if track == "webpage_transformer":
        return "transformer"
    return track or "unknown"


def _maybe_path(value: Any) -> Path | None:
    if not value:
        return None
    return Path(value)


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def _file_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
