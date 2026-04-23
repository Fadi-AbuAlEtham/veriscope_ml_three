from __future__ import annotations

from pathlib import Path
from typing import Any

from veriscope_training.datasets.registry import create_adapter


def validate_raw_dataset(config, name: str, *, preview_limit: int = 3) -> dict[str, Any]:
    source = config.get_source(name)
    snapshot_root = config.snapshot_path_for(name)
    adapter = create_adapter(name, config)
    report = adapter.validation_report()
    files = sorted(path for path in snapshot_root.rglob("*") if path.is_file()) if snapshot_root.exists() else []
    payload = {
        "dataset_name": name,
        "snapshot_root": str(snapshot_root),
        "file_count": len(files),
        "files_preview": [str(path) for path in files[:10]],
        "adapter_validation": report,
        "status": "ready",
        "warnings": [],
        "errors": [],
        "sample_preview": [],
    }
    if not files:
        payload["status"] = "not_ready"
        payload["warnings"].append(f"No raw files were found under {snapshot_root}.")
        return payload
    try:
        samples = [record.to_dict() for record in adapter.preview_records(limit=preview_limit)]
        payload["sample_preview"] = samples
        if not samples:
            payload["status"] = "warning"
            payload["warnings"].append("Files are present but the adapter did not yield preview records.")
    except Exception as exc:
        payload["status"] = "warning"
        payload["errors"].append(str(exc))
    return payload
