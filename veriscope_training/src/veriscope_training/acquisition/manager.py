from __future__ import annotations

from pathlib import Path
from typing import Any

from veriscope_training.acquisition.base import AcquisitionError, DatasetFetcher, FetchResult
from veriscope_training.acquisition.fetch_mendeley import MendeleyFetcher
from veriscope_training.acquisition.fetch_openphish import OpenPhishFetcher
from veriscope_training.acquisition.fetch_oscar_aux import OscarAuxFetcher
from veriscope_training.acquisition.fetch_phishtank import PhishTankFetcher
from veriscope_training.acquisition.fetch_phreshphish import PhreshPhishFetcher
from veriscope_training.acquisition.fetch_uci_phishing import UCIPhishingFetcher
from veriscope_training.acquisition.validation import validate_raw_dataset
from veriscope_training.utils.io import write_json


FETCHER_REGISTRY: dict[str, type[DatasetFetcher]] = {
    "phreshphish": PhreshPhishFetcher,
    "openphish": OpenPhishFetcher,
    "phishtank": PhishTankFetcher,
    "uci_phishing": UCIPhishingFetcher,
    "mendeley": MendeleyFetcher,
    "oscar_aux": OscarAuxFetcher,
}


def create_fetcher(name: str, config) -> DatasetFetcher:
    try:
        fetcher_cls = FETCHER_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"No fetcher is registered for dataset '{name}'.") from exc
    return fetcher_cls(config.get_source(name), config)


def list_fetch_sources(config) -> dict[str, Any]:
    rows = []
    for name, source in sorted(config.sources.items()):
        fetch_cfg = source.extra.get("fetch") or {}
        rows.append(
            {
                "name": name,
                "enabled": source.enabled,
                "fetch_mode": fetch_cfg.get("fetch_mode", "manual_snapshot"),
                "output_dir": str(config.paths.resolve(fetch_cfg.get("output_dir") or source.snapshot_path or f"data/raw/{name}")),
                "remote_source_type": (fetch_cfg.get("remote_source") or {}).get("type"),
                "resumable": bool(fetch_cfg.get("resumable", True)),
                "supported_local_formats": source.extra.get("supported_local_formats", []),
            }
        )
    return {"fetch_sources": rows}


def inspect_fetch_config(config, name: str) -> dict[str, Any]:
    source = config.get_source(name)
    fetch_cfg = source.extra.get("fetch") or {}
    return {
        "name": name,
        "enabled": source.enabled,
        "snapshot_path": str(config.snapshot_path_for(name)),
        "fetch": fetch_cfg,
        "supported_local_formats": source.extra.get("supported_local_formats", []),
        "notes": list(source.notes),
    }


def fetch_dataset(config, name: str, *, force: bool = False) -> dict[str, Any]:
    fetcher = create_fetcher(name, config)
    result = fetcher.fetch(force=force)
    validation = validate_raw_dataset(config, name)
    payload = {
        "dataset_name": name,
        "fetch_result": result.to_dict(),
        "validation": validation,
    }
    manifest_path = _fetch_manifest_path(config, name)
    write_json(manifest_path, payload)
    payload["manifest_path"] = str(manifest_path)
    return payload


def fetch_all_datasets(config, *, force: bool = False) -> dict[str, Any]:
    results = []
    errors = []
    for name, source in sorted(config.sources.items()):
        fetch_mode = (source.extra.get("fetch") or {}).get("fetch_mode", "manual_snapshot")
        if fetch_mode == "disabled":
            continue
        try:
            results.append(fetch_dataset(config, name, force=force))
        except Exception as exc:
            errors.append({"dataset_name": name, "error": str(exc)})
    payload = {"results": results, "errors": errors}
    write_json(_fetch_manifest_path(config, "all_datasets"), payload)
    return payload


def _fetch_manifest_path(config, name: str) -> Path:
    return config.paths.manifest_data / f"fetch_{name}.json"
