from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from veriscope_training.config import AppConfig
from veriscope_training.datasets.manifest import BuildDatasetSummary, ProcessedDatasetManifest
from veriscope_training.datasets.registry import create_adapter, dataset_registry_summary
from veriscope_training.preprocessing.deduplication import ProcessedDeduplicator
from veriscope_training.preprocessing.multilingual_dataset import (
    canonical_language,
    rebalance_multilingual_view,
    write_multilingual_report,
)
from veriscope_training.preprocessing.record_normalization import (
    ProcessedRecord,
    normalize_dataset_record,
    validate_processed_record,
)
from veriscope_training.preprocessing.text_processing import TextNormalizationConfig
from veriscope_training.utils.io import (
    StructuredDatasetWriter,
    available_disk_space_bytes,
    read_json,
    read_jsonl,
    write_json,
)


VIEW_NAME_MAP = {
    "url": "unified_url_dataset",
    "webpage": "unified_webpage_dataset",
    "tabular": "unified_tabular_dataset",
    "auxiliary": "unified_auxiliary_text_dataset",
    "unified_url_dataset": "unified_url_dataset",
    "unified_webpage_dataset": "unified_webpage_dataset",
    "unified_tabular_dataset": "unified_tabular_dataset",
    "unified_auxiliary_text_dataset": "unified_auxiliary_text_dataset",
}


@dataclass(frozen=True)
class BuildDatasetOptions:
    source_names: list[str] | None = None
    view_names: list[str] | None = None
    output_format: str | None = None
    max_records_per_source: int | None = None
    preview_limit: int = 5
    deduplicate: bool | None = None
    skip_completed: bool | None = None
    force_rebuild: bool = False


def build_dataset_plan(config: AppConfig) -> dict[str, object]:
    summary = dataset_registry_summary(config)
    dataset_cfg = config.experiments_config.get("datasets", {})
    preprocessing_cfg = dataset_cfg.get("preprocessing", {})
    return {
        "phase": 3,
        "project_root": str(config.paths.root),
        "enabled_sources": sorted(config.enabled_sources()),
        "dataset_registry": summary,
        "processed_outputs": [
            "unified_url_dataset",
            "unified_webpage_dataset",
            "unified_tabular_dataset",
            "unified_auxiliary_text_dataset",
        ],
        "default_output_format": dataset_cfg.get("output_format", "parquet"),
        "deduplication_enabled": dataset_cfg.get("deduplication", {}).get("enabled", True),
        "resume_enabled": dataset_cfg.get("resume", {}).get("skip_completed_views", True),
        "preprocessing": preprocessing_cfg,
        "next_step": "Run build-dataset to materialize normalized processed datasets and dedupe reports.",
    }


def preview_processed_records(config: AppConfig, options: BuildDatasetOptions) -> dict[str, Any]:
    selected_sources = _resolve_sources(config, options.source_names)
    selected_views = _resolve_views(options.view_names)
    text_config = _text_config(config)
    preprocessing_cfg = _preprocessing_config(config)
    dedup_enabled = options.deduplicate if options.deduplicate is not None else _dedupe_enabled(config)
    deduplicator = ProcessedDeduplicator(enabled=dedup_enabled, policy=_dedupe_policy(config))

    records: list[dict[str, Any]] = []
    source_errors: dict[str, str] = {}
    for source_name in selected_sources:
        adapter = create_adapter(source_name, config)
        try:
            for record_index, raw_record in enumerate(adapter.iterate_records(), start=1):
                if options.max_records_per_source and record_index > options.max_records_per_source:
                    break
                processed = normalize_dataset_record(
                    raw_record,
                    source_config=config.get_source(source_name),
                    text_config=text_config,
                    preprocessing_config=preprocessing_cfg,
                )
                warnings = validate_processed_record(processed)
                if warnings:
                    processed.processing_metadata["validation_warnings"] = warnings
                keep, duplicate_event = deduplicator.process(processed)
                if not keep:
                    continue
                if not any(view in processed.view_names() for view in selected_views):
                    continue
                records.append(processed.to_dict())
                if len(records) >= options.preview_limit:
                    return {
                        "selected_sources": selected_sources,
                        "selected_views": selected_views,
                        "records": records,
                        "dedupe_summary": deduplicator.report().to_dict(),
                        "source_errors": source_errors,
                    }
        except Exception as exc:
            source_errors[source_name] = str(exc)
    if not records:
        error_text = "; ".join(f"{source}: {message}" for source, message in source_errors.items()) or (
            "No processed records were produced. Check snapshot availability and source selection."
        )
        raise RuntimeError(error_text)
    return {
        "selected_sources": selected_sources,
        "selected_views": selected_views,
        "records": records,
        "dedupe_summary": deduplicator.report().to_dict(),
        "source_errors": source_errors,
    }


def build_processed_datasets(config: AppConfig, options: BuildDatasetOptions) -> dict[str, Any]:
    config.paths.ensure_directories()
    selected_sources = _resolve_sources(config, options.source_names)
    selected_views = _resolve_views(options.view_names)
    dataset_cfg = config.experiments_config.get("datasets", {})
    output_format = options.output_format or dataset_cfg.get("output_format", "parquet")
    parquet_batch_size = int(dataset_cfg.get("parquet_batch_size", 1000))
    parquet_compression = str(dataset_cfg.get("parquet_compression", "zstd"))
    text_config = _text_config(config)
    preprocessing_cfg = _preprocessing_config(config)
    dedup_enabled = options.deduplicate if options.deduplicate is not None else _dedupe_enabled(config)
    deduplicator = ProcessedDeduplicator(enabled=dedup_enabled, policy=_dedupe_policy(config))
    skip_completed = (
        options.skip_completed
        if options.skip_completed is not None
        else bool(dataset_cfg.get("resume", {}).get("skip_completed_views", True))
    )

    preflight = _estimate_build_storage(config, selected_sources, selected_views, preprocessing_cfg)
    existing_manifests = {view_name: _load_view_manifest(config, view_name) for view_name in selected_views}
    completed_views = [
        view_name
        for view_name in selected_views
        if skip_completed and not options.force_rebuild and _view_is_materialized(existing_manifests.get(view_name))
    ]
    pending_views = [view_name for view_name in selected_views if view_name not in completed_views]

    build_state_path = config.paths.manifest_data / "build_dataset.state.json"
    write_json(
        build_state_path,
        {
            "status": "running",
            "selected_sources": selected_sources,
            "selected_views": selected_views,
            "pending_views": pending_views,
            "completed_views": completed_views,
            "output_format": output_format,
            "preflight": preflight,
        },
    )

    if not pending_views:
        existing_summary = _load_existing_build_summary(config)
        payload = {
            **(existing_summary or {}),
            "selected_sources": selected_sources,
            "selected_views": selected_views,
            "output_paths": {
                view_name: existing_manifests[view_name]["output_paths"]
                for view_name in selected_views
                if existing_manifests.get(view_name)
            },
            "warnings": (existing_summary or {}).get("warnings", []) + ["all_selected_views_skipped_as_completed"],
            "metadata": {
                **((existing_summary or {}).get("metadata", {})),
                "preflight": preflight,
                "resumed": True,
                "skipped_completed_views": completed_views,
                "build_state_path": str(build_state_path),
            },
        }
        write_json(build_state_path, {"status": "completed", **payload.get("metadata", {})})
        write_json(config.paths.manifest_data / "build_dataset.summary.json", payload)
        return payload

    stage_dir = config.paths.interim_data / "build_dataset_stage"
    staged_views = {
        view_name
        for view_name in pending_views
        if _use_multilingual_stage(view_name=view_name, config=config)
    }
    writers = {
        view_name: StructuredDatasetWriter(
            (stage_dir / view_name) if view_name in staged_views else (config.paths.processed_data / view_name),
            output_format=output_format,
            parquet_batch_size=parquet_batch_size,
            parquet_compression=parquet_compression,
        )
        for view_name in pending_views
    }
    view_stats: dict[str, dict[str, Any]] = {
        view_name: {
            "record_count": 0,
            "source_counts": {},
            "label_counts": {"phishing": 0, "benign": 0, "null": 0},
            "modality_counts": {
                "has_original_url": 0,
                "has_raw_html": 0,
                "has_extracted_text": 0,
                "has_tabular_features": 0,
                "is_auxiliary": 0,
            },
            "validation_warning_count": 0,
        }
        for view_name in pending_views
    }

    duplicate_events_path = config.paths.manifest_data / "build_dataset.duplicates.jsonl"
    duplicate_event_limit = _duplicate_event_limit(config)
    duplicate_events_handle = None
    duplicate_events_written = 0
    if duplicate_events_path.exists() and not (_save_duplicate_events(config) and dedup_enabled and duplicate_event_limit > 0):
        duplicate_events_path.unlink()
    if dedup_enabled and _save_duplicate_events(config) and duplicate_event_limit > 0:
        duplicate_events_handle = duplicate_events_path.open("w", encoding="utf-8")

    source_summaries: dict[str, dict[str, Any]] = {}
    exclusion_summary: dict[str, dict[str, int]] = {}
    total_raw_records = 0
    total_processed_records = 0
    preview_samples: list[dict[str, Any]] = []

    try:
        for source_name in selected_sources:
            adapter = create_adapter(source_name, config)
            summary = {
                "raw_records_seen": 0,
                "processed_records": 0,
                "kept_records": 0,
                "duplicate_records_removed": 0,
                "validation_warning_count": 0,
                "output_views": {view_name: 0 for view_name in selected_views},
                "excluded_counts": {},
                "errors": [],
            }
            try:
                for record_index, raw_record in enumerate(adapter.iterate_records(), start=1):
                    if options.max_records_per_source and record_index > options.max_records_per_source:
                        break
                    summary["raw_records_seen"] += 1
                    total_raw_records += 1
                    if _should_skip_raw_record(raw_record=raw_record, source_name=source_name, config=config):
                        _increment_count(summary["excluded_counts"], "raw_prefiltered:english")
                        continue

                    try:
                        processed = normalize_dataset_record(
                            raw_record,
                            source_config=config.get_source(source_name),
                            text_config=text_config,
                            preprocessing_config=preprocessing_cfg,
                        )
                        warnings = validate_processed_record(processed)
                        if warnings:
                            processed.processing_metadata["validation_warnings"] = warnings
                            summary["validation_warning_count"] += len(warnings)
                        summary["processed_records"] += 1
                        total_processed_records += 1

                        keep, duplicate_event = deduplicator.process(processed)
                        if not keep:
                            summary["duplicate_records_removed"] += 1
                            if (
                                duplicate_events_handle is not None
                                and duplicate_event is not None
                                and duplicate_events_written < duplicate_event_limit
                            ):
                                duplicate_events_handle.write(json.dumps(duplicate_event.to_dict(), sort_keys=True))
                                duplicate_events_handle.write("\n")
                                duplicate_events_written += 1
                            continue

                        target_views = [view_name for view_name in processed.view_names() if view_name in pending_views]
                        if not target_views:
                            _increment_count(summary["excluded_counts"], "no_target_views")
                            continue
                        summary["kept_records"] += 1
                        if len(preview_samples) < options.preview_limit:
                            preview_samples.append(processed.to_dict())
                        for view_name in target_views:
                            writers[view_name].write(processed.to_dict())
                            _update_view_stats(view_stats[view_name], processed)
                            summary["output_views"][view_name] += 1
                    except Exception as exc:
                        _increment_count(summary["excluded_counts"], f"processing_error:{type(exc).__name__}")
            except Exception as exc:
                summary["errors"].append(str(exc))
            source_summaries[source_name] = summary
            exclusion_summary[source_name] = summary["excluded_counts"]
    finally:
        if duplicate_events_handle is not None:
            duplicate_events_handle.close()
            if duplicate_events_written == 0 and duplicate_events_path.exists():
                duplicate_events_path.unlink()
        for writer in writers.values():
            writer.close()

    total_kept_records = deduplicator.report().kept_count
    existing_record_count = sum(
        int((existing_manifests.get(view_name) or {}).get("record_count", 0)) for view_name in completed_views
    )
    if total_kept_records == 0 and existing_record_count == 0:
        errors = [
            f"{source}: {', '.join(summary['errors'])}"
            for source, summary in source_summaries.items()
            if summary["errors"]
        ]
        joined = "; ".join(errors) or "No records were materialized. Check source snapshots and view filters."
        write_json(
            build_state_path,
            {
                "status": "failed",
                "error": joined,
                "selected_sources": selected_sources,
                "selected_views": selected_views,
                "pending_views": pending_views,
                "completed_views": completed_views,
                "preflight": preflight,
            },
        )
        raise RuntimeError(joined)

    dedupe_report_path = config.paths.manifest_data / "build_dataset.dedupe_report.json"
    write_json(dedupe_report_path, deduplicator.report().to_dict())
    exclusion_report_path = config.paths.manifest_data / "build_dataset.exclusions.json"
    write_json(exclusion_report_path, exclusion_summary)

    output_paths: dict[str, dict[str, str]] = {}
    final_view_stats: dict[str, dict[str, Any]] = {}
    multilingual_view_reports: dict[str, Any] = {}
    for view_name in completed_views:
        manifest = existing_manifests.get(view_name) or {}
        output_paths[view_name] = manifest.get("output_paths", {})
        final_view_stats[view_name] = {
            "record_count": int(manifest.get("record_count", 0)),
            "source_counts": manifest.get("source_counts", {}),
            "label_counts": manifest.get("label_counts", {"phishing": 0, "benign": 0, "null": 0}),
            "modality_counts": manifest.get("modality_counts", {}),
            "validation_warning_count": manifest.get("validation_summary", {}).get("warning_count", 0),
        }
    for view_name, writer in writers.items():
        if view_name in staged_views:
            staged_records_path = _preferred_output_path(writer.output_paths)
            rebalance_summary = rebalance_multilingual_view(
                view_name=view_name,
                staged_records_path=staged_records_path,
                final_base_path=config.paths.processed_data / view_name,
                output_format=output_format,
                parquet_batch_size=parquet_batch_size,
                parquet_compression=parquet_compression,
                balance_config=_multilingual_config(config),
            )
            multilingual_view_reports[view_name] = rebalance_summary
            output_paths[view_name] = rebalance_summary["final_output_paths"]
            final_view_stats[view_name] = _view_stats_from_report(rebalance_summary["post_balance"])
        else:
            output_paths[view_name] = writer.output_paths
            final_view_stats[view_name] = view_stats[view_name]

    locally_available_sources = [
        source_name
        for source_name in sorted(config.sources)
        if config.snapshot_path_for(source_name).exists()
    ]
    if multilingual_view_reports:
        report_paths = write_multilingual_report(
            report_base_path=config.paths.manifest_data / "multilingual_dataset_report",
            selected_sources=selected_sources,
            configured_sources=sorted(config.sources),
            locally_available_sources=locally_available_sources,
            source_summaries=source_summaries,
            exclusion_summary=exclusion_summary,
            view_reports=multilingual_view_reports,
        )
    else:
        report_paths = {}

    for view_name in pending_views:
        manifest_output_paths = output_paths[view_name]
        manifest_view_stats = final_view_stats[view_name]
        manifest = ProcessedDatasetManifest(
            view_name=view_name,
            output_paths=manifest_output_paths,
            record_count=manifest_view_stats["record_count"],
            sources=sorted(manifest_view_stats["source_counts"]),
            source_counts=manifest_view_stats["source_counts"],
            label_counts=manifest_view_stats["label_counts"],
            modality_counts=manifest_view_stats["modality_counts"],
            warnings=(["empty_dataset"] if manifest_view_stats["record_count"] == 0 else []),
            validation_summary={"warning_count": manifest_view_stats["validation_warning_count"]},
            dedupe_summary=deduplicator.report().to_dict(),
            metadata={
                "output_format": output_format,
                "parquet_compression": parquet_compression if output_format in {"parquet", "both"} else None,
                "storage_policy": preprocessing_cfg.get("storage", {}),
                "preflight": preflight,
                "resumed": False,
                "multilingual_rebalance": multilingual_view_reports.get(view_name),
            },
        )
        write_json(config.paths.manifest_data / f"{view_name}.manifest.json", manifest.to_dict())

    summary = BuildDatasetSummary(
        selected_sources=selected_sources,
        selected_views=selected_views,
        output_paths=output_paths,
        source_summaries=source_summaries,
        dedupe_report_path=str(dedupe_report_path),
        duplicate_events_path=str(duplicate_events_path) if duplicate_events_path.exists() else None,
        total_raw_records=total_raw_records,
        total_processed_records=total_processed_records,
        total_kept_records=total_kept_records + existing_record_count,
        warnings=[
            f"{view_name} is empty"
            for view_name, stats in final_view_stats.items()
            if stats["record_count"] == 0
        ],
        metadata={
            "preview_samples": preview_samples,
            "preflight": preflight,
            "resumed": bool(completed_views),
            "skipped_completed_views": completed_views,
            "pending_views_built": pending_views,
            "build_state_path": str(build_state_path),
            "duplicate_event_samples_saved": duplicate_events_written,
            "duplicate_event_sample_limit": duplicate_event_limit,
            "exclusion_report_path": str(exclusion_report_path),
            "multilingual_report_paths": report_paths,
            "multilingual_view_reports": multilingual_view_reports,
        },
    )
    summary_path = config.paths.manifest_data / "build_dataset.summary.json"
    write_json(summary_path, summary.to_dict())
    write_json(
        build_state_path,
        {
            "status": "completed",
            "selected_sources": selected_sources,
            "selected_views": selected_views,
            "completed_views": selected_views,
            "pending_views": [],
            "output_paths": output_paths,
            "summary_path": str(summary_path),
            "preflight": preflight,
        },
    )
    return summary.to_dict()


def load_manifest(config: AppConfig, name_or_path: str | None = None) -> dict[str, Any]:
    path = _resolve_manifest_path(config, name_or_path or "build_dataset.summary.json")
    return read_json(path)


def load_dedupe_report(config: AppConfig, *, include_events: bool = False, limit: int = 20) -> dict[str, Any]:
    report_path = config.paths.manifest_data / "build_dataset.dedupe_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Dedupe report not found at {report_path}. Run build-dataset first.")
    payload = {"summary": read_json(report_path), "report_path": str(report_path)}
    if include_events:
        events_path = config.paths.manifest_data / "build_dataset.duplicates.jsonl"
        payload["events_path"] = str(events_path)
        payload["events"] = list(_read_jsonl_limited(events_path, limit)) if events_path.exists() else []
    return payload


def _resolve_sources(config: AppConfig, source_names: list[str] | None) -> list[str]:
    if not source_names:
        return sorted(config.enabled_sources())
    return [config.get_source(name).name for name in source_names]


def _resolve_views(view_names: list[str] | None) -> list[str]:
    if not view_names:
        return [
            "unified_url_dataset",
            "unified_webpage_dataset",
            "unified_tabular_dataset",
            "unified_auxiliary_text_dataset",
        ]
    resolved: list[str] = []
    for view_name in view_names:
        try:
            resolved.append(VIEW_NAME_MAP[view_name])
        except KeyError as exc:
            available = ", ".join(sorted(VIEW_NAME_MAP))
            raise KeyError(f"Unknown view '{view_name}'. Available views: {available}") from exc
    return list(dict.fromkeys(resolved))


def _text_config(config: AppConfig) -> TextNormalizationConfig:
    mapping = config.experiments_config.get("datasets", {}).get("preprocessing", {}).get("text", {})
    return TextNormalizationConfig.from_mapping(mapping)


def _preprocessing_config(config: AppConfig) -> dict[str, Any]:
    return config.experiments_config.get("datasets", {}).get("preprocessing", {})


def _dedupe_enabled(config: AppConfig) -> bool:
    return bool(config.experiments_config.get("datasets", {}).get("deduplication", {}).get("enabled", True))


def _dedupe_policy(config: AppConfig) -> str:
    return str(config.experiments_config.get("datasets", {}).get("deduplication", {}).get("policy", "first_seen"))


def _save_duplicate_events(config: AppConfig) -> bool:
    return bool(
        config.experiments_config.get("datasets", {}).get("deduplication", {}).get("save_duplicate_events", True)
    )


def _duplicate_event_limit(config: AppConfig) -> int:
    return int(
        config.experiments_config.get("datasets", {}).get("deduplication", {}).get("duplicate_event_sample_limit", 200)
    )


def _multilingual_config(config: AppConfig) -> dict[str, Any]:
    return config.experiments_config.get("datasets", {}).get("multilingual", {})


def _use_multilingual_stage(*, view_name: str, config: AppConfig) -> bool:
    multilingual_cfg = _multilingual_config(config)
    if not multilingual_cfg.get("enabled", False):
        return False
    selected_views = set(multilingual_cfg.get("rebalance_views", ["unified_url_dataset", "unified_webpage_dataset"]))
    return view_name in selected_views


def _preferred_output_path(output_paths: dict[str, str]) -> Path:
    if "parquet" in output_paths:
        return Path(output_paths["parquet"])
    if "jsonl" in output_paths:
        return Path(output_paths["jsonl"])
    raise ValueError("No output path available for staged view.")


def _view_stats_from_report(report: dict[str, Any]) -> dict[str, Any]:
    counts_by_label = report.get("counts_by_label", {})
    return {
        "record_count": int(report.get("total_sample_count", 0)),
        "source_counts": report.get("counts_by_dataset", {}),
        "label_counts": {
            "phishing": int(counts_by_label.get("phishing", 0)),
            "benign": int(counts_by_label.get("benign", 0)),
            "null": int(counts_by_label.get("null", 0)),
        },
        "modality_counts": report.get("modality_counts", {}),
        "validation_warning_count": int(report.get("validation_warning_count", 0)),
    }


def _increment_count(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _should_skip_raw_record(*, raw_record: Any, source_name: str, config: AppConfig) -> bool:
    multilingual_cfg = _multilingual_config(config)
    if not multilingual_cfg.get("enabled", False):
        return False
    if source_name != "phreshphish":
        return False
    keep_ratio = float(multilingual_cfg.get("english_prefilter_keep_ratio", 1.0))
    if keep_ratio >= 1.0:
        return False
    language = canonical_language(getattr(raw_record, "language", None))
    if language != "en":
        return False
    sample_id = getattr(raw_record, "sample_id", None)
    if not sample_id:
        return False
    return _stable_fraction(str(sample_id), salt="english_prefilter") > keep_ratio


def _stable_fraction(value: str, *, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big", signed=False)
    return bucket / float(2**64 - 1)


def _load_existing_build_summary(config: AppConfig) -> dict[str, Any] | None:
    path = config.paths.manifest_data / "build_dataset.summary.json"
    if not path.exists():
        return None
    return read_json(path)


def _load_view_manifest(config: AppConfig, view_name: str) -> dict[str, Any] | None:
    path = config.paths.manifest_data / f"{view_name}.manifest.json"
    if not path.exists():
        return None
    return read_json(path)


def _view_is_materialized(manifest: dict[str, Any] | None) -> bool:
    if not manifest:
        return False
    output_paths = manifest.get("output_paths", {})
    if not output_paths:
        return False
    for raw_path in output_paths.values():
        path = Path(raw_path)
        if not path.exists() or path.stat().st_size <= 0:
            return False
    return int(manifest.get("record_count", 0)) > 0


def _estimate_build_storage(
    config: AppConfig,
    selected_sources: list[str],
    selected_views: list[str],
    preprocessing_cfg: dict[str, Any],
) -> dict[str, Any]:
    raw_bytes = 0
    snapshot_roots: dict[str, str] = {}
    for source_name in selected_sources:
        snapshot_root = config.snapshot_path_for(source_name)
        snapshot_roots[source_name] = str(snapshot_root)
        raw_bytes += _directory_size_bytes(snapshot_root)

    storage_cfg = preprocessing_cfg.get("storage", {})
    keep_raw_html = bool(storage_cfg.get("keep_raw_html", False))
    estimate_factor = 0.0
    if "unified_url_dataset" in selected_views:
        estimate_factor += 0.08
    if "unified_webpage_dataset" in selected_views:
        estimate_factor += 1.00 if keep_raw_html else 0.28
    if "unified_tabular_dataset" in selected_views:
        estimate_factor += 0.05
    if "unified_auxiliary_text_dataset" in selected_views:
        estimate_factor += 0.08

    estimated_required_bytes = int(raw_bytes * estimate_factor) + 128 * 1024 * 1024
    free_bytes = available_disk_space_bytes(config.paths.root)
    recommended_minimum_bytes = int(estimated_required_bytes * 1.25)
    return {
        "raw_snapshot_bytes": raw_bytes,
        "estimated_required_bytes": estimated_required_bytes,
        "recommended_minimum_bytes": recommended_minimum_bytes,
        "free_bytes": free_bytes,
        "likely_insufficient": free_bytes < recommended_minimum_bytes,
        "selected_views": selected_views,
        "storage_policy": storage_cfg,
        "snapshot_roots": snapshot_roots,
    }


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for candidate in path.rglob("*"):
        if candidate.is_file():
            total += candidate.stat().st_size
    return total


def _update_view_stats(stats: dict[str, Any], record: ProcessedRecord) -> None:
    stats["record_count"] += 1
    stats["source_counts"][record.source_dataset] = stats["source_counts"].get(record.source_dataset, 0) + 1
    label_bucket = "null"
    if record.normalized_label == 1:
        label_bucket = "phishing"
    elif record.normalized_label == 0:
        label_bucket = "benign"
    stats["label_counts"][label_bucket] += 1
    for key in stats["modality_counts"]:
        if record.modality_flags.get(key):
            stats["modality_counts"][key] += 1
    stats["validation_warning_count"] += len(record.processing_metadata.get("validation_warnings", []))


def _resolve_manifest_path(config: AppConfig, name_or_path: str) -> Path:
    direct_path = Path(name_or_path)
    if direct_path.is_absolute() and direct_path.exists():
        return direct_path
    if direct_path.exists():
        return direct_path.resolve()
    if name_or_path in VIEW_NAME_MAP:
        candidate = config.paths.manifest_data / f"{VIEW_NAME_MAP[name_or_path]}.manifest.json"
        if candidate.exists():
            return candidate
    if name_or_path in VIEW_NAME_MAP.values():
        candidate = config.paths.manifest_data / f"{name_or_path}.manifest.json"
        if candidate.exists():
            return candidate
    candidate = config.paths.manifest_data / name_or_path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Manifest not found for '{name_or_path}'.")


def _read_jsonl_limited(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(path), start=1):
        rows.append(row)
        if index >= limit:
            break
    return rows
