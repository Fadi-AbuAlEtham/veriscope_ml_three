from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from veriscope_training.adaptive.drift_monitoring import generate_drift_report, load_rows_for_drift
from veriscope_training.adaptive.heuristic_proposals import generate_heuristic_proposals
from veriscope_training.adaptive.retraining_candidates import export_retraining_candidates
from veriscope_training.config import AppConfig
from veriscope_training.datasets.registry import create_adapter, dataset_registry_summary
from veriscope_training.logging_utils import configure_logging
from veriscope_training.models.artifacts import load_training_run
from veriscope_training.utils.io import read_json, write_json
from veriscope_training.pipelines.build_dataset import (
    BuildDatasetOptions,
    build_dataset_plan,
    build_processed_datasets,
    load_dedupe_report,
    load_manifest,
    preview_processed_records,
)
from veriscope_training.pipelines.train_all import list_available_models, train_all_enabled_models, train_model


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return {
            key: _to_jsonable(item)
            for key, item in value.__dict__.items()
            if not key.startswith("_")
        }
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veriscope-training")
    parser.add_argument("--project-root", default=None, help="Override project root.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_dirs = subparsers.add_parser("init-dirs", help="Create expected data/output directories.")
    init_dirs.add_argument("--json", action="store_true", help="Return JSON output.")

    show_config = subparsers.add_parser("show-config", help="Print config sections.")
    show_config.add_argument(
        "--section",
        choices=["datasets", "experiments", "models", "all"],
        default="all",
        help="Which config section to print.",
    )

    list_datasets = subparsers.add_parser("list-datasets", help="List configured and registered datasets.")
    list_datasets.add_argument("--json", action="store_true", help="Return JSON output.")

    list_fetch_sources = subparsers.add_parser("list-fetch-sources", help="List server-side raw-data acquisition sources.")
    list_fetch_sources.add_argument("--json", action="store_true", help="Return JSON output.")

    inspect_fetch = subparsers.add_parser("inspect-fetch-config", help="Inspect fetch configuration for one dataset.")
    inspect_fetch.add_argument("name", help="Configured source name.")
    inspect_fetch.add_argument("--json", action="store_true", help="Return JSON output.")

    inspect_source = subparsers.add_parser("inspect-source", help="Inspect a configured data source.")
    inspect_source.add_argument("name", help="Configured source name.")
    inspect_source.add_argument("--json", action="store_true", help="Return JSON output.")

    inspect_adapter = subparsers.add_parser("inspect-adapter", help="Inspect adapter metadata and validation.")
    inspect_adapter.add_argument("name", help="Configured source name.")
    inspect_adapter.add_argument("--json", action="store_true", help="Return JSON output.")

    preview_records = subparsers.add_parser(
        "preview-records",
        help="Preview a few normalized records from an adapter without building datasets.",
    )
    preview_records.add_argument("name", help="Configured source name.")
    preview_records.add_argument("--limit", type=int, default=3, help="Maximum records to preview.")
    preview_records.add_argument("--json", action="store_true", help="Return JSON output.")

    fetch_dataset = subparsers.add_parser("fetch-dataset", help="Fetch or prepare one raw dataset directly on the server.")
    fetch_dataset.add_argument("name", help="Configured source name.")
    fetch_dataset.add_argument("--force", action="store_true", help="Re-fetch even if local files already exist.")
    fetch_dataset.add_argument("--json", action="store_true", help="Return JSON output.")

    fetch_all = subparsers.add_parser("fetch-all-datasets", help="Fetch or prepare all configured raw datasets.")
    fetch_all.add_argument("--force", action="store_true", help="Re-fetch even if local files already exist.")
    fetch_all.add_argument("--json", action="store_true", help="Return JSON output.")

    validate_raw = subparsers.add_parser("validate-raw-dataset", help="Validate raw dataset readiness in data/raw/<name>/.")
    validate_raw.add_argument("name", help="Configured source name.")
    validate_raw.add_argument("--json", action="store_true", help="Return JSON output.")

    build_dataset = subparsers.add_parser(
        "build-dataset",
        help="Normalize, deduplicate, and materialize processed unified datasets.",
    )
    build_dataset.add_argument("--source", action="append", dest="sources", help="Limit to one or more sources.")
    build_dataset.add_argument("--view", action="append", dest="views", help="Limit to one or more output views.")
    build_dataset.add_argument(
        "--output-format",
        choices=["jsonl", "parquet", "both"],
        default=None,
        help="Override configured output format.",
    )
    build_dataset.add_argument(
        "--max-records-per-source",
        type=int,
        default=None,
        help="Optional cap per source for smoke tests.",
    )
    build_dataset.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild selected views even if a completed manifest already exists.",
    )
    build_dataset.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Do not skip already materialized processed views.",
    )
    build_dataset.add_argument("--json", action="store_true", help="Return JSON output.")

    preview_processed = subparsers.add_parser(
        "preview-processed",
        help="Preview processed canonical records after normalization and deduplication.",
    )
    preview_processed.add_argument("--source", action="append", dest="sources", help="Limit to one or more sources.")
    preview_processed.add_argument("--view", action="append", dest="views", help="Limit to one or more output views.")
    preview_processed.add_argument("--limit", type=int, default=5, help="Maximum processed records to preview.")
    preview_processed.add_argument(
        "--max-records-per-source",
        type=int,
        default=None,
        help="Optional cap per source for smoke tests.",
    )
    preview_processed.add_argument("--json", action="store_true", help="Return JSON output.")

    show_manifest = subparsers.add_parser("show-manifest", help="Show a saved build or view manifest.")
    show_manifest.add_argument("name_or_path", nargs="?", default=None, help="Manifest file path or view name.")
    show_manifest.add_argument("--json", action="store_true", help="Return JSON output.")

    show_dedupe = subparsers.add_parser("show-dedupe-report", help="Show the latest dedupe summary/report.")
    show_dedupe.add_argument("--events", action="store_true", help="Include recent duplicate event rows.")
    show_dedupe.add_argument("--limit", type=int, default=20, help="Maximum duplicate events to show.")
    show_dedupe.add_argument("--json", action="store_true", help="Return JSON output.")

    list_models = subparsers.add_parser("list-models", help="List enabled trainable models by track.")
    list_models.add_argument("--json", action="store_true", help="Return JSON output.")

    train_model_cmd = subparsers.add_parser("train-model", help="Train one model for a selected track.")
    train_model_cmd.add_argument("--track", required=True, choices=["url", "webpage", "transformer", "tabular"])
    train_model_cmd.add_argument("--model", required=True, help="Model key from configs/models.yaml.")
    train_model_cmd.add_argument("--run-name", default=None, help="Optional run name override.")
    train_model_cmd.add_argument("--split-strategy", default=None, help="Override split strategy.")
    train_model_cmd.add_argument("--json", action="store_true", help="Return JSON output.")

    train_all_cmd = subparsers.add_parser("train-all-baselines", help="Train all enabled models.")
    train_all_cmd.add_argument("--include-transformers", action="store_true", help="Include enabled transformer runs.")
    train_all_cmd.add_argument("--split-strategy", default=None, help="Override split strategy.")
    train_all_cmd.add_argument("--json", action="store_true", help="Return JSON output.")

    show_training = subparsers.add_parser("show-training-run", help="Show a saved training run summary.")
    show_training.add_argument("path_or_dir", help="Run directory or run_summary.json path.")
    show_training.add_argument("--json", action="store_true", help="Return JSON output.")

    evaluate_all = subparsers.add_parser(
        "evaluate-all",
        help="Run a named experiment suite, compare completed runs, calibrate thresholds, and export integration artifacts.",
    )
    evaluate_all.add_argument("--group", default="full_suite", help="Experiment group name from configs/experiments.yaml.")
    evaluate_all.add_argument("--output-dir", default=None, help="Optional output directory.")
    evaluate_all.add_argument("--rerun", action="store_true", help="Ignore existing completed runs and retrain.")
    evaluate_all.add_argument("--json", action="store_true", help="Return JSON output.")

    compare_runs = subparsers.add_parser("compare-runs", help="Aggregate and compare saved training runs.")
    compare_runs.add_argument("--run-dir", action="append", dest="run_dirs", help="One or more run directories.")
    compare_runs.add_argument("--output-dir", default=None, help="Optional report output directory.")
    compare_runs.add_argument("--split", default=None, help="Split name to compare, default from config.")
    compare_runs.add_argument("--json", action="store_true", help="Return JSON output.")

    calibrate_cmd = subparsers.add_parser("calibrate-thresholds", help="Calibrate thresholds for a run or prediction file.")
    calibrate_cmd.add_argument("--run-dir", default=None, help="Training run directory.")
    calibrate_cmd.add_argument("--predictions", default=None, help="Prediction JSONL path.")
    calibrate_cmd.add_argument("--output-dir", default=None, help="Optional calibration output directory.")
    calibrate_cmd.add_argument("--split", default="test", help="Prediction split name when using --run-dir.")
    calibrate_cmd.add_argument("--json", action="store_true", help="Return JSON output.")

    show_comparison = subparsers.add_parser("show-comparison-report", help="Show a saved comparison report JSON.")
    show_comparison.add_argument("path", help="Comparison JSON path.")
    show_comparison.add_argument("--json", action="store_true", help="Return JSON output.")

    show_recommendation = subparsers.add_parser("show-recommendation", help="Show a saved recommendation JSON.")
    show_recommendation.add_argument("path", help="Recommendation JSON path.")
    show_recommendation.add_argument("--json", action="store_true", help="Return JSON output.")

    export_integration = subparsers.add_parser(
        "export-integration-configs",
        help="Compare saved runs, calibrate recommended models, and export VeriScope integration-ready configs.",
    )
    export_integration.add_argument("--run-dir", action="append", dest="run_dirs", help="Optional run directories.")
    export_integration.add_argument("--output-dir", default=None, help="Optional integration output directory.")
    export_integration.add_argument("--json", action="store_true", help="Return JSON output.")

    error_analysis = subparsers.add_parser("error-analysis", help="Generate structured error analysis for a training run.")
    error_analysis.add_argument("--run-dir", required=True, help="Training run directory.")
    error_analysis.add_argument("--split", default="test", help="Prediction split name.")
    error_analysis.add_argument("--output-dir", default=None, help="Optional output directory.")
    error_analysis.add_argument("--json", action="store_true", help="Return JSON output.")

    export_candidates = subparsers.add_parser(
        "export-retraining-candidates",
        help="Export uncertain or reviewed prediction rows as retraining candidates.",
    )
    export_candidates.add_argument("--predictions", required=True, help="Prediction JSONL path.")
    export_candidates.add_argument("--feedback", default=None, help="Optional feedback JSONL path.")
    export_candidates.add_argument("--output-dir", default=None, help="Optional output directory.")
    export_candidates.add_argument("--json", action="store_true", help="Return JSON output.")

    heuristic_cmd = subparsers.add_parser(
        "generate-heuristic-proposals",
        help="Generate review-only heuristic proposals from processed samples.",
    )
    heuristic_cmd.add_argument("--processed", default=None, help="Processed dataset JSONL path.")
    heuristic_cmd.add_argument("--feedback", default=None, help="Optional feedback JSONL path.")
    heuristic_cmd.add_argument("--from-errors", default=None, help="Run error analysis directory.")
    heuristic_cmd.add_argument("--output", default=None, help="Optional output JSON path.")
    heuristic_cmd.add_argument("--top-k", type=int, default=25, help="Maximum proposals to return.")
    heuristic_cmd.add_argument("--json", action="store_true", help="Return JSON output.")

    fusion_cmd = subparsers.add_parser("run-fusion-experiment", help="Run fusion experiment between URL and webpage models.")
    fusion_cmd.add_argument("--url-run", required=True, help="URL model run directory.")
    fusion_cmd.add_argument("--webpage-run", required=True, help="Webpage model run directory.")
    fusion_cmd.add_argument("--output-dir", default=None, help="Optional output directory.")
    fusion_cmd.add_argument("--strategy", choices=["weighted", "cascade", "both"], default="both")
    fusion_cmd.add_argument("--json", action="store_true", help="Return JSON output.")

    predict_cmd = subparsers.add_parser("predict", help="Manual prediction test for a trained run.")
    predict_cmd.add_argument("--run-dir", required=True, help="Training run directory.")
    predict_cmd.add_argument("--text", required=True, help="Text to test.")
    predict_cmd.add_argument("--json", action="store_true", help="Return JSON output.")

    drift_cmd = subparsers.add_parser("show-drift-report", help="Compare a reference batch and a current batch.")
    drift_cmd.add_argument("--reference", required=True, help="Reference JSONL/JSON path or processed view name.")
    drift_cmd.add_argument("--current", required=True, help="Current JSONL/JSON path or processed view name.")
    drift_cmd.add_argument("--output", default=None, help="Optional output JSON path.")
    drift_cmd.add_argument("--json", action="store_true", help="Return JSON output.")

    return parser


def _print_json(payload: Any) -> None:
    print(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True))


def command_init_dirs(config: AppConfig, as_json: bool) -> int:
    config.paths.ensure_directories()
    payload = {
        "project_root": str(config.paths.root),
        "created_or_verified": [
            str(config.paths.data_root),
            str(config.paths.raw_data),
            str(config.paths.interim_data),
            str(config.paths.processed_data),
            str(config.paths.manifest_data),
            str(config.paths.outputs),
        ],
    }
    if as_json:
        _print_json(payload)
    else:
        print(f"Project root: {payload['project_root']}")
        for entry in payload["created_or_verified"]:
            print(entry)
    return 0


def command_show_config(config: AppConfig, section: str) -> int:
    if section == "all":
        payload = {
            "datasets": config.datasets_config,
            "experiments": config.experiments_config,
            "models": config.models_config,
        }
    else:
        payload = config.raw_section(section)
    _print_json(payload)
    return 0


def command_list_datasets(config: AppConfig, as_json: bool) -> int:
    summary = dataset_registry_summary(config)
    if as_json:
        _print_json(summary)
    else:
        for row in summary["datasets"]:
            status = "registered" if row["adapter_registered"] else "config-only"
            enabled = "enabled" if row["enabled"] else "disabled"
            formats = ",".join(row["supported_formats"]) if row["supported_formats"] else "n/a"
            print(
                f"{row['name']}: {enabled}, {status}, modalities={','.join(row['modalities'])}, "
                f"formats={formats}"
            )
    return 0


def command_list_fetch_sources(config: AppConfig, as_json: bool) -> int:
    from veriscope_training.acquisition.manager import list_fetch_sources

    payload = list_fetch_sources(config)
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_inspect_fetch_config(config: AppConfig, source_name: str, as_json: bool) -> int:
    from veriscope_training.acquisition.manager import inspect_fetch_config

    payload = inspect_fetch_config(config, source_name)
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_inspect_source(config: AppConfig, source_name: str, as_json: bool) -> int:
    source = config.get_source(source_name)
    payload = {
        "name": source.name,
        "enabled": source.enabled,
        "description": source.description,
        "access": source.access,
        "snapshot_path": str(config.snapshot_path_for(source.name)),
        "snapshot_glob": source.snapshot_glob,
        "modalities": list(source.modalities),
        "intended_use": list(source.intended_use),
        "label_strategy": source.label_strategy,
        "notes": list(source.notes),
        "extra": source.extra,
    }
    if as_json:
        _print_json(payload)
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def command_inspect_adapter(config: AppConfig, source_name: str, as_json: bool) -> int:
    adapter = create_adapter(source_name, config)
    payload = {
        "adapter": adapter.adapter_metadata(),
        "validation": adapter.validation_report(),
    }
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True))
    return 0


def command_preview_records(config: AppConfig, source_name: str, limit: int, as_json: bool) -> int:
    adapter = create_adapter(source_name, config)
    payload = {
        "adapter": adapter.adapter_metadata(),
        "preview_count": limit,
        "records": [record.to_dict() for record in adapter.preview_records(limit=limit)],
    }
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_fetch_dataset(config: AppConfig, source_name: str, force: bool, as_json: bool) -> int:
    try:
        from veriscope_training.acquisition.manager import fetch_dataset

        payload = fetch_dataset(config, source_name, force=force)
    except Exception as exc:
        payload = {"error": str(exc), "source_name": source_name}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    validation = payload.get("validation") or {}
    return 0 if validation.get("status") != "not_ready" else 1


def command_fetch_all_datasets(config: AppConfig, force: bool, as_json: bool) -> int:
    try:
        from veriscope_training.acquisition.manager import fetch_all_datasets

        payload = fetch_all_datasets(config, force=force)
    except Exception as exc:
        payload = {"error": str(exc)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not payload.get("errors") else 1


def command_validate_raw_dataset(config: AppConfig, source_name: str, as_json: bool) -> int:
    try:
        from veriscope_training.acquisition.validation import validate_raw_dataset

        payload = validate_raw_dataset(config, source_name)
    except Exception as exc:
        payload = {"error": str(exc), "source_name": source_name}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "ready" else 1


def command_build_dataset(
    config: AppConfig,
    *,
    sources: list[str] | None,
    views: list[str] | None,
    output_format: str | None,
    max_records_per_source: int | None,
    force_rebuild: bool,
    skip_completed: bool | None,
    as_json: bool,
) -> int:
    options = BuildDatasetOptions(
        source_names=sources,
        view_names=views,
        output_format=output_format,
        max_records_per_source=max_records_per_source,
        preview_limit=int(config.experiments_config.get("datasets", {}).get("preview_limit", 5)),
        force_rebuild=force_rebuild,
        skip_completed=skip_completed,
    )
    try:
        payload = build_processed_datasets(config, options)
    except Exception as exc:
        payload = {"error": str(exc), "plan": build_dataset_plan(config)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_preview_processed(
    config: AppConfig,
    *,
    sources: list[str] | None,
    views: list[str] | None,
    limit: int,
    max_records_per_source: int | None,
    as_json: bool,
) -> int:
    options = BuildDatasetOptions(
        source_names=sources,
        view_names=views,
        max_records_per_source=max_records_per_source,
        preview_limit=limit,
    )
    try:
        payload = preview_processed_records(config, options)
    except Exception as exc:
        payload = {"error": str(exc), "plan": build_dataset_plan(config)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_show_manifest(config: AppConfig, name_or_path: str | None, as_json: bool) -> int:
    try:
        payload = load_manifest(config, name_or_path)
    except Exception as exc:
        payload = {"error": str(exc)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_show_dedupe_report(config: AppConfig, include_events: bool, limit: int, as_json: bool) -> int:
    try:
        payload = load_dedupe_report(config, include_events=include_events, limit=limit)
    except Exception as exc:
        payload = {"error": str(exc)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_list_models(config: AppConfig, as_json: bool) -> int:
    payload = list_available_models(config)
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_train_model(
    config: AppConfig,
    *,
    track: str,
    model_name: str,
    run_name: str | None,
    split_strategy: str | None,
    as_json: bool,
) -> int:
    try:
        payload = train_model(config, track=track, model_name=model_name, run_name=run_name, split_strategy=split_strategy)
    except Exception as exc:
        payload = {"error": str(exc), "available_models": list_available_models(config)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_train_all_baselines(config: AppConfig, *, include_transformers: bool, split_strategy: str | None, as_json: bool) -> int:
    payload = train_all_enabled_models(config, include_transformers=include_transformers, split_strategy=split_strategy)
    exit_code = 0 if not payload["errors"] else 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


def command_show_training_run(path_or_dir: str, as_json: bool) -> int:
    try:
        payload = load_training_run(path_or_dir)
    except Exception as exc:
        payload = {"error": str(exc)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_evaluate_all(
    config: AppConfig,
    *,
    group_name: str,
    output_dir: str | None,
    rerun: bool,
    as_json: bool,
) -> int:
    try:
        from veriscope_training.pipelines.evaluate_all import run_experiment_group

        payload = run_experiment_group(config, group_name=group_name, output_dir=output_dir, rerun=rerun)
    except Exception as exc:
        payload = {"error": str(exc)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not payload.get("errors") else 1


def command_compare_runs(
    config: AppConfig,
    *,
    run_dirs: list[str] | None,
    output_dir: str | None,
    split_name: str | None,
    as_json: bool,
) -> int:
    try:
        from veriscope_training.pipelines.compare_runs import compare_training_runs

        payload = compare_training_runs(config, run_dirs=run_dirs, output_dir=output_dir, split_name=split_name)
    except Exception as exc:
        payload = {"error": str(exc)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_calibrate_thresholds(
    config: AppConfig,
    *,
    run_dir: str | None,
    predictions: str | None,
    output_dir: str | None,
    split_name: str,
    as_json: bool,
) -> int:
    try:
        from veriscope_training.pipelines.calibrate_thresholds import calibrate_run_thresholds

        payload = calibrate_run_thresholds(
            config,
            run_dir=run_dir,
            prediction_path=predictions,
            output_dir=output_dir,
            split_name=split_name,
        )
    except Exception as exc:
        payload = {"error": str(exc)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_show_json_report(path: str, as_json: bool) -> int:
    try:
        payload = read_json(path)
    except Exception as exc:
        payload = {"error": str(exc)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_export_integration_configs(
    config: AppConfig,
    *,
    run_dirs: list[str] | None,
    output_dir: str | None,
    as_json: bool,
) -> int:
    try:
        from veriscope_training.integration.export_configs import export_integration_configs
        from veriscope_training.pipelines.calibrate_thresholds import calibrate_run_thresholds
        from veriscope_training.pipelines.compare_runs import compare_training_runs

        compared = compare_training_runs(config, run_dirs=run_dirs, output_dir=output_dir)
        threshold_configs = {}
        risk_mapping_configs = {}
        for key, value in compared["recommendations"].get("track_recommendations", {}).items():
            winner = value.get("winner")
            if not winner or not winner.get("predictions_path"):
                continue
            calibrated = calibrate_run_thresholds(
                config,
                run_dir=winner.get("run_dir"),
                prediction_path=winner.get("predictions_path"),
                output_dir=str(Path(output_dir or compared["output_dir"]) / "integration_calibration" / key),
            )
            threshold_configs[key] = calibrated["payload"]["binary_threshold"]
            risk_mapping_configs[key] = calibrated["payload"]["risk_mapping"]
        integration_dir = output_dir or str(Path(compared["output_dir"]) / "integration")
        payload = {
            "compare_output_dir": compared["output_dir"],
            "integration_files": export_integration_configs(
                output_dir=integration_dir,
                recommendations=compared["recommendations"],
                threshold_configs=threshold_configs,
                risk_mapping_configs=risk_mapping_configs,
            ),
        }
    except Exception as exc:
        payload = {"error": str(exc)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_error_analysis(
    config: AppConfig,
    *,
    run_dir: str,
    split_name: str,
    output_dir: str | None,
    as_json: bool,
) -> int:
    try:
        from veriscope_training.evaluation.error_analysis import analyze_run_errors, save_error_analysis

        run_summary = load_training_run(run_dir)
        payload = analyze_run_errors(config, run_summary, split_name=split_name, top_k=25)
        if output_dir is None:
            output_dir = str(Path(run_dir) / "error_analysis")
        files = save_error_analysis(payload, output_dir=output_dir)
        payload = {"summary": payload, "files": files}
    except Exception as exc:
        payload = {"error": str(exc)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_export_retraining_candidates(
    config: AppConfig,
    *,
    predictions: str,
    feedback: str | None,
    output_dir: str | None,
    as_json: bool,
) -> int:
    try:
        if output_dir is None:
            output_dir = str(_adaptive_root(config) / "retraining_candidates" / _timestamp_slug())
        band = config.experiments_config.get("adaptive", {}).get("uncertainty_band", [0.4, 0.6])
        payload = export_retraining_candidates(
            prediction_path=predictions,
            feedback_path=feedback,
            output_dir=output_dir,
            uncertainty_band=(float(band[0]), float(band[1])),
        )
    except Exception as exc:
        payload = {"error": str(exc)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_generate_heuristic_proposals(
    config: AppConfig,
    *,
    processed: str | None,
    feedback: str | None,
    from_errors: str | None,
    output: str | None,
    top_k: int,
    as_json: bool,
) -> int:
    try:
        from veriscope_training.adaptive.heuristic_proposals import (
            generate_heuristic_proposals,
            generate_proposals_from_errors,
        )

        if from_errors:
            if output is None:
                output = str(_adaptive_root(config) / "heuristic_proposals" / f"from_errors_{_timestamp_slug()}.json")
            payload = generate_proposals_from_errors(
                error_analysis_dir=from_errors,
                output_path=output,
                top_k=top_k,
            )
        else:
            if not processed:
                raise ValueError("Either --processed or --from-errors must be provided.")
            if output is None:
                output = str(_adaptive_root(config) / "heuristic_proposals" / f"standard_{_timestamp_slug()}.json")
            payload = generate_heuristic_proposals(
                processed_path=processed,
                feedback_path=feedback,
                output_path=output,
                top_k=top_k,
            )
    except Exception as exc:
        payload = {"error": str(exc)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_run_fusion_experiment(
    config: AppConfig,
    *,
    url_run: str,
    webpage_run: str,
    output_dir: str | None,
    strategy: str,
    as_json: bool,
) -> int:
    try:
        from veriscope_training.fusion.evaluation import align_predictions, compute_fusion_metrics
        from veriscope_training.fusion.weighted_fusion import apply_weighted_fusion
        from veriscope_training.fusion.cascade_fusion import apply_cascade_fusion

        url_summary = load_training_run(url_run)
        web_summary = load_training_run(webpage_run)

        url_preds = url_summary["artifact_paths"].get("test_predictions")
        web_preds = web_summary["artifact_paths"].get("test_predictions")

        if not url_preds or not web_preds:
            raise FileNotFoundError("Test predictions missing in one or both runs.")

        df, stats = align_predictions(url_preds, web_preds)

        results = {"stats": stats, "experiments": {}}

        if strategy in ("weighted", "both"):
            weighted_scores = apply_weighted_fusion(df)
            results["experiments"]["weighted"] = compute_fusion_metrics(df["normalized_label"], weighted_scores)

        if strategy in ("cascade", "both"):
            cascade_scores = apply_cascade_fusion(df)
            results["experiments"]["cascade"] = compute_fusion_metrics(df["normalized_label"], cascade_scores)

        if output_dir is None:
            output_dir = str(config.paths.outputs / "fusion" / _timestamp_slug())
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        write_json(out_path / "fusion_results.json", results)
        
        payload = results
    except Exception as exc:
        payload = {"error": str(exc)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_predict(
    run_dir: str,
    text: str,
    as_json: bool,
) -> int:
    try:
        from veriscope_training.models.predict import predict_single
        payload = predict_single(run_dir, text)
    except Exception as exc:
        payload = {"error": str(exc)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def command_show_drift_report(
    config: AppConfig,
    *,
    reference: str,
    current: str,
    output: str | None,
    as_json: bool,
) -> int:
    try:
        reference_rows = load_rows_for_drift(reference, config=config)
        current_rows = load_rows_for_drift(current, config=config)
        if output is None:
            output = str(_adaptive_root(config) / "drift_reports" / f"{_timestamp_slug()}.json")
        payload = generate_drift_report(reference_rows, current_rows, output_path=output)
        payload["output_path"] = output
    except Exception as exc:
        payload = {"error": str(exc)}
        if as_json:
            _print_json(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if as_json:
        _print_json(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _adaptive_root(config: AppConfig) -> Path:
    root = config.paths.resolve(config.experiments_config.get("artifacts", {}).get("adaptive_root", "outputs/adaptive"))
    if root is None:
        raise ValueError("Adaptive root could not be resolved.")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    config = AppConfig.load(root=args.project_root)

    if args.command == "init-dirs":
        return command_init_dirs(config, args.json)
    if args.command == "show-config":
        return command_show_config(config, args.section)
    if args.command == "list-datasets":
        return command_list_datasets(config, args.json)
    if args.command == "list-fetch-sources":
        return command_list_fetch_sources(config, args.json)
    if args.command == "inspect-fetch-config":
        return command_inspect_fetch_config(config, args.name, args.json)
    if args.command == "inspect-source":
        return command_inspect_source(config, args.name, args.json)
    if args.command == "inspect-adapter":
        return command_inspect_adapter(config, args.name, args.json)
    if args.command == "preview-records":
        return command_preview_records(config, args.name, args.limit, args.json)
    if args.command == "fetch-dataset":
        return command_fetch_dataset(config, args.name, args.force, args.json)
    if args.command == "fetch-all-datasets":
        return command_fetch_all_datasets(config, args.force, args.json)
    if args.command == "validate-raw-dataset":
        return command_validate_raw_dataset(config, args.name, args.json)
    if args.command == "build-dataset":
        return command_build_dataset(
            config,
            sources=args.sources,
            views=args.views,
            output_format=args.output_format,
            max_records_per_source=args.max_records_per_source,
            force_rebuild=args.force_rebuild,
            skip_completed=False if args.no_skip_completed else None,
            as_json=args.json,
        )
    if args.command == "preview-processed":
        return command_preview_processed(
            config,
            sources=args.sources,
            views=args.views,
            limit=args.limit,
            max_records_per_source=args.max_records_per_source,
            as_json=args.json,
        )
    if args.command == "show-manifest":
        return command_show_manifest(config, args.name_or_path, args.json)
    if args.command == "show-dedupe-report":
        return command_show_dedupe_report(config, args.events, args.limit, args.json)
    if args.command == "list-models":
        return command_list_models(config, args.json)
    if args.command == "train-model":
        return command_train_model(
            config,
            track=args.track,
            model_name=args.model,
            run_name=args.run_name,
            split_strategy=args.split_strategy,
            as_json=args.json,
        )
    if args.command == "train-all-baselines":
        return command_train_all_baselines(
            config,
            include_transformers=args.include_transformers,
            split_strategy=args.split_strategy,
            as_json=args.json,
        )
    if args.command == "show-training-run":
        return command_show_training_run(args.path_or_dir, args.json)
    if args.command == "evaluate-all":
        return command_evaluate_all(
            config,
            group_name=args.group,
            output_dir=args.output_dir,
            rerun=args.rerun,
            as_json=args.json,
        )
    if args.command == "compare-runs":
        return command_compare_runs(
            config,
            run_dirs=args.run_dirs,
            output_dir=args.output_dir,
            split_name=args.split,
            as_json=args.json,
        )
    if args.command == "calibrate-thresholds":
        return command_calibrate_thresholds(
            config,
            run_dir=args.run_dir,
            predictions=args.predictions,
            output_dir=args.output_dir,
            split_name=args.split,
            as_json=args.json,
        )
    if args.command == "show-comparison-report":
        return command_show_json_report(args.path, args.json)
    if args.command == "show-recommendation":
        return command_show_json_report(args.path, args.json)
    if args.command == "export-integration-configs":
        return command_export_integration_configs(
            config,
            run_dirs=args.run_dirs,
            output_dir=args.output_dir,
            as_json=args.json,
        )
    if args.command == "error-analysis":
        return command_error_analysis(
            config,
            run_dir=args.run_dir,
            split_name=args.split,
            output_dir=args.output_dir,
            as_json=args.json,
        )
    if args.command == "export-retraining-candidates":
        return command_export_retraining_candidates(
            config,
            predictions=args.predictions,
            feedback=args.feedback,
            output_dir=args.output_dir,
            as_json=args.json,
        )
    if args.command == "generate-heuristic-proposals":
        return command_generate_heuristic_proposals(
            config,
            processed=args.processed,
            feedback=args.feedback,
            from_errors=args.from_errors,
            output=args.output,
            top_k=args.top_k,
            as_json=args.json,
        )
    if args.command == "run-fusion-experiment":
        return command_run_fusion_experiment(
            config,
            url_run=args.url_run,
            webpage_run=args.webpage_run,
            output_dir=args.output_dir,
            strategy=args.strategy,
            as_json=args.json,
        )
    if args.command == "predict":
        return command_predict(
            run_dir=args.run_dir,
            text=args.text,
            as_json=args.json,
        )
    if args.command == "show-drift-report":
        return command_show_drift_report(
            config,
            reference=args.reference,
            current=args.current,
            output=args.output,
            as_json=args.json,
        )

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
