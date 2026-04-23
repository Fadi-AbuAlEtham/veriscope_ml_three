from __future__ import annotations

from pathlib import Path
from typing import Any

from veriscope_training.evaluation.aggregate import (
    aggregate_run_bundles,
    discover_training_run_dirs,
    load_run_bundle,
    save_aggregate_payload,
)
from veriscope_training.evaluation.plots import save_metric_comparison_bar
from veriscope_training.evaluation.reports import save_comparison_reports, save_recommendation_reports
from veriscope_training.models.recommendation import recommend_models
from veriscope_training.utils.io import write_json


def compare_training_runs(
    config,
    *,
    run_dirs: list[str] | None = None,
    output_dir: str | None = None,
    split_name: str | None = None,
) -> dict[str, Any]:
    training_root = config.paths.resolve(config.experiments_config.get("artifacts", {}).get("training_root", "outputs/training"))
    if output_dir is None:
        reports_root = config.paths.resolve(config.experiments_config.get("artifacts", {}).get("reports_root", "outputs/reports"))
        output_dir = str(Path(reports_root) / "compare_runs")
    discovered = [Path(path) for path in run_dirs] if run_dirs else discover_training_run_dirs(training_root)
    if not discovered:
        raise FileNotFoundError("No training runs were found to compare.")
    bundles = [load_run_bundle(path) for path in discovered]
    aggregate_payload = aggregate_run_bundles(
        bundles,
        split_name=split_name or config.experiments_config.get("evaluation", {}).get("default_split", "test"),
    )
    comparison_files = save_aggregate_payload(aggregate_payload, output_dir=output_dir, basename="comparison")
    report_files = save_comparison_reports(aggregate_payload, output_dir=output_dir, title="VeriScope Model Comparison")
    plot_files = {
        "f1_bar": save_metric_comparison_bar(
            aggregate_payload["rows"],
            metric="f1",
            output_path=Path(output_dir) / "f1_comparison.png",
            title="F1 Comparison Across Runs",
        ),
        "pr_auc_bar": save_metric_comparison_bar(
            aggregate_payload["rows"],
            metric="pr_auc",
            output_path=Path(output_dir) / "pr_auc_comparison.png",
            title="PR-AUC Comparison Across Runs",
        ),
        "transformer_f1_bar": save_metric_comparison_bar(
            [row for row in aggregate_payload["rows"] if row.get("track_group") == "transformer"],
            metric="f1",
            output_path=Path(output_dir) / "transformer_f1_comparison.png",
            title="Transformer F1 Comparison",
        ),
    }
    weights = config.experiments_config.get("evaluation", {}).get("selection_weights", {})
    recommendations = recommend_models(aggregate_payload, metric_weights=weights)
    recommendation_files = save_recommendation_reports(recommendations, output_dir=output_dir)
    index_path = write_json(
        Path(output_dir) / "run_index.json",
        {"run_dirs": [str(path) for path in discovered], "run_count": len(discovered)},
    )
    return {
        "output_dir": str(output_dir),
        "aggregate": aggregate_payload,
        "comparison_files": {**comparison_files, **report_files, **plot_files},
        "recommendations": recommendations,
        "recommendation_files": recommendation_files,
        "run_index": str(index_path),
    }
