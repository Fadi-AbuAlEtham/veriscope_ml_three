from __future__ import annotations

from pathlib import Path
from typing import Any

from veriscope_training.evaluation.calibration import calibrate_predictions, load_prediction_rows, save_calibration_payload
from veriscope_training.evaluation.plots import (
    save_pr_curve_plot,
    save_risk_bucket_distribution_plot,
    save_roc_curve_plot,
    save_score_distribution_plot,
    save_threshold_performance_plot,
)
from veriscope_training.models.artifacts import load_training_run
from veriscope_training.utils.io import write_json


def calibrate_run_thresholds(
    config,
    *,
    run_dir: str | None = None,
    prediction_path: str | None = None,
    output_dir: str | None = None,
    split_name: str = "test",
) -> dict[str, Any]:
    if run_dir is None and prediction_path is None:
        raise ValueError("Provide either run_dir or prediction_path for threshold calibration.")
    run_summary = None
    if run_dir is not None:
        run_summary = load_training_run(run_dir)
        prediction_path = prediction_path or run_summary.get("artifact_paths", {}).get(f"{split_name}_predictions") or run_summary.get("artifact_paths", {}).get("test_predictions")
    if not prediction_path:
        raise FileNotFoundError("Prediction path could not be resolved for threshold calibration.")
    predictions = load_prediction_rows(prediction_path)
    calibration_cfg = config.experiments_config.get("evaluation", {}).get("threshold_calibration", {})
    payload = calibrate_predictions(
        predictions,
        objective=calibration_cfg.get("binary_objective", "maximize_f1"),
        precision_floor=calibration_cfg.get("precision_floor"),
        recall_floor=calibration_cfg.get("recall_floor"),
        low_bucket_max_phishing_rate=float(calibration_cfg.get("low_bucket_max_phishing_rate", 0.05)),
        min_low_bucket_coverage=float(calibration_cfg.get("min_low_bucket_coverage", 0.1)),
        high_threshold_strategy=calibration_cfg.get("high_threshold_strategy", "binary_threshold"),
        high_bucket_precision_floor=float(calibration_cfg.get("high_bucket_precision_floor", 0.9)),
        high_bucket_recall_floor=float(calibration_cfg.get("high_bucket_recall_floor", 0.4)),
        max_thresholds=int(calibration_cfg.get("max_thresholds", 200)),
        min_medium_width=float(calibration_cfg.get("min_medium_width", 0.05)),
        min_bucket_support=int(calibration_cfg.get("min_bucket_support", 10)),
    )
    if output_dir is None:
        reports_root = config.paths.resolve(config.experiments_config.get("artifacts", {}).get("reports_root", "outputs/reports"))
        run_slug = Path(run_dir).name if run_dir else Path(prediction_path).stem
        output_dir = str(Path(reports_root) / "calibration" / run_slug)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    calibration_file = save_calibration_payload(payload, output_path / "calibration.json")
    threshold_plot = save_threshold_performance_plot(payload["threshold_table"], output_path / "threshold_f1_curve.png")
    score_plot = save_score_distribution_plot(predictions, output_path / "score_distribution.png")
    risk_plot = save_risk_bucket_distribution_plot(payload["risk_mapping"]["risk_bucket_summary"], output_path / "risk_buckets.png")
    scored_rows = [row for row in predictions if row.get("normalized_label") in (0, 1) and row.get("score") is not None]
    y_true = [int(row["normalized_label"]) for row in scored_rows]
    y_score = [float(row["score"]) for row in scored_rows]
    pr_plot = save_pr_curve_plot(y_true, y_score, output_path / "pr_curve.png", title="Precision-Recall Curve")
    roc_plot = save_roc_curve_plot(y_true, y_score, output_path / "roc_curve.png", title="ROC Curve")
    summary_path = write_json(
        output_path / "calibration_summary.json",
        {
            "run_dir": run_dir,
            "prediction_path": prediction_path,
            "binary_threshold": payload["binary_threshold"],
            "risk_mapping": payload["risk_mapping"],
            "plots": {
                "threshold_performance": threshold_plot,
                "score_distribution": score_plot,
                "risk_buckets": risk_plot,
                "pr_curve": pr_plot,
                "roc_curve": roc_plot,
            },
            "run_summary": run_summary,
        },
    )
    return {
        "output_dir": str(output_path),
        "calibration_file": calibration_file,
        "summary_file": str(summary_path),
        "plots": {
            "threshold_performance": threshold_plot,
            "score_distribution": score_plot,
            "risk_buckets": risk_plot,
            "pr_curve": pr_plot,
            "roc_curve": roc_plot,
        },
        "payload": payload,
    }
