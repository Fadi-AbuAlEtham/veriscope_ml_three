from __future__ import annotations

from pathlib import Path
from typing import Any

from veriscope_training.models.artifacts import load_training_run
from veriscope_training.pipelines.calibrate_thresholds import calibrate_run_thresholds
from veriscope_training.pipelines.compare_runs import compare_training_runs
from veriscope_training.pipelines.train_all import train_model
from veriscope_training.integration.export_configs import export_integration_configs
from veriscope_training.evaluation.error_analysis import analyze_run_errors, save_error_analysis
from veriscope_training.utils.io import write_json


TRANSFORMER_ALIASES = {
    "xlmr": "xlmr_sequence_classifier",
    "mbert": "mbert_sequence_classifier",
}


def run_experiment_group(
    config,
    *,
    group_name: str,
    output_dir: str | None = None,
    rerun: bool = False,
) -> dict[str, Any]:
    groups = config.experiments_config.get("experiment_groups", {})
    group = groups.get(group_name)
    if not group:
        raise KeyError(f"Experiment group '{group_name}' is not defined in configs/experiments.yaml.")
    experiment_root = config.paths.resolve(config.experiments_config.get("artifacts", {}).get("experiment_root", "outputs/experiments"))
    if output_dir is None:
        output_dir = str(Path(experiment_root) / group_name)
    continue_on_error = bool(group.get("continue_on_error", True))
    skip_completed = bool(group.get("skip_completed", True)) and not rerun

    runs = []
    errors = []
    run_dirs = []
    for spec in group.get("runs", []):
        track = spec["track"]
        model_name = spec["model"]
        split_strategy = spec.get("split_strategy")
        run_name = spec.get("run_name") or _default_run_name(group_name, track, model_name, split_strategy)
        existing_dir = _expected_run_dir(config, track=track, model_name=model_name, run_name=run_name)
        try:
            if skip_completed and (existing_dir / "run_summary.json").exists():
                summary = load_training_run(existing_dir)
                status = "skipped_existing"
            else:
                summary = train_model(
                    config,
                    track=track,
                    model_name=model_name,
                    run_name=run_name,
                    split_strategy=split_strategy,
                )
                status = "trained"
            runs.append({"track": track, "model_name": model_name, "run_name": run_name, "status": status, "run_dir": summary.get("run_dir")})
            run_dirs.append(summary.get("run_dir"))
        except Exception as exc:
            errors.append({"track": track, "model_name": model_name, "run_name": run_name, "error": str(exc)})
            if not continue_on_error:
                raise

    successful_run_dirs = [path for path in run_dirs if path]
    if not successful_run_dirs:
        manifest = {
            "group_name": group_name,
            "output_dir": str(output_dir),
            "runs": runs,
            "errors": errors,
            "comparison": None,
            "integration_files": None,
        }
        manifest_path = write_json(Path(output_dir) / "experiment_manifest.json", manifest)
        return {
            "output_dir": str(output_dir),
            "runs": runs,
            "errors": errors,
            "comparison": None,
            "calibrations": None,
            "integration": None,
            "manifest": str(manifest_path),
        }

    comparison = compare_training_runs(config, run_dirs=successful_run_dirs, output_dir=str(Path(output_dir) / "comparison"))
    recommendations = comparison["recommendations"]
    calibrations = _calibrate_recommended_runs(config, recommendations)
    error_analyses = _analyze_recommended_runs(config, recommendations, base_output_dir=Path(output_dir) / "error_analysis")
    integration = export_integration_configs(
        output_dir=Path(output_dir) / "integration",
        recommendations=recommendations,
        threshold_configs=calibrations["threshold_configs"],
        risk_mapping_configs=calibrations["risk_mapping_configs"],
    )
    manifest = {
        "group_name": group_name,
        "output_dir": str(output_dir),
        "runs": runs,
        "errors": errors,
        "comparison": comparison["comparison_files"],
        "recommendation_files": comparison["recommendation_files"],
        "integration_files": integration,
        "calibrations": calibrations["calibration_outputs"],
        "error_analysis": error_analyses,
    }
    manifest_path = write_json(Path(output_dir) / "experiment_manifest.json", manifest)
    return {
        "output_dir": str(output_dir),
        "runs": runs,
        "errors": errors,
        "comparison": comparison,
        "calibrations": calibrations,
        "error_analysis": error_analyses,
        "integration": integration,
        "manifest": str(manifest_path),
    }


def _expected_run_dir(config, *, track: str, model_name: str, run_name: str) -> Path:
    training_root = config.paths.resolve(config.experiments_config.get("artifacts", {}).get("training_root", "outputs/training"))
    if track == "transformer":
        return Path(training_root) / "webpage_transformer" / TRANSFORMER_ALIASES.get(model_name, model_name) / run_name
    return Path(training_root) / track / model_name / run_name


def _default_run_name(group_name: str, track: str, model_name: str, split_strategy: str | None) -> str:
    suffix = split_strategy or "default"
    return f"{group_name}-{track}-{model_name}-{suffix}".replace("/", "_")


def _calibrate_recommended_runs(config, recommendations: dict[str, Any]) -> dict[str, Any]:
    threshold_configs = {}
    risk_mapping_configs = {}
    calibration_outputs = {}
    for key, value in recommendations.get("track_recommendations", {}).items():
        winner = value.get("winner")
        if not winner or not winner.get("predictions_path"):
            continue
        calibrated = calibrate_run_thresholds(
            config,
            run_dir=winner.get("run_dir"),
            prediction_path=winner.get("predictions_path"),
            output_dir=str(Path(winner["run_dir"]) / "calibration"),
        )
        threshold_configs[key] = calibrated["payload"]["binary_threshold"]
        risk_mapping_configs[key] = calibrated["payload"]["risk_mapping"]
        calibration_outputs[key] = {
            "output_dir": calibrated["output_dir"],
            "summary_file": calibrated["summary_file"],
        }
    return {
        "threshold_configs": threshold_configs,
        "risk_mapping_configs": risk_mapping_configs,
        "calibration_outputs": calibration_outputs,
    }


def _analyze_recommended_runs(config, recommendations: dict[str, Any], *, base_output_dir: Path) -> dict[str, Any]:
    payload = {}
    for key, value in recommendations.get("track_recommendations", {}).items():
        winner = value.get("winner")
        if not winner or not winner.get("run_dir"):
            continue
        run_summary = load_training_run(winner["run_dir"])
        analysis = analyze_run_errors(config, run_summary, split_name="test", top_k=25)
        saved = save_error_analysis(analysis, output_dir=base_output_dir / key)
        payload[key] = saved
    return payload
