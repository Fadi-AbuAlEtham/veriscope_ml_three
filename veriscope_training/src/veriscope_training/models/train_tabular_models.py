from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from veriscope_training.config import AppConfig
from veriscope_training.evaluation.metrics import compute_binary_classification_metrics
from veriscope_training.models.artifacts import (
    create_training_run_context,
    load_processed_view_records,
    save_config_snapshot,
    save_label_metadata,
    save_metrics,
    save_model_bundle,
    save_package_versions,
    save_predictions,
    save_run_metadata,
)
from veriscope_training.splits.splitters import create_dataset_split, save_split_manifest, subset_by_indices


def train_tabular_model(
    config: AppConfig,
    *,
    model_name: str,
    run_name: str | None = None,
    split_strategy: str | None = None,
) -> dict[str, Any]:
    model_config = config.models_config.get("tabular_models", {}).get(model_name)
    if not model_config or not model_config.get("enabled", False):
        raise KeyError(f"Tabular model '{model_name}' is not enabled in configs/models.yaml.")

    records = load_processed_view_records(config, "unified_tabular_dataset")
    excluded_counts = {
        "missing_binary_label": sum(1 for row in records if row.get("normalized_label") not in (0, 1)),
        "missing_tabular_features": sum(1 for row in records if not row.get("tabular_features")),
    }
    supervised = [
        row for row in records if row.get("normalized_label") in (0, 1) and row.get("tabular_features")
    ]
    if not supervised:
        raise RuntimeError("No supervised tabular records were found.")

    split = create_dataset_split(
        supervised,
        strategy=split_strategy or "random_stratified",
        seed=int(config.experiments_config.get("reproducibility", {}).get("seed", 42)),
        validation_fraction=float(config.experiments_config.get("splits", {}).get("validation_fraction", 0.1)),
        test_fraction=float(config.experiments_config.get("splits", {}).get("test_fraction", 0.2)),
    )
    train_rows = subset_by_indices(supervised, split.train_indices)
    validation_rows = subset_by_indices(supervised, split.validation_indices)
    test_rows = subset_by_indices(supervised, split.test_indices)
    y_train = np.asarray([row["normalized_label"] for row in train_rows], dtype=int)
    y_val = np.asarray([row["normalized_label"] for row in validation_rows], dtype=int)
    y_test = np.asarray([row["normalized_label"] for row in test_rows], dtype=int)

    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform([row["tabular_features"] for row in train_rows])
    x_val = vectorizer.transform([row["tabular_features"] for row in validation_rows])
    x_test = vectorizer.transform([row["tabular_features"] for row in test_rows])

    estimator = _build_estimator(model_name, model_config)
    estimator.fit(x_train, y_train)

    context = create_training_run_context(config, track="tabular", model_name=model_name, run_name=run_name)
    split_paths = save_split_manifest(split, records=supervised, output_dir=context.run_dir / "splits")

    train_eval = _evaluate(estimator, train_rows, x_train, y_train)
    val_eval = _evaluate(estimator, validation_rows, x_val, y_val)
    test_eval = _evaluate(estimator, test_rows, x_test, y_test)

    from veriscope_training.evaluation.plots import save_confusion_matrix_plot

    confusion_plot = save_confusion_matrix_plot(
        test_eval["metrics"]["confusion_matrix"],
        context.reports_dir / "confusion_matrix_test.png",
        title=f"Tabular {model_name} Test Confusion Matrix",
    )
    predictions_path = save_predictions(context, "test", test_eval["prediction_rows"])
    artifact_path = save_model_bundle(
        context,
        "model_bundle",
        {"estimator": estimator, "vectorizer": vectorizer, "track": "tabular", "model_name": model_name},
    )
    metrics_payload = {
        "train": train_eval["metrics"],
        "validation": val_eval["metrics"],
        "test": test_eval["metrics"],
        "class_distribution": split.label_counts,
        "split_strategy": split.strategy,
        "confusion_matrix_plot": confusion_plot,
        "excluded_counts": excluded_counts,
        "input_fields": ["tabular_features"],
    }
    metrics_path = save_metrics(context, metrics_payload)
    label_metadata_path = save_label_metadata(context, {"label_map": {"benign": 0, "phishing": 1}})
    config_snapshot_path = save_config_snapshot(
        context,
        {
            "track": "tabular",
            "model_name": model_name,
            "model_config": model_config,
            "split": split.to_dict(),
            "excluded_counts": excluded_counts,
            "input_fields": ["tabular_features"],
            "tabular_feature_count": len(train_rows[0]["tabular_features"]) if train_rows else 0,
        },
    )
    versions_path = save_package_versions(
        context,
        ["numpy", "scikit-learn", "scipy", "pandas", "pyarrow", "xgboost", "lightgbm"],
    )

    summary = {
        "track": "tabular",
        "model_name": model_name,
        "run_dir": str(context.run_dir),
        "artifact_paths": {
            "model_bundle": artifact_path,
            "metrics": metrics_path,
            "label_metadata": label_metadata_path,
            "config_snapshot": config_snapshot_path,
            "split_manifest": split_paths["manifest"],
            "split_assignments": split_paths["assignments"],
            "test_predictions": predictions_path,
            "package_versions": versions_path,
            "confusion_matrix_plot": confusion_plot,
        },
        "metrics": metrics_payload,
        "excluded_counts": excluded_counts,
    }
    save_run_metadata(context, payload=summary)
    return summary


def _build_estimator(model_name: str, model_config: dict[str, Any]):
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=int(model_config.get("max_iter", 2000)),
            class_weight=model_config.get("class_weight"),
            solver="liblinear",
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(model_config.get("n_estimators", 400)),
            class_weight=model_config.get("class_weight"),
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "gradient_boosting_optional":
        backends = model_config.get("backend_preference", ["xgboost", "lightgbm"])
        for backend in backends:
            if backend == "xgboost":
                try:
                    from xgboost import XGBClassifier

                    return XGBClassifier(
                        n_estimators=int(model_config.get("n_estimators", 300)),
                        max_depth=int(model_config.get("max_depth", 6)),
                        learning_rate=float(model_config.get("learning_rate", 0.1)),
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=42,
                    )
                except ImportError:
                    continue
            if backend == "lightgbm":
                try:
                    from lightgbm import LGBMClassifier

                    return LGBMClassifier(
                        n_estimators=int(model_config.get("n_estimators", 300)),
                        learning_rate=float(model_config.get("learning_rate", 0.05)),
                        random_state=42,
                    )
                except ImportError:
                    continue
        raise RuntimeError(
            "Tabular gradient boosting requested, but neither xgboost nor lightgbm is installed."
        )
    raise KeyError(f"Unsupported tabular model '{model_name}'.")


def _evaluate(estimator: Any, rows: list[dict[str, Any]], x: Any, y_true: np.ndarray) -> dict[str, Any]:
    y_pred = estimator.predict(x)
    scores = None
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(x)[:, 1]
    metrics = compute_binary_classification_metrics(y_true.tolist(), y_pred.tolist(), y_score=None if scores is None else np.asarray(scores).tolist())
    prediction_rows = []
    for row, true_label, pred_label, score in zip(rows, y_true.tolist(), y_pred.tolist(), _expand_scores(scores, len(rows))):
        prediction_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "source_dataset": row.get("source_dataset"),
                "normalized_label": true_label,
                "predicted_label": int(pred_label),
                "score": score,
            }
        )
    return {"metrics": metrics, "prediction_rows": prediction_rows}


def _expand_scores(scores: Any, expected: int) -> list[float | None]:
    if scores is None:
        return [None] * expected
    if hasattr(scores, "tolist"):
        return [float(value) for value in scores.tolist()]
    return [float(value) for value in scores]
