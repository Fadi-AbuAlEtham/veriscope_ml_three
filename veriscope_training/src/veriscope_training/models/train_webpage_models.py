from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

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


def train_webpage_text_baseline(
    config: AppConfig,
    *,
    model_name: str,
    run_name: str | None = None,
    split_strategy: str | None = None,
) -> dict[str, Any]:
    model_config = config.models_config.get("webpage_models", {}).get(model_name)
    if not model_config or not model_config.get("enabled", False):
        raise KeyError(f"Webpage text model '{model_name}' is not enabled in configs/models.yaml.")

    records = load_processed_view_records(config, "unified_webpage_dataset")
    excluded_counts = {
        "missing_binary_label": sum(1 for row in records if row.get("normalized_label") not in (0, 1)),
        "missing_usable_text": sum(1 for row in records if not _text_value(row)),
        "raw_html_only_excluded": sum(
            1
            for row in records
            if not _text_value(row) and row.get("raw_html")
        ),
    }
    supervised = [
        row
        for row in records
        if row.get("normalized_label") in (0, 1) and _text_value(row)
    ]
    if not supervised:
        raise RuntimeError("No supervised webpage records with usable text were found.")

    split = create_dataset_split(
        supervised,
        strategy=split_strategy or config.experiments_config.get("splits", {}).get("default_strategy", "domain_aware"),
        seed=int(config.experiments_config.get("reproducibility", {}).get("seed", 42)),
        validation_fraction=float(config.experiments_config.get("splits", {}).get("validation_fraction", 0.1)),
        test_fraction=float(config.experiments_config.get("splits", {}).get("test_fraction", 0.2)),
    )
    train_rows = subset_by_indices(supervised, split.train_indices)
    validation_rows = subset_by_indices(supervised, split.validation_indices)
    test_rows = subset_by_indices(supervised, split.test_indices)
    train_texts = [_text_value(row) for row in train_rows]
    val_texts = [_text_value(row) for row in validation_rows]
    test_texts = [_text_value(row) for row in test_rows]
    y_train = np.asarray([row["normalized_label"] for row in train_rows], dtype=int)
    y_val = np.asarray([row["normalized_label"] for row in validation_rows], dtype=int)
    y_test = np.asarray([row["normalized_label"] for row in test_rows], dtype=int)

    context = create_training_run_context(config, track="webpage", model_name=model_name, run_name=run_name)
    split_paths = save_split_manifest(split, records=supervised, output_dir=context.run_dir / "splits")

    vectorizer_cfg = model_config.get("vectorizer", {})
    vectorizer = TfidfVectorizer(
        analyzer=vectorizer_cfg.get("analyzer", "word"),
        ngram_range=tuple(vectorizer_cfg.get("ngram_range", [1, 2])),
        min_df=int(vectorizer_cfg.get("min_df", 2)),
        max_features=vectorizer_cfg.get("max_features"),
    )
    classifier_cfg = model_config.get("classifier", {})
    if "logreg" in model_name:
        classifier = LogisticRegression(
            max_iter=int(classifier_cfg.get("max_iter", 3000)),
            class_weight=classifier_cfg.get("class_weight"),
            solver="liblinear",
        )
    else:
        classifier = LinearSVC(class_weight=classifier_cfg.get("class_weight"))
    estimator = Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])
    estimator.fit(train_texts, y_train)

    train_eval = _evaluate(estimator, train_rows, train_texts, y_train)
    val_eval = _evaluate(estimator, validation_rows, val_texts, y_val)
    test_eval = _evaluate(estimator, test_rows, test_texts, y_test)

    from veriscope_training.evaluation.plots import save_confusion_matrix_plot

    confusion_plot = save_confusion_matrix_plot(
        test_eval["metrics"]["confusion_matrix"],
        context.reports_dir / "confusion_matrix_test.png",
        title=f"Webpage {model_name} Test Confusion Matrix",
    )
    predictions_path = save_predictions(context, "test", test_eval["prediction_rows"])
    artifact_path = save_model_bundle(
        context,
        "model_bundle",
        {"estimator": estimator, "track": "webpage", "model_name": model_name},
    )
    metrics_payload = {
        "train": train_eval["metrics"],
        "validation": val_eval["metrics"],
        "test": test_eval["metrics"],
        "class_distribution": split.label_counts,
        "split_strategy": split.strategy,
        "confusion_matrix_plot": confusion_plot,
        "excluded_counts": excluded_counts,
        "text_field_priority": ["normalized_text", "extracted_text"],
    }
    metrics_path = save_metrics(context, metrics_payload)
    label_metadata_path = save_label_metadata(context, {"label_map": {"benign": 0, "phishing": 1}})
    config_snapshot_path = save_config_snapshot(
        context,
        {
            "track": "webpage",
            "model_name": model_name,
            "model_config": model_config,
            "split": split.to_dict(),
            "excluded_counts": excluded_counts,
            "text_field_priority": ["normalized_text", "extracted_text"],
            "raw_html_as_training_text": False,
        },
    )
    versions_path = save_package_versions(context, ["numpy", "scikit-learn", "scipy", "pandas", "pyarrow"])

    summary = {
        "track": "webpage",
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


def _evaluate(estimator: Pipeline, rows: list[dict[str, Any]], texts: list[str], y_true: np.ndarray) -> dict[str, Any]:
    y_pred = estimator.predict(texts)
    classifier = estimator.named_steps["classifier"]
    scores = None
    if hasattr(classifier, "predict_proba"):
        scores = estimator.predict_proba(texts)[:, 1]
    elif hasattr(classifier, "decision_function"):
        scores = classifier.decision_function(estimator.named_steps["vectorizer"].transform(texts))
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
                "normalized_text": _text_value(row),
            }
        )
    return {"metrics": metrics, "prediction_rows": prediction_rows}


def _text_value(row: dict[str, Any]) -> str | None:
    return row.get("normalized_text") or row.get("extracted_text")


def _expand_scores(scores: Any, expected: int) -> list[float | None]:
    if scores is None:
        return [None] * expected
    if hasattr(scores, "tolist"):
        return [float(value) for value in scores.tolist()]
    return [float(value) for value in scores]
