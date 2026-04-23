from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
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


def train_url_baseline(
    config: AppConfig,
    *,
    model_name: str,
    run_name: str | None = None,
    split_strategy: str | None = None,
) -> dict[str, Any]:
    model_config = config.models_config.get("url_models", {}).get(model_name)
    if not model_config or not model_config.get("enabled", False):
        raise KeyError(f"URL model '{model_name}' is not enabled in configs/models.yaml.")

    records = load_processed_view_records(config, "unified_url_dataset")
    excluded_counts = {
        "missing_binary_label": sum(1 for row in records if row.get("normalized_label") not in (0, 1)),
        "missing_normalized_url": sum(1 for row in records if not row.get("normalized_url")),
    }
    supervised = [
        row for row in records if row.get("normalized_label") in (0, 1) and row.get("normalized_url")
    ]
    if not supervised:
        raise RuntimeError("No supervised URL records with normalized_url were found. Build processed datasets first.")

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
    train_texts = [row["normalized_url"] for row in train_rows]
    val_texts = [row["normalized_url"] for row in validation_rows]
    test_texts = [row["normalized_url"] for row in test_rows]
    y_train = np.asarray([row["normalized_label"] for row in train_rows], dtype=int)
    y_val = np.asarray([row["normalized_label"] for row in validation_rows], dtype=int)
    y_test = np.asarray([row["normalized_label"] for row in test_rows], dtype=int)

    context = create_training_run_context(config, track="url", model_name=model_name, run_name=run_name)
    split_paths = save_split_manifest(split, records=supervised, output_dir=context.run_dir / "splits")

    estimator, vectorizer_info = _fit_url_model(model_name, model_config, train_rows, train_texts, y_train)
    train_eval = _evaluate_url_model(estimator, train_rows, train_texts, y_train)
    val_eval = _evaluate_url_model(estimator, validation_rows, val_texts, y_val)
    test_eval = _evaluate_url_model(estimator, test_rows, test_texts, y_test)

    from veriscope_training.evaluation.plots import save_confusion_matrix_plot

    confusion_plot = save_confusion_matrix_plot(
        test_eval["metrics"]["confusion_matrix"],
        context.reports_dir / "confusion_matrix_test.png",
        title=f"URL {model_name} Test Confusion Matrix",
    )

    predictions_path = save_predictions(context, "test", test_eval["prediction_rows"])
    artifact_path = save_model_bundle(
        context,
        "model_bundle",
        {
            "estimator": estimator,
            "track": "url",
            "model_name": model_name,
            "vectorizer_info": vectorizer_info,
            "input_fields": ["normalized_url"] if model_name != "handcrafted_boosting" else ["url_features"],
        },
    )
    metrics_payload = {
        "train": train_eval["metrics"],
        "validation": val_eval["metrics"],
        "test": test_eval["metrics"],
        "class_distribution": split.label_counts,
        "split_strategy": split.strategy,
        "confusion_matrix_plot": confusion_plot,
        "excluded_counts": excluded_counts,
    }
    metrics_path = save_metrics(context, metrics_payload)
    label_metadata_path = save_label_metadata(context, {"label_map": {"benign": 0, "phishing": 1}})
    config_snapshot_path = save_config_snapshot(
        context,
        {
            "track": "url",
            "model_name": model_name,
            "model_config": model_config,
            "split": split.to_dict(),
            "excluded_counts": excluded_counts,
            "input_fields": ["normalized_url"] if model_name != "handcrafted_boosting" else ["url_features"],
            "feature_names_used": vectorizer_info.get("feature_names_used"),
        },
    )
    versions_path = save_package_versions(
        context,
        ["numpy", "scikit-learn", "scipy", "pandas", "pyarrow"],
    )

    summary = {
        "track": "url",
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


def _fit_url_model(model_name: str, model_config: dict[str, Any], rows: list[dict[str, Any]], texts: list[str], y: np.ndarray):
    if model_name in {"tfidf_logreg", "tfidf_linear_svm"}:
        vectorizer_cfg = model_config.get("vectorizer", {})
        vectorizer = TfidfVectorizer(
            analyzer=vectorizer_cfg.get("analyzer", "char_wb"),
            ngram_range=tuple(vectorizer_cfg.get("ngram_range", [3, 5])),
            min_df=int(vectorizer_cfg.get("min_df", 2)),
            max_features=vectorizer_cfg.get("max_features"),
        )
        classifier_cfg = model_config.get("classifier", {})
        if model_name == "tfidf_logreg":
            classifier = LogisticRegression(
                max_iter=int(classifier_cfg.get("max_iter", 2000)),
                class_weight=classifier_cfg.get("class_weight"),
                solver="liblinear",
            )
        else:
            classifier = LinearSVC(class_weight=classifier_cfg.get("class_weight"))
        estimator = Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])
        estimator.fit(texts, y)
        return estimator, {"kind": "tfidf_pipeline", "feature_names_used": ["normalized_url"]}

    if model_name == "handcrafted_boosting":
        feature_dicts = [_select_url_features(row) for row in rows]
        vectorizer = DictVectorizer(sparse=True)
        x_train = vectorizer.fit_transform(feature_dicts)
        classifier_cfg = model_config.get("classifier", {})
        classifier = RandomForestClassifier(
            n_estimators=int(classifier_cfg.get("n_estimators", 300)),
            max_depth=classifier_cfg.get("max_depth"),
            class_weight=classifier_cfg.get("class_weight"),
            random_state=42,
            n_jobs=-1,
        )
        classifier.fit(x_train, y)
        return {
            "vectorizer": vectorizer,
            "classifier": classifier,
        }, {
            "kind": "dict_vectorizer",
            "feature_names_used": sorted({key for row in rows for key in _select_url_features(row).keys()}),
        }

    raise KeyError(f"Unsupported URL model '{model_name}'.")


def _evaluate_url_model(estimator: Any, rows: list[dict[str, Any]], texts: list[str], y_true: np.ndarray) -> dict[str, Any]:
    if isinstance(estimator, Pipeline):
        y_pred = estimator.predict(texts)
        scores = None
        classifier = estimator.named_steps["classifier"]
        if hasattr(classifier, "predict_proba"):
            scores = estimator.predict_proba(texts)[:, 1]
        elif hasattr(classifier, "decision_function"):
            scores = classifier.decision_function(estimator.named_steps["vectorizer"].transform(texts))
    else:
        vectorizer = estimator["vectorizer"]
        classifier = estimator["classifier"]
        x = vectorizer.transform([_select_url_features(row) for row in rows])
        y_pred = classifier.predict(x)
        scores = classifier.predict_proba(x)[:, 1] if hasattr(classifier, "predict_proba") else None

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
                "normalized_url": row.get("normalized_url"),
            }
        )
    return {"metrics": metrics, "prediction_rows": prediction_rows}


def _select_url_features(row: dict[str, Any]) -> dict[str, Any]:
    url_features = row.get("url_features") or {}
    allowed = {
        "is_ip_address",
        "has_punycode",
        "has_percent_encoding",
        "url_length",
        "path_length",
        "query_length",
        "num_subdomains",
        "suspicious_keyword_count",
        "digit_count",
        "digit_ratio",
        "hostname_digit_count",
        "contains_at_symbol",
        "contains_double_slash_path",
        "contains_hyphenated_hostname",
        "path_depth",
        "query_param_count",
    }
    return {key: value for key, value in url_features.items() if key in allowed and value is not None}


def _expand_scores(scores: Any, expected: int) -> list[float | None]:
    if scores is None:
        return [None] * expected
    if hasattr(scores, "tolist"):
        return [float(value) for value in scores.tolist()]
    return [float(value) for value in scores]
