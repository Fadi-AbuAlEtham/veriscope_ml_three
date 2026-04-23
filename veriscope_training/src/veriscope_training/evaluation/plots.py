from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def save_confusion_matrix_plot(confusion: dict[str, int], output_path: str | Path, *, title: str) -> str:
    plt = _load_pyplot()
    if plt is None:
        return ""
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = [
            [confusion.get("tn", 0), confusion.get("fp", 0)],
            [confusion.get("fn", 0), confusion.get("tp", 0)],
        ]
        figure, axis = plt.subplots(figsize=(4, 4))
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set_title(title)
        axis.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        axis.set_yticks([0, 1], labels=["True 0", "True 1"])
        for row in range(2):
            for col in range(2):
                axis.text(col, row, str(matrix[row][col]), ha="center", va="center", color="black")
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return str(path)
    except Exception:
        return ""


def save_pr_curve_plot(y_true: list[int], y_score: list[float], output_path: str | Path, *, title: str) -> str:
    if not y_true or not y_score or len(set(y_true)) < 2:
        return ""
    plt = _load_pyplot()
    if plt is None:
        return ""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(recall, precision)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    return _save_figure(figure, output_path, plt)


def save_roc_curve_plot(y_true: list[int], y_score: list[float], output_path: str | Path, *, title: str) -> str:
    if not y_true or not y_score or len(set(y_true)) < 2:
        return ""
    plt = _load_pyplot()
    if plt is None:
        return ""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(fpr, tpr)
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    return _save_figure(figure, output_path, plt)


def save_metric_comparison_bar(
    rows: list[dict[str, object]],
    *,
    metric: str,
    output_path: str | Path,
    title: str,
) -> str:
    if not rows:
        return ""
    plt = _load_pyplot()
    if plt is None:
        return ""
    labels = [f"{row.get('track_group')}:{row.get('model_name')}" for row in rows]
    values = [float(row.get(metric) or 0.0) for row in rows]
    figure, axis = plt.subplots(figsize=(max(6, len(rows) * 1.2), 4))
    axis.bar(labels, values)
    axis.set_title(title)
    axis.set_ylabel(metric)
    axis.tick_params(axis="x", rotation=45, labelsize=8)
    axis.grid(axis="y", alpha=0.3)
    return _save_figure(figure, output_path, plt)


def save_threshold_performance_plot(
    threshold_table: list[dict[str, object]],
    output_path: str | Path,
    *,
    metric: str = "f1",
) -> str:
    if not threshold_table:
        return ""
    plt = _load_pyplot()
    if plt is None:
        return ""
    thresholds = [float(row.get("threshold") or 0.0) for row in threshold_table]
    values = [float(((row.get("metrics") or {}).get(metric) or 0.0)) for row in threshold_table]
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(thresholds, values)
    axis.set_xlabel("Threshold")
    axis.set_ylabel(metric)
    axis.set_title(f"Threshold vs {metric}")
    axis.grid(alpha=0.3)
    return _save_figure(figure, output_path, plt)


def save_score_distribution_plot(predictions: list[dict[str, object]], output_path: str | Path) -> str:
    scored = [row for row in predictions if row.get("score") is not None and row.get("normalized_label") in (0, 1)]
    if not scored:
        return ""
    plt = _load_pyplot()
    if plt is None:
        return ""
    positives = [float(row["score"]) for row in scored if row.get("normalized_label") == 1]
    negatives = [float(row["score"]) for row in scored if row.get("normalized_label") == 0]
    figure, axis = plt.subplots(figsize=(5, 4))
    bins = np.linspace(0, 1, 30)
    if negatives:
        axis.hist(negatives, bins=bins, alpha=0.6, label="benign")
    if positives:
        axis.hist(positives, bins=bins, alpha=0.6, label="phishing")
    axis.set_title("Score Distribution by Class")
    axis.set_xlabel("Score")
    axis.set_ylabel("Count")
    axis.legend()
    return _save_figure(figure, output_path, plt)


def save_risk_bucket_distribution_plot(bucket_summary: dict[str, dict[str, int]], output_path: str | Path) -> str:
    if not bucket_summary:
        return ""
    plt = _load_pyplot()
    if plt is None:
        return ""
    labels = ["low", "medium", "high"]
    counts = [int((bucket_summary.get(label) or {}).get("count", 0)) for label in labels]
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.bar(labels, counts, color=["#8fb339", "#f2c14e", "#d1495b"])
    axis.set_title("Risk Bucket Distribution")
    axis.set_ylabel("Count")
    axis.grid(axis="y", alpha=0.3)
    return _save_figure(figure, output_path, plt)


def _load_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def _save_figure(figure, output_path: str | Path, plt) -> str:
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return str(path)
    except Exception:
        try:
            plt.close(figure)
        except Exception:
            pass
        return ""
