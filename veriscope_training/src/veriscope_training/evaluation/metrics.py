from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    *,
    y_score: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    if y_true_array.size == 0:
        raise ValueError("Cannot compute metrics for an empty dataset.")

    labels_present = sorted(set(y_true_array.tolist()))
    tn, fp, fn, tp = _confusion_values(y_true_array, y_pred_array)
    metrics: dict[str, Any] = {
        "sample_count": int(y_true_array.size),
        "labels_present": labels_present,
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else None,
        "roc_auc": None,
        "pr_auc": None,
    }
    if y_score is not None and len(set(labels_present)) > 1:
        score_array = np.asarray(y_score, dtype=float)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_array, score_array))
        except ValueError:
            metrics["roc_auc"] = None
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true_array, score_array))
        except ValueError:
            metrics["pr_auc"] = None
    return metrics


def _confusion_values(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if matrix.shape != (2, 2):
        padded = np.zeros((2, 2), dtype=int)
        padded[: matrix.shape[0], : matrix.shape[1]] = matrix
        matrix = padded
    return int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1])
