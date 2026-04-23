from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from veriscope_training.evaluation.metrics import compute_binary_classification_metrics
from veriscope_training.utils.io import read_jsonl, write_json


def load_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def compute_threshold_table(
    prediction_rows: list[dict[str, Any]],
    *,
    max_thresholds: int = 200,
) -> list[dict[str, Any]]:
    scored_rows = [row for row in prediction_rows if row.get("normalized_label") in (0, 1) and row.get("score") is not None]
    if not scored_rows:
        raise ValueError("No scored prediction rows with normalized_label were found for calibration.")
    y_true = np.asarray([int(row["normalized_label"]) for row in scored_rows], dtype=int)
    scores = np.asarray([float(row["score"]) for row in scored_rows], dtype=float)
    thresholds = _candidate_thresholds(scores, max_thresholds=max_thresholds)
    table: list[dict[str, Any]] = []
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        metrics = compute_binary_classification_metrics(y_true, y_pred, y_score=scores)
        below_mask = scores < threshold
        above_mask = scores >= threshold
        table.append(
            {
                "threshold": round(float(threshold), 6),
                "metrics": metrics,
                "low_bucket_count": int(below_mask.sum()),
                "medium_bucket_count": 0,
                "high_bucket_count": int(above_mask.sum()),
                "low_bucket_phishing_rate": _bucket_positive_rate(y_true, below_mask),
                "high_bucket_phishing_rate": _bucket_positive_rate(y_true, above_mask),
                "high_bucket_precision": _bucket_precision(y_true, above_mask),
                "low_bucket_negative_predictive_value": _bucket_negative_predictive_value(y_true, below_mask),
            }
        )
    return table


def select_binary_threshold(
    threshold_table: list[dict[str, Any]],
    *,
    objective: str,
    precision_floor: float | None = None,
    recall_floor: float | None = None,
) -> dict[str, Any]:
    if not threshold_table:
        raise ValueError("Threshold table is empty.")
    candidates = list(threshold_table)
    if objective == "recall_under_precision_floor":
        floor = precision_floor if precision_floor is not None else 0.9
        filtered = [row for row in candidates if (row["metrics"].get("precision") or 0.0) >= floor]
        if filtered:
            candidates = filtered
        chosen = max(candidates, key=lambda row: ((row["metrics"].get("recall") or 0.0), (row["metrics"].get("f1") or 0.0)))
    elif objective == "min_false_positives_under_recall":
        floor = recall_floor if recall_floor is not None else 0.8
        filtered = [row for row in candidates if (row["metrics"].get("recall") or 0.0) >= floor]
        if filtered:
            candidates = filtered
        chosen = min(candidates, key=lambda row: ((row["metrics"].get("false_positive_rate") or math.inf), -(row["metrics"].get("f1") or 0.0)))
    else:
        chosen = max(candidates, key=lambda row: ((row["metrics"].get("f1") or 0.0), (row["metrics"].get("pr_auc") or 0.0)))
        objective = "maximize_f1"
    return {
        "objective": objective,
        "selected_threshold": chosen["threshold"],
        "selected_metrics": chosen["metrics"],
        "rationale": _binary_threshold_rationale(objective, chosen, precision_floor=precision_floor, recall_floor=recall_floor),
    }


def build_risk_mapping(
    prediction_rows: list[dict[str, Any]],
    *,
    binary_threshold: float,
    low_threshold_strategy: str = "max_low_bucket_coverage_under_phishing_rate",
    low_bucket_max_phishing_rate: float = 0.05,
    min_low_bucket_coverage: float = 0.1,
    high_threshold_strategy: str = "binary_threshold",
    high_bucket_precision_floor: float = 0.9,
    high_bucket_recall_floor: float = 0.4,
    max_thresholds: int = 200,
    min_medium_width: float = 0.05,
    min_bucket_support: int = 10,
) -> dict[str, Any]:
    threshold_table = compute_threshold_table(prediction_rows, max_thresholds=max_thresholds)
    low_threshold = _select_low_threshold(
        threshold_table,
        binary_threshold=binary_threshold,
        strategy=low_threshold_strategy,
        max_phishing_rate=low_bucket_max_phishing_rate,
        min_coverage=min_low_bucket_coverage,
    )
    high_threshold = _select_high_threshold(
        threshold_table,
        binary_threshold=binary_threshold,
        strategy=high_threshold_strategy,
        precision_floor=high_bucket_precision_floor,
        recall_floor=high_bucket_recall_floor,
    )

    degenerate_warning = None
    if (high_threshold - low_threshold) < min_medium_width:
        degenerate_warning = "degenerate_three_level_mapping"
        # If it's degenerate, we align them to binary_threshold but keep them as is for reporting
        # User said "do not force a medium-risk bucket", so we report it as degenerate.

    buckets = summarize_risk_buckets(prediction_rows, low_threshold=low_threshold, high_threshold=high_threshold)

    # Check for min_bucket_support
    if buckets["medium"]["count"] < min_bucket_support:
        degenerate_warning = "degenerate_three_level_mapping"

    return {
        "binary_threshold": binary_threshold,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "low_threshold_strategy": low_threshold_strategy,
        "high_threshold_strategy": high_threshold_strategy,
        "risk_bucket_summary": buckets,
        "degenerate_warning": degenerate_warning,
        "future_fusion_note": (
            "These thresholds apply to server-side model scores only. "
            "Future VeriScope fusion can combine them with heuristic scores, permission-abuse signals, and structural indicators."
        ),
    }


def summarize_risk_buckets(
    prediction_rows: list[dict[str, Any]],
    *,
    low_threshold: float,
    high_threshold: float,
) -> dict[str, Any]:
    counts = {
        "low": {"count": 0, "phishing": 0, "benign": 0},
        "medium": {"count": 0, "phishing": 0, "benign": 0},
        "high": {"count": 0, "phishing": 0, "benign": 0},
    }
    for row in prediction_rows:
        score = row.get("score")
        if score is None:
            continue
        label = row.get("normalized_label")
        bucket = "medium"
        if score < low_threshold:
            bucket = "low"
        elif score >= high_threshold:
            bucket = "high"
        counts[bucket]["count"] += 1
        if label == 1:
            counts[bucket]["phishing"] += 1
        elif label == 0:
            counts[bucket]["benign"] += 1
    return counts


def calibrate_predictions(
    prediction_rows: list[dict[str, Any]],
    *,
    objective: str,
    precision_floor: float | None = None,
    recall_floor: float | None = None,
    low_bucket_max_phishing_rate: float = 0.05,
    min_low_bucket_coverage: float = 0.1,
    high_threshold_strategy: str = "binary_threshold",
    high_bucket_precision_floor: float = 0.9,
    high_bucket_recall_floor: float = 0.4,
    max_thresholds: int = 200,
    min_medium_width: float = 0.05,
    min_bucket_support: int = 10,
) -> dict[str, Any]:
    threshold_table = compute_threshold_table(prediction_rows, max_thresholds=max_thresholds)
    binary = select_binary_threshold(
        threshold_table,
        objective=objective,
        precision_floor=precision_floor,
        recall_floor=recall_floor,
    )
    risk_mapping = build_risk_mapping(
        prediction_rows,
        binary_threshold=float(binary["selected_threshold"]),
        low_bucket_max_phishing_rate=low_bucket_max_phishing_rate,
        min_low_bucket_coverage=min_low_bucket_coverage,
        high_threshold_strategy=high_threshold_strategy,
        high_bucket_precision_floor=high_bucket_precision_floor,
        high_bucket_recall_floor=high_bucket_recall_floor,
        max_thresholds=max_thresholds,
        min_medium_width=min_medium_width,
        min_bucket_support=min_bucket_support,
    )
    return {
        "threshold_table": threshold_table,
        "binary_threshold": binary,
        "risk_mapping": risk_mapping,
    }


def save_calibration_payload(payload: dict[str, Any], output_path: str | Path) -> str:
    write_json(output_path, payload)
    return str(output_path)


def _candidate_thresholds(scores: np.ndarray, *, max_thresholds: int) -> list[float]:
    unique = sorted(set(float(value) for value in scores.tolist()))
    if not unique:
        return [0.5]
    if len(unique) > max_thresholds:
        return np.linspace(0.0, 1.0, max_thresholds).round(6).tolist()
    thresholds = sorted(set([0.0, 0.5, 1.0, *unique]))
    return [float(value) for value in thresholds]


def _bucket_positive_rate(y_true: np.ndarray, mask: np.ndarray) -> float | None:
    total = int(mask.sum())
    if total == 0:
        return None
    return float(y_true[mask].sum() / total)


def _bucket_precision(y_true: np.ndarray, mask: np.ndarray) -> float | None:
    total = int(mask.sum())
    if total == 0:
        return None
    positives = int(y_true[mask].sum())
    return float(positives / total)


def _bucket_negative_predictive_value(y_true: np.ndarray, mask: np.ndarray) -> float | None:
    total = int(mask.sum())
    if total == 0:
        return None
    benign = int((y_true[mask] == 0).sum())
    return float(benign / total)


def _select_low_threshold(
    threshold_table: list[dict[str, Any]],
    *,
    binary_threshold: float,
    strategy: str,
    max_phishing_rate: float,
    min_coverage: float,
) -> float:
    eligible = [row for row in threshold_table if row["threshold"] <= binary_threshold]
    if not eligible:
        return binary_threshold
    if strategy != "max_low_bucket_coverage_under_phishing_rate":
        return eligible[0]["threshold"]
    total = max(row["low_bucket_count"] + row["high_bucket_count"] for row in threshold_table)
    filtered = []
    for row in eligible:
        rate = row.get("low_bucket_phishing_rate")
        coverage = row["low_bucket_count"] / total if total else 0.0
        if rate is not None and rate <= max_phishing_rate and coverage >= min_coverage:
            filtered.append((coverage, row["threshold"]))
    if filtered:
        filtered.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return float(filtered[0][1])
    return float(min(eligible, key=lambda row: abs(row["threshold"] - binary_threshold))["threshold"])


def _select_high_threshold(
    threshold_table: list[dict[str, Any]],
    *,
    binary_threshold: float,
    strategy: str,
    precision_floor: float,
    recall_floor: float,
) -> float:
    if strategy == "min_precision_floor":
        filtered = [
            row
            for row in threshold_table
            if (row.get("high_bucket_precision") or 0.0) >= precision_floor
            and (row["metrics"].get("recall") or 0.0) >= recall_floor
        ]
        if filtered:
            return float(min(filtered, key=lambda row: row["threshold"])["threshold"])
    return float(binary_threshold)


def _binary_threshold_rationale(
    objective: str,
    chosen: dict[str, Any],
    *,
    precision_floor: float | None,
    recall_floor: float | None,
) -> str:
    if objective == "recall_under_precision_floor":
        return (
            f"Selected threshold {chosen['threshold']} as the highest-recall operating point "
            f"meeting precision >= {precision_floor}."
        )
    if objective == "min_false_positives_under_recall":
        return (
            f"Selected threshold {chosen['threshold']} as the lowest false-positive operating point "
            f"meeting recall >= {recall_floor}."
        )
    return f"Selected threshold {chosen['threshold']} by maximizing F1."
