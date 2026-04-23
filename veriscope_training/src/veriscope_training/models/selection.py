from __future__ import annotations

from typing import Any


POSITIVE_METRICS = {"precision", "recall", "f1", "pr_auc", "roc_auc", "accuracy"}
NEGATIVE_METRICS = {"false_positive_rate", "false_negative_rate", "artifact_size_mb"}


def score_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    metric_weights: dict[str, float],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    normalized = {metric: _normalized_metric_values(rows, metric) for metric in metric_weights}
    scored = []
    for index, row in enumerate(rows):
        score = 0.0
        used_weight = 0.0
        for metric, weight in metric_weights.items():
            value = normalized[metric][index]
            if value is None:
                continue
            score += weight * value
            used_weight += abs(weight)
        enriched = dict(row)
        enriched["selection_score"] = float(score / used_weight) if used_weight else 0.0
        scored.append(enriched)
    scored.sort(key=lambda item: item["selection_score"], reverse=True)
    return scored


def choose_best_row(rows: list[dict[str, Any]], *, metric_weights: dict[str, float]) -> dict[str, Any] | None:
    ranked = score_candidate_rows(rows, metric_weights=metric_weights)
    return ranked[0] if ranked else None


def _normalized_metric_values(rows: list[dict[str, Any]], metric: str) -> list[float | None]:
    raw_values = [row.get(metric) for row in rows]
    numeric = [float(value) for value in raw_values if isinstance(value, (int, float))]
    if not numeric:
        return [None] * len(rows)
    minimum = min(numeric)
    maximum = max(numeric)
    values: list[float | None] = []
    for value in raw_values:
        if not isinstance(value, (int, float)):
            values.append(None)
            continue
        if maximum == minimum:
            normalized = 1.0
        else:
            normalized = (float(value) - minimum) / (maximum - minimum)
        if metric in NEGATIVE_METRICS:
            normalized = 1.0 - normalized
        values.append(normalized)
    return values
