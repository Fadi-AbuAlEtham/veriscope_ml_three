from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from veriscope_training.models.artifacts import load_processed_view_records
from veriscope_training.utils.io import read_json, read_records_file, write_json


def generate_drift_report(
    reference_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_count": len(reference_rows),
        "current_count": len(current_rows),
        "label_distribution": _distribution_shift(reference_rows, current_rows, lambda row: str(row.get("normalized_label"))),
        "language_distribution": _distribution_shift(reference_rows, current_rows, lambda row: row.get("language") or "unknown"),
        "source_distribution": _distribution_shift(reference_rows, current_rows, lambda row: row.get("source_dataset") or "unknown"),
        "tld_distribution": _distribution_shift(
            reference_rows,
            current_rows,
            lambda row: ((row.get("url_features") or {}).get("suffix") or "unknown"),
        ),
        "score_distribution": _score_shift(reference_rows, current_rows),
        "top_token_shift": _token_shift(reference_rows, current_rows),
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


def load_rows_for_drift(path_or_view: str, *, config=None) -> list[dict[str, Any]]:
    path = Path(path_or_view)
    if path.exists():
        if path.suffix == ".json":
            payload = read_json(path)
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict) and "records" in payload and isinstance(payload["records"], list):
                return payload["records"]
            raise ValueError(f"Unsupported JSON payload for drift input at {path}.")
        return list(read_records_file(path))
    if path.suffix:
        raise FileNotFoundError(f"Drift input file was not found: {path}")
    if config is None:
        raise FileNotFoundError(f"Drift input '{path_or_view}' was not found as a file path.")
    return load_processed_view_records(config, path_or_view)


def _distribution_shift(reference_rows, current_rows, value_getter):
    reference = Counter(value_getter(row) for row in reference_rows)
    current = Counter(value_getter(row) for row in current_rows)
    return {
        "reference": dict(reference),
        "current": dict(current),
        "js_divergence": _js_divergence(reference, current),
    }


def _score_shift(reference_rows, current_rows):
    reference_scores = [row.get("score") for row in reference_rows if row.get("score") is not None]
    current_scores = [row.get("score") for row in current_rows if row.get("score") is not None]
    if not reference_scores or not current_scores:
        return {"available": False}
    return {
        "available": True,
        "reference_mean": sum(reference_scores) / len(reference_scores),
        "current_mean": sum(current_scores) / len(current_scores),
        "reference_std": _std(reference_scores),
        "current_std": _std(current_scores),
    }


def _token_shift(reference_rows, current_rows, *, top_k: int = 25):
    reference_counter = Counter()
    current_counter = Counter()
    for row in reference_rows:
        reference_counter.update(_tokens(row.get("normalized_text") or row.get("extracted_text") or ""))
    for row in current_rows:
        current_counter.update(_tokens(row.get("normalized_text") or row.get("extracted_text") or ""))
    combined = set(reference_counter) | set(current_counter)
    scored = []
    for token in combined:
        scored.append(
            {
                "token": token,
                "reference_count": reference_counter.get(token, 0),
                "current_count": current_counter.get(token, 0),
                "delta": current_counter.get(token, 0) - reference_counter.get(token, 0),
            }
        )
    scored.sort(key=lambda item: abs(item["delta"]), reverse=True)
    return scored[:top_k]


def _tokens(text: str) -> list[str]:
    return [token for token in text.lower().split() if len(token) >= 4][:200]


def _js_divergence(counter_a: Counter, counter_b: Counter) -> float:
    keys = sorted(set(counter_a) | set(counter_b))
    total_a = sum(counter_a.values()) or 1
    total_b = sum(counter_b.values()) or 1
    p = [counter_a.get(key, 0) / total_a for key in keys]
    q = [counter_b.get(key, 0) / total_b for key in keys]
    m = [(x + y) / 2 for x, y in zip(p, q)]
    return round((_kl_divergence(p, m) + _kl_divergence(q, m)) / 2, 6)


def _kl_divergence(p: list[float], q: list[float]) -> float:
    total = 0.0
    for px, qx in zip(p, q):
        if px > 0 and qx > 0:
            total += px * math.log(px / qx, 2)
    return total


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance ** 0.5
