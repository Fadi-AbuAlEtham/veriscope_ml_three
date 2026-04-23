from __future__ import annotations

from typing import Any
import pandas as pd
from veriscope_training.evaluation.metrics import compute_binary_classification_metrics
from veriscope_training.utils.io import read_jsonl

def align_predictions(
    url_preds_path: str,
    webpage_preds_path: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    url_df = pd.DataFrame(read_jsonl(url_preds_path))
    web_df = pd.DataFrame(read_jsonl(webpage_preds_path))

    if "sample_id" not in url_df.columns or "sample_id" not in web_df.columns:
        raise ValueError("Both prediction files must contain 'sample_id' for alignment.")

    url_count = len(url_df)
    web_count = len(web_df)

    # Use suffixes to distinguish scores
    merged = pd.merge(
        url_df[["sample_id", "score", "normalized_label"]],
        web_df[["sample_id", "score"]],
        on="sample_id",
        suffixes=("_url", "_webpage")
    )

    intersection_count = len(merged)
    stats = {
        "url_total": url_count,
        "webpage_total": web_count,
        "intersected": intersection_count,
        "dropped_url": url_count - intersection_count,
        "dropped_webpage": web_count - intersection_count,
    }

    return merged, stats

def compute_fusion_metrics(
    y_true: Any,
    y_score: Any,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    return compute_binary_classification_metrics(y_true, y_pred, y_score=y_score)
