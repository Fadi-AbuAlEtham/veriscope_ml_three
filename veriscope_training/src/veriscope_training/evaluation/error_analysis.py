from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from veriscope_training.models.artifacts import load_processed_view_records
from veriscope_training.utils.io import read_jsonl, write_json, write_jsonl


TRACK_VIEW_MAP = {
    "url": "unified_url_dataset",
    "webpage": "unified_webpage_dataset",
    "webpage_transformer": "unified_webpage_dataset",
    "tabular": "unified_tabular_dataset",
}


def analyze_prediction_errors(
    prediction_rows: list[dict[str, Any]],
    *,
    processed_rows: list[dict[str, Any]] | None = None,
    top_k: int = 25,
) -> dict[str, Any]:
    processed_by_sample = {
        row.get("sample_id"): row for row in (processed_rows or []) if row.get("sample_id")
    }
    false_positives = []
    false_negatives = []
    for row in prediction_rows:
        true_label = row.get("normalized_label")
        pred_label = row.get("predicted_label")
        if true_label == 0 and pred_label == 1:
            false_positives.append(_enrich_prediction_row(row, processed_by_sample))
        elif true_label == 1 and pred_label == 0:
            false_negatives.append(_enrich_prediction_row(row, processed_by_sample))
    payload = {
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "errors_by_language": _count_by(false_positives + false_negatives, "language"),
        "errors_by_source": _count_by(false_positives + false_negatives, "source_dataset"),
        "errors_by_domain": _count_by(false_positives + false_negatives, "registered_domain"),
        "errors_by_tld": _count_by(false_positives + false_negatives, "suffix"),
        "top_false_positive_tokens": _top_tokens(false_positives, top_k=top_k),
        "top_false_negative_tokens": _top_tokens(false_negatives, top_k=top_k),
        "heuristic_opportunities": _heuristic_opportunities(false_negatives, false_positives, top_k=top_k),
    }
    return payload


def analyze_run_errors(config, run_summary: dict[str, Any], *, split_name: str = "test", top_k: int = 25) -> dict[str, Any]:
    artifact_paths = run_summary.get("artifact_paths", {})
    prediction_path = artifact_paths.get(f"{split_name}_predictions") or artifact_paths.get("test_predictions")
    if not prediction_path:
        raise FileNotFoundError(f"No {split_name} predictions path recorded for run '{run_summary.get('run_dir')}'.")
    prediction_rows = list(read_jsonl(prediction_path))
    view_name = TRACK_VIEW_MAP.get(run_summary.get("track"))
    processed_rows = load_processed_view_records(config, view_name) if view_name else []
    return analyze_prediction_errors(prediction_rows, processed_rows=processed_rows, top_k=top_k)


def save_error_analysis(payload: dict[str, Any], *, output_dir: str | Path) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = write_json(target / "error_analysis.json", payload)
    false_positive_path = write_jsonl(target / "false_positives.jsonl", payload.get("false_positives", []))
    false_negative_path = write_jsonl(target / "false_negatives.jsonl", payload.get("false_negatives", []))
    markdown_path = target / "error_analysis.md"
    markdown_path.write_text(_error_markdown(payload), encoding="utf-8")
    return {
        "summary": str(summary_path),
        "false_positives": str(false_positive_path),
        "false_negatives": str(false_negative_path),
        "markdown": str(markdown_path),
    }


def _enrich_prediction_row(row: dict[str, Any], processed_by_sample: dict[str, dict[str, Any]]) -> dict[str, Any]:
    enriched = dict(row)
    processed = processed_by_sample.get(row.get("sample_id")) or {}
    url_features = processed.get("url_features") or {}
    enriched["language"] = processed.get("language")
    enriched["registered_domain"] = url_features.get("registered_domain")
    enriched["suffix"] = url_features.get("suffix")
    enriched["normalized_url"] = processed.get("normalized_url") or processed.get("original_url")
    enriched["text_snippet"] = (processed.get("normalized_text") or processed.get("extracted_text") or "")[:240]
    enriched["action_texts"] = (processed.get("html_features") or {}).get("action_texts", [])[:10]
    return enriched


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter((row.get(key) or "unknown") for row in rows)
    return dict(counter.most_common())


def _top_tokens(rows: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    counter = Counter()
    for row in rows:
        text = row.get("text_snippet") or ""
        counter.update(token for token in text.lower().split() if len(token) >= 4)
    return [{"token": token, "count": count} for token, count in counter.most_common(top_k)]


def _heuristic_opportunities(
    false_negatives: list[dict[str, Any]],
    false_positives: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    cta_counter = Counter()
    domain_counter = Counter()
    for row in false_negatives:
        cta_counter.update(action.lower() for action in row.get("action_texts") or [] if action)
        if row.get("registered_domain"):
            domain_counter[row["registered_domain"]] += 1
    proposals = []
    for pattern, count in cta_counter.most_common(top_k // 2):
        proposals.append(
            {
                "proposal_type": "cta_pattern_from_false_negatives",
                "candidate_pattern": pattern,
                "supporting_count": count,
                "rationale": "Recurring CTA pattern appears in false negatives and may support heuristic review.",
            }
        )
    for pattern, count in domain_counter.most_common(top_k // 2):
        proposals.append(
            {
                "proposal_type": "domain_family_review",
                "candidate_pattern": pattern,
                "supporting_count": count,
                "rationale": "Repeated domain family appears in misclassified phishing pages and may merit deeper review.",
            }
        )
    if false_positives:
        proposals.append(
            {
                "proposal_type": "false_positive_guardrail",
                "candidate_pattern": "review recurring benign tokens",
                "supporting_count": len(false_positives),
                "rationale": "Use false-positive clusters to refine future guardrails and threshold choices.",
            }
        )
    return proposals[:top_k]


def _error_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Error Analysis",
        "",
        f"- False positives: {payload.get('false_positive_count', 0)}",
        f"- False negatives: {payload.get('false_negative_count', 0)}",
        "",
        "## By Source",
        "",
    ]
    for key, value in payload.get("errors_by_source", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## By Language", ""])
    for key, value in payload.get("errors_by_language", {}).items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"
