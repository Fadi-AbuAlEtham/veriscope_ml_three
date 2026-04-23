from __future__ import annotations

from typing import Any

from veriscope_training.models.selection import choose_best_row, score_candidate_rows


TRACK_BUCKETS = {
    "best_url_model": "url",
    "best_webpage_classical_model": "webpage",
    "best_transformer_model": "transformer",
    "best_tabular_model": "tabular",
}


def recommend_models(
    comparison_payload: dict[str, Any],
    *,
    metric_weights: dict[str, float],
) -> dict[str, Any]:
    rows = comparison_payload.get("rows", [])
    track_recommendations: dict[str, Any] = {}
    for output_key, track_group in TRACK_BUCKETS.items():
        candidates = [row for row in rows if row.get("track_group") == track_group]
        ranked = score_candidate_rows(candidates, metric_weights=metric_weights)
        track_recommendations[output_key] = {
            "winner": ranked[0] if ranked else None,
            "ranked_candidates": ranked[:5],
        }

    overall_stack = {
        "fast_api_scoring": track_recommendations["best_url_model"]["winner"],
        "webpage_classical_fallback": track_recommendations["best_webpage_classical_model"]["winner"],
        "deep_analysis_primary": track_recommendations["best_transformer_model"]["winner"]
        or track_recommendations["best_webpage_classical_model"]["winner"],
        "tabular_research_baseline": track_recommendations["best_tabular_model"]["winner"],
        "server_side_only": True,
    }
    rationale = {
        "metric_weights": metric_weights,
        "notes": [
            "Recommendations are data-driven from saved run metrics.",
            "XLM-R remains the expected primary multilingual candidate only if comparison metrics support it.",
            "mBERT is treated as a comparison baseline, not a hardcoded default.",
            "All ML/NLP inference remains server-side only.",
        ],
    }
    return {
        "track_recommendations": track_recommendations,
        "overall_stack": overall_stack,
        "rationale": rationale,
    }
