from __future__ import annotations

from typing import Any


def apply_risk_mapping(score: float | None, *, config: dict[str, Any]) -> dict[str, Any]:
    if score is None:
        return {
            "score": None,
            "predicted_label": None,
            "risk_level": "unknown",
            "thresholds_used": {
                "low_threshold": config.get("low_threshold"),
                "high_threshold": config.get("high_threshold"),
                "binary_threshold": config.get("binary_threshold"),
            },
        }
    low_threshold = float(config.get("low_threshold", 0.3))
    high_threshold = float(config.get("high_threshold", 0.7))
    binary_threshold = float(config.get("binary_threshold", high_threshold))
    if score < low_threshold:
        risk_level = "low"
    elif score >= high_threshold:
        risk_level = "high"
    else:
        risk_level = "medium"
    return {
        "score": float(score),
        "predicted_label": int(score >= binary_threshold),
        "risk_level": risk_level,
        "thresholds_used": {
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "binary_threshold": binary_threshold,
        },
    }
