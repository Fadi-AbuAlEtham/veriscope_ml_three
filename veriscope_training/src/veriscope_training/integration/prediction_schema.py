from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class IntegrationPrediction:
    request_id: str | None
    model_name: str
    model_version: str | None
    track: str
    score: float | None
    predicted_label: int | None
    risk_level: str
    thresholds_used: dict[str, Any]
    top_contributing_signals: list[str] = field(default_factory=list)
    source_metadata_summary: dict[str, Any] = field(default_factory=dict)
    language: str | None = None
    explanation_stub: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prediction_schema_example() -> dict[str, Any]:
    return IntegrationPrediction(
        request_id="req_example_001",
        model_name="xlmr_sequence_classifier",
        model_version="2026.04.22",
        track="webpage_transformer",
        score=0.91,
        predicted_label=1,
        risk_level="high",
        thresholds_used={"low_threshold": 0.22, "high_threshold": 0.76, "binary_threshold": 0.76},
        top_contributing_signals=["server_model_score_high", "password_input_present", "phishing_cta_pattern"],
        source_metadata_summary={"language": "en", "registered_domain": "example-login.com"},
        language="en",
        explanation_stub="High-risk server-side phishing classification. Reason codes are placeholders for future evidence fusion.",
        reason_codes=["server_model_high_score", "html_password_input", "cta_verify_account"],
    ).to_dict()
