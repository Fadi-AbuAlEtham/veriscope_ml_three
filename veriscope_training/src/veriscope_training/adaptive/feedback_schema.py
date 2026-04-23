from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from veriscope_training.utils.io import read_jsonl, write_jsonl


@dataclass
class FeedbackRecord:
    sample_id: str | None = None
    request_id: str | None = None
    source_dataset: str | None = None
    live_source: str | None = None
    model_used: str | None = None
    prediction_score: float | None = None
    predicted_label: int | None = None
    final_reviewed_label: int | None = None
    review_status: str | None = None
    user_reported: bool | None = None
    analyst_reviewed: bool | None = None
    false_positive: bool | None = None
    false_negative: bool | None = None
    language: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    signals_snapshot: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_feedback_records(path: str | Path, rows: list[FeedbackRecord | dict[str, Any]]) -> str:
    payload = [row.to_dict() if isinstance(row, FeedbackRecord) else row for row in rows]
    write_jsonl(path, payload)
    return str(path)


def load_feedback_records(path: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))
