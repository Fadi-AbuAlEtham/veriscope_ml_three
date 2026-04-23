from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from veriscope_training.adaptive.feedback_schema import load_feedback_records
from veriscope_training.utils.io import read_jsonl, write_json, write_jsonl


def export_retraining_candidates(
    *,
    prediction_path: str | Path,
    output_dir: str | Path,
    feedback_path: str | Path | None = None,
    uncertainty_band: tuple[float, float] = (0.4, 0.6),
) -> dict[str, Any]:
    predictions = list(read_jsonl(prediction_path))
    feedback_rows = load_feedback_records(feedback_path) if feedback_path else []
    feedback_by_sample = {row.get("sample_id"): row for row in feedback_rows if row.get("sample_id")}

    candidates = []
    summary = {"uncertain": 0, "misclassified": 0, "reviewed": 0}
    for row in predictions:
        sample_id = row.get("sample_id")
        reasons = []
        score = row.get("score")
        if isinstance(score, (int, float)) and uncertainty_band[0] <= score <= uncertainty_band[1]:
            reasons.append("uncertain_score")
            summary["uncertain"] += 1
        feedback = feedback_by_sample.get(sample_id)
        if feedback:
            final_label = feedback.get("final_reviewed_label")
            if final_label in (0, 1):
                reasons.append("reviewed_label_available")
                summary["reviewed"] += 1
                if row.get("predicted_label") in (0, 1) and row.get("predicted_label") != final_label:
                    reasons.append("misclassified_after_review")
                    summary["misclassified"] += 1
        if not reasons:
            continue
        candidates.append(
            {
                "sample_id": sample_id,
                "source_dataset": row.get("source_dataset"),
                "predicted_label": row.get("predicted_label"),
                "prediction_score": score,
                "reasons": sorted(set(reasons)),
                "feedback": feedback,
            }
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prediction_path": str(prediction_path),
        "feedback_path": str(feedback_path) if feedback_path else None,
        "candidate_count": len(candidates),
        "summary": summary,
    }
    manifest_path = output_root / "retraining_candidates.manifest.json"
    candidates_path = output_root / "retraining_candidates.jsonl"
    write_json(manifest_path, manifest)
    write_jsonl(candidates_path, candidates)
    return {
        "manifest_path": str(manifest_path),
        "candidates_path": str(candidates_path),
        "candidate_count": len(candidates),
        "summary": summary,
    }
