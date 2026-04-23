from __future__ import annotations

from pathlib import Path
from typing import Any

from veriscope_training.integration.prediction_schema import prediction_schema_example
from veriscope_training.utils.io import write_json, write_yaml


def export_integration_configs(
    *,
    output_dir: str | Path,
    recommendations: dict[str, Any],
    threshold_configs: dict[str, Any],
    risk_mapping_configs: dict[str, Any],
) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    recommendation_path = write_json(target / "model_recommendations.json", recommendations)
    threshold_path = write_json(target / "threshold_configs.json", threshold_configs)
    risk_mapping_path = write_json(target / "risk_mapping_configs.json", risk_mapping_configs)
    schema_json_path = write_json(target / "prediction_schema.example.json", prediction_schema_example())
    schema_yaml_path = write_yaml(
        target / "prediction_schema.schema.yaml",
        {
            "server_side_only": True,
            "schema_fields": prediction_schema_example(),
            "future_fusion_placeholders": [
                "heuristic_score",
                "permission_abuse_signals",
                "structural_webpage_indicators",
                "reason_codes",
            ],
        },
    )
    score_interpretation_path = write_json(
        target / "score_interpretation.json",
        {
            "note": "Model scores map to low/medium/high server-side risk levels through per-model thresholds.",
            "future_fusion_note": "Final VeriScope risk may later combine heuristics, permissions, and structural indicators.",
        },
    )
    return {
        "model_recommendations": str(recommendation_path),
        "threshold_configs": str(threshold_path),
        "risk_mapping_configs": str(risk_mapping_path),
        "prediction_schema_json": str(schema_json_path),
        "prediction_schema_yaml": str(schema_yaml_path),
        "score_interpretation": str(score_interpretation_path),
    }
