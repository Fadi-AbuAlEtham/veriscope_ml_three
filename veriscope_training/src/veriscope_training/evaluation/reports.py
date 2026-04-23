from __future__ import annotations

from pathlib import Path
from typing import Any

from veriscope_training.utils.io import ensure_parent_dir, write_json
from veriscope_training.utils.serialization import save_text_artifact


DEFAULT_COLUMNS = [
    "track_group",
    "model_name",
    "split_strategy",
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "false_positive_rate",
    "false_negative_rate",
    "artifact_size_mb",
]


def rows_to_markdown_table(rows: list[dict[str, Any]], *, columns: list[str] | None = None) -> str:
    columns = columns or DEFAULT_COLUMNS
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append("" if value is None else str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def save_comparison_reports(
    payload: dict[str, Any],
    *,
    output_dir: str | Path,
    title: str = "Experiment Comparison",
) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = write_json(target / "comparison_report.json", payload)
    markdown = build_comparison_markdown(payload, title=title)
    markdown_path = save_text_artifact(target / "comparison_report.md", markdown)
    return {"json": str(json_path), "markdown": str(markdown_path)}


def build_comparison_markdown(payload: dict[str, Any], *, title: str) -> str:
    rows = payload.get("rows", [])
    lines = [
        f"# {title}",
        "",
        f"- Run count: {payload.get('row_count', len(rows))}",
        f"- Split: {payload.get('split_name', 'test')}",
        "",
        "## Overall Table",
        "",
        rows_to_markdown_table(rows),
    ]
    by_track = {}
    for row in rows:
        by_track.setdefault(row.get("track_group") or "unknown", []).append(row)
    for track_name, track_rows in sorted(by_track.items()):
        lines.extend(
            [
                "",
                f"## {track_name.title()}",
                "",
                rows_to_markdown_table(track_rows),
            ]
        )
    return "\n".join(lines) + "\n"


def save_recommendation_reports(
    recommendation_payload: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = write_json(target / "recommendation.json", recommendation_payload)
    markdown_path = save_text_artifact(target / "recommendation.md", build_recommendation_markdown(recommendation_payload))
    return {"json": str(json_path), "markdown": str(markdown_path)}


def build_recommendation_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Recommended Production Candidates",
        "",
        "## Track Winners",
        "",
    ]
    for key, value in payload.get("track_recommendations", {}).items():
        winner = value.get("winner")
        if not winner:
            lines.append(f"- {key}: no eligible run")
            continue
        lines.append(
            f"- {key}: `{winner.get('model_name')}` on `{winner.get('track')}` "
            f"(selection_score={winner.get('selection_score'):.4f})"
        )
    overall = payload.get("overall_stack", {})
    if overall:
        lines.extend(["", "## Overall Stack", ""])
        for key, value in overall.items():
            if isinstance(value, dict) and value.get("model_name"):
                lines.append(f"- {key}: `{value['model_name']}`")
            else:
                lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"
