from __future__ import annotations

import hashlib
import heapq
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from veriscope_training.utils.io import StructuredDatasetWriter, read_records_file, write_json


LANGUAGE_ALIASES = {
    "ara": "ar",
    "arabic": "ar",
    "eng": "en",
    "english": "en",
    "fra": "fr",
    "fre": "fr",
    "french": "fr",
}
TRACKED_LANGUAGES = ("en", "ar", "fr")


def rebalance_multilingual_view(
    *,
    view_name: str,
    staged_records_path: Path,
    final_base_path: Path,
    output_format: str,
    parquet_batch_size: int,
    parquet_compression: str,
    balance_config: dict[str, Any],
) -> dict[str, Any]:
    pre_stats = compute_dataset_statistics([staged_records_path], view_name=view_name)
    selection_plan = _build_selection_plan(pre_stats=pre_stats, balance_config=balance_config)

    english_targets = selection_plan["english_unique_targets"]
    english_selected_ids = _select_english_sample_ids(
        staged_records_path=staged_records_path,
        english_targets=english_targets,
    )

    oversample_cfg = selection_plan["oversample_targets"]
    oversample_pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    writer = StructuredDatasetWriter(
        final_base_path,
        output_format=output_format,
        parquet_batch_size=parquet_batch_size,
        parquet_compression=parquet_compression,
    )
    final_stats = _empty_stats(view_name)
    selected_counts = Counter()

    try:
        for row in read_records_file(staged_records_path):
            language = canonical_language(row.get("language"))
            label_bucket = label_bucket_for_row(row)
            keep = True
            if language == "en":
                keep = row.get("sample_id") in english_selected_ids
            if not keep:
                continue
            writer.write(row)
            _update_stats(final_stats, row, tracked_languages=TRACKED_LANGUAGES)
            selected_counts[(language, label_bucket)] += 1
            if (language, label_bucket) in oversample_cfg:
                oversample_pools[(language, label_bucket)].append(row)

        resampled_counts = Counter()
        for bucket, target_count in oversample_cfg.items():
            current = selected_counts[bucket]
            if current >= target_count:
                continue
            pool = oversample_pools.get(bucket) or []
            if not pool:
                continue
            deficit = target_count - current
            for index in range(deficit):
                template = pool[index % len(pool)]
                duplicated = _duplicate_row(
                    template,
                    duplicate_index=index + 1,
                    reason="priority_language_oversample",
                )
                writer.write(duplicated)
                _update_stats(final_stats, duplicated, tracked_languages=TRACKED_LANGUAGES)
                resampled_counts[bucket] += 1
    finally:
        writer.close()

    post_stats = compute_dataset_statistics([Path(writer.output_paths["parquet"])], view_name=view_name)
    summary = {
        "view_name": view_name,
        "staged_records_path": str(staged_records_path),
        "final_output_paths": writer.output_paths,
        "selection_plan": selection_plan,
        "pre_balance": pre_stats,
        "post_balance": post_stats,
        "resampled_counts": {
            f"{language}:{label}": count
            for (language, label), count in sorted(resampled_counts.items())
        },
    }
    return summary


def compute_dataset_statistics(
    record_paths: list[Path],
    *,
    view_name: str,
) -> dict[str, Any]:
    stats = _empty_stats(view_name)
    for path in record_paths:
        for row in read_records_file(path):
            _update_stats(stats, row, tracked_languages=TRACKED_LANGUAGES)
    return _finalize_stats(stats)


def write_multilingual_report(
    *,
    report_base_path: Path,
    selected_sources: list[str],
    configured_sources: list[str],
    locally_available_sources: list[str],
    source_summaries: dict[str, Any],
    exclusion_summary: dict[str, Any],
    view_reports: dict[str, Any],
) -> dict[str, str]:
    report_payload = {
        "selected_sources": selected_sources,
        "configured_sources": configured_sources,
        "locally_available_sources": locally_available_sources,
        "source_summaries": source_summaries,
        "exclusion_summary": exclusion_summary,
        "views": view_reports,
    }
    json_path = write_json(report_base_path.with_suffix(".json"), report_payload)
    markdown_path = _write_markdown_report(report_base_path.with_suffix(".md"), report_payload)
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }


def canonical_language(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if not text or text in {"none", "null", "nan", "unknown", "und"}:
        return "unknown"
    text = text.replace("_", "-")
    text = LANGUAGE_ALIASES.get(text, text)
    base = text.split("-", 1)[0]
    return LANGUAGE_ALIASES.get(base, base)


def label_bucket_for_row(row: dict[str, Any]) -> str:
    label = row.get("normalized_label")
    if label == 1:
        return "phishing"
    if label == 0:
        return "benign"
    return "null"


def _empty_stats(view_name: str) -> dict[str, Any]:
    return {
        "view_name": view_name,
        "total_sample_count": 0,
        "counts_by_dataset": Counter(),
        "counts_by_language": Counter(),
        "counts_by_label": Counter(),
        "counts_by_dataset_x_language": defaultdict(Counter),
        "tracked_language_counts": {language: Counter() for language in TRACKED_LANGUAGES},
        "modality_counts": Counter(),
        "validation_warning_count": 0,
    }


def _update_stats(stats: dict[str, Any], row: dict[str, Any], *, tracked_languages: tuple[str, ...]) -> None:
    language = canonical_language(row.get("language"))
    source = str(row.get("source_dataset") or "unknown_source")
    label_bucket = label_bucket_for_row(row)
    stats["total_sample_count"] += 1
    stats["counts_by_dataset"][source] += 1
    stats["counts_by_language"][language] += 1
    stats["counts_by_label"][label_bucket] += 1
    stats["counts_by_dataset_x_language"][source][language] += 1
    if language in tracked_languages:
        stats["tracked_language_counts"][language][label_bucket] += 1
    for key, value in (row.get("modality_flags") or {}).items():
        if value:
            stats["modality_counts"][str(key)] += 1
    stats["validation_warning_count"] += len((row.get("processing_metadata") or {}).get("validation_warnings", []))


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "view_name": stats["view_name"],
        "total_sample_count": int(stats["total_sample_count"]),
        "counts_by_dataset": dict(sorted(stats["counts_by_dataset"].items())),
        "counts_by_language": dict(stats["counts_by_language"].most_common()),
        "counts_by_label": dict(sorted(stats["counts_by_label"].items())),
        "counts_by_dataset_x_language": {
            dataset: dict(counter.most_common())
            for dataset, counter in sorted(stats["counts_by_dataset_x_language"].items())
        },
        "tracked_language_counts": {
            language: dict(sorted(counter.items()))
            for language, counter in stats["tracked_language_counts"].items()
        },
        "modality_counts": dict(sorted(stats["modality_counts"].items())),
        "validation_warning_count": int(stats["validation_warning_count"]),
    }


def _build_selection_plan(
    *,
    pre_stats: dict[str, Any],
    balance_config: dict[str, Any],
) -> dict[str, Any]:
    counts_by_language = Counter(pre_stats["counts_by_language"])
    tracked_language_counts = {
        language: Counter(pre_stats["tracked_language_counts"].get(language, {}))
        for language in TRACKED_LANGUAGES
    }
    english_total = counts_by_language.get("en", 0)
    non_english_total = max(sum(counts_by_language.values()) - english_total, 0)
    max_english_share = float(balance_config.get("max_english_share", 0.6))
    min_english_records = int(balance_config.get("min_english_records", 50000))

    english_cap = english_total
    if english_total and non_english_total:
        english_cap = min(
            english_total,
            max(
                min_english_records,
                int(math.floor(non_english_total * max_english_share / max(1e-6, 1.0 - max_english_share))),
            ),
        )
    english_cap = min(english_cap, english_total)

    english_label_counts = tracked_language_counts.get("en", Counter())
    english_unique_targets = _allocate_targets(english_label_counts, english_cap)

    oversample_targets: dict[tuple[str, str], int] = {}
    oversample_cfg = balance_config.get("oversample_language_label_minimums", {}) or {}
    for language, label_targets in oversample_cfg.items():
        canonical = canonical_language(language)
        available = tracked_language_counts.get(canonical, Counter())
        for label_name, target in (label_targets or {}).items():
            label_bucket = str(label_name)
            if available.get(label_bucket, 0) > 0:
                oversample_targets[(canonical, label_bucket)] = max(int(target), available.get(label_bucket, 0))

    return {
        "max_english_share": max_english_share,
        "min_english_records": min_english_records,
        "english_total_before": english_total,
        "non_english_total_before": non_english_total,
        "english_total_after_target": english_cap,
        "english_unique_targets": {
            label: count for (_, label), count in sorted(english_unique_targets.items())
        },
        "oversample_targets": {
            f"{language}:{label}": count
            for (language, label), count in sorted(oversample_targets.items())
        },
        "tracked_languages_before": {
            language: dict(sorted(counter.items()))
            for language, counter in tracked_language_counts.items()
        },
    }


def _allocate_targets(counts: Counter, total_target: int) -> dict[tuple[str, str], int]:
    total_available = sum(counts.values())
    if not total_available or total_target >= total_available:
        return {
            ("en", label): int(count)
            for label, count in counts.items()
        }

    allocated: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, str]] = []
    running_total = 0
    for label, count in sorted(counts.items()):
        exact = (count / total_available) * total_target
        base = min(count, int(math.floor(exact)))
        allocated[("en", label)] = base
        running_total += base
        remainders.append((exact - base, label))

    remaining = total_target - running_total
    for _, label in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        key = ("en", label)
        if allocated[key] >= counts[label]:
            continue
        allocated[key] += 1
        remaining -= 1
    return allocated


def _select_english_sample_ids(
    *,
    staged_records_path: Path,
    english_targets: dict[str, int],
) -> set[str]:
    if not english_targets:
        return set()

    heaps: dict[str, list[tuple[int, str]]] = {label: [] for label, target in english_targets.items() if target > 0}
    for row in read_records_file(staged_records_path):
        if canonical_language(row.get("language")) != "en":
            continue
        label_bucket = label_bucket_for_row(row)
        target = english_targets.get(label_bucket, 0)
        if target <= 0:
            continue
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            continue
        rank = _stable_rank(sample_id)
        heap = heaps[label_bucket]
        item = (-rank, sample_id)
        if len(heap) < target:
            heapq.heappush(heap, item)
            continue
        if item > heap[0]:
            heapq.heapreplace(heap, item)

    selected: set[str] = set()
    for heap in heaps.values():
        for _, sample_id in heap:
            selected.add(sample_id)
    return selected


def _stable_rank(sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _duplicate_row(row: dict[str, Any], *, duplicate_index: int, reason: str) -> dict[str, Any]:
    payload = dict(row)
    original_sample_id = str(row.get("sample_id") or "")
    payload["sample_id"] = f"{original_sample_id}::dup::{duplicate_index}"
    processing_metadata = dict(payload.get("processing_metadata") or {})
    processing_metadata["resampled"] = True
    processing_metadata["resampled_from_sample_id"] = original_sample_id
    processing_metadata["resampling_reason"] = reason
    processing_metadata["resampling_index"] = duplicate_index
    payload["processing_metadata"] = processing_metadata
    return payload


def _write_markdown_report(path: Path, payload: dict[str, Any]) -> Path:
    lines = [
        "# Multilingual Dataset Report",
        "",
        f"- Selected sources: {', '.join(payload['selected_sources']) or 'none'}",
        f"- Configured sources: {', '.join(payload['configured_sources']) or 'none'}",
        f"- Locally available sources: {', '.join(payload['locally_available_sources']) or 'none'}",
        "",
        "## Source exclusions",
        "",
    ]
    for source, summary in sorted(payload["source_summaries"].items()):
        excluded = summary.get("excluded_counts") or {}
        if excluded:
            excluded_text = ", ".join(f"{reason}={count}" for reason, count in sorted(excluded.items()))
        else:
            excluded_text = "none"
        lines.append(f"- {source}: {excluded_text}")
    for view_name, report in sorted(payload["views"].items()):
        lines.extend(
            [
                "",
                f"## {view_name}",
                "",
                f"- Total before: {report['pre_balance']['total_sample_count']}",
                f"- Total after: {report['post_balance']['total_sample_count']}",
                f"- English/Arabic/French before: "
                f"{_tracked_counts_text(report['pre_balance'])}",
                f"- English/Arabic/French after: "
                f"{_tracked_counts_text(report['post_balance'])}",
            ]
        )
    markdown = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def _tracked_counts_text(report: dict[str, Any]) -> str:
    parts = []
    for language in TRACKED_LANGUAGES:
        counts = report.get("tracked_language_counts", {}).get(language, {})
        total = sum(int(value) for value in counts.values())
        parts.append(f"{language}={total}")
    return ", ".join(parts)
