from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DeduplicationEvent:
    removed_sample_id: str
    kept_sample_id: str
    removed_source: str
    kept_source: str
    dedupe_scope: str
    matched_on: list[str]
    key_values: dict[str, str]
    candidate_signatures: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DedupeSummary:
    enabled: bool
    policy: str
    kept_count: int = 0
    removed_count: int = 0
    match_counts: dict[str, int] = field(default_factory=dict)
    scope_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProcessedDeduplicator:
    """First-seen deterministic deduplication using exact canonical keys."""

    def __init__(self, *, enabled: bool = True, policy: str = "first_seen") -> None:
        self.enabled = enabled
        self.policy = policy
        self.summary = DedupeSummary(enabled=enabled, policy=policy)
        self._seen: dict[str, dict[str, tuple[str, str]]] = {}

    def process(self, record: Any) -> tuple[bool, DeduplicationEvent | None]:
        if not self.enabled:
            self.summary.kept_count += 1
            return True, None

        scope = self._scope(record)
        scope_index = self._seen.setdefault(scope, {})
        key_pairs = _dedupe_key_pairs(record)
        matches: dict[str, tuple[str, str]] = {}
        for key_type, key_value in key_pairs.items():
            namespaced_key = f"{key_type}::{key_value}"
            if namespaced_key in scope_index:
                matches[key_type] = scope_index[namespaced_key]

        if matches:
            kept_sample_id, kept_source = sorted(matches.values(), key=lambda item: item[0])[0]
            event = DeduplicationEvent(
                removed_sample_id=record.sample_id,
                kept_sample_id=kept_sample_id,
                removed_source=record.source_dataset,
                kept_source=kept_source,
                dedupe_scope=scope,
                matched_on=sorted(matches),
                key_values={key: value for key, value in key_pairs.items() if key in matches},
                candidate_signatures={
                    "registered_domain": record.url_features.get("registered_domain"),
                    "normalized_text_prefix_hash": record.processing_metadata.get("text_prefix_hash"),
                },
            )
            self.summary.removed_count += 1
            self.summary.scope_counts[scope] = self.summary.scope_counts.get(scope, 0) + 1
            for key_type in matches:
                self.summary.match_counts[key_type] = self.summary.match_counts.get(key_type, 0) + 1
            return False, event

        for key_type, key_value in key_pairs.items():
            scope_index[f"{key_type}::{key_value}"] = (record.sample_id, record.source_dataset)
        self.summary.kept_count += 1
        return True, None

    def report(self) -> DedupeSummary:
        return self.summary

    def _scope(self, record: Any) -> str:
        if record.is_auxiliary:
            return "auxiliary"
        if record.modality_flags.get("has_tabular_features") and not record.modality_flags.get("has_original_url"):
            return "tabular"
        return "supervised"


def _dedupe_key_pairs(record: Any) -> dict[str, str]:
    pairs: dict[str, str] = {}
    if record.source_original_id:
        pairs["source_original_id"] = f"{record.source_dataset}:{record.source_original_id}"
    if record.original_url:
        pairs["original_url"] = record.original_url
    if record.normalized_url:
        pairs["normalized_url"] = record.normalized_url
    html_hash = record.processing_metadata.get("raw_html_hash")
    if html_hash:
        pairs["raw_html_hash"] = html_hash
    text_hash = record.processing_metadata.get("normalized_text_hash")
    if text_hash:
        pairs["normalized_text_hash"] = text_hash
    return pairs
