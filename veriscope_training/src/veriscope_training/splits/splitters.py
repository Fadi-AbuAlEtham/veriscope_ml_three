from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from veriscope_training.utils.io import write_json, write_jsonl


VALID_SPLIT_NAMES = {"train", "validation", "test"}


@dataclass
class SplitResult:
    strategy: str
    seed: int
    train_indices: list[int]
    validation_indices: list[int]
    test_indices: list[int]
    counts: dict[str, int]
    label_counts: dict[str, dict[str, int]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_dataset_split(
    records: list[dict[str, Any]],
    *,
    strategy: str,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> SplitResult:
    if not records:
        raise ValueError("Cannot create a split for an empty dataset.")
    if strategy == "predefined_source":
        return _predefined_split(
            records,
            seed=seed,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
    if strategy == "time_aware":
        return _time_aware_split(records, seed=seed, validation_fraction=validation_fraction, test_fraction=test_fraction)
    if strategy == "source_aware":
        return _group_based_split(
            records,
            strategy="source_aware",
            seed=seed,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            grouping_field="source_dataset",
            group_getter=lambda row: row.get("source_dataset") or "unknown_source",
        )
    if strategy == "domain_aware":
        return _group_based_split(
            records,
            strategy="domain_aware",
            seed=seed,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            grouping_field="registered_domain_or_hostname",
            group_getter=lambda row: (
                (row.get("url_features") or {}).get("registered_domain")
                or (row.get("url_features") or {}).get("hostname")
                or row.get("source_dataset")
                or "unknown_domain"
            ),
        )
    if strategy == "random_stratified":
        return _random_split(records, seed=seed, validation_fraction=validation_fraction, test_fraction=test_fraction)
    raise KeyError(f"Unsupported split strategy '{strategy}'.")


def save_split_manifest(
    split: SplitResult,
    *,
    records: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "split_manifest.json"
    assignments_path = target / "split_assignments.jsonl"

    assignments = []
    for split_name, indices in (
        ("train", split.train_indices),
        ("validation", split.validation_indices),
        ("test", split.test_indices),
    ):
        for index in indices:
            record = records[index]
            assignments.append(
                {
                    "sample_id": record.get("sample_id"),
                    "split": split_name,
                    "source_dataset": record.get("source_dataset"),
                    "normalized_label": record.get("normalized_label"),
                }
            )

    write_json(manifest_path, split.to_dict())
    write_jsonl(assignments_path, assignments)
    return {"manifest": str(manifest_path), "assignments": str(assignments_path)}


def subset_by_indices(records: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    return [records[index] for index in indices]


def _random_split(
    records: list[dict[str, Any]],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> SplitResult:
    indices = np.arange(len(records))
    labels = [row.get("normalized_label") for row in records]
    stratify_labels = labels if _can_stratify(labels) else None
    train_val_indices, test_indices, stratified_test = _safe_train_test_split(
        indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=stratify_labels,
    )
    train_val_labels = [labels[index] for index in train_val_indices]
    val_size_relative = validation_fraction / max(1e-9, (1.0 - test_fraction))
    stratify_train_val = train_val_labels if _can_stratify(train_val_labels) else None
    train_indices, validation_indices, stratified_validation = _safe_train_test_split(
        train_val_indices,
        test_size=val_size_relative,
        random_state=seed,
        stratify=stratify_train_val,
    )
    return _make_split_result(
        strategy="random_stratified",
        seed=seed,
        train_indices=train_indices.tolist(),
        validation_indices=validation_indices.tolist(),
        test_indices=test_indices.tolist(),
        records=records,
        metadata={
            "stratified_test": stratified_test,
            "stratified_validation": stratified_validation,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
        },
    )


def _group_based_split(
    records: list[dict[str, Any]],
    *,
    strategy: str,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    grouping_field: str,
    group_getter,
) -> SplitResult:
    indices = np.arange(len(records))
    groups = np.array([group_getter(row) for row in records], dtype=object)
    if len(set(groups.tolist())) <= 1:
        result = _random_split(records, seed=seed, validation_fraction=validation_fraction, test_fraction=test_fraction)
        result.metadata.update(
            {
                "requested_strategy": strategy,
                "fallback_reason": "insufficient_distinct_groups",
            }
        )
        return result

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_positions, test_positions = next(splitter.split(indices, groups=groups))
    train_val_indices = indices[train_val_positions]
    test_indices = indices[test_positions]

    train_val_groups = groups[train_val_positions]
    val_size_relative = validation_fraction / max(1e-9, (1.0 - test_fraction))
    if len(set(train_val_groups.tolist())) <= 1:
        train_indices, validation_indices = train_test_split(
            train_val_indices,
            test_size=val_size_relative,
            random_state=seed,
        )
    else:
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_size_relative, random_state=seed)
        train_positions, validation_positions = next(val_splitter.split(train_val_indices, groups=train_val_groups))
        train_indices = train_val_indices[train_positions]
        validation_indices = train_val_indices[validation_positions]

    return _make_split_result(
        strategy=strategy,
        seed=seed,
        train_indices=train_indices.tolist(),
        validation_indices=validation_indices.tolist(),
        test_indices=test_indices.tolist(),
        records=records,
        metadata={
            "grouping_field": grouping_field,
            "group_count": len(set(groups.tolist())),
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
        },
    )


def _time_aware_split(
    records: list[dict[str, Any]],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> SplitResult:
    def sort_key(index: int) -> tuple[int, str]:
        timestamp = records[index].get("timestamp")
        normalized = _normalized_timestamp(timestamp)
        if normalized:
            return (0, normalized)
        return (1, str(records[index].get("sample_id")))

    ordered = sorted(range(len(records)), key=sort_key)
    total = len(ordered)
    test_count = max(1, int(round(total * test_fraction)))
    validation_count = max(1, int(round(total * validation_fraction)))
    if total <= test_count + validation_count:
        result = _random_split(records, seed=seed, validation_fraction=validation_fraction, test_fraction=test_fraction)
        result.metadata.update(
            {
                "requested_strategy": "time_aware",
                "fallback_reason": "dataset_too_small_for_temporal_holdout",
            }
        )
        return result
    train_count = total - validation_count - test_count
    train_indices = ordered[:train_count]
    validation_indices = ordered[train_count : train_count + validation_count]
    test_indices = ordered[train_count + validation_count :]
    return _make_split_result(
        strategy="time_aware",
        seed=seed,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        records=records,
        metadata={
            "ordered_by": "timestamp_then_sample_id",
            "timestamp_coverage": sum(1 for row in records if _normalized_timestamp(row.get("timestamp"))),
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
        },
    )


def _predefined_split(
    records: list[dict[str, Any]], *, seed: int, validation_fraction: float, test_fraction: float
) -> SplitResult:
    split_buckets = {"train": [], "validation": [], "test": []}
    passthrough_count = 0
    for index, row in enumerate(records):
        raw_split = (row.get("source_split") or "").strip().lower()
        if raw_split in {"val", "dev"}:
            raw_split = "validation"
        if raw_split in VALID_SPLIT_NAMES:
            split_buckets[raw_split].append(index)
            passthrough_count += 1
        else:
            split_buckets["train"].append(index)
    if passthrough_count == 0 or not split_buckets["test"]:
        result = _random_split(
            records,
            seed=seed,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        result.metadata.update(
            {
                "requested_strategy": "predefined_source",
                "fallback_reason": "missing_or_incomplete_source_splits",
                "passthrough_count": passthrough_count,
            }
        )
        return result
    return _make_split_result(
        strategy="predefined_source",
        seed=seed,
        train_indices=split_buckets["train"],
        validation_indices=split_buckets["validation"],
        test_indices=split_buckets["test"],
        records=records,
        metadata={
            "passthrough_count": passthrough_count,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
        },
    )


def _make_split_result(
    *,
    strategy: str,
    seed: int,
    train_indices: list[int],
    validation_indices: list[int],
    test_indices: list[int],
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> SplitResult:
    train_indices, validation_indices, test_indices = _ensure_non_empty_splits(
        train_indices,
        validation_indices,
        test_indices,
        total=len(records),
    )
    counts = {
        "train": len(train_indices),
        "validation": len(validation_indices),
        "test": len(test_indices),
    }
    label_counts = {
        "train": _label_counts(records, train_indices),
        "validation": _label_counts(records, validation_indices),
        "test": _label_counts(records, test_indices),
    }
    metadata = {
        **metadata,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return SplitResult(
        strategy=strategy,
        seed=seed,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        counts=counts,
        label_counts=label_counts,
        metadata=metadata,
    )


def _label_counts(records: list[dict[str, Any]], indices: list[int]) -> dict[str, int]:
    counts = {"phishing": 0, "benign": 0, "null": 0}
    for index in indices:
        label = records[index].get("normalized_label")
        if label == 1:
            counts["phishing"] += 1
        elif label == 0:
            counts["benign"] += 1
        else:
            counts["null"] += 1
    return counts


def _can_stratify(labels: list[Any]) -> bool:
    non_null = [label for label in labels if label in (0, 1)]
    return len(non_null) == len(labels) and len(set(non_null)) > 1


def _safe_train_test_split(
    indices: np.ndarray,
    *,
    test_size: float,
    random_state: int,
    stratify: list[Any] | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if stratify is not None:
        try:
            train_indices, test_indices = train_test_split(
                indices,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
            return np.asarray(train_indices), np.asarray(test_indices), True
        except ValueError:
            pass
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )
    return np.asarray(train_indices), np.asarray(test_indices), False


def _ensure_non_empty_splits(
    train_indices: list[int],
    validation_indices: list[int],
    test_indices: list[int],
    *,
    total: int,
) -> tuple[list[int], list[int], list[int]]:
    train = list(train_indices)
    validation = list(validation_indices)
    test = list(test_indices)
    if total < 3:
        return train, validation, test
    if not validation and len(train) > 1:
        validation.append(train.pop())
    if not test and len(train) > 1:
        test.append(train.pop())
    if not train:
        if validation:
            train.append(validation.pop(0))
        elif test:
            train.append(test.pop(0))
    return train, validation, test


def _normalized_timestamp(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).isoformat()
    except ValueError:
        return text
