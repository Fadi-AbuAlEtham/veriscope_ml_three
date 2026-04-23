"""Dataset abstractions and adapters for VeriScope training."""

from veriscope_training.datasets.base import DatasetAdapter, DatasetRecord
from veriscope_training.datasets.registry import (
    create_adapter,
    dataset_registry_summary,
    ensure_builtin_adapters_registered,
    get_registered_adapter,
    register_dataset,
)

ensure_builtin_adapters_registered()

__all__ = [
    "DatasetAdapter",
    "DatasetRecord",
    "create_adapter",
    "dataset_registry_summary",
    "ensure_builtin_adapters_registered",
    "get_registered_adapter",
    "register_dataset",
]
