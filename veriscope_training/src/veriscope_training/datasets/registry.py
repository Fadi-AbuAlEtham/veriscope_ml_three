from __future__ import annotations

from importlib import import_module
from typing import Type

from veriscope_training.config import AppConfig
from veriscope_training.datasets.base import DatasetAdapter


_DATASET_REGISTRY: dict[str, Type[DatasetAdapter]] = {}
_BUILTIN_MODULES = (
    "veriscope_training.datasets.phreshphish",
    "veriscope_training.datasets.openphish",
    "veriscope_training.datasets.phishtank",
    "veriscope_training.datasets.uci_phishing",
    "veriscope_training.datasets.mendeley",
    "veriscope_training.datasets.oscar_aux",
)
_BUILTINS_IMPORTED = False


def ensure_builtin_adapters_registered() -> None:
    global _BUILTINS_IMPORTED
    if _BUILTINS_IMPORTED:
        return
    for module_name in _BUILTIN_MODULES:
        import_module(module_name)
    _BUILTINS_IMPORTED = True


def register_dataset(name: str):
    def decorator(adapter_cls: Type[DatasetAdapter]) -> Type[DatasetAdapter]:
        if name in _DATASET_REGISTRY:
            raise ValueError(f"Dataset adapter '{name}' is already registered.")
        adapter_cls.dataset_name = name
        _DATASET_REGISTRY[name] = adapter_cls
        return adapter_cls

    return decorator


def get_registered_adapter(name: str) -> Type[DatasetAdapter] | None:
    ensure_builtin_adapters_registered()
    return _DATASET_REGISTRY.get(name)


def create_adapter(name: str, config: AppConfig) -> DatasetAdapter:
    adapter_cls = get_registered_adapter(name)
    if adapter_cls is None:
        available = ", ".join(sorted(_DATASET_REGISTRY)) or "none"
        raise KeyError(f"No dataset adapter registered for '{name}'. Registered adapters: {available}")
    return adapter_cls(config.get_source(name), config)


def registered_names() -> list[str]:
    ensure_builtin_adapters_registered()
    return sorted(_DATASET_REGISTRY)


def dataset_registry_summary(config: AppConfig) -> dict[str, object]:
    ensure_builtin_adapters_registered()
    configured_names = sorted(config.sources)
    rows = []
    for name in configured_names:
        source = config.get_source(name)
        adapter_cls = get_registered_adapter(name)
        rows.append(
            {
                "name": name,
                "enabled": source.enabled,
                "adapter_registered": name in _DATASET_REGISTRY,
                "adapter_class": (
                    f"{adapter_cls.__module__}.{adapter_cls.__name__}" if adapter_cls is not None else None
                ),
                "modalities": list(source.modalities),
                "snapshot_path": str(config.snapshot_path_for(name)),
                "access": source.access,
                "label_strategy": source.label_strategy,
                "supported_formats": list(getattr(adapter_cls, "supported_formats", ())),
            }
        )
    return {
        "configured_count": len(configured_names),
        "registered_count": len(_DATASET_REGISTRY),
        "registered_names": registered_names(),
        "datasets": rows,
    }
