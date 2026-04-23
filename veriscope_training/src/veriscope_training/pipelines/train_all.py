from __future__ import annotations

from typing import Any

from veriscope_training.config import AppConfig


def list_available_models(config: AppConfig) -> dict[str, list[str]]:
    transformer_models = sorted(
        name for name, cfg in config.models_config.get("transformer_models", {}).items() if cfg.get("enabled", False)
    )
    return {
        "url": sorted(
            name for name, cfg in config.models_config.get("url_models", {}).items() if cfg.get("enabled", False)
        ),
        "webpage": sorted(
            name for name, cfg in config.models_config.get("webpage_models", {}).items() if cfg.get("enabled", False)
        ),
        "transformer": transformer_models,
        "transformer_aliases": ["xlmr", "mbert"],
        "tabular": sorted(
            name for name, cfg in config.models_config.get("tabular_models", {}).items() if cfg.get("enabled", False)
        ),
    }


def train_model(
    config: AppConfig,
    *,
    track: str,
    model_name: str,
    run_name: str | None = None,
    split_strategy: str | None = None,
) -> dict[str, Any]:
    transformer_names = set(config.models_config.get("transformer_models", {}))
    if track == "webpage" and model_name in transformer_names | {"xlmr", "mbert"}:
        from veriscope_training.models.train_transformer_text import train_transformer_text_model

        return train_transformer_text_model(
            config,
            model_name=model_name,
            run_name=run_name,
            split_strategy=split_strategy,
        )
    if track == "url":
        from veriscope_training.models.train_url_baselines import train_url_baseline

        return train_url_baseline(config, model_name=model_name, run_name=run_name, split_strategy=split_strategy)
    if track == "webpage":
        from veriscope_training.models.train_webpage_models import train_webpage_text_baseline

        return train_webpage_text_baseline(
            config,
            model_name=model_name,
            run_name=run_name,
            split_strategy=split_strategy,
        )
    if track == "transformer":
        from veriscope_training.models.train_transformer_text import train_transformer_text_model

        return train_transformer_text_model(
            config,
            model_name=model_name,
            run_name=run_name,
            split_strategy=split_strategy,
        )
    if track == "tabular":
        from veriscope_training.models.train_tabular_models import train_tabular_model

        return train_tabular_model(
            config,
            model_name=model_name,
            run_name=run_name,
            split_strategy=split_strategy,
        )
    raise KeyError(f"Unsupported track '{track}'.")


def train_all_enabled_models(
    config: AppConfig,
    *,
    include_transformers: bool = True,
    split_strategy: str | None = None,
) -> dict[str, Any]:
    available = list_available_models(config)
    runs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for track_name in ("url", "webpage", "transformer", "tabular"):
        model_names = available.get(track_name, [])
        if track_name == "transformer" and not include_transformers:
            continue
        actual_track = "webpage" if track_name == "webpage" else track_name
        for model_name in model_names:
            try:
                runs.append(
                    train_model(
                        config,
                        track=actual_track if track_name != "transformer" else "transformer",
                        model_name=model_name,
                        split_strategy=split_strategy,
                    )
                )
            except Exception as exc:
                errors.append({"track": track_name, "model_name": model_name, "error": str(exc)})
    return {"runs": runs, "errors": errors, "available_models": available}
