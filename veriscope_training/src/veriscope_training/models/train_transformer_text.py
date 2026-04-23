from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from veriscope_training.config import AppConfig
from veriscope_training.evaluation.metrics import compute_binary_classification_metrics
from veriscope_training.models.artifacts import (
    create_training_run_context,
    load_processed_view_records,
    save_config_snapshot,
    save_label_metadata,
    save_metrics,
    save_model_bundle,
    save_package_versions,
    save_predictions,
    save_run_metadata,
)
from veriscope_training.splits.splitters import create_dataset_split, save_split_manifest, subset_by_indices


TRANSFORMER_ALIASES = {
    "xlmr": "xlmr_sequence_classifier",
    "mbert": "mbert_sequence_classifier",
}


def train_transformer_text_model(
    config: AppConfig,
    *,
    model_name: str,
    run_name: str | None = None,
    split_strategy: str | None = None,
) -> dict[str, Any]:
    resolved_name = TRANSFORMER_ALIASES.get(model_name, model_name)
    model_config = config.models_config.get("transformer_models", {}).get(resolved_name)
    if not model_config or not model_config.get("enabled", False):
        raise KeyError(f"Transformer model '{model_name}' is not enabled in configs/models.yaml.")

    records = load_processed_view_records(config, "unified_webpage_dataset")
    excluded_counts = {
        "missing_binary_label": sum(1 for row in records if row.get("normalized_label") not in (0, 1)),
        "missing_usable_text": sum(1 for row in records if not _text_value(row)),
        "raw_html_only_excluded": sum(
            1
            for row in records
            if not _text_value(row) and row.get("raw_html")
        ),
    }
    supervised = [
        row
        for row in records
        if row.get("normalized_label") in (0, 1) and _text_value(row)
    ]
    if not supervised:
        raise RuntimeError("No supervised webpage text records were found for transformer training.")

    split = create_dataset_split(
        supervised,
        strategy=split_strategy or config.experiments_config.get("splits", {}).get("default_strategy", "domain_aware"),
        seed=int(config.experiments_config.get("reproducibility", {}).get("seed", 42)),
        validation_fraction=float(config.experiments_config.get("splits", {}).get("validation_fraction", 0.1)),
        test_fraction=float(config.experiments_config.get("splits", {}).get("test_fraction", 0.2)),
    )
    train_rows = subset_by_indices(supervised, split.train_indices)
    validation_rows = subset_by_indices(supervised, split.validation_indices)
    test_rows = subset_by_indices(supervised, split.test_indices)
    train_texts = [_text_value(row) for row in train_rows]
    val_texts = [_text_value(row) for row in validation_rows]
    test_texts = [_text_value(row) for row in test_rows]
    y_train = np.asarray([row["normalized_label"] for row in train_rows], dtype=int)
    y_val = np.asarray([row["normalized_label"] for row in validation_rows], dtype=int)
    y_test = np.asarray([row["normalized_label"] for row in test_rows], dtype=int)

    context = create_training_run_context(config, track="webpage_transformer", model_name=resolved_name, run_name=run_name)
    split_paths = save_split_manifest(split, records=supervised, output_dir=context.run_dir / "splits")

    training_mode = model_config.get("training_mode", "full_finetune")
    if training_mode == "frozen_encoder":
        result = _train_frozen_encoder(
            context=context,
            model_config=model_config,
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            train_texts=train_texts,
            val_texts=val_texts,
            test_texts=test_texts,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )
    else:
        result = _train_full_finetune(
            context=context,
            model_config=model_config,
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            train_texts=train_texts,
            val_texts=val_texts,
            test_texts=test_texts,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )

    if result.get("metrics", {}).get("skipped"):
        summary = {
            "track": "webpage_transformer",
            "model_name": resolved_name,
            "run_dir": str(context.run_dir),
            "gpu_unavailable": True,
            "metrics": result["metrics"],
            "status": "skipped",
        }
        save_run_metadata(context, payload=summary)
        return summary

    from veriscope_training.evaluation.plots import save_confusion_matrix_plot

    confusion_plot = save_confusion_matrix_plot(
        result["metrics"]["test"]["confusion_matrix"],
        context.reports_dir / "confusion_matrix_test.png",
        title=f"Transformer {resolved_name} Test Confusion Matrix",
    )
    result["metrics"]["confusion_matrix_plot"] = confusion_plot
    metrics_path = save_metrics(context, result["metrics"])
    label_metadata_path = save_label_metadata(context, {"label_map": {"benign": 0, "phishing": 1}})
    config_snapshot_path = save_config_snapshot(
        context,
        {
            "track": "webpage_transformer",
            "model_name": resolved_name,
            "model_config": model_config,
            "split": split.to_dict(),
            "server_side_inference_only": True,
            "text_field_priority": ["normalized_text", "extracted_text"],
            "raw_html_as_training_text": False,
            "excluded_counts": excluded_counts,
            "gpu_unavailable": result.get("gpu_unavailable", False),
        },
    )
    versions_path = save_package_versions(
        context,
        ["numpy", "scikit-learn", "transformers", "datasets", "torch", "accelerate"],
    )
    summary = {
        "track": "webpage_transformer",
        "model_name": resolved_name,
        "run_dir": str(context.run_dir),
        "artifact_paths": {
            **result["artifact_paths"],
            "metrics": metrics_path,
            "label_metadata": label_metadata_path,
            "config_snapshot": config_snapshot_path,
            "split_manifest": split_paths["manifest"],
            "split_assignments": split_paths["assignments"],
            "package_versions": versions_path,
            "confusion_matrix_plot": confusion_plot,
        },
        "metrics": result["metrics"],
        "training_mode": training_mode,
        "server_side_inference_only": True,
        "excluded_counts": excluded_counts,
    }
    save_run_metadata(context, payload=summary)
    return summary


def _train_full_finetune(
    *,
    context,
    model_config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    try:
        import accelerate  # noqa: F401
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Transformer fine-tuning requires transformers, datasets, torch, and accelerate. "
            "Install veriscope-training[transformers] or switch the model to training_mode=frozen_encoder."
        ) from exc

    model_id = model_config.get("model_name", "xlm-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenized_datasets = {}
    for split_name, texts, labels, rows in (
        ("train", train_texts, y_train.tolist(), train_rows),
        ("validation", val_texts, y_val.tolist(), validation_rows),
        ("test", test_texts, y_test.tolist(), test_rows),
    ):
        dataset = Dataset.from_dict(
            {
                "text": texts,
                "label": labels,
                "sample_id": [row.get("sample_id") for row in rows],
            }
        )
        tokenized = dataset.map(
            lambda batch: tokenizer(
                batch["text"],
                truncation=True,
                max_length=int(model_config.get("max_length", 256)),
            ),
            batched=True,
        )
        tokenized_datasets[split_name] = tokenized

    id2label = {0: "benign", 1: "phishing"}
    label2id = {"benign": 0, "phishing": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    output_dir = context.artifact_dir / "hf_model"
    if model_config.get("force_gpu", False) and not torch.cuda.is_available():
        print("WARNING: force_gpu=true but CUDA is unavailable. Skipping full fine-tuning.")
        return {
            "metrics": {"skipped": True},
            "gpu_unavailable": True,
            "artifact_paths": {},
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(model_config.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(model_config.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(model_config.get("per_device_eval_batch_size", 8)),
        num_train_epochs=float(model_config.get("num_train_epochs", 2)),
        weight_decay=float(model_config.get("weight_decay", 0.01)),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=True,
        report_to=[],
        fp16=bool(model_config.get("fp16", False)),
        gradient_accumulation_steps=int(model_config.get("gradient_accumulation_steps", 1)),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir / "tokenizer")

    metrics = {}
    prediction_paths = {}
    for split_name, dataset_rows, dataset in (
        ("train", train_rows, tokenized_datasets["train"]),
        ("validation", validation_rows, tokenized_datasets["validation"]),
        ("test", test_rows, tokenized_datasets["test"]),
    ):
        predictions = trainer.predict(dataset)
        logits = predictions.predictions
        probs = _softmax(logits)[:, 1]
        pred_labels = logits.argmax(axis=1)
        true_labels = dataset["label"]
        metrics[split_name] = compute_binary_classification_metrics(true_labels, pred_labels.tolist(), y_score=probs.tolist())
        prediction_rows = []
        for row, true_label, pred_label, score in zip(dataset_rows, true_labels, pred_labels.tolist(), probs.tolist()):
            prediction_rows.append(
                {
                    "sample_id": row.get("sample_id"),
                    "source_dataset": row.get("source_dataset"),
                    "normalized_label": true_label,
                    "predicted_label": int(pred_label),
                    "score": float(score),
                    "model_name": model_id,
                }
            )
        prediction_paths[split_name] = save_predictions(context, split_name, prediction_rows)
    return {
        "metrics": metrics,
        "artifact_paths": {
            "hf_model": str(output_dir),
            "tokenizer": str(output_dir / "tokenizer"),
            "train_predictions": prediction_paths["train"],
            "validation_predictions": prediction_paths["validation"],
            "test_predictions": prediction_paths["test"],
        },
    }


def _train_frozen_encoder(
    *,
    context,
    model_config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Frozen encoder mode requires transformers and torch. "
            "Install veriscope-training[transformers]."
        ) from exc

    model_id = model_config.get("model_name", "xlm-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoder = AutoModel.from_pretrained(model_id)
    device = "cpu" if model_config.get("force_cpu", False) or not torch.cuda.is_available() else "cuda"
    encoder.to(device)
    encoder.eval()

    train_embeddings = _encode_texts(encoder, tokenizer, train_texts, device=device, max_length=int(model_config.get("max_length", 256)), batch_size=int(model_config.get("per_device_eval_batch_size", 8)))
    val_embeddings = _encode_texts(encoder, tokenizer, val_texts, device=device, max_length=int(model_config.get("max_length", 256)), batch_size=int(model_config.get("per_device_eval_batch_size", 8)))
    test_embeddings = _encode_texts(encoder, tokenizer, test_texts, device=device, max_length=int(model_config.get("max_length", 256)), batch_size=int(model_config.get("per_device_eval_batch_size", 8)))

    classifier = LogisticRegression(
        max_iter=int(model_config.get("classifier_max_iter", 2000)),
        class_weight="balanced",
        solver="liblinear",
    )
    classifier.fit(train_embeddings, y_train)

    encoder_dir = context.artifact_dir / "encoder"
    encoder.save_pretrained(encoder_dir)
    tokenizer.save_pretrained(context.artifact_dir / "tokenizer")

    metrics = {}
    prediction_paths = {}
    for split_name, rows, embeddings, y_true in (
        ("train", train_rows, train_embeddings, y_train),
        ("validation", validation_rows, val_embeddings, y_val),
        ("test", test_rows, test_embeddings, y_test),
    ):
        pred_labels = classifier.predict(embeddings)
        probs = classifier.predict_proba(embeddings)[:, 1]
        metrics[split_name] = compute_binary_classification_metrics(y_true.tolist(), pred_labels.tolist(), y_score=probs.tolist())
        prediction_rows = []
        for row, true_label, pred_label, score in zip(rows, y_true.tolist(), pred_labels.tolist(), probs.tolist()):
            prediction_rows.append(
                {
                    "sample_id": row.get("sample_id"),
                    "source_dataset": row.get("source_dataset"),
                    "normalized_label": true_label,
                    "predicted_label": int(pred_label),
                    "score": float(score),
                    "model_name": model_id,
                }
            )
        prediction_paths[split_name] = save_predictions(context, split_name, prediction_rows)

    classifier_path = save_model_bundle(
        context,
        "model_bundle",
        {"estimator": classifier, "track": "webpage_transformer", "model_name": model_id, "training_mode": "frozen_encoder"},
    )
    return {
        "metrics": metrics,
        "artifact_paths": {
            "encoder": str(encoder_dir),
            "tokenizer": str(context.artifact_dir / "tokenizer"),
            "classifier_bundle": classifier_path,
            "train_predictions": prediction_paths["train"],
            "validation_predictions": prediction_paths["validation"],
            "test_predictions": prediction_paths["test"],
        },
    }


def _encode_texts(encoder, tokenizer, texts: list[str], *, device: str, max_length: int, batch_size: int) -> np.ndarray:
    import torch

    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = encoder(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vectors.append(pooled.cpu().numpy())
    return np.vstack(vectors)


def _text_value(row: dict[str, Any]) -> str | None:
    return row.get("normalized_text") or row.get("extracted_text")


def _softmax(logits: Any) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)
