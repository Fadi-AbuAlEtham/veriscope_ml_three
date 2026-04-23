from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def predict_single(
    run_dir: str | Path,
    text: str,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    model_dir = run_path / "artifacts" / "hf_model"
    if not model_dir.exists():
        model_dir = run_path / "artifacts" / "encoder"
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found in {run_path / 'artifacts'}")

    tokenizer_dir = run_path / "artifacts" / "tokenizer"
    if not tokenizer_dir.exists():
        tokenizer_dir = model_dir / "tokenizer"
    
    if not tokenizer_dir.exists():
        tokenizer_dir = model_dir # Fallback to same dir

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        score = probs[0][1].item()
        pred_label = torch.argmax(logits, dim=1).item()

    id2label = model.config.id2label
    label = id2label.get(pred_label, str(pred_label))

    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "score": round(score, 6),
        "label": label,
        "label_id": pred_label,
        "device": device,
    }
