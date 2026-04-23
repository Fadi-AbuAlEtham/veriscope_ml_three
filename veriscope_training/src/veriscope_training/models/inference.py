from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from veriscope_training.utils.serialization import load_joblib


def load_sklearn_bundle(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "artifacts" / "model_bundle.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Sklearn model bundle not found at {path}.")
    return load_joblib(path)


def predict_with_sklearn_bundle(bundle: dict[str, Any], features: Any) -> dict[str, Any]:
    estimator = bundle["estimator"]
    predictions = estimator.predict(features)
    scores = None
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(features)[:, 1]
    elif hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(features)
        scores = _sigmoid(decision)
    return {
        "predictions": predictions.tolist() if hasattr(predictions, "tolist") else list(predictions),
        "scores": scores.tolist() if hasattr(scores, "tolist") else (list(scores) if scores is not None else None),
    }


def _sigmoid(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-array))
