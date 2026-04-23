import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "src"))

from veriscope_training.fusion.evaluation import align_predictions, compute_fusion_metrics
from veriscope_training.fusion.weighted_fusion import apply_weighted_fusion
from veriscope_training.fusion.cascade_fusion import apply_cascade_fusion

url_preds = "outputs/training/url/tfidf_linear_svm/fr1-url-svm/splits/test_predictions.jsonl"
web_preds = "outputs/training/webpage_transformer/xlmr_sequence_classifier/fr1-xlmr/splits/test_predictions.jsonl"

df, stats = align_predictions(url_preds, web_preds)
print(f"Stats: {stats}")

weighted_scores = apply_weighted_fusion(df)
weighted_metrics = compute_fusion_metrics(df["normalized_label"], weighted_scores)
print(f"Weighted Metrics: {weighted_metrics}")

cascade_scores = apply_cascade_fusion(df)
cascade_metrics = compute_fusion_metrics(df["normalized_label"], cascade_scores)
print(f"Cascade Metrics: {cascade_metrics}")
