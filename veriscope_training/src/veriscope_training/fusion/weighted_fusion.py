from __future__ import annotations

import pandas as pd

def apply_weighted_fusion(
    df: pd.DataFrame,
    url_weight: float = 0.4,
    webpage_weight: float = 0.6,
) -> pd.Series:
    """
    Computes weighted fusion score: a * url_score + b * webpage_score
    """
    return (df["score_url"] * url_weight) + (df["score_webpage"] * webpage_weight)
