from __future__ import annotations

import pandas as pd

def apply_cascade_fusion(
    df: pd.DataFrame,
    url_high_threshold: float = 0.9,
    url_low_threshold: float = 0.1,
) -> pd.Series:
    """
    If url_score is very high/low, short-circuit.
    Else defer to webpage model (or an average).
    """
    def _fusion_logic(row):
        url_score = row["score_url"]
        web_score = row["score_webpage"]
        
        if url_score >= url_high_threshold:
            return url_score
        if url_score <= url_low_threshold:
            return url_score
            
        # Defer to webpage score in the middle
        return web_score

    return df.apply(_fusion_logic, axis=1)
