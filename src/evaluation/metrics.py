from __future__ import annotations
import numpy as np
import pandas as pd


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute evaluation metrics from a DataFrame that has "position",
    "true_value", and "predicted_value" columns.

    Returns per-scale metrics (outer, inner) AND overall combined metrics.

    Args:
        df: DataFrame with at least position, true_value, predicted_value

    Returns:
        dict with keys:
            overall: combined metrics across all scales
            outer:   metrics for outer scale rows only
            inner:   metrics for inner scale rows only
        Each sub-dict contains:
            total_images, successful_reads, failed_reads,
            mae, mape, within_5pct, within_10pct,
            within_5pct_count, within_10pct_count
    """
    result = {}

    result["overall"] = _compute_single_metrics(df)

    for position in ["outer", "inner"]:
        subset = df[df["position"] == position]
        result[position] = _compute_single_metrics(subset)

    return result


def _compute_single_metrics(df: pd.DataFrame) -> dict:
    """Compute metrics for a single subset of data."""
    total = len(df)
    valid = df.dropna(subset=["predicted_value", "true_value"])
    failed = total - len(valid)

    if len(valid) == 0:
        return _empty_metrics(total, failed)

    abs_errors = (valid["predicted_value"] - valid["true_value"]).abs()

    nonzero = valid[valid["true_value"] != 0]
    mape = (
        ((nonzero["predicted_value"] - nonzero["true_value"]).abs()
         / nonzero["true_value"].abs())
        .mean() * 100
        if len(nonzero) > 0 else None
    )

    pct_errors = (abs_errors / valid["true_value"].abs().replace(0, np.nan)) * 100

    within_5 = (pct_errors <= 5).sum()
    within_10 = (pct_errors <= 10).sum()

    return {
        "total_images": total,
        "successful_reads": len(valid),
        "failed_reads": failed,
        "mae": round(abs_errors.mean(), 4),
        "mape": round(mape, 2) if mape is not None else None,
        "within_5pct": round(within_5 / len(valid) * 100, 1),
        "within_10pct": round(within_10 / len(valid) * 100, 1),
        "within_5pct_count": int(within_5),
        "within_10pct_count": int(within_10),
    }


def _empty_metrics(total: int, failed: int) -> dict:
    return {
        "total_images": total,
        "successful_reads": 0,
        "failed_reads": failed,
        "mae": None,
        "mape": None,
        "within_5pct": None,
        "within_10pct": None,
        "within_5pct_count": 0,
        "within_10pct_count": 0,
    }
