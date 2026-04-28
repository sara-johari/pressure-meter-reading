from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path


def load_and_merge(
    predictions_path: str | Path,
    ground_truth_path: str | Path,
) -> pd.DataFrame:
    """
    Load predictions and ground truth CSVs and join them on
    (filename, position).

    The ground truth CSV uses wide dual-scale columns:
        true_value_outer, true_value_inner, unit_outer, unit_inner, ...
    These are melted into long format with a "position" column before
    merging.

    Args:
        predictions_path: path to outputs/predictions.csv
        ground_truth_path: path to data/ground_truth.csv

    Returns:
        Merged DataFrame with per-row error columns added
    """
    preds = pd.read_csv(predictions_path)
    truth = pd.read_csv(ground_truth_path)

    truth_long = _melt_ground_truth(truth)

    preds["_key"] = preds["filename"].apply(lambda p: Path(p).name)
    truth_long["_key"] = truth_long["filename"].apply(lambda p: Path(p).name)
    # Drop filename from truth_long to avoid suffix collision (we match on _key)
    truth_long = truth_long.drop(columns=["filename"])

    merged = pd.merge(
        preds,
        truth_long,
        on=["_key", "position"],
        how="left",
        suffixes=("_pred", "_truth"),
    ).drop(columns=["_key"])

    # Rename to avoid ambiguity where both sides have same column
    rename_map = {}
    # Truth notes column (no collision since preds renamed notes->agent_notes)
    if "notes" in merged.columns:
        rename_map["notes"] = "gt_notes"
    if "notes_truth" in merged.columns:
        rename_map["notes_truth"] = "gt_notes"
    if "notes_pred" in merged.columns:
        rename_map["notes_pred"] = "agent_notes"
    if "unit_truth" in merged.columns:
        rename_map["unit_truth"] = "true_unit"
    if "scale_min_truth" in merged.columns:
        rename_map["scale_min_truth"] = "true_scale_min"
    if "scale_max_truth" in merged.columns:
        rename_map["scale_max_truth"] = "true_scale_max"
    merged = merged.rename(columns=rename_map)

    merged["predicted_value"] = pd.to_numeric(merged["predicted_value"], errors="coerce")
    merged["true_value"] = pd.to_numeric(merged["true_value"], errors="coerce")

    merged["abs_error"] = (merged["predicted_value"] - merged["true_value"]).abs()
    merged["pct_error"] = np.where(
        merged["true_value"] != 0,
        merged["abs_error"] / merged["true_value"].abs() * 100,
        np.nan,
    )
    merged["within_5pct"] = merged["pct_error"] <= 5
    merged["within_10pct"] = merged["pct_error"] <= 10

    return merged


def _melt_ground_truth(truth: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape wide dual-scale ground truth into long format.

    Input columns:
        filename, true_value_outer, unit_outer, scale_min_outer,
        scale_max_outer, true_value_inner, unit_inner,
        scale_min_inner, scale_max_inner, notes

    Output columns:
        filename, position, true_value, unit, scale_min, scale_max, notes
    """
    rows = []
    for _, row in truth.iterrows():
        filename = row["filename"]
        notes = row.get("notes", "")

        # Always add outer scale row
        rows.append({
            "filename": filename,
            "position": "outer",
            "true_value": _to_float_or_nan(row.get("true_value_outer")),
            "unit": row.get("unit_outer"),
            "scale_min": _to_float_or_nan(row.get("scale_min_outer")),
            "scale_max": _to_float_or_nan(row.get("scale_max_outer")),
            "notes": notes,
        })

        # Add inner scale row only if present (not nil/NaN/empty)
        inner_val = row.get("true_value_inner")
        if pd.notna(inner_val) and str(inner_val).strip().lower() not in ("nil", "", "none"):
            rows.append({
                "filename": filename,
                "position": "inner",
                "true_value": _to_float_or_nan(inner_val),
                "unit": row.get("unit_inner"),
                "scale_min": _to_float_or_nan(row.get("scale_min_inner")),
                "scale_max": _to_float_or_nan(row.get("scale_max_inner")),
                "notes": notes,
            })

    return pd.DataFrame(rows)


def _to_float_or_nan(val):
    try:
        return float(val) if pd.notna(val) else np.nan
    except (TypeError, ValueError):
        return np.nan
