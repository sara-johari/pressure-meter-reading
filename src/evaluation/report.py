from __future__ import annotations
import pandas as pd
from pathlib import Path

from .metrics import compute_metrics


def generate_report(merged_df: pd.DataFrame, output_path: str | Path) -> dict:
    """
    Write a per-image, per-scale evaluation report CSV and return
    a summary dict.

    Args:
        merged_df:   output of compare.load_and_merge()
        output_path: where to save the CSV report

    Returns:
        metrics dict with "overall", "outer", "inner" keys
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_cols = [
        "filename",
        "position",
        "true_value",
        "predicted_value",
        "unit",
        "true_unit",
        "true_scale_min",
        "true_scale_max",
        "abs_error",
        "pct_error",
        "within_5pct",
        "within_10pct",
        "confidence",
        "primary_scale",
        "prompt_version",
        "agent_notes",
        "gt_notes",
    ]

    cols = [c for c in report_cols if c in merged_df.columns]
    report_df = merged_df[cols].copy()

    for col in ["abs_error", "pct_error", "predicted_value", "true_value"]:
        if col in report_df.columns:
            report_df[col] = report_df[col].round(4)

    report_df.to_csv(output_path, index=False)

    metrics = compute_metrics(merged_df)
    return metrics


def print_summary(metrics: dict) -> None:
    """Print a human-readable summary with per-scale breakdowns."""
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)

    _print_section("OVERALL", metrics.get("overall", {}))

    for position in ["outer", "inner"]:
        section = metrics.get(position, {})
        if section.get("total_images", 0) > 0:
            _print_section(position.upper(), section)

    print("=" * 60 + "\n")


def _print_section(label: str, m: dict) -> None:
    print("\n  --- %s ---" % label)
    print("  Total readings:      %d" % m.get("total_images", 0))
    print("  Successful reads:    %d" % m.get("successful_reads", 0))
    print("  Failed reads:        %d" % m.get("failed_reads", 0))
    if m.get("mae") is not None:
        print("  MAE:                 %s" % m["mae"])
        mape_val = m.get("mape")
        if mape_val is not None:
            print("  MAPE:                %s%%" % mape_val)
        else:
            print("  MAPE:                N/A")
        w5 = m["within_5pct"]
        w5c = m["within_5pct_count"]
        print("  Within +/-5%%:       %s%%  (%d readings)" % (w5, w5c))
        w10 = m["within_10pct"]
        w10c = m["within_10pct_count"]
        print("  Within +/-10%%:      %s%%  (%d readings)" % (w10, w10c))
    else:
        print("  No valid predictions to evaluate.")
