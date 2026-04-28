"""
run_evaluation.py
-----------------
Compare VLM predictions against ground truth and produce a report.

Supports dual-scale predictions: merges on (filename, position) and
computes per-scale metrics alongside overall metrics.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --predictions outputs/predictions.csv
    python scripts/run_evaluation.py --truth data/ground_truth.csv --output outputs/my_report.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.compare import load_and_merge
from src.evaluation.report import generate_report, print_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent predictions vs ground truth.")
    parser.add_argument(
        "--predictions", "-p",
        default="outputs/predictions.csv",
        help="Path to predictions CSV. Default: outputs/predictions.csv",
    )
    parser.add_argument(
        "--truth", "-t",
        default="data/ground_truth.csv",
        help="Path to ground truth CSV. Default: data/ground_truth.csv",
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/evaluation_report.csv",
        help="Path for the output report CSV. Default: outputs/evaluation_report.csv",
    )
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    truth_path = Path(args.truth)
    output_path = Path(args.output)

    if not predictions_path.exists():
        print(f"[ERROR] Predictions file not found: {predictions_path}")
        print("  Run scripts/run_agent.py first.")
        sys.exit(1)

    if not truth_path.exists():
        print(f"[ERROR] Ground truth file not found: {truth_path}")
        print("  Expected columns: filename, true_value_outer, unit_outer, ..., true_value_inner, ...")
        sys.exit(1)

    print(f"Loading predictions from: {predictions_path}")
    print(f"Loading ground truth from: {truth_path}")

    merged = load_and_merge(predictions_path, truth_path)
    metrics = generate_report(merged, output_path)

    print_summary(metrics)
    print(f"Full report saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
