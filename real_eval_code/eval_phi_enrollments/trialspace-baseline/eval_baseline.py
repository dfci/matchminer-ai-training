#!/usr/bin/env python3
"""
Evaluate trialspace baseline retrieval model performance.

This script evaluates the baseline trialspace embedding model by computing:
- MAP@K metrics for ranking quality
- Eligibility result distribution

Usage:
    python eval_baseline.py --mode patient_centric --data-dir /path/to/data --output-dir /path/to/output
    python eval_baseline.py --mode trial_centric --data-dir /path/to/data --output-dir /path/to/output
"""

import argparse
import sys
from pathlib import Path

# Add scripts directory to path for eval_utils
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import pandas as pd
import numpy as np
from eval_utils import (
    average_precision_at_k,
    generate_ranking_report,
    load_and_combine_csv_files
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trialspace baseline retrieval model"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["patient_centric", "trial_centric"],
                        help="Evaluation mode")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing candidate CSV files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save evaluation outputs")
    parser.add_argument("--k", type=int, default=20,
                        help="K value for MAP@K calculation (default: 20)")
    return parser.parse_args()


def evaluate_patient_centric(data_dir: Path, output_dir: Path, k: int = 20):
    """
    Evaluate patient-centric retrieval.

    For each patient, evaluate how well the model ranks trials
    (with enrolled trial at the top).
    """
    print("=" * 60)
    print("EVALUATING PATIENT-CENTRIC BASELINE RETRIEVAL")
    print("=" * 60)

    # Load candidate files
    candidates_dir = data_dir / "spaces_for_patient_checks"
    if not candidates_dir.exists():
        # Try alternate path pattern
        candidates_dir = data_dir / "shards_patient_centric"

    print(f"Loading data from: {candidates_dir}")

    try:
        combined_df = load_and_combine_csv_files(str(candidates_dir))
    except FileNotFoundError:
        # Try loading consolidated file
        consolidated_path = data_dir / "patient_centric_candidates.csv"
        if consolidated_path.exists():
            print(f"Loading consolidated file: {consolidated_path}")
            combined_df = pd.read_csv(consolidated_path)
        else:
            print(f"No data found in {candidates_dir} or {consolidated_path}")
            return

    print(f"Loaded {len(combined_df)} rows")
    print(f"Unique patients: {combined_df.patient_summary.nunique()}")
    print(f"Unique trial spaces: {combined_df.this_space.nunique()}")

    # Check for eligibility_result column
    if 'eligibility_result' not in combined_df.columns:
        print("Warning: eligibility_result column not found")
        return

    # Print eligibility distribution
    print("\nEligibility distribution:")
    print(combined_df.eligibility_result.value_counts())
    print(f"Positive rate: {combined_df.eligibility_result.mean():.4f}")

    # Calculate MAP@K
    print(f"\nCalculating MAP@{k}...")
    validation_set = combined_df.copy()

    # Group by patient and calculate AP
    temp = validation_set.groupby('patient_summary').eligibility_result.apply(
        lambda x: average_precision_at_k(x.head(k).values)
    )
    map_k = temp.mean()

    print(f"MAP@{k}: {map_k:.4f}")

    # Generate PDF report
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "baseline_patient_centric_eval.pdf"

    stats = generate_ranking_report(
        validation_set,
        group_col='patient_summary',
        label_col='eligibility_result',
        pdf_path=str(pdf_path),
        title_prefix="Baseline Patient-Centric",
        k=k
    )

    print(f"\nEvaluation complete. Report saved to: {pdf_path}")
    return stats


def evaluate_trial_centric(data_dir: Path, output_dir: Path, k: int = 20):
    """
    Evaluate trial-centric retrieval.

    For each trial, evaluate how well the model ranks patients
    (with enrolled patients at the top).
    """
    print("=" * 60)
    print("EVALUATING TRIAL-CENTRIC BASELINE RETRIEVAL")
    print("=" * 60)

    # Load candidate files
    candidates_dir = data_dir / "patients_for_spaces_checks"
    if not candidates_dir.exists():
        candidates_dir = data_dir / "shards_trial_centric"

    print(f"Loading data from: {candidates_dir}")

    try:
        combined_df = load_and_combine_csv_files(str(candidates_dir))
    except FileNotFoundError:
        # Try loading consolidated file
        consolidated_path = data_dir / "trial_centric_candidates.csv"
        if consolidated_path.exists():
            print(f"Loading consolidated file: {consolidated_path}")
            combined_df = pd.read_csv(consolidated_path)
        else:
            print(f"No data found in {candidates_dir} or {consolidated_path}")
            return

    print(f"Loaded {len(combined_df)} rows")
    print(f"Unique patients: {combined_df.patient_summary.nunique()}")
    print(f"Unique trial spaces: {combined_df.this_space.nunique()}")

    # Check for eligibility_result column
    if 'eligibility_result' not in combined_df.columns:
        print("Warning: eligibility_result column not found")
        return

    # Print eligibility distribution
    print("\nEligibility distribution:")
    print(combined_df.eligibility_result.value_counts())
    print(f"Positive rate: {combined_df.eligibility_result.mean():.4f}")

    # Calculate MAP@K (group by trial space instead of patient)
    print(f"\nCalculating MAP@{k}...")
    validation_set = combined_df.copy()

    temp = validation_set.groupby('this_space').eligibility_result.apply(
        lambda x: average_precision_at_k(x.head(k).values)
    )
    map_k = temp.mean()

    print(f"MAP@{k}: {map_k:.4f}")

    # Generate PDF report
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "baseline_trial_centric_eval.pdf"

    stats = generate_ranking_report(
        validation_set,
        group_col='this_space',
        label_col='eligibility_result',
        pdf_path=str(pdf_path),
        title_prefix="Baseline Trial-Centric",
        k=k
    )

    print(f"\nEvaluation complete. Report saved to: {pdf_path}")
    return stats


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.mode == "patient_centric":
        evaluate_patient_centric(data_dir, output_dir, args.k)
    else:
        evaluate_trial_centric(data_dir, output_dir, args.k)


if __name__ == "__main__":
    main()
