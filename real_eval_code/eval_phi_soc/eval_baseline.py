#!/usr/bin/env python3
"""
Evaluate trialspace baseline retrieval model performance for SOC data.

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

# eval_utils is in the same directory now

import pandas as pd
import numpy as np
from eval_utils import (
    average_precision_at_k,
    generate_ranking_report,
    load_and_combine_csv_files
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trialspace baseline retrieval model for SOC"
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
    parser.add_argument("--split-filter", type=str, default=None,
                        help="Filter to specific split (e.g., 'test', 'validation')")
    parser.add_argument("--prefix", type=str, default="",
                        help="Prefix for data files (e.g., 'baseline_' for baseline evaluation)")
    return parser.parse_args()


def evaluate_patient_centric(data_dir: Path, output_dir: Path, k: int = 20,
                              split_filter: str = None, prefix: str = ""):
    """
    Evaluate patient-centric retrieval for SOC data.

    For each patient, evaluate how well the model ranks trials.

    Args:
        data_dir: Directory containing data files
        output_dir: Directory to save outputs
        k: K value for MAP@K
        split_filter: Filter to specific split
        prefix: Prefix for file names (e.g., "baseline_" for baseline evaluation)
    """
    print("=" * 60)
    print(f"EVALUATING {prefix.upper().rstrip('_') + ' ' if prefix else ''}PATIENT-CENTRIC BASELINE RETRIEVAL (SOC)")
    print("=" * 60)

    # For baseline evaluation, load from consolidated eligibility file which has eligibility_result
    # For non-baseline (no prefix), load from candidate files
    if prefix:
        # Baseline: load from consolidated eligibility file (has eligibility_result from LLM checks)
        consolidated_path = data_dir / f"{prefix}consolidated_eligibility_patient_centric.csv"
        if not consolidated_path.exists():
            print(f"Consolidated eligibility file not found: {consolidated_path}")
            return
        print(f"Loading consolidated eligibility file: {consolidated_path}")
        combined_df = pd.read_csv(consolidated_path)
    else:
        # Non-baseline: try candidate files
        consolidated_path = data_dir / "patient_centric_candidates.csv"
        if consolidated_path.exists():
            print(f"Loading consolidated file: {consolidated_path}")
            combined_df = pd.read_csv(consolidated_path)
        else:
            # Fallback to shard directories
            candidates_dir = data_dir / "spaces_for_patients_checks"
            if not candidates_dir.exists():
                candidates_dir = data_dir / "shards_patient_centric"
            print(f"Loading data from: {candidates_dir}")
            try:
                combined_df = load_and_combine_csv_files(str(candidates_dir))
            except FileNotFoundError:
                print(f"No data found")
                return

    print(f"Loaded {len(combined_df)} rows")

    # Filter by split if specified
    if split_filter and 'split' in combined_df.columns:
        combined_df = combined_df[combined_df.split.str.contains(split_filter)]
        print(f"Filtered to {split_filter} split: {len(combined_df)} rows")

    # Filter out null patient summaries
    combined_df = combined_df[~combined_df.patient_summary.isnull()]

    print(f"Unique patients: {combined_df.patient_summary.nunique()}")
    print(f"Unique trial spaces: {combined_df.this_space.nunique()}")

    if 'eligibility_result' not in combined_df.columns:
        print("Warning: eligibility_result column not found")
        return

    print("\nEligibility distribution:")
    print(combined_df.eligibility_result.value_counts())
    print(f"Positive rate: {combined_df.eligibility_result.mean():.4f}")

    # Calculate MAP@K
    print(f"\nCalculating MAP@{k}...")
    validation_set = combined_df.copy()

    temp = validation_set.groupby('patient_summary').eligibility_result.apply(
        lambda x: average_precision_at_k(x.head(k).values)
    )
    map_k = temp.mean()

    print(f"MAP@{k}: {map_k:.4f}")

    # Generate PDF report
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "baseline_patient_centric_eval_soc.pdf"

    stats = generate_ranking_report(
        validation_set,
        group_col='patient_summary',
        label_col='eligibility_result',
        pdf_path=str(pdf_path),
        title_prefix="SOC Baseline Patient-Centric",
        k=k
    )

    print(f"\nEvaluation complete. Report saved to: {pdf_path}")
    return stats


def evaluate_trial_centric(data_dir: Path, output_dir: Path, k: int = 20,
                            split_filter: str = None, prefix: str = ""):
    """
    Evaluate trial-centric retrieval for SOC data.

    For each trial, evaluate how well the model ranks patients.

    Args:
        data_dir: Directory containing data files
        output_dir: Directory to save outputs
        k: K value for MAP@K
        split_filter: Filter to specific split
        prefix: Prefix for file names (e.g., "baseline_" for baseline evaluation)
    """
    print("=" * 60)
    print(f"EVALUATING {prefix.upper().rstrip('_') + ' ' if prefix else ''}TRIAL-CENTRIC BASELINE RETRIEVAL (SOC)")
    print("=" * 60)

    # For baseline evaluation, load from consolidated eligibility file which has eligibility_result
    # For non-baseline (no prefix), load from candidate files
    if prefix:
        # Baseline: load from consolidated eligibility file (has eligibility_result from LLM checks)
        consolidated_path = data_dir / f"{prefix}consolidated_eligibility_trial_centric.csv"
        if not consolidated_path.exists():
            print(f"Consolidated eligibility file not found: {consolidated_path}")
            return
        print(f"Loading consolidated eligibility file: {consolidated_path}")
        combined_df = pd.read_csv(consolidated_path)
    else:
        # Non-baseline: try candidate files
        consolidated_path = data_dir / "trial_centric_candidates.csv"
        if consolidated_path.exists():
            print(f"Loading consolidated file: {consolidated_path}")
            combined_df = pd.read_csv(consolidated_path)
        else:
            # Fallback to shard directories
            candidates_dir = data_dir / "patients_for_spaces_checks"
            if not candidates_dir.exists():
                candidates_dir = data_dir / "shards_trial_centric"
            print(f"Loading data from: {candidates_dir}")
            try:
                combined_df = load_and_combine_csv_files(str(candidates_dir))
            except FileNotFoundError:
                print(f"No data found")
                return

    print(f"Loaded {len(combined_df)} rows")

    if split_filter and 'split' in combined_df.columns:
        combined_df = combined_df[combined_df.split.str.contains(split_filter)]
        print(f"Filtered to {split_filter} split: {len(combined_df)} rows")

    combined_df = combined_df[~combined_df.patient_summary.isnull()]

    print(f"Unique patients: {combined_df.patient_summary.nunique()}")
    print(f"Unique trial spaces: {combined_df.this_space.nunique()}")

    if 'eligibility_result' not in combined_df.columns:
        print("Warning: eligibility_result column not found")
        return

    print("\nEligibility distribution:")
    print(combined_df.eligibility_result.value_counts())
    print(f"Positive rate: {combined_df.eligibility_result.mean():.4f}")

    print(f"\nCalculating MAP@{k}...")
    validation_set = combined_df.copy()

    temp = validation_set.groupby('this_space').eligibility_result.apply(
        lambda x: average_precision_at_k(x.head(k).values)
    )
    map_k = temp.mean()

    print(f"MAP@{k}: {map_k:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "baseline_trial_centric_eval_soc.pdf"

    stats = generate_ranking_report(
        validation_set,
        group_col='this_space',
        label_col='eligibility_result',
        pdf_path=str(pdf_path),
        title_prefix="SOC Baseline Trial-Centric",
        k=k
    )

    print(f"\nEvaluation complete. Report saved to: {pdf_path}")
    return stats


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    prefix = args.prefix

    if args.mode == "patient_centric":
        evaluate_patient_centric(data_dir, output_dir, args.k, args.split_filter, prefix)
    else:
        evaluate_trial_centric(data_dir, output_dir, args.k, args.split_filter, prefix)


if __name__ == "__main__":
    main()
