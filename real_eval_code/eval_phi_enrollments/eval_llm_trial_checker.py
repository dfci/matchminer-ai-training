#!/usr/bin/env python3
"""
Evaluate OncoReasoning-3B LLM trial checker performance.

This script evaluates the OncoReasoning-3B LLM-based trial eligibility checker by:
- Loading pre-computed LLM inference results
- Computing classification metrics (AUC, F1, precision-recall)
- Computing ranking metrics (MAP@K)
- Generating PDF reports

The script expects:
- LLM results file with prompt_id and eligibility_result columns
- Gold standard candidate files with actual eligibility labels

Usage:
    python eval_llm_trial_checker.py --mode patient_centric --data-dir /path/to/data --output-dir /path/to/output
    python eval_llm_trial_checker.py --mode trial_centric --data-dir /path/to/data --output-dir /path/to/output
"""

import argparse
import sys
from pathlib import Path

# eval_utils is in the same directory now

import pandas as pd
import numpy as np
from eval_utils import (
    eval_model,
    average_precision_at_k,
    generate_ranking_report,
    load_and_combine_csv_files
)
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate OncoReasoning-3B LLM trial checker"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["patient_centric", "trial_centric"],
                        help="Evaluation mode")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing candidate and LLM result files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save evaluation outputs")
    parser.add_argument("--llm-results-file", type=str, default=None,
                        help="Path to LLM results CSV file (overrides auto-detection)")
    parser.add_argument("--k", type=int, default=20,
                        help="K value for MAP@K calculation (default: 20)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for positive classification (default: 0.5)")
    return parser.parse_args()


def load_llm_results(results_file: Path) -> pd.Series:
    """
    Load and aggregate LLM results by prompt_id.

    Args:
        results_file: Path to LLM results CSV

    Returns:
        Series of aggregated predictions indexed by prompt_id
    """
    print(f"Loading LLM results from: {results_file}")
    input_frame = pd.read_csv(results_file)
    print(f"Loaded {len(input_frame)} rows")

    # Aggregate predictions by prompt_id (mean of eligibility_result)
    predictions = input_frame.groupby(['prompt_id'])['eligibility_result'].mean()
    print(f"Aggregated to {len(predictions)} unique prompts")

    return predictions


def evaluate_patient_centric(data_dir: Path, output_dir: Path,
                              llm_results_file: Path = None,
                              k: int = 20, threshold: float = 0.5):
    """Evaluate patient-centric LLM trial checker performance."""
    print("=" * 60)
    print("EVALUATING PATIENT-CENTRIC LLM TRIAL CHECKER")
    print("=" * 60)

    # Find LLM results file - only OncoReasoning output
    if llm_results_file is None:
        llm_results_file = data_dir / "oncoreasoning_trialcheck_patient_centric.csv"

    if not llm_results_file.exists():
        print(f"LLM results file not found: {llm_results_file}")
        return

    # Load LLM predictions
    predictions = load_llm_results(llm_results_file)

    # Load gold standard labels - prefer consolidated candidate file
    consolidated_path = data_dir / "patient_centric_candidates.csv"
    if consolidated_path.exists():
        print(f"Loading consolidated file: {consolidated_path}")
        gold = pd.read_csv(consolidated_path)
    else:
        # Fallback to shard directories
        candidates_dir = data_dir / "spaces_for_patient_checks"
        if not candidates_dir.exists():
            candidates_dir = data_dir / "shards_patient_centric"

        print(f"Loading gold standard from: {candidates_dir}")

        try:
            gold = load_and_combine_csv_files(str(candidates_dir))
        except FileNotFoundError:
            print(f"No gold standard data found")
            return

    # Filter nulls and sort
    gold = gold[~gold.patient_summary.isnull()]
    if 'Unnamed: 0' in gold.columns:
        gold = gold.sort_values(by='Unnamed: 0').reset_index(drop=True)
    else:
        gold = gold.reset_index(drop=True)

    print(f"Loaded {len(gold)} gold standard rows")

    # Align predictions with gold standard
    # Predictions are indexed by prompt_id which corresponds to row order
    if len(predictions) != len(gold):
        print(f"Warning: prediction count ({len(predictions)}) != gold count ({len(gold)})")
        # Try to align by index
        min_len = min(len(predictions), len(gold))
        predictions = predictions.head(min_len)
        gold = gold.head(min_len)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute classification metrics
    print("\n--- Classification Metrics ---")
    auc = roc_auc_score(gold.eligibility_result, predictions.values)
    print(f"AUC: {auc:.4f}")

    # Generate classification PDF report
    pdf_path = output_dir / "llm_trial_checker_patient_centric_classification.pdf"
    eval_model(
        predictions.values,
        gold.eligibility_result.values,
        pdf_path=str(pdf_path),
        title_prefix="LLM Trial Checker Patient-Centric"
    )

    # Compute ranking metrics after filtering to positive predictions
    print(f"\n--- Ranking Metrics (threshold={threshold}) ---")

    # Filter to positive predictions
    positive_mask = predictions.values >= threshold
    pruned_gold = gold[positive_mask].copy()

    print(f"Samples after filtering: {len(pruned_gold)}")
    print(f"Median results per patient: {pruned_gold.groupby('patient_summary').size().median():.1f}")
    print(f"Mean results per patient: {pruned_gold.groupby('patient_summary').size().mean():.1f}")
    print(f"Unique patients: {pruned_gold.patient_summary.nunique()}")
    print(f"Unique trial spaces: {pruned_gold.this_space.nunique()}")
    print(f"Positive rate after filtering: {pruned_gold.eligibility_result.mean():.4f}")

    # Calculate MAP@K on filtered set
    temp = pruned_gold.groupby('patient_summary').head(k)
    ap_scores = pruned_gold.groupby('patient_summary').eligibility_result.apply(
        lambda x: average_precision_at_k(x.values)
    )
    map_k = ap_scores.mean()
    print(f"MAP@{k} (after LLM check): {map_k:.4f}")

    # Generate ranking PDF report
    pdf_path = output_dir / "llm_trial_checker_patient_centric_ranking.pdf"
    generate_ranking_report(
        pruned_gold,
        group_col='patient_summary',
        label_col='eligibility_result',
        pdf_path=str(pdf_path),
        title_prefix="LLM Trial Checker Patient-Centric (Filtered)",
        k=k
    )

    print(f"\nEvaluation complete. Reports saved to: {output_dir}")


def evaluate_trial_centric(data_dir: Path, output_dir: Path,
                            llm_results_file: Path = None,
                            k: int = 20, threshold: float = 0.5):
    """Evaluate trial-centric LLM trial checker performance."""
    print("=" * 60)
    print("EVALUATING TRIAL-CENTRIC LLM TRIAL CHECKER")
    print("=" * 60)

    # Find LLM results file - only OncoReasoning output
    if llm_results_file is None:
        llm_results_file = data_dir / "oncoreasoning_trialcheck_trial_centric.csv"

    if not llm_results_file.exists():
        print(f"LLM results file not found: {llm_results_file}")
        return

    # Load LLM predictions
    predictions = load_llm_results(llm_results_file)

    # Load gold standard labels - prefer consolidated candidate file
    consolidated_path = data_dir / "trial_centric_candidates.csv"
    if consolidated_path.exists():
        print(f"Loading consolidated file: {consolidated_path}")
        gold = pd.read_csv(consolidated_path)
    else:
        # Fallback to shard directories
        candidates_dir = data_dir / "patients_for_spaces_checks"
        if not candidates_dir.exists():
            candidates_dir = data_dir / "shards_trial_centric"

        print(f"Loading gold standard from: {candidates_dir}")

        try:
            gold = load_and_combine_csv_files(str(candidates_dir))
        except FileNotFoundError:
            print(f"No gold standard data found")
            return

    # Filter nulls and sort
    gold = gold[~gold.patient_summary.isnull()]
    if 'Unnamed: 0' in gold.columns:
        gold = gold.sort_values(by='Unnamed: 0').reset_index(drop=True)
    else:
        gold = gold.reset_index(drop=True)

    print(f"Loaded {len(gold)} gold standard rows")

    # Align predictions with gold standard
    if len(predictions) != len(gold):
        print(f"Warning: prediction count ({len(predictions)}) != gold count ({len(gold)})")
        min_len = min(len(predictions), len(gold))
        predictions = predictions.head(min_len)
        gold = gold.head(min_len)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute classification metrics
    print("\n--- Classification Metrics ---")
    auc = roc_auc_score(gold.eligibility_result, predictions.values)
    print(f"AUC: {auc:.4f}")

    pdf_path = output_dir / "llm_trial_checker_trial_centric_classification.pdf"
    eval_model(
        predictions.values,
        gold.eligibility_result.values,
        pdf_path=str(pdf_path),
        title_prefix="LLM Trial Checker Trial-Centric"
    )

    # Compute ranking metrics
    print(f"\n--- Ranking Metrics (threshold={threshold}) ---")

    positive_mask = predictions.values >= threshold
    pruned_gold = gold[positive_mask].copy()

    print(f"Samples after filtering: {len(pruned_gold)}")
    print(f"Median results per trial: {pruned_gold.groupby('this_space').size().median():.1f}")
    print(f"Mean results per trial: {pruned_gold.groupby('this_space').size().mean():.1f}")
    print(f"Unique patients: {pruned_gold.patient_summary.nunique()}")
    print(f"Unique trial spaces: {pruned_gold.this_space.nunique()}")
    print(f"Positive rate after filtering: {pruned_gold.eligibility_result.mean():.4f}")

    # For trial-centric, group by trial space
    ap_scores = pruned_gold.groupby('this_space').eligibility_result.apply(
        lambda x: average_precision_at_k(x.values)
    )
    map_k = ap_scores.mean()
    print(f"MAP@{k} (after LLM check): {map_k:.4f}")

    pdf_path = output_dir / "llm_trial_checker_trial_centric_ranking.pdf"
    generate_ranking_report(
        pruned_gold,
        group_col='this_space',
        label_col='eligibility_result',
        pdf_path=str(pdf_path),
        title_prefix="LLM Trial Checker Trial-Centric (Filtered)",
        k=k
    )

    print(f"\nEvaluation complete. Reports saved to: {output_dir}")


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    llm_results_file = Path(args.llm_results_file) if args.llm_results_file else None

    if args.mode == "patient_centric":
        evaluate_patient_centric(
            data_dir, output_dir,
            llm_results_file=llm_results_file,
            k=args.k,
            threshold=args.threshold
        )
    else:
        evaluate_trial_centric(
            data_dir, output_dir,
            llm_results_file=llm_results_file,
            k=args.k,
            threshold=args.threshold
        )


if __name__ == "__main__":
    main()
