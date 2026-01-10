#!/usr/bin/env python3
"""
Evaluate OncoReasoning-3B LLM boilerplate checker performance for SOC data.

Usage:
    python eval_llm_boilerplate_checker.py --mode patient_centric --data-dir /path/to/data --output-dir /path/to/output
    python eval_llm_boilerplate_checker.py --mode trial_centric --data-dir /path/to/data --output-dir /path/to/output
"""

import argparse
import sys
from pathlib import Path

# Add enrollments scripts directory to path for eval_utils
ENROLLMENTS_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "eval_phi_enrollments" / "scripts"
sys.path.insert(0, str(ENROLLMENTS_SCRIPTS_DIR))

import pandas as pd
import numpy as np
from eval_utils import (
    eval_model,
    load_and_combine_csv_files
)
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate OncoReasoning-3B LLM boilerplate checker for SOC"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["patient_centric", "trial_centric"],
                        help="Evaluation mode")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing candidate and LLM result files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save evaluation outputs")
    parser.add_argument("--llm-results-file", type=str, default=None,
                        help="Path to LLM results CSV file")
    parser.add_argument("--split-filter", type=str, default=None,
                        help="Filter to specific split")
    return parser.parse_args()


def load_llm_results(results_file: Path, label_col: str = 'exclusion_result') -> pd.Series:
    """Load and aggregate LLM results by prompt_id."""
    print(f"Loading LLM results from: {results_file}")
    input_frame = pd.read_csv(results_file)
    print(f"Loaded {len(input_frame)} rows")
    predictions = input_frame.groupby(['prompt_id'])[label_col].mean()
    print(f"Aggregated to {len(predictions)} unique prompts")
    return predictions


def evaluate_patient_centric(data_dir: Path, output_dir: Path,
                              llm_results_file: Path = None,
                              split_filter: str = None):
    """Evaluate patient-centric LLM boilerplate checker for SOC."""
    print("=" * 60)
    print("EVALUATING PATIENT-CENTRIC LLM BOILERPLATE CHECKER (SOC)")
    print("=" * 60)

    if llm_results_file is None:
        possible_files = [
            data_dir / "patient_centric_boilerplate_checks" / "consolidated_results.csv",
            data_dir / "boilerplate_patient_centric_soc.csv",
            Path(__file__).parent / "boilerplate_patient_centric_soc.csv",
        ]
        for f in possible_files:
            if f.exists():
                llm_results_file = f
                break

    if llm_results_file is None or not llm_results_file.exists():
        print(f"LLM results file not found")
        return

    predictions = load_llm_results(llm_results_file, 'exclusion_result')

    candidates_dir = data_dir / "boilerplates_for_spaces_for_patient_checks"
    if not candidates_dir.exists():
        candidates_dir = data_dir / "patient_centric_boilerplate_checks"

    print(f"Loading gold standard from: {candidates_dir}")

    try:
        gold = load_and_combine_csv_files(str(candidates_dir))
    except FileNotFoundError:
        print(f"No gold standard data found in {candidates_dir}")
        return

    if split_filter and 'split' in gold.columns:
        gold = gold[gold.split.str.contains(split_filter)]
        print(f"Filtered to {split_filter} split: {len(gold)} rows")

    if 'Unnamed: 0' in gold.columns:
        gold = gold.sort_values(by='Unnamed: 0').reset_index(drop=True)
    else:
        gold = gold.reset_index(drop=True)

    print(f"Loaded {len(gold)} gold standard rows")

    label_col = 'exclusion_result'
    if label_col not in gold.columns:
        print(f"Warning: {label_col} column not found")
        return

    if len(predictions) != len(gold):
        print(f"Warning: prediction count ({len(predictions)}) != gold count ({len(gold)})")
        min_len = min(len(predictions), len(gold))
        predictions = predictions.head(min_len)
        gold = gold.head(min_len)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Classification Metrics ---")
    auc = roc_auc_score(gold[label_col], predictions.values)
    print(f"AUC: {auc:.4f}")

    pdf_path = output_dir / "llm_boilerplate_checker_patient_centric_classification_soc.pdf"
    eval_model(
        predictions.values,
        gold[label_col].values,
        pdf_path=str(pdf_path),
        title_prefix="SOC LLM Boilerplate Checker Patient-Centric"
    )

    print(f"\nEvaluation complete. Reports saved to: {output_dir}")


def evaluate_trial_centric(data_dir: Path, output_dir: Path,
                            llm_results_file: Path = None,
                            split_filter: str = None):
    """Evaluate trial-centric LLM boilerplate checker for SOC."""
    print("=" * 60)
    print("EVALUATING TRIAL-CENTRIC LLM BOILERPLATE CHECKER (SOC)")
    print("=" * 60)

    if llm_results_file is None:
        possible_files = [
            data_dir / "trial_centric_boilerplate_checks" / "consolidated_results.csv",
            data_dir / "boilerplate_trial_centric_soc.csv",
            Path(__file__).parent / "boilerplate_trial_centric_soc.csv",
        ]
        for f in possible_files:
            if f.exists():
                llm_results_file = f
                break

    if llm_results_file is None or not llm_results_file.exists():
        print(f"LLM results file not found")
        return

    predictions = load_llm_results(llm_results_file, 'exclusion_result')

    candidates_dir = data_dir / "boilerplates_for_patients_for_spaces_checks"
    if not candidates_dir.exists():
        candidates_dir = data_dir / "trial_centric_boilerplate_checks"

    print(f"Loading gold standard from: {candidates_dir}")

    try:
        gold = load_and_combine_csv_files(str(candidates_dir))
    except FileNotFoundError:
        print(f"No gold standard data found in {candidates_dir}")
        return

    if split_filter and 'split' in gold.columns:
        gold = gold[gold.split.str.contains(split_filter)]
        print(f"Filtered to {split_filter} split: {len(gold)} rows")

    if 'Unnamed: 0' in gold.columns:
        gold = gold.sort_values(by='Unnamed: 0').reset_index(drop=True)
    else:
        gold = gold.reset_index(drop=True)

    print(f"Loaded {len(gold)} gold standard rows")

    label_col = 'exclusion_result'
    if label_col not in gold.columns:
        print(f"Warning: {label_col} column not found")
        return

    if len(predictions) != len(gold):
        print(f"Warning: prediction count ({len(predictions)}) != gold count ({len(gold)})")
        min_len = min(len(predictions), len(gold))
        predictions = predictions.head(min_len)
        gold = gold.head(min_len)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Classification Metrics ---")
    auc = roc_auc_score(gold[label_col], predictions.values)
    print(f"AUC: {auc:.4f}")

    pdf_path = output_dir / "llm_boilerplate_checker_trial_centric_classification_soc.pdf"
    eval_model(
        predictions.values,
        gold[label_col].values,
        pdf_path=str(pdf_path),
        title_prefix="SOC LLM Boilerplate Checker Trial-Centric"
    )

    print(f"\nEvaluation complete. Reports saved to: {output_dir}")


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    llm_results_file = Path(args.llm_results_file) if args.llm_results_file else None

    if args.mode == "patient_centric":
        evaluate_patient_centric(data_dir, output_dir, llm_results_file, args.split_filter)
    else:
        evaluate_trial_centric(data_dir, output_dir, llm_results_file, args.split_filter)


if __name__ == "__main__":
    main()
