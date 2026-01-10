#!/usr/bin/env python3
"""
prepare_trials.py

Prepares the trial denominator for SOC evaluation by selecting trials that
were NOT used in training.

Logic (from 1_pull_trials.ipynb):
1. Load all trial spaces from v20_public_data/trial_space_lineitems.csv
2. Load training trials from v20_public_data/sample_trial_space_lineitems.csv
3. Filter out trials that were used for training
4. Sample N unique trials from the remaining trials
5. Output the trial spaces for those sampled trials
"""

import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare trial denominator for SOC evaluation"
    )
    parser.add_argument(
        "--all-trials",
        type=str,
        required=True,
        help="Path to all trial spaces CSV (e.g., v20_public_data/trial_space_lineitems.csv)"
    )
    parser.add_argument(
        "--training-trials",
        type=str,
        required=True,
        help="Path to training trial spaces CSV (e.g., v20_public_data/sample_trial_space_lineitems.csv)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output path for selected trial spaces CSV"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=500,
        help="Number of unique trials to sample (default: 500)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Step 1: Load all trial spaces
    print(f"Loading all trial spaces from {args.all_trials}...")
    all_trials = pd.read_csv(args.all_trials).reset_index(drop=True)
    print(f"  Loaded {len(all_trials)} trial space line items")
    print(f"  From {all_trials.nct_id.nunique()} unique trials")

    # Step 2: Load training trials
    print(f"\nLoading training trials from {args.training_trials}...")
    training_trials = pd.read_csv(args.training_trials)
    print(f"  Loaded {len(training_trials)} training trial space line items")
    print(f"  From {training_trials.nct_id.nunique()} unique trials")

    # Step 3: Filter out training trials
    print("\nFiltering out training trials...")
    remaining_trials = all_trials[~all_trials.nct_id.isin(training_trials.nct_id)]
    print(f"  {len(remaining_trials)} trial space line items remaining")
    print(f"  From {remaining_trials.nct_id.nunique()} unique trials")

    # Step 4: Sample N unique trials
    print(f"\nSampling {args.n_trials} unique trials (seed={args.random_seed})...")
    unique_trial_sample = (
        remaining_trials
        .groupby('nct_id')
        .first()
        .reset_index()[['nct_id']]
        .sample(n=min(args.n_trials, remaining_trials.nct_id.nunique()), random_state=args.random_seed)
    )
    print(f"  Selected {len(unique_trial_sample)} unique trials")

    # Step 5: Get all spaces for sampled trials
    sample_spaces = remaining_trials[remaining_trials.nct_id.isin(unique_trial_sample.nct_id)]
    print(f"  {len(sample_spaces)} trial space line items for sampled trials")

    # Save output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_spaces.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  All trials: {all_trials.nct_id.nunique()}")
    print(f"  Training trials (excluded): {training_trials.nct_id.nunique()}")
    print(f"  Remaining trials: {remaining_trials.nct_id.nunique()}")
    print(f"  Sampled trials: {unique_trial_sample.shape[0]}")
    print(f"  Output line items: {len(sample_spaces)}")


if __name__ == "__main__":
    main()
