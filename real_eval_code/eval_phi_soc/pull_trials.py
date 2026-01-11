#!/usr/bin/env python3
"""
Pulls and samples trial spaces for SOC evaluation.

This script:
1. Loads trial spaces from the public data
2. Excludes trials used in training
3. Samples a subset of trials for evaluation

Usage:
    python pull_trials.py [--n-trials 500] [--random-seed 42]
"""

import argparse
import pandas as pd
from pathlib import Path

# Repo root for default paths
REPO_ROOT = Path(__file__).resolve().parents[3]  # matchminer-ai-training


def parse_args():
    parser = argparse.ArgumentParser(description="Pull and sample trial spaces")
    parser.add_argument("--trial-spaces", type=str, default=None,
                        help="Input CSV with all trial spaces (default: v20_public_data/trial_space_lineitems.csv)")
    parser.add_argument("--training-trials", type=str, default=None,
                        help="Input CSV with training trials to exclude (default: v20_public_data/sample_trial_space_lineitems.csv)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output CSV file (default: data/phi_soc/sample_spaces.csv)")
    parser.add_argument("--n-trials", type=int, default=500,
                        help="Number of unique trials to sample")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set defaults
    if args.trial_spaces is None:
        args.trial_spaces = str(REPO_ROOT / "v20_public_data/trial_space_lineitems.csv")
    if args.training_trials is None:
        args.training_trials = str(REPO_ROOT / "v20_public_data/sample_trial_space_lineitems.csv")
    if args.output_file is None:
        args.output_file = str(REPO_ROOT.parent / "data/phi_soc/sample_spaces.csv")

    # Ensure output directory exists
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Load trial spaces
    print(f"Loading trial spaces from {args.trial_spaces}...")
    trials = pd.read_csv(args.trial_spaces).reset_index(drop=True)
    print(f"Loaded {trials.shape[0]} trial space records")

    # Load training trials to exclude
    print(f"Loading training trials from {args.training_trials}...")
    training_trials = pd.read_csv(args.training_trials)
    print(f"Found {training_trials.nct_id.nunique()} training trial NCT IDs to exclude")

    # Exclude training trials
    remaining_trials = trials[~trials.nct_id.isin(training_trials.nct_id)]
    print(f"Remaining trials after exclusion: {remaining_trials.shape[0]} records, {remaining_trials.nct_id.nunique()} unique trials")

    # Sample unique trials
    unique_trial_sample = remaining_trials.groupby('nct_id').first().reset_index()[['nct_id']].sample(
        n=min(args.n_trials, remaining_trials.nct_id.nunique()),
        random_state=args.random_seed
    )
    print(f"Sampled {len(unique_trial_sample)} unique trials")

    # Get all spaces for sampled trials
    sample_spaces = remaining_trials[remaining_trials.nct_id.isin(unique_trial_sample.nct_id)]
    print(f"Sample contains {sample_spaces.shape[0]} space records")

    # Save
    sample_spaces.to_csv(args.output_file, index=False)
    print(f"Saved to {args.output_file}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Unique trials: {sample_spaces.nct_id.nunique()}")
    print(f"Unique spaces: {sample_spaces.this_space.nunique()}")
    print(f"Total records: {sample_spaces.shape[0]}")


if __name__ == "__main__":
    main()
