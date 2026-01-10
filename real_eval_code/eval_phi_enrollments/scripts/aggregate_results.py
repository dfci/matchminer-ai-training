#!/usr/bin/env python3
"""
Aggregates LLM checking results and computes inter-rater reliability metrics.

This script:
1. Loads shard files from eligibility/boilerplate checks
2. Consolidates into a single file
3. Optionally computes IRR metrics if a second run exists

Usage:
    # Aggregate eligibility results for patient_centric
    python aggregate_results.py --mode eligibility --direction patient_centric

    # Aggregate boilerplate results for patient_centric
    python aggregate_results.py --mode boilerplate --direction patient_centric

    # Aggregate eligibility results for trial_centric
    python aggregate_results.py --mode eligibility --direction trial_centric

    # With custom directories
    python aggregate_results.py --mode eligibility --direction patient_centric \
        --primary-dir ./custom_checks

    # With IRR comparison
    python aggregate_results.py --mode eligibility --direction patient_centric \
        --irr-dir ./patient_centric_eligibility_checks_irr
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# Repo root for default paths
REPO_ROOT = Path(__file__).resolve().parents[3]  # matchminer-ai-training


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate LLM checking results")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["eligibility", "boilerplate"],
                        help="Type of check to aggregate")
    parser.add_argument("--direction", type=str, required=True,
                        choices=["patient_centric", "trial_centric"],
                        help="Direction of matching")
    parser.add_argument("--primary-dir", type=str, default=None,
                        help="Directory containing primary run shard files (default: derived from mode/direction)")
    parser.add_argument("--irr-dir", type=str, default=None,
                        help="Directory containing IRR run shard files (for reliability calculation)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output consolidated CSV file (default: derived from mode/direction)")
    return parser.parse_args()


def load_and_consolidate_shards(shard_dir):
    """Load all shard files and consolidate."""
    pattern = os.path.join(shard_dir, "*.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"No CSV files found in {shard_dir}")
        return None

    list_of_dfs = []
    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            list_of_dfs.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if not list_of_dfs:
        print("No DataFrames could be successfully loaded.")
        return None

    combined_df = pd.concat(list_of_dfs, ignore_index=True)

    # Sort by original index if available
    if 'Unnamed: 0' in combined_df.columns:
        combined_df = combined_df.sort_values(by='Unnamed: 0').reset_index(drop=True)

    return combined_df


def compute_irr_metrics(df_primary, df_secondary, result_col):
    """Compute inter-rater reliability metrics."""
    from sklearn.metrics import cohen_kappa_score

    # Ensure same ordering
    if 'Unnamed: 0' in df_primary.columns and 'Unnamed: 0' in df_secondary.columns:
        df_primary = df_primary.sort_values(by='Unnamed: 0').reset_index(drop=True)
        df_secondary = df_secondary.sort_values(by='Unnamed: 0').reset_index(drop=True)

    # Check alignment
    if len(df_primary) != len(df_secondary):
        print(f"Warning: Primary ({len(df_primary)}) and secondary ({len(df_secondary)}) have different lengths")

    # Compute metrics
    primary_results = df_primary[result_col].values
    secondary_results = df_secondary[result_col].values

    # Agreement rate
    agreement = (primary_results == secondary_results).mean()

    # Cohen's Kappa
    kappa = cohen_kappa_score(primary_results, secondary_results)

    # Cross-tabulation
    crosstab = pd.crosstab(
        pd.Series(primary_results, name='Primary'),
        pd.Series(secondary_results, name='Secondary')
    )

    return {
        'agreement_rate': agreement,
        'cohen_kappa': kappa,
        'crosstab': crosstab,
        'n_samples': len(primary_results)
    }


def main():
    args = parse_args()

    # Set defaults based on mode and direction
    data_dir = REPO_ROOT.parent / "data/phi"

    if args.primary_dir is None:
        if args.mode == "eligibility":
            args.primary_dir = str(data_dir / f"{args.direction}_eligibility_checks")
        else:
            args.primary_dir = str(data_dir / f"{args.direction}_boilerplate_checks")

    if args.output_file is None:
        args.output_file = str(data_dir / f"consolidated_{args.mode}_{args.direction}.csv")

    # Determine result column name
    if args.mode == "eligibility":
        result_col = "eligibility_result"
        response_col = "llama_response"
    else:
        result_col = "exclusion_result"
        response_col = "llm_boilerplate_response"

    # Load and consolidate primary results
    print(f"Loading primary results from {args.primary_dir}...")
    df_primary = load_and_consolidate_shards(args.primary_dir)

    if df_primary is None:
        print("Failed to load primary results")
        return

    print(f"Loaded {len(df_primary)} records")

    # Ensure output directory exists
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Save consolidated primary
    df_primary.to_csv(args.output_file, index=False)
    print(f"Saved consolidated results to {args.output_file}")

    # Report basic statistics
    if result_col in df_primary.columns:
        print(f"\n=== Primary Results Summary ===")
        print(f"Total records: {len(df_primary)}")
        print(f"Positive rate: {df_primary[result_col].mean():.4f}")
        print(f"Value counts:")
        print(df_primary[result_col].value_counts())

    # If IRR directory provided, compute reliability
    if args.irr_dir:
        print(f"\n=== Loading IRR results from {args.irr_dir} ===")
        df_secondary = load_and_consolidate_shards(args.irr_dir)

        if df_secondary is not None and result_col in df_secondary.columns:
            print(f"Loaded {len(df_secondary)} IRR records")

            metrics = compute_irr_metrics(df_primary, df_secondary, result_col)

            print(f"\n=== Inter-Rater Reliability Metrics ===")
            print(f"N samples: {metrics['n_samples']}")
            print(f"Agreement rate: {metrics['agreement_rate']:.4f}")
            print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
            print(f"\nCross-tabulation:")
            print(metrics['crosstab'])

            # Check if response texts match (they usually won't due to reasoning variation)
            if response_col in df_primary.columns and response_col in df_secondary.columns:
                response_match_rate = (df_primary[response_col] == df_secondary[response_col]).mean()
                print(f"\nExact response match rate: {response_match_rate:.4f}")
        else:
            print("Failed to load or process IRR results")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
