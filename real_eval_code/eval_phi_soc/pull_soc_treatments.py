#!/usr/bin/env python3
"""
Pulls standard-of-care (SOC) treatment data for evaluation.

This script:
1. Loads treatment plan data from structured EHR data
2. Filters to standard oncology chemo plans (not research/trial)
3. Filters to date range (2016-2023)
4. Merges with train/validation/test split
5. Saves filtered treatments

Usage:
    python pull_soc_treatments.py [--start-date 2016-01-01] [--end-date 2023-01-01]
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Repo root for default paths
REPO_ROOT = Path(__file__).resolve().parents[3]  # matchminer-ai-training


def parse_args():
    parser = argparse.ArgumentParser(description="Pull SOC treatment data")
    parser.add_argument("--structured-folder", type=str,
                        default="/data1/ken/pan_dfci_2024/structured_data/",
                        help="Folder containing structured EHR data")
    parser.add_argument("--split-file", type=str,
                        default="/data1/ken/pan_dfci_2024/derived_data/split_5-2024.csv",
                        help="CSV file with train/validation/test split")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output CSV file (default: data/phi_soc/soc_treatments.csv)")
    parser.add_argument("--start-date", type=str, default="2016-01-01",
                        help="Start date for filtering treatments")
    parser.add_argument("--end-date", type=str, default="2023-01-01",
                        help="End date for filtering treatments")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set defaults
    if args.output_file is None:
        args.output_file = str(REPO_ROOT.parent / "data/phi_soc/soc_treatments.csv")

    # Ensure output directory exists
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Load treatment plan data
    print(f"Loading treatment data from {args.structured_folder}...")
    treatments = pd.read_csv(
        args.structured_folder + 'TREATMENT_PLAN.txt',
        sep="|",
        encoding='latin1',
        low_memory=False
    ).rename(columns={
        'TPLAN_GOAL': 'tplan_goal',
        'PATIENT_ID': 'patient_id',
        'TPLAN_START_DT': 'tplan_start_dt'
    })

    print(f"Loaded {treatments.shape[0]} treatment records")

    # Process treatments
    treatments['tplan_start_dt'] = pd.to_datetime(treatments.tplan_start_dt)

    # Filter to those with chemo plans
    treatments = treatments[
        ~(treatments.STD_CHEMO_PLAN.isnull() & treatments.RESEARCH_CHEMO_PLAN.isnull())
    ]

    # Create plan column
    treatments['plan'] = np.where(
        treatments.STD_CHEMO_PLAN.isnull(),
        treatments.RESEARCH_CHEMO_PLAN,
        treatments.STD_CHEMO_PLAN
    )
    treatments['is_trial'] = np.where(treatments.STD_CHEMO_PLAN.isnull(), 1, 0)
    treatments['dfci_mrn'] = treatments['DFCI_MRN']
    treatments['tplan_id'] = treatments['TPLAN_ID']
    treatments['dx'] = treatments.TPLAN_ICD_DX_CODES.str[:3]

    # Filter to non-null goal
    treatments = treatments[~treatments.tplan_goal.isnull()]
    treatments['is_palliative'] = np.where(
        treatments.tplan_goal.str.contains('PALLIATIVE|CONTROL'), 1, 0
    )
    treatments['protocol_nbr'] = treatments['RESEARCH_CHEMO_PLAN_NBR']

    # Filter to standard oncology plans only (not research)
    treatments = treatments[treatments.TREATMENT_PLAN_CATEGORY == 'ONCOLOGY STANDARD CHEMO PLAN']
    print(f"After filtering to SOC: {treatments.shape[0]} records")

    # Filter by date range
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    treatments = treatments[
        (treatments.tplan_start_dt >= start_date) &
        (treatments.tplan_start_dt <= end_date)
    ]
    print(f"After date filtering ({args.start_date} to {args.end_date}): {treatments.shape[0]} records")

    # Load and merge split
    print(f"Loading split from {args.split_file}...")
    split = pd.read_csv(args.split_file)
    treatments = pd.merge(split, treatments, on='dfci_mrn')

    print(f"\nSplit distribution:")
    print(treatments.split.value_counts())

    # Save
    treatments.to_csv(args.output_file, index=False)
    print(f"\nSaved {treatments.shape[0]} records to {args.output_file}")
    print(f"Unique patients: {treatments.dfci_mrn.nunique()}")


if __name__ == "__main__":
    main()
