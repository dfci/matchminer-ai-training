#!/usr/bin/env python3
"""
Creates trial "spaces" (structured representations) from trial texts.

Usage:
    python 3_spacify_dfci_trials.py --gpus 0,1 [--gpus-per-instance 2]
"""

import argparse
import os
import subprocess
import sys
import pandas as pd
from pathlib import Path

# Repo root for default paths
REPO_ROOT = Path(__file__).resolve().parents[3]  # matchminer-ai-training


def parse_args():
    parser = argparse.ArgumentParser(description="Create trial spaces from trial texts")
    parser.add_argument("--gpus", type=str, required=True,
                        help="Comma-separated GPU IDs (e.g., '0,1')")
    parser.add_argument("--gpus-per-instance", type=int, default=2,
                        help="GPUs per vLLM instance")
    parser.add_argument("--input-file", type=str, default=None,
                        help="Input parquet with trial enrollments (default: data/phi/phi_enrollments_longnotes.parquet)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/phi)")
    parser.add_argument("--download-dir", type=str,
                        default="/data1/ken/meta/2024/meta_ai",
                        help="Download directory for model weights")
    parser.add_argument("--gpu-mem-util", type=float, default=0.90,
                        help="GPU memory utilization")
    parser.add_argument("--create-spaces-script", type=str,
                        default=None,
                        help="Path to trial spaces creation script (default: searches common locations)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set defaults
    if args.input_file is None:
        args.input_file = str(REPO_ROOT.parent / "data/phi/phi_enrollments_longnotes.parquet")
    if args.output_dir is None:
        args.output_dir = str(REPO_ROOT.parent / "data/phi")
    if args.create_spaces_script is None:
        # Try common locations
        possible_paths = [
            REPO_ROOT / "0b_create_trial_spaces.py",
            Path("/ksg/kehl_mm_data/meta/2024/v20/v20_training_code/0b_create_trial_spaces.py"),
        ]
        for p in possible_paths:
            if p.exists():
                args.create_spaces_script = str(p)
                break
        if args.create_spaces_script is None:
            print("Error: Could not find trial spaces creation script. Please specify --create-spaces-script")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract unique trials from the enrollment data
    print("=== Step 1: Extracting unique trials ===")
    input_path = Path(args.input_file)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if input_path.suffix == '.csv':
        temp = pd.read_csv(str(input_path))
    else:
        temp = pd.read_parquet(str(input_path))
    print(f"Loaded {temp.shape[0]} enrollment records")

    # Get unique trials
    trials = temp[['protocol_number', 'nct_id', 'trial_text']].groupby(['nct_id']).first().reset_index()
    print(f"Found {trials.shape[0]} unique trials")

    trials_csv = output_dir / "enrolled_trials_for_analysis.csv"
    trials.to_csv(str(trials_csv))
    print(f"Saved trials to {trials_csv}")

    # Step 2: Run the trial spaces creation script
    print("\n=== Step 2: Creating trial spaces ===")

    output_csv = output_dir / "trial_space_lineitems.csv"

    cmd = [
        "python", args.create_spaces_script,
        "--input", str(trials_csv),
        "--output-spaces", str(output_csv),
        "--output-trials", str(output_dir / "trials_with_spaces.csv"),
        "--work-dir", str(output_dir / "trial_space_shards"),
        "--gpus", args.gpus,
        "--gpus-per-instance", str(args.gpus_per_instance),
        "--download-dir", args.download_dir,
        "--gpu-mem-util", str(args.gpu_mem_util)
    ]

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"Error: Trial spaces creation failed with code {result.returncode}")
        sys.exit(result.returncode)

    # Verify output
    if output_csv.exists():
        spaces = pd.read_csv(str(output_csv))
        print(f"\n=== Done ===")
        print(f"Created {spaces.shape[0]} trial space line items")
    else:
        print(f"Warning: Expected output file not found: {output_csv}")


if __name__ == "__main__":
    main()
