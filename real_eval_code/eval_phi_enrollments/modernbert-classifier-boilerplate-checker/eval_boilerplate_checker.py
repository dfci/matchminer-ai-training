#!/usr/bin/env python3
"""
Evaluate ModernBERT boilerplate checker classifier performance.

This script evaluates the ModernBERT-based boilerplate exclusion classifier by:
- Running inference on patient-trial boilerplate pairs
- Computing classification metrics (AUC, F1, precision-recall)
- Generating PDF reports

Usage:
    python eval_boilerplate_checker.py --mode patient_centric --data-dir /path/to/data --output-dir /path/to/output
    python eval_boilerplate_checker.py --mode trial_centric --data-dir /path/to/data --output-dir /path/to/output
"""

import argparse
import os
import sys
from pathlib import Path

# Add scripts directory to path for eval_utils
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import pandas as pd
import numpy as np
from eval_utils import (
    eval_model,
    load_and_combine_csv_files
)
from sklearn.metrics import roc_auc_score, cohen_kappa_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ModernBERT boilerplate checker classifier"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["patient_centric", "trial_centric"],
                        help="Evaluation mode")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing boilerplate candidate CSV files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save evaluation outputs")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to boilerplate checker model")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device to use")
    parser.add_argument("--run-inference", action="store_true",
                        help="Run model inference (requires GPU)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference")
    return parser.parse_args()


def run_boilerplate_checker_inference(df: pd.DataFrame, model_path: str,
                                       device: str = "cuda",
                                       batch_size: int = 32) -> pd.DataFrame:
    """
    Run boilerplate checker model inference.

    Args:
        df: DataFrame with patient_boilerplate_text and trial_boilerplate_text columns
        model_path: Path to the boilerplate checker model
        device: Device to use for inference
        batch_size: Batch size

    Returns:
        DataFrame with predictions added
    """
    from transformers import pipeline, AutoTokenizer

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline(
        'text-classification',
        model_path,
        tokenizer=tokenizer,
        truncation=True,
        padding='max_length',
        max_length=3196,
        device=device,
        batch_size=batch_size
    )

    # Create boilerplate pair text
    df = df.copy()

    # Filter out null values
    df = df[~df.trial_boilerplate_text.isnull()]
    df = df[~df.patient_boilerplate_text.isnull()]

    df['boilerplate_pair'] = (
        "Patient history: " + df['patient_boilerplate_text'] +
        "\nTrial exclusions:" + df['trial_boilerplate_text']
    )
    df = df[~df.boilerplate_pair.isnull()]

    print(f"Running inference on {len(df)} samples...")
    predictions = pipe(df.boilerplate_pair.tolist())

    # Process predictions
    predictions_df = pd.DataFrame(predictions)
    predictions_df['score'] = np.where(
        predictions_df.label == 'NEGATIVE',
        1 - predictions_df.score,
        predictions_df.score
    )
    predictions_df['logit_score'] = np.log(
        predictions_df.score + 1e-6 / (1 - predictions_df.score + 1e-6)
    )

    # Merge predictions back
    df = df.reset_index(drop=True)
    df['prediction_label'] = predictions_df['label']
    df['prediction_score'] = predictions_df['score']
    df['prediction_logit'] = predictions_df['logit_score']

    return df


def evaluate_patient_centric(data_dir: Path, output_dir: Path,
                              model_path: str = None, gpu: str = "0",
                              run_inference: bool = False,
                              batch_size: int = 32):
    """Evaluate patient-centric boilerplate checker performance."""
    print("=" * 60)
    print("EVALUATING PATIENT-CENTRIC BOILERPLATE CHECKER")
    print("=" * 60)

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Load boilerplate candidate files
    candidates_dir = data_dir / "boilerplates_for_spaces_for_patient_checks"
    if not candidates_dir.exists():
        candidates_dir = data_dir / "patient_centric_boilerplate_checks"

    print(f"Looking for data in: {candidates_dir}")

    try:
        combined_df = load_and_combine_csv_files(str(candidates_dir))
    except FileNotFoundError:
        print(f"No boilerplate data found in {candidates_dir}")
        return

    print(f"Loaded {len(combined_df)} rows")

    validation_set = combined_df.copy()

    # Filter nulls
    if 'trial_boilerplate_text' in validation_set.columns:
        validation_set = validation_set[~validation_set.trial_boilerplate_text.isnull()]

    print(f"Unique trial spaces: {validation_set.this_space.nunique()}")
    print(f"Unique patients: {validation_set.patient_summary.nunique()}")

    # Run inference if requested
    if run_inference and model_path:
        validation_set = run_boilerplate_checker_inference(
            validation_set, model_path, device='cuda', batch_size=batch_size
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate_path = output_dir / "boilerplate_patient_centric_with_predictions.csv"
        validation_set.to_csv(intermediate_path, index=False)
        print(f"Saved predictions to: {intermediate_path}")
    elif 'prediction_score' not in validation_set.columns:
        precomputed_path = output_dir / "boilerplate_patient_centric_with_predictions.csv"
        if precomputed_path.exists():
            print(f"Loading pre-computed predictions from: {precomputed_path}")
            validation_set = pd.read_csv(precomputed_path)
        else:
            print("No predictions available. Use --run-inference to generate them.")
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute classification metrics
    label_col = 'exclusion_result'
    if label_col not in validation_set.columns:
        print(f"Warning: {label_col} column not found")
        return

    if 'prediction_score' in validation_set.columns:
        print("\n--- Classification Metrics ---")
        print(f"Total samples: {len(validation_set)}")

        auc = roc_auc_score(validation_set[label_col], validation_set.prediction_score)
        print(f"AUC: {auc:.4f}")

        # Generate classification PDF report
        pdf_path = output_dir / "boilerplate_checker_patient_centric_classification.pdf"
        eval_model(
            validation_set.prediction_score.values,
            validation_set[label_col].values,
            pdf_path=str(pdf_path),
            title_prefix="Boilerplate Checker Patient-Centric"
        )

        # Cohen's kappa
        if 'prediction_label' in validation_set.columns:
            actual_labels = np.where(
                validation_set[label_col] == 0.0, 'NEGATIVE', 'POSITIVE'
            )
            kappa = cohen_kappa_score(actual_labels, validation_set.prediction_label)
            print(f"Cohen's Kappa: {kappa:.4f}")

    print(f"\nEvaluation complete. Reports saved to: {output_dir}")


def evaluate_trial_centric(data_dir: Path, output_dir: Path,
                            model_path: str = None, gpu: str = "0",
                            run_inference: bool = False,
                            batch_size: int = 32):
    """Evaluate trial-centric boilerplate checker performance."""
    print("=" * 60)
    print("EVALUATING TRIAL-CENTRIC BOILERPLATE CHECKER")
    print("=" * 60)

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Load boilerplate candidate files
    candidates_dir = data_dir / "boilerplates_for_patients_for_spaces_checks"
    if not candidates_dir.exists():
        candidates_dir = data_dir / "trial_centric_boilerplate_checks"

    print(f"Looking for data in: {candidates_dir}")

    try:
        combined_df = load_and_combine_csv_files(str(candidates_dir))
    except FileNotFoundError:
        print(f"No boilerplate data found in {candidates_dir}")
        return

    print(f"Loaded {len(combined_df)} rows")

    validation_set = combined_df.copy()

    # Filter nulls
    if 'trial_boilerplate_text' in validation_set.columns:
        validation_set = validation_set[~validation_set.trial_boilerplate_text.isnull()]

    print(f"Unique trial spaces: {validation_set.this_space.nunique()}")
    print(f"Unique patients: {validation_set.patient_summary.nunique()}")

    # Run inference if requested
    if run_inference and model_path:
        validation_set = run_boilerplate_checker_inference(
            validation_set, model_path, device='cuda', batch_size=batch_size
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate_path = output_dir / "boilerplate_trial_centric_with_predictions.csv"
        validation_set.to_csv(intermediate_path, index=False)
        print(f"Saved predictions to: {intermediate_path}")
    elif 'prediction_score' not in validation_set.columns:
        precomputed_path = output_dir / "boilerplate_trial_centric_with_predictions.csv"
        if precomputed_path.exists():
            print(f"Loading pre-computed predictions from: {precomputed_path}")
            validation_set = pd.read_csv(precomputed_path)
        else:
            print("No predictions available. Use --run-inference to generate them.")
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute classification metrics
    label_col = 'exclusion_result'
    if label_col not in validation_set.columns:
        print(f"Warning: {label_col} column not found")
        return

    if 'prediction_score' in validation_set.columns:
        print("\n--- Classification Metrics ---")
        print(f"Total samples: {len(validation_set)}")

        auc = roc_auc_score(validation_set[label_col], validation_set.prediction_score)
        print(f"AUC: {auc:.4f}")

        pdf_path = output_dir / "boilerplate_checker_trial_centric_classification.pdf"
        eval_model(
            validation_set.prediction_score.values,
            validation_set[label_col].values,
            pdf_path=str(pdf_path),
            title_prefix="Boilerplate Checker Trial-Centric"
        )

        if 'prediction_label' in validation_set.columns:
            actual_labels = np.where(
                validation_set[label_col] == 0.0, 'NEGATIVE', 'POSITIVE'
            )
            kappa = cohen_kappa_score(actual_labels, validation_set.prediction_label)
            print(f"Cohen's Kappa: {kappa:.4f}")

    print(f"\nEvaluation complete. Reports saved to: {output_dir}")


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.mode == "patient_centric":
        evaluate_patient_centric(
            data_dir, output_dir,
            model_path=args.model_path,
            gpu=args.gpu,
            run_inference=args.run_inference,
            batch_size=args.batch_size
        )
    else:
        evaluate_trial_centric(
            data_dir, output_dir,
            model_path=args.model_path,
            gpu=args.gpu,
            run_inference=args.run_inference,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
