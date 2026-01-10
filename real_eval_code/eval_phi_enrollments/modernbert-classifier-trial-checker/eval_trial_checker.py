#!/usr/bin/env python3
"""
Evaluate ModernBERT trial checker classifier performance.

This script evaluates the ModernBERT-based trial eligibility classifier by:
- Running inference on patient-trial pairs
- Computing classification metrics (AUC, F1, precision-recall)
- Computing ranking metrics (MAP@K)
- Generating PDF reports

Usage:
    python eval_trial_checker.py --mode patient_centric --data-dir /path/to/data --output-dir /path/to/output
    python eval_trial_checker.py --mode trial_centric --data-dir /path/to/data --output-dir /path/to/output
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
    average_precision_at_k,
    generate_ranking_report,
    load_and_combine_csv_files
)
from sklearn.metrics import roc_auc_score, cohen_kappa_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ModernBERT trial checker classifier"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["patient_centric", "trial_centric"],
                        help="Evaluation mode")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing candidate CSV files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save evaluation outputs")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trial checker model")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device to use")
    parser.add_argument("--k", type=int, default=20,
                        help="K value for MAP@K calculation (default: 20)")
    parser.add_argument("--run-inference", action="store_true",
                        help="Run model inference (requires GPU)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference")
    return parser.parse_args()


def run_trial_checker_inference(df: pd.DataFrame, model_path: str,
                                 device: str = "cuda", batch_size: int = 32) -> pd.DataFrame:
    """
    Run trial checker model inference on patient-trial pairs.

    Args:
        df: DataFrame with patient_summary and this_space columns
        model_path: Path to the trial checker model
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
        max_length=4096,
        device=device,
        batch_size=batch_size
    )

    # Create patient-trial pair text
    df = df.copy()
    df['pt_trial_pair'] = (
        df['this_space'] +
        "\nNow here is the patient summary:" +
        df['patient_summary']
    )
    df = df[~df.pt_trial_pair.isnull()]

    print(f"Running inference on {len(df)} samples...")
    predictions = pipe(df.pt_trial_pair.tolist())

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
                              k: int = 20, run_inference: bool = False,
                              batch_size: int = 32):
    """Evaluate patient-centric trial checker performance."""
    print("=" * 60)
    print("EVALUATING PATIENT-CENTRIC TRIAL CHECKER")
    print("=" * 60)

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Load candidate files
    candidates_dir = data_dir / "spaces_for_patient_checks"
    if not candidates_dir.exists():
        candidates_dir = data_dir / "shards_patient_centric"

    print(f"Loading data from: {candidates_dir}")

    try:
        combined_df = load_and_combine_csv_files(str(candidates_dir))
    except FileNotFoundError:
        consolidated_path = data_dir / "patient_centric_candidates.csv"
        if consolidated_path.exists():
            print(f"Loading consolidated file: {consolidated_path}")
            combined_df = pd.read_csv(consolidated_path)
        else:
            print(f"No data found")
            return

    print(f"Loaded {len(combined_df)} rows")

    # Filter out null patient summaries
    combined_df = combined_df[~combined_df.patient_summary.isnull()]

    validation_set = combined_df.copy()
    print(f"Unique trial spaces: {validation_set.this_space.nunique()}")
    print(f"Unique patients: {validation_set.patient_summary.nunique()}")

    # Run inference if requested
    if run_inference and model_path:
        validation_set = run_trial_checker_inference(
            validation_set, model_path, device='cuda', batch_size=batch_size
        )
        # Save intermediate results
        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate_path = output_dir / "patient_centric_with_predictions.csv"
        validation_set.to_csv(intermediate_path, index=False)
        print(f"Saved predictions to: {intermediate_path}")
    elif 'prediction_score' not in validation_set.columns:
        # Try loading pre-computed predictions
        precomputed_path = output_dir / "patient_centric_with_predictions.csv"
        if precomputed_path.exists():
            print(f"Loading pre-computed predictions from: {precomputed_path}")
            validation_set = pd.read_csv(precomputed_path)
        else:
            print("No predictions available. Use --run-inference to generate them.")
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute classification metrics
    if 'prediction_score' in validation_set.columns and 'eligibility_result' in validation_set.columns:
        print("\n--- Classification Metrics ---")
        auc = roc_auc_score(validation_set.eligibility_result, validation_set.prediction_score)
        print(f"AUC: {auc:.4f}")

        # Generate classification PDF report
        pdf_path = output_dir / "trial_checker_patient_centric_classification.pdf"
        eval_model(
            validation_set.prediction_logit.values,
            validation_set.eligibility_result.values,
            pdf_path=str(pdf_path),
            title_prefix="Trial Checker Patient-Centric"
        )

        # Cohen's kappa
        if 'prediction_label' in validation_set.columns:
            actual_labels = np.where(
                validation_set.eligibility_result == 0.0, 'NEGATIVE', 'POSITIVE'
            )
            kappa = cohen_kappa_score(actual_labels, validation_set.prediction_label)
            print(f"Cohen's Kappa: {kappa:.4f}")

            # Crosstab
            print("\nCrosstab (actual vs predicted):")
            print(pd.crosstab(validation_set.eligibility_result, validation_set.prediction_label))

    # Compute ranking metrics after filtering to positive predictions
    if 'prediction_label' in validation_set.columns:
        print("\n--- Ranking Metrics (after filtering to POSITIVE predictions) ---")

        # Take top k per patient first, then filter to positive
        pruned_set = validation_set.groupby('patient_summary').head(k)
        pruned_set = pruned_set[pruned_set.prediction_label == 'POSITIVE']

        print(f"Samples after filtering: {len(pruned_set)}")
        print(f"Median results per patient: {pruned_set.groupby('patient_summary').size().median():.1f}")
        print(f"Mean results per patient: {pruned_set.groupby('patient_summary').size().mean():.1f}")
        print(f"Unique patients: {pruned_set.patient_summary.nunique()}")
        print(f"Unique trial spaces: {pruned_set.this_space.nunique()}")

        if 'eligibility_result' in pruned_set.columns:
            # Calculate MAP@K on filtered set
            print(f"\nPositive rate after filtering: {pruned_set.eligibility_result.mean():.4f}")

            temp = pruned_set.groupby('patient_summary').eligibility_result.apply(
                lambda x: average_precision_at_k(x.head(k).values)
            )
            map_k = temp.mean()
            print(f"MAP@{k} (after trial checker): {map_k:.4f}")

            # Generate ranking PDF report
            pdf_path = output_dir / "trial_checker_patient_centric_ranking.pdf"
            generate_ranking_report(
                pruned_set,
                group_col='patient_summary',
                label_col='eligibility_result',
                pdf_path=str(pdf_path),
                title_prefix="Trial Checker Patient-Centric (Filtered)",
                k=k
            )

    print(f"\nEvaluation complete. Reports saved to: {output_dir}")


def evaluate_trial_centric(data_dir: Path, output_dir: Path,
                            model_path: str = None, gpu: str = "0",
                            k: int = 20, run_inference: bool = False,
                            batch_size: int = 32):
    """Evaluate trial-centric trial checker performance."""
    print("=" * 60)
    print("EVALUATING TRIAL-CENTRIC TRIAL CHECKER")
    print("=" * 60)

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # Load candidate files
    candidates_dir = data_dir / "patients_for_spaces_checks"
    if not candidates_dir.exists():
        candidates_dir = data_dir / "shards_trial_centric"

    print(f"Loading data from: {candidates_dir}")

    try:
        combined_df = load_and_combine_csv_files(str(candidates_dir))
    except FileNotFoundError:
        consolidated_path = data_dir / "trial_centric_candidates.csv"
        if consolidated_path.exists():
            print(f"Loading consolidated file: {consolidated_path}")
            combined_df = pd.read_csv(consolidated_path)
        else:
            print(f"No data found")
            return

    print(f"Loaded {len(combined_df)} rows")

    # Filter out null patient summaries
    combined_df = combined_df[~combined_df.patient_summary.isnull()]

    validation_set = combined_df.copy()
    print(f"Unique trial spaces: {validation_set.this_space.nunique()}")
    print(f"Unique patients: {validation_set.patient_summary.nunique()}")

    # Run inference if requested
    if run_inference and model_path:
        validation_set = run_trial_checker_inference(
            validation_set, model_path, device='cuda', batch_size=batch_size
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate_path = output_dir / "trial_centric_with_predictions.csv"
        validation_set.to_csv(intermediate_path, index=False)
        print(f"Saved predictions to: {intermediate_path}")
    elif 'prediction_score' not in validation_set.columns:
        precomputed_path = output_dir / "trial_centric_with_predictions.csv"
        if precomputed_path.exists():
            print(f"Loading pre-computed predictions from: {precomputed_path}")
            validation_set = pd.read_csv(precomputed_path)
        else:
            print("No predictions available. Use --run-inference to generate them.")
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute classification metrics
    if 'prediction_score' in validation_set.columns and 'eligibility_result' in validation_set.columns:
        print("\n--- Classification Metrics ---")
        auc = roc_auc_score(validation_set.eligibility_result, validation_set.prediction_score)
        print(f"AUC: {auc:.4f}")

        pdf_path = output_dir / "trial_checker_trial_centric_classification.pdf"
        eval_model(
            validation_set.prediction_logit.values,
            validation_set.eligibility_result.values,
            pdf_path=str(pdf_path),
            title_prefix="Trial Checker Trial-Centric"
        )

        if 'prediction_label' in validation_set.columns:
            actual_labels = np.where(
                validation_set.eligibility_result == 0.0, 'NEGATIVE', 'POSITIVE'
            )
            kappa = cohen_kappa_score(actual_labels, validation_set.prediction_label)
            print(f"Cohen's Kappa: {kappa:.4f}")

            print("\nCrosstab (actual vs predicted):")
            print(pd.crosstab(validation_set.eligibility_result, validation_set.prediction_label))

    # Compute ranking metrics (grouped by trial space instead of patient)
    if 'prediction_label' in validation_set.columns:
        print("\n--- Ranking Metrics (after filtering to POSITIVE predictions) ---")

        pruned_set = validation_set[validation_set.prediction_label == 'POSITIVE']

        print(f"Samples after filtering: {len(pruned_set)}")
        print(f"Median results per trial: {pruned_set.groupby('this_space').size().median():.1f}")
        print(f"Mean results per trial: {pruned_set.groupby('this_space').size().mean():.1f}")
        print(f"Unique patients: {pruned_set.patient_summary.nunique()}")
        print(f"Unique trial spaces: {pruned_set.this_space.nunique()}")

        if 'eligibility_result' in pruned_set.columns:
            print(f"\nPositive rate after filtering: {pruned_set.eligibility_result.mean():.4f}")

            # For trial-centric, group by trial space
            temp = pruned_set.groupby('this_space').eligibility_result.apply(
                lambda x: average_precision_at_k(x.head(k).values)
            )
            map_k = temp.mean()
            print(f"MAP@{k} (after trial checker): {map_k:.4f}")

            pdf_path = output_dir / "trial_checker_trial_centric_ranking.pdf"
            generate_ranking_report(
                pruned_set,
                group_col='this_space',
                label_col='eligibility_result',
                pdf_path=str(pdf_path),
                title_prefix="Trial Checker Trial-Centric (Filtered)",
                k=k
            )

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
            k=args.k,
            run_inference=args.run_inference,
            batch_size=args.batch_size
        )
    else:
        evaluate_trial_centric(
            data_dir, output_dir,
            model_path=args.model_path,
            gpu=args.gpu,
            k=args.k,
            run_inference=args.run_inference,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
