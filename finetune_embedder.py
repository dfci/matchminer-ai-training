#!/usr/bin/env python3
import argparse
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401  (kept to match your original deps)

PROMPT_PREFIX = (
    "Instruct: Given a cancer patient summary, retrieve clinical trial options that are "
    "reasonable for that patient; or, given a clinical trial option, retrieve cancer "
    "patients who are reasonable candidates for that trial. "
)
DEFAULT_BASE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_INPUT = "space_specific_eligibility_checks.parquet"
MAX_TOKENS = 2500


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a text embedding model for trial–patient retrieval."
    )
    # Allow multiple -i flags; also supports glob patterns
    parser.add_argument(
        "-i", "--input-parquet",
        action="append",
        help="Path(s) to input parquet file(s). May be provided multiple times "
             "and may include glob patterns. If omitted, defaults to "
             f"'{DEFAULT_INPUT}'."
    )
    parser.add_argument(
        "-c", "--ckpt-dir",
        default="./initial_embedder_training",
        help="Checkpoint/output directory for the SentenceTransformer trainer."
    )
    parser.add_argument(
        "-o", "--output-model",
        default="pt_trial_summary_pertrial_finetuned.model",
        help="Directory name to save the final fine-tuned model (directory will be created)."
    )
    parser.add_argument(
        "-m", "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Starting model name or local path for tokenizer and SentenceTransformer."
    )
    return parser.parse_args()


def expand_inputs(input_args):
    """Expand a list of input paths (which may include globs) to a concrete file list."""
    if not input_args:
        input_args = [DEFAULT_INPUT]
    files = []
    for pattern in input_args:
        expanded = sorted(glob.glob(pattern)) if any(ch in pattern for ch in "*?[]") else [pattern]
        files.extend(expanded)
    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    if not unique_files:
        raise FileNotFoundError("No input parquet files found after expanding patterns.")
    return unique_files


def load_and_concat(parquet_files):
    """Load multiple parquet files and concatenate them."""
    frames = []
    for p in parquet_files:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Input parquet not found: {p}")
        df = pd.read_parquet(p)
        print(f"[INFO] Loaded {p} with shape {df.shape}")
        frames.append(df)
    if len(frames) == 1:
        return frames[0]
    concat_df = pd.concat(frames, axis=0, ignore_index=True)
    print(f"[INFO] Concatenated {len(frames)} files -> shape {concat_df.shape}")
    return concat_df


def main():
    args = parse_args()
    input_files = expand_inputs(args.input_parquet)
    print(f"[INFO] Using {len(input_files)} input file(s):")
    for f in input_files:
        print(f"       - {f}")
    print(f"[INFO] Checkpoint dir: {args.ckpt_dir}")
    print(f"[INFO] Output model dir: {args.output_model}")
    print(f"[INFO] Base model: {args.base_model}")

    # Deferred imports
    from sentence_transformers import (
        SentenceTransformer, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    )
    from datasets import Dataset
    from transformers import AutoTokenizer

    # --- Load & filter data ---
    trial_checks = load_and_concat(input_files)

    # Optional filter if present
    if "patient_long_text" in trial_checks.columns:
        trial_checks = trial_checks[trial_checks.patient_long_text.fillna("") != ""]

    req_cols = {"patient_summary", "this_space", "eligibility_result"}
    missing = req_cols - set(trial_checks.columns)
    if missing:
        raise ValueError(f"Input parquet missing required columns: {missing}")

    trial_checks = trial_checks[
        ~trial_checks.patient_summary.fillna("").str.contains(
            "No cancer|no cancer|No primary|No evidence of malignancy", na=False
        )
    ]
    trial_checks = trial_checks[
        ~trial_checks.patient_summary.fillna("").str.startswith("No information")
    ]

    # remove leading digit and period and space from trial space if present
    trial_checks["this_space"] = trial_checks['this_space'].str.replace(r'^\s*\d+\.', '', regex=True)
    
    print("\n[INFO] Dataframe info after filtering:")
    trial_checks.info()
    print("\n[INFO] eligibility_result counts:")
    print(trial_checks.eligibility_result.value_counts(dropna=False))

    # --- Tokenizer & truncation (from the base model) ---
    try:
        tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.base_model)

    def truncate(text: str, max_tokens: int = MAX_TOKENS) -> str:
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)
        return tok.decode(
            tok.encode(text, add_special_tokens=True, truncation=True, max_length=max_tokens),
            skip_special_tokens=True
        )

    trial_checks["patient_summary_trunc"] = PROMPT_PREFIX + trial_checks["patient_summary"].map(truncate)
    trial_checks["this_space_trunc"] = PROMPT_PREFIX + trial_checks["this_space"].map(truncate)

    eligible = trial_checks[trial_checks.eligibility_result == 1]
    print("\n[INFO] Eligible subset info:")
    eligible.info()

    # --- Model (from the base model) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Using device: {device}")
    model = SentenceTransformer(args.base_model, device=device)

    # Set prompt hook if supported (Qwen embedders expose `prompts`)
    if hasattr(model, "prompts") and isinstance(getattr(model, "prompts"), dict):
        model.prompts["query"] = PROMPT_PREFIX

    model.max_seq_length = MAX_TOKENS

    # --- Datasets ---
    mnri_dataset = Dataset.from_pandas(
        eligible[["patient_summary_trunc", "this_space_trunc"]],
        preserve_index=False
    )
    contrastive_dataset = Dataset.from_pandas(
        trial_checks[["patient_summary_trunc", "this_space_trunc", "eligibility_result"]]
        .rename(columns={"eligibility_result": "label"}),
        preserve_index=False
    )
    train_dataset = {
        "mnri_dataset": mnri_dataset,
        "contrastive_dataset": contrastive_dataset,
    }

    # --- Losses ---
    mll_train_loss = losses.MultipleNegativesRankingLoss(model=model)
    contrastive_train_loss = losses.OnlineContrastiveLoss(model=model)
    losses_map = {
        "mnri_dataset": mll_train_loss,
        "contrastive_dataset": contrastive_train_loss,
    }

    # --- Trainer ---
    args_st = SentenceTransformerTrainingArguments(
        output_dir=args.ckpt_dir,
        per_device_train_batch_size=18,
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.01,
        save_strategy="steps",
        save_steps=2500,
        save_total_limit=2,
        logging_steps=100,
        num_train_epochs=3,
        bf16=True if torch.cuda.is_available() else False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args_st,
        train_dataset=train_dataset,
        loss=losses_map,
    )

    # Check if checkpoint directory exists and has checkpoints
    checkpoint_exists = False
    if os.path.exists(args.ckpt_dir):
        checkpoints = [d for d in os.listdir(args.ckpt_dir) if d.startswith("checkpoint-")]
        checkpoint_exists = len(checkpoints) > 0
    
    print("\n[INFO] Starting training...")
    if checkpoint_exists:
        print(f"[INFO] Resuming from checkpoint in {args.ckpt_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("[INFO] Starting fresh training")
        trainer.train()
    print("\n[INFO] Training complete.")

    # --- Save final model ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output_model)), exist_ok=True)
    model.save(args.output_model)
    print(f"[INFO] Saved fine-tuned model to: {args.output_model}")


if __name__ == "__main__":
    main()
