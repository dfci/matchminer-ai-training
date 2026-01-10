#!/usr/bin/env python3
"""
Combines individual parquet training files into a single dataset,
balances by replicating underrepresented data, shuffles, and pre-tokenizes
for LLM fine-tuning.
"""

import os
import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for fine-tuning")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="../../data/oncoreasoning_training_data",
        help="Directory containing input parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../data/oncoreasoning_training_data",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Tokenizer model name",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=13000,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=32,
        help="Number of processes for tokenization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Skip combining step if all_training_data.parquet already exists",
    )
    parser.add_argument(
        "--writer-batch-size",
        type=int,
        default=1000,
        help="Writer batch size for tokenization to reduce memory usage",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    combined_parquet_path = os.path.join(output_dir, "all_training_data.parquet")

    # Check if we can skip combining
    if args.skip_combine and os.path.exists(combined_parquet_path):
        print(f"Skipping combine step, loading existing {combined_parquet_path}")
    else:
        # Find all parquet files in input directory (excluding output files)
        parquet_files = [
            f for f in os.listdir(input_dir)
            if f.endswith(".parquet") and f not in ("all_training_data.parquet",)
        ]
        print(f"Found {len(parquet_files)} parquet files: {parquet_files}")

        # Load all parquet files
        frames = []
        for f in parquet_files:
            filepath = os.path.join(input_dir, f)
            df = pd.read_parquet(filepath)
            print(f"Loaded {f}: {len(df)} rows")

            # Balance: replicate spacified_trial 10x since it's much smaller
            if "spacified_trial" in f:
                df = pd.concat([df] * 10, ignore_index=True)
                print(f"  -> Replicated 10x to {len(df)} rows for balancing")

            frames.append(df)

        # Combine all frames
        print("\nCombining all dataframes...")
        combined = pd.concat(frames, ignore_index=True)
        print(f"Combined dataset: {len(combined)} rows")

        # Keep only the 'text' column (drop 'label' if present)
        if "label" in combined.columns:
            combined = combined[["text"]]
            print("Dropped 'label' column, keeping only 'text'")

        # Shuffle the dataset
        print(f"\nShuffling with seed={args.seed}...")
        shuffled = combined.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        # Free memory
        del combined, frames

        # Save combined parquet
        print(f"\nSaving combined parquet to {combined_parquet_path}...")
        shuffled.to_parquet(combined_parquet_path)
        print(f"Saved {len(shuffled)} rows")

        # Free memory
        del shuffled

    # Load from parquet directly (more memory efficient)
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset from {combined_parquet_path}...")
    hf_ds = Dataset.from_parquet(combined_parquet_path)
    print(f"Loaded {len(hf_ds)} examples")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
        )

    print(f"Tokenizing with max_length={args.max_length}, num_proc={args.num_proc}, writer_batch_size={args.writer_batch_size}...")
    tokenized_dataset = hf_ds.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_proc,
        writer_batch_size=args.writer_batch_size,
    )

    # Save tokenized dataset
    tokenized_path = os.path.join(output_dir, "tokenized_training_data.dataset")
    print(f"\nSaving tokenized dataset to {tokenized_path}...")
    tokenized_dataset.save_to_disk(tokenized_path)

    print(f"\nDone! Final dataset has {len(tokenized_dataset)} examples")
    print(f"  - Combined parquet: {combined_parquet_path}")
    print(f"  - Tokenized dataset: {tokenized_path}")


if __name__ == "__main__":
    main()
