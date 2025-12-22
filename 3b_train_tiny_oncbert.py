#!/usr/bin/env python3
# train_mlm_from_tables.py
"""
Fine-tune a tiny BERT (Masked LM) from tabular files (CSV/Parquet) and per-file text columns.

- Base model: prajjwal1/bert-tiny (override with --model_name)
- Sequence length: 512 total tokens ([CLS] + 510 chunk + [SEP])
- For texts >512: take a random middle chunk (start jittered around the center) of length 510

Examples:
accelerate launch train_tiny_oncbert.py \
    --data dfci_trial_space_lineitems_9-24-25.csv:trial_text \
    dfci_trial_space_lineitems_9-24-25.csv:this_space \
    dfci_trial_space_lineitems_9-24-25.csv:trial_boilerplate_text \
    ./synthetic_notes/synthetic_notes.parquet:synthetic_note \
    ./synthetic_negative_notes/synthetic_notes.parquet:synthetic_note \
    --output_dir ./onc_bert_tiny_mlm_512
"""


"""
Fine-tune a tiny BERT (MLM) from tabular files (CSV/Parquet) where each row in a specified text column
is a training document.


Base model: prajjwal1/bert-tiny (override with --model_name)
Masked LM: DataCollatorForLanguageModeling (mlm_probability=0.15)

Multi-GPU: Launch with `accelerate launch ...` (Trainer uses Accelerate under the hood).
"""

import argparse
import os
import json
import re
import random
from typing import List, Tuple, Dict, Iterable

import numpy as np
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune BERT-tiny MLM from CSV/Parquet with dynamic chunking and drug-perturbation.")
    # I/O spec: either --data (path:col pairs) OR --files + --text_cols (parallel lists)
    p.add_argument("--data", nargs="*", default=None, help="List like path:column. Example: notes.parquet:text_col")
    p.add_argument("--files", nargs="*", default=None, help="Alternative to --data: list of files.")
    p.add_argument("--text_cols", nargs="*", default=None, help="Alternative to --data: list of text columns (same length as --files).")

    # Note: Each input dataframe must have a 'split' column with 'train' and 'val' values

    # Model / training
    p.add_argument("--model_name", type=str, default="prajjwal1/bert-tiny")
    p.add_argument("--output_dir", type=str, default="./bert_tiny_mlm_dynamic")
    p.add_argument("--max_length", type=int, default=512, help="Total sequence length incl. special tokens.")
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine",
                   choices=["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"])
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_strategy", type=str, default="epoch", choices=["steps","epoch","no"])
    p.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["steps","epoch","no"])
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--num_proc", type=int, default=max(1, (os.cpu_count() or 1) // 2), help="Workers for map() when building DS from pandas.")

    return p.parse_args()

def _parse_pairs(pairs: List[str]) -> List[Tuple[str, str]]:
    out = []
    for item in pairs or []:
        if ":" not in item:
            raise ValueError(f"--data entry must be path:column, got: {item}")
        path, col = item.split(":", 1)
        out.append((path, col))
    return out

def _resolve_io(args) -> List[Tuple[str,str]]:
    if args.data:
        train_pairs = _parse_pairs(args.data)
    elif args.files and args.text_cols:
        if len(args.files) != len(args.text_cols):
            raise ValueError("--files and --text_cols must have the same length.")
        train_pairs = list(zip(args.files, args.text_cols))
    else:
        raise ValueError("Specify either --data path:col ... OR both --files and --text_cols.")
    return train_pairs

def _load_one(path: str, col: str, split_filter: str = None) -> pd.DataFrame:
    """Load a file and extract text column and split column.
    
    Args:
        path: Path to CSV or Parquet file
        col: Name of text column to extract
        split_filter: If provided, only return rows where split column equals this value
    
    Returns:
        DataFrame with 'text' and 'split' columns
    """
    lower = path.lower()
    if lower.endswith((".parquet", ".parq")):
        df = pd.read_parquet(path)
    elif lower.endswith((".csv", ".csv.gz")):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type for {path}. Use CSV or Parquet.")
    
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {path}. Available: {list(df.columns)[:10]}...")
    
    if 'split' not in df.columns:
        raise KeyError(f"Column 'split' not found in {path}. Available: {list(df.columns)[:10]}...")
    
    # Filter by split if requested
    if split_filter is not None:
        df = df[df['split'] == split_filter].copy()
    
    # Extract text column and clean
    df['text'] = df[col].fillna('').astype(str)
    df = df[df['text'].str.strip() != '']
    
    return df[['text', 'split']]

def load_dataframe(pairs: List[Tuple[str,str]], split_filter: str = None) -> pd.DataFrame:
    """Load and concatenate multiple file:column pairs, optionally filtering by split.
    
    Args:
        pairs: List of (filepath, column_name) tuples
        split_filter: If provided, only load rows where split column equals this value
    
    Returns:
        DataFrame with 'text' column only
    """
    parts = []
    for path, col in pairs:
        df = _load_one(path, col, split_filter=split_filter)
        parts.append(df)
    
    combo = pd.concat(parts, axis=0, ignore_index=True)
    # Return only text column since split filtering is already done
    return pd.DataFrame({"text": combo['text']})

def build_datasets(args) -> DatasetDict:
    """Build train and validation datasets from input files using 'split' column."""
    pairs = _resolve_io(args)
    
    # Load train split
    train_df = load_dataframe(pairs, split_filter='train')
    print(f"Loaded {len(train_df)} training examples")
    
    # Load validation split
    val_df = load_dataframe(pairs, split_filter='val')
    print(f"Loaded {len(val_df)} validation examples")
    
    if len(val_df) == 0:
        raise ValueError("No validation examples found. Ensure your data has rows with split='val'")
    
    ds = DatasetDict(
        train=Dataset.from_pandas(train_df, preserve_index=False),
        validation=Dataset.from_pandas(val_df, preserve_index=False),
    )
    return ds

# ------------------------ Dynamic collator (re-chunk + mask + drug-perturb) ------------------------

class DynamicChunkMaskCollator:
    """
    A collator that:
      1) Optionally perturbs oncology drug *generic* names to brand names (per occurrence, p=drug_perturb_prob).
      2) Tokenizes WITHOUT specials, chooses a fresh random middle chunk per example to fit `eff_len`,
         then adds [CLS]/[SEP] and pads to `max_length`.
      3) Delegates to HuggingFace's DataCollatorForLanguageModeling for masking.

    This yields different chunks each epoch/batch (re-chunking), and independent drug-perturb decisions.
    """
    def __init__(
        self,
        tokenizer,
        max_length: int,
        mlm_probability: float,
        seed: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eff_len = max_length - 2
        self.inner_mlm = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)
        self.rng = random.Random(seed if seed is not None else 0xC0FFEE)
        self.numpy_rng = np.random.default_rng(seed if seed is not None else None)

    def _random_middle_start(self, n: int) -> int:
        if n <= self.eff_len:
            return 0
        max_start = n - self.eff_len
        center = max_start // 2
        radius = max(1, int(0.25 * max_start))  # ±25% jitter around the valid center
        start = int(np.clip(center + self.numpy_rng.integers(-radius, radius + 1), 0, max_start))
        return start

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        # 1) Drug perturbation at raw text level
        texts = [ex['text'] for ex in examples]

        # 2) Tokenize (no specials) then make fixed-length sequences with fresh random chunk each call
        enc = self.tokenizer(texts, add_special_tokens=False, truncation=False)
        input_ids_batch: List[List[int]] = []
        attn_masks_batch: List[List[int]] = []
        token_type_ids_batch: List[List[int]] = []

        pad_id = self.tokenizer.pad_token_id
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        for ids in enc["input_ids"]:
            start = self._random_middle_start(len(ids))
            mid = ids[start : start + self.eff_len]
            seq = [cls_id] + mid + [sep_id]
            # Right-pad to max_length
            deficit = self.max_length - len(seq)
            if deficit > 0:
                seq = seq + [pad_id] * deficit
            mask = [1] * (self.max_length - deficit) + ([0] * deficit if deficit > 0 else [])
            tt = [0] * self.max_length
            input_ids_batch.append(seq)
            attn_masks_batch.append(mask)
            token_type_ids_batch.append(tt)

        # 3) Apply MLM masking
        batch = [
            {
                "input_ids": torch.tensor(input_ids_batch[i], dtype=torch.long),
                "attention_mask": torch.tensor(attn_masks_batch[i], dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids_batch[i], dtype=torch.long),
            }
            for i in range(len(input_ids_batch))
        ]
        masked = self.inner_mlm(batch)  # adds 'labels'
        return masked

def main():
    args = parse_args()
    set_seed(args.seed)

    # Build raw datasets (just a single "text" column)
    ds = build_datasets(args)

    # Tokenizer / model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, model_max_length=args.max_length)
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name, config=config)

    # Dynamic collator (handles re-chunking and drug-perturbation) + masking
    data_collator = DynamicChunkMaskCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
	seed=args.seed,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        fp16=args.fp16,
        bf16=args.bf16,
        # Length grouping not critical since we force fixed length; leave True/False per your preference.
        group_by_length=False,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
