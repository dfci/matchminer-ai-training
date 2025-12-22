#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-GPU parallel extractor for patient long notes using a HF text-classification pipeline.

Usage (example):
python 5_make_patient_longnotes.py \
  --input_parquet all_synthetic_notes.parquet \
  --spaces_csv trial_spaces_lineitems.csv \
  --model_id ./auto-tiny-bert-tagger-with-pretraining \
  --max_length 128 \
  --prob_threshold 0.5 \
  --batch_size 1024 \
  --out_parquet useful_trial_enrollments_with_longnotes.parquet \
  --tmp_dir ./longnotes_shards \
  --gpus 0,1,2,3,4,5,6,7

Notes:
- By default, uses *all visible GPUs* if --gpus is omitted.
- Creates shard files in --tmp_dir/worker{K}.parquet and then concatenates them.
"""

import os
import re
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import List, Tuple

# Torch/HF imports inside worker too; but importing here helps fail fast if missing.
import torch
from transformers import AutoTokenizer, pipeline


def parse_args():
    ap = argparse.ArgumentParser("Multi-GPU patient long-note extractor")
    ap.add_argument("--input_parquet", required=True, help="Path to input notes parquet")
    ap.add_argument("--spaces_csv", default=None, help="Optional spaces CSV (merged at end on space_index)")
    ap.add_argument("--model_id", default="ksg-dfci/TinyBertOncoTagger-0825")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--prob_threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--gpus", default=None, help="Comma-separated GPU ids, e.g. '0,1,2,3'. Default: all visible.")
    ap.add_argument("--tmp_dir", default="./longnotes_shards")
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--start_date_min", default="2010-01-01")
    ap.add_argument("--start_date_max", default="2021-12-31")
    return ap.parse_args()


# ---------- Utility helpers ----------

def get_gpu_ids(arg: str | None) -> List[int]:
    if arg:
        return [int(x.strip()) for x in arg.split(",") if x.strip() != ""]
    cnt = torch.cuda.device_count()
    if cnt == 0:
        raise RuntimeError("No CUDA GPUs visible. Set --gpus or ensure CUDA is available.")
    return list(range(cnt))


def chunked_iterable(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


_CLEAN_RE_1 = re.compile(r"[\n\r]+")
_CLEAN_RE_2 = re.compile(r"\s+")
# We'll keep a close analogue to your original sentence split:
def split_into_excerpts(text: str) -> List[str]:
    t = _CLEAN_RE_1.sub(" ", (text or "").strip())
    t = _CLEAN_RE_2.sub(" ", t)
    if not t:
        return []
    # emulate: chunks = "<excerpt break>" + re.sub("\\. ", "<excerpt break>", t) + "<excerpt break>"
    t2 = t.replace(". ", "<excerpt break>")
    parts = [p.strip() for p in t2.split("<excerpt break>") if p.strip()]
    return parts


def deterministic_rng_for_mrn(mrn: int) -> np.random.RandomState:
    # reproducible per MRN
    seed = (int(mrn) * 2654435761) & 0xFFFFFFFF
    return np.random.RandomState(seed)


def assign_fake_dates(n_rows: int, rng: np.random.RandomState, start_min: str, start_max: str) -> pd.Series:
    start_dates = pd.date_range(start=start_min, end=start_max, freq="D")
    if len(start_dates) == 0:
        raise ValueError("Invalid start_date range")
    # One base date per patient row, then cumulative random gaps
    base = pd.to_datetime([rng.choice(start_dates)] * n_rows)
    gaps = np.cumsum(rng.randint(0, 180, size=n_rows))
    return base + pd.to_timedelta(gaps, unit='D')


# ---------- Core per-worker routine ----------

def worker_run(
    gpu_global_id: int,
    worker_idx: int,
    input_path: str,
    mrn_subset: List[int],
    model_id: str,
    max_length: int,
    prob_threshold: float,
    batch_size: int,
    start_date_min: str,
    start_date_max: str,
    shard_path: str
):
    # Isolate one GPU per worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_global_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = 0  # after setting CUDA_VISIBLE_DEVICES, local index is 0
    # Load small classifier model/tokenizer on this GPU
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    clf = pipeline(
        "text-classification",
        model=model_id,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        padding="max_length"
    )

    # Read only needed columns to reduce IO/mem
    cols = None  # load all; adjust if you know exact column names
    all_reports = pd.read_parquet(input_path, columns=cols)
    all_reports = all_reports.copy()

    # Ensure columns match downstream expectations
    if "event_type" in all_reports.columns and "note_type" not in all_reports.columns:
        all_reports = all_reports.rename(columns={"event_type": "note_type"})
    if "synthetic_note" in all_reports.columns and "text" not in all_reports.columns:
        all_reports = all_reports.rename(columns={"synthetic_note": "text"})

    # Filter to this worker's MRNs
    all_reports = all_reports[all_reports["pseudo_mrn"].isin(mrn_subset)].reset_index(drop=True)
    if all_reports.empty:
        # Write empty shard to keep downstream simple
        pd.DataFrame().to_parquet(shard_path, index=False)
        return

    # Build excerpt table for ALL rows (vectorized) to maximize batching
    # Per MRN random date series
    def process_patient_group(df_patient: pd.DataFrame) -> pd.DataFrame:
        dfp = df_patient.copy()
        rng = deterministic_rng_for_mrn(int(dfp["pseudo_mrn"].iloc[0]))
        dfp["date"] = assign_fake_dates(dfp.shape[0], rng, start_date_min, start_date_max)
        # explode excerpts
        # build one big list, then explode
        excerpts = []
        note_types = []
        dates = []
        mrns = []
        idxs = dfp.index.tolist()

        for i in idxs:
            exs = split_into_excerpts(str(dfp.at[i, "text"]))
            if not exs:
                continue
            n = len(exs)
            excerpts.extend(exs)
            note_types.extend([dfp.at[i, "note_type"]] * n)
            dates.extend([dfp.at[i, "date"]] * n)
            mrns.extend([dfp.at[i, "pseudo_mrn"]] * n)

        if not excerpts:
            return pd.DataFrame(columns=["pseudo_mrn","date","note_type","excerpt"])

        out = pd.DataFrame({
            "pseudo_mrn": mrns,
            "date": dates,
            "note_type": note_types,
            "excerpt": excerpts
        })
        # drop dup excerpts within patient to match your behavior
        out = out.drop_duplicates(subset=["excerpt"], keep="first")
        return out

    # Build per-patient excerpts, concat, then score in big batches
    ex_frames = []
    for mrn, g in all_reports.groupby("pseudo_mrn"):
        ex_frames.append(process_patient_group(g))
    if len(ex_frames) == 0:
        pd.DataFrame().to_parquet(shard_path, index=False)
        return
    excerpts_df = pd.concat(ex_frames, axis=0, ignore_index=True)
    if excerpts_df.empty:
        pd.DataFrame().to_parquet(shard_path, index=False)
        return

    # Run classifier in large batches on this GPU
    texts = excerpts_df["excerpt"].tolist()
    predict_frame_list = []

    with torch.inference_mode():
        for batch in chunked_iterable(texts, batch_size):
            preds = clf(
                batch,
                batch_size=len(batch),  # inner batch handled by pipeline; we pass exact size for throughput
                truncation=True,
                padding=True,
                max_length=max_length,
                top_k=None
            )
            # pipeline returns list of dicts: {'label': 'NEGATIVE'|'POSITIVE', 'score': float}
            prediction_output = pd.DataFrame([x[1] for x in preds])
            predict_frame_list.append(prediction_output)

    prediction_frame = pd.concat(predict_frame_list, ignore_index=True)
    excerpts_df["label"] = prediction_frame['label']
    excerpts_df["score"] = prediction_frame['score']
    # Convert to positive probability as in your original code
    excerpts_df["positive_prob"] = np.where(excerpts_df["label"] == "NEGATIVE", 1.0 - excerpts_df["score"], excerpts_df["score"])

    # Threshold and group to build per-date/note_type strings, then per-patient long text
    keep = excerpts_df[excerpts_df["positive_prob"] > prob_threshold].copy()
    if keep.empty:
        # Still write shard structure with columns expected downstream
        shard = pd.DataFrame(columns=list(all_reports.columns) + ["patient_long_text"])
        shard.to_parquet(shard_path, index=False)
        return

    keep_grouped = (
        keep.groupby(["pseudo_mrn", "date", "note_type"])["excerpt"]
            .agg(". ".join)
            .reset_index()
    )
    keep_grouped["date_text"] = keep_grouped["date"].astype(str) + " " + keep_grouped["note_type"] + " " + keep_grouped["excerpt"]

    per_patient_text = (
        keep_grouped.groupby("pseudo_mrn")["date_text"]
            .agg("\n".join)
            .reset_index()
            .rename(columns={"date_text": "patient_long_text"})
    )

    # Join with one "enrollment" row per patient (take first row per mrn like your script)
    first_rows = (
        all_reports.sort_index()
        .groupby("pseudo_mrn")
        .head(1)
        .reset_index(drop=True)
    )
    shard = pd.merge(first_rows, per_patient_text, on="pseudo_mrn", how="left")
    shard.to_parquet(shard_path, index=False)


def main():
    args = parse_args()
    os.makedirs(args.tmp_dir, exist_ok=True)

    # Pre-read to get MRNs & keep memory small here (drop big text columns not needed for partitioning)
    input_df = pd.read_parquet(args.input_parquet, columns=["pseudo_mrn"])
    all_mrns = input_df["pseudo_mrn"].unique().tolist()

    gpu_ids = get_gpu_ids(args.gpus)
    n_gpus = len(gpu_ids)
    if n_gpus < 1:
        raise RuntimeError("No GPUs selected/visible.")

    # Partition MRNs roughly evenly
    mrn_slices = []
    per = math.ceil(len(all_mrns) / n_gpus)
    for i in range(n_gpus):
        mrn_slices.append(all_mrns[i * per : (i + 1) * per])

    # Use spawn (CUDA-safe)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set, that's fine.
        pass

    jobs = []
    with mp.Pool(processes=n_gpus) as pool:
        for widx, (gpu, mrns) in enumerate(zip(gpu_ids, mrn_slices)):
            shard_path = os.path.join(args.tmp_dir, f"worker{widx}.parquet")
            jobs.append(
                pool.apply_async(
                    worker_run,
                    kwds=dict(
                        gpu_global_id=gpu,
                        worker_idx=widx,
                        input_path=args.input_parquet,
                        mrn_subset=mrns,
                        model_id=args.model_id,
                        max_length=args.max_length,
                        prob_threshold=args.prob_threshold,
                        batch_size=args.batch_size,
                        start_date_min=args.start_date_min,
                        start_date_max=args.start_date_max,
                        shard_path=shard_path,
                    ),
                )
            )
        # Wait for all
        for j in jobs:
            j.get()

    # Concat shards
    shard_files = [os.path.join(args.tmp_dir, f) for f in os.listdir(args.tmp_dir) if f.endswith(".parquet")]
    if not shard_files:
        warnings.warn("No shard files produced; writing empty output.")
        pd.DataFrame().to_parquet(args.out_parquet, index=False)
        return

    parts = []
    for sp in shard_files:
        try:
            df = pd.read_parquet(sp)
            if not df.empty:
                parts.append(df)
        except Exception as e:
            warnings.warn(f"Failed reading shard {sp}: {e}")

    if len(parts) == 0:
        warnings.warn("All shards empty or unreadable; writing empty output.")
        pd.DataFrame().to_parquet(args.out_parquet, index=False)
        return

    long_notes_enrollments = pd.concat(parts, axis=0, ignore_index=True)

    # Optional merge with spaces CSV if provided and column exists
    if args.spaces_csv is not None and os.path.exists(args.spaces_csv):
        spaces = pd.read_csv(args.spaces_csv)
        spaces["trial_boilerplate_text"] = spaces["trial_boilerplate_text"].fillna("")
        spaces["criteria"] = spaces["this_space"] + "\nNo history of:\n" + spaces["trial_boilerplate_text"]
        spaces["criteria_index"] = spaces.index
        # If your enrollment rows already have 'space_index', this will work:
        if "space_index" in long_notes_enrollments.columns:
            long_notes_enrollments = pd.merge(
                long_notes_enrollments,
                spaces.rename(columns={"criteria_index": "space_index"}),
                on="space_index",
                how="left"
            )
        else:
            warnings.warn("spaces_csv provided but 'space_index' not found in enrollment rows; skipping merge.")

    # Persist final
    long_notes_enrollments.to_parquet(args.out_parquet, index=False)
    print(f"Wrote {args.out_parquet} with shape {long_notes_enrollments.shape}")


if __name__ == "__main__":
    main()