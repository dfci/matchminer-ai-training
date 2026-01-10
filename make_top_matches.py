#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel retrieval (patients->trials and trials->patients) with SentenceTransformers
- Each GPU loads the model and processes a shard
- Samples trials at the trial level, ranks at the space level, outputs spaces
- For patients->trials: samples N trials per patient, ranks their spaces, returns top-K spaces
- For trials->patients: samples N patients per trial space, ranks them, returns top-K patients
- Rejoins shards to produce the same output files (now configurable via CLI):
    - default: top_cohorts_tocheck_round1.parquet
    - default: top_patients_tocheck_round1.parquet

Example:
python make_top_matches.py \
  --parquet trial_specific_eligibility_checks.parquet \
  --model pt_trial_summary_pertrial_finetuned.model \
  --gpus 0,1,2,3,4,5,6,7 \
  --sample_trials_per_patient 500 \
  --sample_patients_per_trial 20000 \
  --top_k_spaces 20 \
  --top_k_patients 40 \
  --encode_batch_size 128 \
  --score_batch_size 2048 \
  --max_seq_length 2500 \
  --out_cohorts_parquet top_cohorts_tocheck_round1.parquet \
  --out_patients_parquet top_patients_tocheck_round1.parquet
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor
from sentence_transformers import SentenceTransformer

# -------------------------
# Helpers
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        default="trial_specific_eligibility_checks.parquet",
        help="Input parquet containing columns: patient_summary, patient_boilerplate_text, this_space, nct_id, eligibility_result",
    )
    ap.add_argument("--model", default="pt_trial_summary_pertrial_finetuned.model")
    ap.add_argument(
        "--gpus", default="0", help="Comma-separated CUDA device indices, e.g. '0,1,2,3'"
    )
    ap.add_argument(
        "--sample_trials_per_patient",
        type=int,
        required=True,
        help="Number of trials to randomly sample for consideration per patient",
    )
    ap.add_argument(
        "--sample_patients_per_trial",
        type=int,
        required=True,
        help="Number of patients to randomly sample for consideration per trial space",
    )
    ap.add_argument(
        "--top_k_spaces",
        type=int,
        default=10,
        help="Top spaces per patient for patients->trials ranking (from sampled trials)",
    )
    ap.add_argument(
        "--top_k_patients",
        type=int,
        default=20,
        help="Top patients per trial space for trials->patients ranking (from sampled patients)",
    )
    ap.add_argument(
        "--encode_batch_size", type=int, default=256, help="Batch size for encoding texts"
    )
    ap.add_argument(
        "--score_batch_size",
        type=int,
        default=2048,
        help="Batch size for similarity/topk scoring per worker",
    )
    ap.add_argument("--max_seq_length", type=int, default=1500)
    ap.add_argument(
        "--query_prompt",
        default=(
            "Instruct: Given a cancer patient summary, retrieve clinical trial options "
            "that are reasonable for that patient; or, given a clinical trial option, "
            "retrieve cancer patients who are reasonable candidates for that trial."
        ),
    )
    ap.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for sampling reproducibility",
    )
    # Output filenames
    ap.add_argument(
        "--out_cohorts_parquet",
        default="top_cohorts_tocheck_round1.parquet",
        help="Output parquet for patients->trials results",
    )
    ap.add_argument(
        "--out_patients_parquet",
        default="top_patients_tocheck_round1.parquet",
        help="Output parquet for trials->patients results",
    )
    return ap.parse_args()


def chunk_ranges(n_items: int, n_chunks: int):
    """Return [(start, end), ...] covering range(n_items) in ~equal chunks."""
    base = n_items // n_chunks
    rem = n_items % n_chunks
    ranges = []
    start = 0
    for i in range(n_chunks):
        add = base + (1 if i < rem else 0)
        end = start + add
        ranges.append((start, end))
        start = end
    return ranges


def _encode_worker(texts, device_str, model_path, encode_batch_size, max_seq_length, query_prompt):
    """Runs in a subprocess on one GPU; returns a float32 numpy (N, D)."""
    torch.cuda.set_device(int(device_str.split(":")[-1]))
    model = SentenceTransformer(model_path, trust_remote_code=True, device=device_str)
    # Apply same prompt / length as original
    try:
        model.prompts["query"] = query_prompt
    except Exception:
        pass
    try:
        model.max_seq_length = max_seq_length
    except Exception:
        pass

    with torch.no_grad():
        embs = model.encode(
            texts,
            batch_size=encode_batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,  # makes dot product == cosine similarity
            show_progress_bar=False,
            prompt="query",
        ).cpu().to(dtype=torch.float32).numpy()
    return embs


def parallel_encode(all_texts, gpu_ids, model_path, encode_batch_size, max_seq_length, query_prompt):
    """Encode a list of texts across GPUs; returns (N,D) float32 numpy aligned to original order."""
    if len(all_texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    n = len(all_texts)
    shards = chunk_ranges(n, len(gpu_ids))
    futures = []
    outputs = [None] * len(shards)
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as ex:
        for wi, (s, e) in enumerate(shards):
            if s == e:
                outputs[wi] = np.zeros((0, 0), dtype=np.float32)
                continue
            texts_slice = all_texts[s:e]
            fut = ex.submit(
                _encode_worker,
                texts_slice,
                f"cuda:{gpu_ids[wi]}",
                model_path,
                encode_batch_size,
                max_seq_length,
                query_prompt,
            )
            futures.append((wi, s, e, fut))
        for wi, s, e, fut in futures:
            outputs[wi] = fut.result()

    # Concatenate in-order
    embs = []
    for (s, e), arr in zip(shards, outputs):
        if s == e:
            continue
        embs.append(arr)
    embs = np.concatenate(embs, axis=0) if embs else np.zeros((0, 0), dtype=np.float32)
    assert embs.shape[0] == n, f"Encoded rows {embs.shape[0]} != expected {n}"
    return embs


def _topk_with_sampling_worker(query_emb_batch, all_ref_embs, sample_indices_batch, topk, device_str):
    """
    query_emb_batch: (B, D) float32 numpy
    all_ref_embs: (M, D) float32 numpy
    sample_indices_batch: list of B arrays, each containing indices to sample from all_ref_embs
    topk: int
    
    Returns: list of B arrays, each containing topk indices into the ORIGINAL all_ref_embs space
    """
    torch.cuda.set_device(int(device_str.split(":")[-1]))
    device = torch.device(device_str)
    
    results = []
    with torch.no_grad():
        q_tensor = torch.from_numpy(query_emb_batch).to(device, non_blocking=True)
        all_ref_tensor = torch.from_numpy(all_ref_embs).to(device, non_blocking=True)
        
        for i, sample_indices in enumerate(sample_indices_batch):
            # Get the sampled reference embeddings
            if len(sample_indices) == 0:
                results.append(np.array([], dtype=np.int64))
                continue
                
            sampled_refs = all_ref_tensor[sample_indices]  # (n_sampled, D)
            query_vec = q_tensor[i:i+1]  # (1, D)
            
            # Compute similarities
            sims = query_vec @ sampled_refs.t()  # (1, n_sampled)
            
            # Get topk from sampled set
            k_actual = min(topk, len(sample_indices))
            _, topk_in_sample = torch.topk(sims, k=k_actual, dim=1, largest=True, sorted=True)
            
            # Map back to original indices
            topk_in_sample_np = topk_in_sample.cpu().numpy()[0]
            original_indices = sample_indices[topk_in_sample_np]
            results.append(original_indices)
    
    return results


def parallel_topk_with_sampling(all_query_embs, all_ref_embs, sample_indices_per_query, 
                                 gpu_ids, topk, score_batch_size):
    """
    Splits queries across GPUs with per-query sampling.
    
    all_query_embs: (N_queries, D) float32 numpy
    all_ref_embs: (M, D) float32 numpy
    sample_indices_per_query: list of N_queries arrays, each containing indices to sample
    topk: int
    
    Returns: list of N_queries arrays, each containing topk indices into all_ref_embs
    """
    n = all_query_embs.shape[0]
    if n == 0:
        return []

    ranges = chunk_ranges(n, len(gpu_ids))
    results = [None] * n

    def batches_for_range(start, end, bsz):
        for s in range(start, end, bsz):
            e = min(s + bsz, end)
            yield s, e

    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as ex:
        futs = []
        for wi, (rg_s, rg_e) in enumerate(ranges):
            if rg_s == rg_e:
                continue
            device_str = f"cuda:{gpu_ids[wi]}"
            for bs, be in batches_for_range(rg_s, rg_e, score_batch_size):
                fut = ex.submit(
                    _topk_with_sampling_worker,
                    all_query_embs[bs:be, :],
                    all_ref_embs,
                    sample_indices_per_query[bs:be],
                    topk,
                    device_str,
                )
                futs.append((bs, be, fut))

        for bs, be, fut in futs:
            batch_results = fut.result()
            for i, result in enumerate(batch_results):
                results[bs + i] = result

    return results


# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
    if len(gpu_ids) == 0:
        raise ValueError("No GPUs specified. Use --gpus like '0,1'.")

    # Set random seed
    np.random.seed(args.random_seed)

    # Ensure output directories exist
    for out_path in [args.out_cohorts_parquet, args.out_patients_parquet]:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    # Load & filter
    df = pd.read_parquet(args.parquet)
    df = df[~df.patient_summary.isnull()]
    df = df[~df.this_space.isnull()]
    df = df[~df.nct_id.isnull()]

    df['this_space'] = df['this_space'].str.replace(r'^\s*\d+\.', '', regex=True)


    # Deduplicate to get unique patients
    patients = df.groupby("patient_summary", as_index=False).first()[
        ["patient_summary", "patient_boilerplate_text"]
    ].copy()
    
    # Get all unique trials (with their spaces)
    trials_full = df[["nct_id", "this_space","trial_boilerplate_text"]].drop_duplicates().copy()
    
    # Get unique trial spaces for encoding
    trials_spaces = df.groupby("this_space", as_index=False).first()[["nct_id", "this_space", "trial_boilerplate_text"]].copy()

    # Encode patients and trial spaces across GPUs
    patient_texts = patients["patient_summary"].astype(str).tolist()
    this_spaces = trials_spaces["this_space"].astype(str).tolist()

    print(f"Encoding {len(patient_texts)} patients...")
    patient_embs = parallel_encode(
        patient_texts,
        gpu_ids=gpu_ids,
        model_path=args.model,
        encode_batch_size=args.encode_batch_size,
        max_seq_length=args.max_seq_length,
        query_prompt=args.query_prompt,
    )  # (P, D)

    print(f"Encoding {len(this_spaces)} trial spaces...")
    trial_space_embs = parallel_encode(
        this_spaces,
        gpu_ids=gpu_ids,
        model_path=args.model,
        encode_batch_size=args.encode_batch_size,
        max_seq_length=args.max_seq_length,
        query_prompt=args.query_prompt,
    )  # (S, D)

    # Build mapping from space to space_index
    space_to_idx = {space: idx for idx, space in enumerate(trials_spaces["this_space"])}

    # -------------------------
    # Patients -> Trials (sample trials, rank spaces, return top spaces)
    # -------------------------
    print(f"\nProcessing patients -> spaces...")
    print(f"  Sampling {args.sample_trials_per_patient} trials per patient")
    print(f"  Ranking their unique spaces and returning top {args.top_k_spaces} spaces")
    
    # For each patient, sample trials and get their unique spaces
    sample_space_indices_per_patient = []
    n_patients = len(patients)
    n_all_trials = len(trials_full)
    
    for p_idx in range(n_patients):
        # Randomly sample trial indices
        sample_size = min(args.sample_trials_per_patient, n_all_trials)
        sampled_trial_indices = np.random.choice(n_all_trials, size=sample_size, replace=False)
        
        # Get the unique spaces for these sampled trials
        sampled_trials = trials_full.iloc[sampled_trial_indices]
        sampled_spaces = sampled_trials["this_space"].unique()
        
        # Convert to space indices
        space_indices = np.array([space_to_idx[space] for space in sampled_spaces])
        sample_space_indices_per_patient.append(space_indices)
    
    # Run parallel topk with sampling
    pts_to_spaces_results = parallel_topk_with_sampling(
        all_query_embs=patient_embs,
        all_ref_embs=trial_space_embs,
        sample_indices_per_query=sample_space_indices_per_patient,
        gpu_ids=gpu_ids,
        topk=args.top_k_spaces,
        score_batch_size=args.score_batch_size,
    )

    # Build output dataframe (patient + space pairs)
    out_rows = []
    for p_idx, top_space_indices in enumerate(pts_to_spaces_results):
        if len(top_space_indices) == 0:
            continue
            
        patient_row = patients.iloc[p_idx]
        
        # Get the top-ranked spaces
        for space_idx in top_space_indices:
            space_row = trials_spaces.iloc[space_idx]
            out_rows.append({
                "patient_summary": patient_row["patient_summary"],
                "patient_boilerplate_text": patient_row["patient_boilerplate_text"],
                "nct_id": space_row["nct_id"],
                "this_space": space_row["this_space"],
                "trial_boilerplate_text": space_row["trial_boilerplate_text"]
            })

    pts_to_spaces_df = pd.DataFrame(out_rows) if out_rows else pd.DataFrame()
    if not pts_to_spaces_df.empty:
        pts_to_spaces_df["patient_summary"] = pts_to_spaces_df["patient_summary"].str.strip()
        pts_to_spaces_df["split"] = "train"
    pts_to_spaces_df.to_parquet(args.out_cohorts_parquet, index=False)

    # -------------------------
    # Trials -> Patients (sample patients, rank them, return top patients)
    # -------------------------
    print(f"\nProcessing trial spaces -> patients...")
    print(f"  Sampling {args.sample_patients_per_trial} patients per trial space")
    print(f"  Ranking them and returning top {args.top_k_patients} patients")
    
    # For each trial space, sample patients
    sample_patient_indices_per_space = []
    n_spaces = len(trials_spaces)
    n_all_patients = len(patients)
    
    for s_idx in range(n_spaces):
        # Randomly sample patient indices
        sample_size = min(args.sample_patients_per_trial, n_all_patients)
        sampled_patient_indices = np.random.choice(n_all_patients, size=sample_size, replace=False)
        sample_patient_indices_per_space.append(sampled_patient_indices)
    
    # Run parallel topk with sampling
    spaces_to_pts_results = parallel_topk_with_sampling(
        all_query_embs=trial_space_embs,
        all_ref_embs=patient_embs,
        sample_indices_per_query=sample_patient_indices_per_space,
        gpu_ids=gpu_ids,
        topk=args.top_k_patients,
        score_batch_size=args.score_batch_size,
    )

    # Build output dataframe (space + patient pairs)
    out_rows = []
    for s_idx, top_patient_indices in enumerate(spaces_to_pts_results):
        if len(top_patient_indices) == 0:
            continue
            
        space_row = trials_spaces.iloc[s_idx]
        
        # Get the top-ranked patients
        for p_idx in top_patient_indices:
            patient_row = patients.iloc[p_idx]
            out_rows.append({
                "patient_summary": patient_row["patient_summary"],
                "patient_boilerplate_text": patient_row["patient_boilerplate_text"],
                "nct_id": space_row["nct_id"],
                "this_space": space_row["this_space"],
                "trial_boilerplate_text": space_row["trial_boilerplate_text"]
            })

    spaces_to_pts_df = pd.DataFrame(out_rows) if out_rows else pd.DataFrame()
    if not spaces_to_pts_df.empty:
        spaces_to_pts_df["patient_summary"] = spaces_to_pts_df["patient_summary"].str.strip()
        spaces_to_pts_df["split"] = "train"
    spaces_to_pts_df.to_parquet(args.out_patients_parquet, index=False)

    print("\nWrote:")
    print(f"  {args.out_cohorts_parquet} ({len(pts_to_spaces_df)} rows)")
    print(f"  {args.out_patients_parquet} ({len(spaces_to_pts_df)} rows)")


if __name__ == "__main__":
    # Use 'spawn' to avoid CUDA re-init issues in subprocesses
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
