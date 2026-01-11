#!/usr/bin/env python3
"""
Trial-centric patient retrieval for SOC evaluation.
For each trial space, finds the top N matching patients.

Uses embedding similarity to match trial space descriptions to patient summaries.

Usage:
    python trial_centric_retrieval.py --gpu 0 [--top-k 40]
"""

import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

# Repo root for default paths
REPO_ROOT = Path(__file__).resolve().parents[3]  # matchminer-ai-training


def parse_args():
    parser = argparse.ArgumentParser(description="Trial-centric patient retrieval")
    parser.add_argument("--gpu", type=str, required=True,
                        help="GPU ID to use (e.g., '0')")
    parser.add_argument("--top-k", type=int, default=40,
                        help="Number of top matching patients per space")
    parser.add_argument("--patient-summaries", type=str, default=None,
                        help="Input parquet with patient summaries (default: data/phi_soc/patient_summaries.parquet)")
    parser.add_argument("--trial-spaces", type=str, default=None,
                        help="Input CSV with trial spaces (default: data/phi_soc/sample_spaces.csv)")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output CSV file (default: data/phi_soc/trial_centric_candidates.csv)")
    parser.add_argument("--embedding-model", type=str, default=None,
                        help="Path to sentence transformer model (default: models/trialspace)")
    parser.add_argument("--shard-dir", type=str, default=None,
                        help="Directory for checkpoint shards (default: data/phi_soc/shards_trial_centric)")
    parser.add_argument("--shard-size", type=int, default=100,
                        help="Number of spaces per shard")
    return parser.parse_args()


def get_completed_shards(shard_dir):
    """Get list of completed shard indices."""
    pattern = os.path.join(shard_dir, "shard_*.csv")
    files = glob.glob(pattern)
    indices = set()
    for f in files:
        try:
            idx = int(os.path.basename(f).split("_")[1].replace(".csv", ""))
            indices.add(idx)
        except:
            pass
    return indices


def consolidate_shards(shard_dir, output_file):
    """Consolidate all shards into a single file."""
    pattern = os.path.join(shard_dir, "shard_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No shards found")
        return None

    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, axis=0).reset_index(drop=True)
    combined.to_csv(output_file, index=False)
    print(f"Consolidated {len(files)} shards into {output_file}")
    return combined


def main():
    args = parse_args()

    # Set defaults relative to repo root
    if args.patient_summaries is None:
        args.patient_summaries = str(REPO_ROOT.parent / "data/phi_soc/patient_summaries.parquet")
    if args.trial_spaces is None:
        args.trial_spaces = str(REPO_ROOT.parent / "data/phi_soc/sample_spaces.csv")
    if args.output_file is None:
        args.output_file = str(REPO_ROOT.parent / "data/phi_soc/trial_centric_candidates.csv")
    if args.embedding_model is None:
        args.embedding_model = str(REPO_ROOT.parent / "models/trialspace")
    if args.shard_dir is None:
        args.shard_dir = str(REPO_ROOT.parent / "data/phi_soc/shards_trial_centric")

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    from sentence_transformers import SentenceTransformer

    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Load patient summaries
    print("Loading patient summaries...")
    patients = pd.read_parquet(args.patient_summaries)
    patients = patients[~patients.patient_summary.isnull()]
    print(f"Loaded {patients.shape[0]} patient records")

    # Load trial spaces
    print("Loading trial spaces...")
    spaces = pd.read_csv(args.trial_spaces)
    print(f"Loaded {spaces.shape[0]} trial space records")

    # Prepare patient summaries dataframe - deduplicate
    patient_summaries = patients.groupby('patient_summary').first().reset_index()[
        ['dfci_mrn', 'patient_summary', 'patient_boilerplate_text', 'split']
    ]

    # Prepare spaces dataframe - deduplicate
    spaces = spaces.groupby(['nct_id', 'this_space']).first().reset_index()
    if 'trial_boilerplate_text' not in spaces.columns:
        spaces['trial_boilerplate_text'] = None
    spaces = spaces[['nct_id', 'this_space', 'trial_boilerplate_text']]
    print(f"Unique spaces: {spaces.shape[0]}")

    # Load embedding model
    print(f"Loading embedding model from {args.embedding_model}...")
    embedding_model = SentenceTransformer(
        args.embedding_model, trust_remote_code=True, device='cuda'
    )
    embedding_model.prompts['query'] = (
        "Instruct: Given a cancer patient summary, retrieve clinical trial options "
        "that are reasonable for that patient; or, given a clinical trial option, "
        "retrieve cancer patients who are reasonable candidates for that trial. "
    )
    embedding_model.max_seq_length = 2500

    # Compute embeddings
    print("Computing patient embeddings...")
    with torch.no_grad():
        patient_embeddings = embedding_model.encode(
            patient_summaries.patient_summary.tolist(),
            convert_to_tensor=True,
            prompt='query'
        )

    print("Computing space embeddings...")
    with torch.no_grad():
        space_embeddings = embedding_model.encode(
            spaces.this_space.tolist(),
            convert_to_tensor=True,
            prompt='query'
        )

    # Process spaces with sharding
    print("Matching spaces to patients...")
    completed_shards = get_completed_shards(str(shard_dir))
    total_spaces = spaces.shape[0]

    for shard_idx in range(0, total_spaces, args.shard_size):
        shard_num = shard_idx // args.shard_size
        if shard_num in completed_shards:
            print(f"Shard {shard_num} already complete, skipping...")
            continue

        end_idx = min(shard_idx + args.shard_size, total_spaces)
        print(f"Processing shard {shard_num} (spaces {shard_idx} to {end_idx})...")

        output_list = []
        for i in range(shard_idx, end_idx):
            space_embedding = space_embeddings[i, :]

            # Compute similarities to all patients
            similarities = F.cosine_similarity(space_embedding.unsqueeze(0), patient_embeddings)
            sorted_similarities, sorted_indices = torch.sort(similarities, descending=True)

            top_k = min(args.top_k, len(patient_summaries))
            top_indices = sorted_indices[0:top_k].cpu().numpy()

            relevant_patients = patient_summaries.iloc[top_indices]

            output = pd.DataFrame({
                'dfci_mrn': relevant_patients.dfci_mrn.values,
                'split': relevant_patients.split.values,
                'patient_summary': relevant_patients.patient_summary.values,
                'patient_boilerplate_text': relevant_patients.patient_boilerplate_text.values,
                'this_space': spaces.iloc[i].this_space,
                'trial_boilerplate_text': spaces.iloc[i].trial_boilerplate_text
            })
            output_list.append(output)

        if output_list:
            shard_df = pd.concat(output_list, axis=0).reset_index(drop=True)
            shard_df['patient_summary'] = shard_df.patient_summary.str.strip()
            shard_file = shard_dir / f"shard_{shard_num}.csv"
            shard_df.to_csv(str(shard_file), index=False)
            print(f"Saved shard {shard_num} ({len(shard_df)} records)")

    # Consolidate shards
    print("Consolidating shards...")
    consolidate_shards(str(shard_dir), args.output_file)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
