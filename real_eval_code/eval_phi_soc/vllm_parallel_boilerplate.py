#!/usr/bin/env python3
"""
Parallelized vLLM inference script for patient boilerplate exclusion checking.

Evaluates whether patients have underlying medical conditions that would
exclude them from specific clinical trials based on boilerplate criteria.

Supports:
- Multi-GPU parallelization with configurable tensor parallelism
- Resumable execution via output shards
- Configurable sampling parameters


python vllm_parallel_boilerplate.py \
    --model ksg-dfci/OncoReasoning-3B-1225 \
    --input-file ../top_cohorts_tocheck_only_open_phi.csv \
    --output-file ./boilerplate_results.csv \
    --temp-dir ./boilerplate_shards \
    --gpus 0,1,2,3,4,5,6,7 \
    --tensor-parallel-size 1 \
    --temperature 1.0 \
    --top-p 0.9 \ 
    --n-samples 100 
"""

import argparse
import os
import sys
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run parallelized vLLM inference for boilerplate exclusion checking"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model or HuggingFace model ID",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input CSV file with patient_boilerplate_text and trial_boilerplate_text columns",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to final merged output CSV file",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        required=True,
        help="Directory for temporary output shards (enables resumability)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs per vLLM instance (tensor parallelism)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default 1.0 for boilerplate task)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples to generate per prompt",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Number of prompts per batch",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=7500,
        help="Maximum tokens to generate per response",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=10000,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization fraction for vLLM",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty for generation",
    )
    parser.add_argument(
        "--filter-val",
        action="store_true",
        help="Filter to only validation split rows",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to process (for testing)",
    )
    return parser.parse_args()


def get_shard_filename(temp_dir: str, batch_start: int, batch_end: int) -> str:
    """Generate consistent shard filename."""
    return os.path.join(temp_dir, f"shard_{batch_start:08d}_{batch_end:08d}.csv")


def get_completed_shards(temp_dir: str) -> set:
    """Get set of (batch_start, batch_end) tuples for completed shards."""
    completed = set()
    if not os.path.exists(temp_dir):
        return completed
    
    pattern = re.compile(r"shard_(\d+)_(\d+)\.csv")
    for filename in os.listdir(temp_dir):
        match = pattern.match(filename)
        if match:
            batch_start = int(match.group(1))
            batch_end = int(match.group(2))
            # Verify file is not empty/corrupted
            filepath = os.path.join(temp_dir, filename)
            try:
                df = pd.read_csv(filepath, nrows=1)
                if len(df) > 0:
                    completed.add((batch_start, batch_end))
            except Exception:
                pass  # File is corrupted, will be reprocessed
    
    return completed


def worker_process(
    worker_id: int,
    gpu_ids: list,
    batches: list,
    df: pd.DataFrame,
    args: argparse.Namespace,
):
    """
    Worker process that runs vLLM inference on assigned batches.
    
    Args:
        worker_id: Unique worker identifier
        gpu_ids: List of GPU IDs assigned to this worker
        batches: List of (batch_start, batch_end) tuples to process
        df: The full dataframe
        args: Command line arguments
    """
    # Set CUDA_VISIBLE_DEVICES for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    # Import vLLM after setting CUDA_VISIBLE_DEVICES
    from vllm import LLM, SamplingParams
    
    print(f"[Worker {worker_id}] Starting on GPUs {gpu_ids}, processing {len(batches)} batches")
    
    # Initialize vLLM model
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    
    tokenizer = llm.get_tokenizer()
    
    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    
    for batch_start, batch_end in batches:
        shard_path = get_shard_filename(args.temp_dir, batch_start, batch_end)
        
        # Double-check shard doesn't exist (race condition protection)
        if os.path.exists(shard_path):
            try:
                test_df = pd.read_csv(shard_path, nrows=1)
                if len(test_df) > 0:
                    print(f"[Worker {worker_id}] Skipping batch {batch_start}-{batch_end} (already exists)")
                    continue
            except Exception:
                pass
        
        print(f"[Worker {worker_id}] Processing batch {batch_start}-{batch_end}")
        
        batch_df = df.iloc[batch_start:batch_end].copy()
        
        # Build prompts
        prompts = []
        for _, row in batch_df.iterrows():
            patient_boilerplate = row["_patient_bp_str"]
            trial_boilerplate = row["_trial_bp_str"]
            
            messages = [
                {"role": "system", "content": "Reasoning: high"},
                {"role": "user", "content": (
                    "You are a brilliant oncologist with encyclopedic knowledge about cancer and its treatment.\n"
                    "Your job is to evaluate whether a patient has any underlying medical conditions that would exclude him or her from a specific clinical trial.\n\n"
                    f"Here is an extract of the patient's history:\n{patient_boilerplate}\n"
                    f"Here are the exclusion criteria for the trial:\n{trial_boilerplate}\n"
                    "Note that the extract was generated by prompting an LLM to determine whether the patient meets specific common exclusion criteria, "
                    "such as uncontrolled brain metastases, lack of measurable disease, congestive heart failure, pneumonitis, renal dysfunction, "
                    "liver dysfunction, and HIV or hepatitis infection, and to present evidence for whether the patient met the criterion.\n"
                    "You should therefore not assume that mention of such condition means the patient has the condition; it may represent the LLM reasoning "
                    "about whether the patient has the condition.\n"
                    "Based on the extract, you should determine whether the patient clearly meets one of the exclusion criteria for this specific trial.\n"
                    "Do not evaluate exclusion criteria other than those listed for this trial.\n"
                    "Reason through one exclusion criterion at a time. Generate a numbered list of the criteria as you go. For each one, decide whether the patient clearly "
                    "meets the exclusion criteron. If it is not completely clear that the patient meets the exclusion criterion, give the patient the benefit of the doubt, "
                    "and err on the side of deciding the patient is not excluded. A description in the patient extract that a condition is mild, low-grade, or resolved is even "
                    "more of a reason not to exclude the patient based on that condition.\n"
                    'Once you have evaluated all exclusion criteria, answer the question "Is this patient clearly excluded from this trial?" with a one-word "Yes!" or "No!" answer, '
                    "based on whether the patient clearly met any of the individual exclusion criteria. It is critical that your final word be either \"Yes!\" or \"No!\", verbatim, and case-sensitive.\n"
                    "Make sure to include the exclamation point in your final one-word answer.\n"
                    "No introductory text or concluding text after that final answer."
                )}
            ]
            
            prompt = tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=False
            )
            prompts.append(prompt)
        
        # Run inference
        responses = llm.generate(prompts, sampling_params)
        
        # Process results
        results = []
        batch_df_reset = batch_df.reset_index(drop=True)
        
        for prompt_id, request_output in enumerate(responses):
            original_idx = batch_start + prompt_id
            row_data = batch_df_reset.iloc[prompt_id]
            
            for completion_output in request_output.outputs:
                response_text = completion_output.text
                response_id = completion_output.index
                
                # Parse exclusion result
                if ("Yes!" in response_text[-10:]) or ("YES!" in response_text[-10:]):
                    exclusion_result = 1.0
                else:
                    exclusion_result = 0.0
                
                result = {
                    "prompt_id": original_idx,
                    "response_id": response_id,
                    "llm_boilerplate_response": response_text,
                    "exclusion_result": exclusion_result,
                    # Include dedup keys for later joining back to original rows
                    "_patient_bp_str": row_data["_patient_bp_str"],
                    "_trial_bp_str": row_data["_trial_bp_str"],
                }
                results.append(result)
        
        # Save shard
        results_df = pd.DataFrame(results)
        results_df.to_csv(shard_path, index=False)
        print(f"[Worker {worker_id}] Saved batch {batch_start}-{batch_end} ({len(results_df)} rows)")
    
    print(f"[Worker {worker_id}] Finished all batches")
    return worker_id


def merge_shards(temp_dir: str, output_file: str):
    """Merge all shards and map back to original rows."""
    shard_files = sorted(Path(temp_dir).glob("shard_*.csv"))
    
    if not shard_files:
        print("No shards found to merge!")
        return
    
    print(f"Merging {len(shard_files)} shards...")
    
    dfs = []
    for shard_file in shard_files:
        try:
            df = pd.read_csv(shard_file)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {shard_file}: {e}")
    
    if not dfs:
        print("No valid shards to merge!")
        return
    
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Merged shards: {len(merged_df)} rows (from deduplicated inference)")
    
    # Check if we have the original mapping file for expansion
    mapping_file = os.path.join(temp_dir, "_original_mapping.csv")
    if os.path.exists(mapping_file):
        print("Expanding results back to all original rows...")
        
        # Load the original mapping
        original_mapping = pd.read_csv(mapping_file, index_col=0)
        
        # The merged_df has _patient_bp_str and _trial_bp_str columns
        # We need to join the inference results back to all original rows
        
        # Select only the inference result columns (not the original data columns that were duplicated)
        result_cols = ["_patient_bp_str", "_trial_bp_str", "response_id", 
                       "llm_boilerplate_response", "exclusion_result"]
        results_only = merged_df[result_cols].copy()
        
        # Join back to original mapping to expand to all original rows
        original_mapping = original_mapping.reset_index()
        original_mapping.columns = ["original_row_idx", "_patient_bp_str", "_trial_bp_str"]
        
        expanded_df = original_mapping.merge(
            results_only,
            on=["_patient_bp_str", "_trial_bp_str"],
            how="left"
        )
        
        # Clean up temp columns
        expanded_df = expanded_df.drop(columns=["_patient_bp_str", "_trial_bp_str"])
        expanded_df = expanded_df.rename(columns={"original_row_idx": "prompt_id"})
        
        print(f"Expanded to {len(expanded_df)} rows")
        expanded_df.to_csv(output_file, index=False)
        print(f"Final output saved to {output_file}")
    else:
        # No deduplication was done, just save merged results
        merged_df.to_csv(output_file, index=False)
        print(f"Merged output saved to {output_file} ({len(merged_df)} total rows)")


def main():
    args = parse_args()
    
    # Parse GPU list
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    num_gpus = len(gpu_ids)
    tp_size = args.tensor_parallel_size
    
    if num_gpus % tp_size != 0:
        print(f"Error: Number of GPUs ({num_gpus}) must be divisible by tensor_parallel_size ({tp_size})")
        sys.exit(1)
    
    num_workers = num_gpus // tp_size
    print(f"Configuration: {num_gpus} GPUs, tensor_parallel_size={tp_size}, {num_workers} parallel workers")
    
    # Create temp directory
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Load input data
    print(f"Loading input file: {args.input_file}")
    df_original = pd.read_csv(args.input_file)
    print(f"Loaded {len(df_original)} rows")
    
    # Apply filters
    if args.filter_val:
        if "split" in df_original.columns:
            df_original = df_original[df_original["split"].str.contains("val", na=False)]
            print(f"After filtering to validation split: {len(df_original)} rows")
    
    if args.max_rows is not None:
        df_original = df_original.head(args.max_rows)
        print(f"Limited to {len(df_original)} rows")
    
    # Reset index for consistent indexing
    df_original = df_original.reset_index(drop=True)
    
    # Deduplicate based on unique (patient_boilerplate_text, trial_boilerplate_text) combinations
    # We'll run inference on unique pairs only, then map back to all original rows
    dedup_cols = ["patient_boilerplate_text", "trial_boilerplate_text"]
    
    # Convert to string and handle NaN for consistent deduplication
    df_original["_patient_bp_str"] = df_original["patient_boilerplate_text"].astype(str)
    df_original["_trial_bp_str"] = df_original["trial_boilerplate_text"].astype(str)
    
    # Create unique combinations dataframe
    df = df_original.drop_duplicates(subset=["_patient_bp_str", "_trial_bp_str"]).copy()
    df = df.reset_index(drop=True)
    
    print(f"Unique (patient_boilerplate_text, trial_boilerplate_text) combinations: {len(df)} "
          f"(reduced from {len(df_original)} rows, {100*(1 - len(df)/len(df_original)):.1f}% savings)")
    
    # Save the mapping info for later reconstruction
    mapping_file = os.path.join(args.temp_dir, "_original_mapping.csv")
    df_original[["_patient_bp_str", "_trial_bp_str"]].to_csv(mapping_file, index=True)
    
    # Also save the unique df's string columns for joining later
    df["_dedup_idx"] = df.index
    
    # Generate batch list
    all_batches = []
    for batch_start in range(0, len(df), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(df))
        all_batches.append((batch_start, batch_end))
    
    print(f"Total batches: {len(all_batches)}")
    
    # Check for completed shards
    completed = get_completed_shards(args.temp_dir)
    remaining_batches = [b for b in all_batches if b not in completed]
    
    if completed:
        print(f"Found {len(completed)} completed shards, {len(remaining_batches)} remaining")
    
    if not remaining_batches:
        print("All batches already completed!")
        merge_shards(args.temp_dir, args.output_file)
        return
    
    # Distribute batches across workers
    worker_batches = [[] for _ in range(num_workers)]
    for i, batch in enumerate(remaining_batches):
        worker_batches[i % num_workers].append(batch)
    
    # Assign GPUs to workers
    worker_gpus = []
    for i in range(num_workers):
        start_idx = i * tp_size
        end_idx = start_idx + tp_size
        worker_gpus.append(gpu_ids[start_idx:end_idx])
    
    # Print worker assignments
    for i, (gpus, batches) in enumerate(zip(worker_gpus, worker_batches)):
        print(f"Worker {i}: GPUs {gpus}, {len(batches)} batches")
    
    # Use spawn method for CUDA compatibility
    mp.set_start_method("spawn", force=True)
    
    # Run workers in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for worker_id, (gpus, batches) in enumerate(zip(worker_gpus, worker_batches)):
            if batches:  # Only start workers with work to do
                future = executor.submit(
                    worker_process,
                    worker_id,
                    gpus,
                    batches,
                    df,
                    args,
                )
                futures.append(future)
        
        # Wait for all workers to complete
        for future in as_completed(futures):
            try:
                worker_id = future.result()
                print(f"Worker {worker_id} completed successfully")
            except Exception as e:
                print(f"Worker failed with error: {e}")
                raise
    
    # Merge all shards
    merge_shards(args.temp_dir, args.output_file)
    print("Done!")


if __name__ == "__main__":
    main()