#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel trial 'reasonableness' checker with per-kernel vLLM instances.
NOW WITH RESUME CAPABILITY - skips batches that already have output files.

- You specify a list of GPU ids (e.g., "0,1,2,3,4,5,6,7") and how many GPUs per kernel (e.g., 2).
- The script shards the input rows across kernels and spawns one worker per kernel.
- Each worker:
    * Sets CUDA_VISIBLE_DEVICES to its GPU group (size = gpus_per_kernel)
    * Checks if output shard parquet already exists; if so, SKIPS that batch
    * Launches its own vLLM instance with tensor_parallel_size = gpus_per_kernel
    * Processes its shard in prompt-sized batches
    * Streams results to disk as parquets in: {out_dir}/shards/worker{K}_batch{B}_{start}_{end}.parquet
- Optionally, the main process combines all shard parquets into a single file (--final_output).

Example:
python llm_check_trials.py \
  --input_parquet top_cohorts_tocheck_round1.parquet \
  --out_dir ./round1_patientcentric_checks \
  --final_output top_cohorts_checked_round1.parquet \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 2 \
  --prompt_batch_size 2000 \
  --model openai/gpt-oss-120b \
  --download_dir /data1/ken/meta/2024/meta_ai \
  --max_model_len 20000 \
  --gpu_memory_utilization 0.92
"""

import os
import re
import argparse
import math
import multiprocessing as mp
from typing import List, Tuple

import pandas as pd
import numpy as np

# ---------- Prompting logic (same behavior as your single-GPU version) ----------

def ask_about_trials_loosely(patient_summaries: List[str],
                             trial_summaries: List[str],
                             llm_model):
    """
    Given aligned lists of patient_summaries and trial_summaries, return
    (responses, response_texts, eligibility_results)
    """
    tokenizer = llm_model.get_tokenizer()
    prompts = []

    for patient_summary, trial_summary in zip(patient_summaries, trial_summaries):
        messages = [
            {'role': 'system', 'content': "Reasoning: high"},
            {'role': 'user', 'content': (
                "You are a brilliant oncologist with encyclopedic knowledge about cancer and its treatment. "
                "Your job is to evaluate whether a given clinical trial is a reasonable consideration for a patient, "
                "given a clinical trial summary and a patient summary.\n\n"
                f"Here is a summary of the clinical trial:\n{trial_summary}\n"
                f"Here is a summary of the patient:\n{patient_summary}\n"
                "Base your judgment on whether the patient generally fits the age requirements if any, sex requirements if any, cancer type(s), cancer burden, prior treatment(s), "
                "and biomarker criteria specified for the trial.\n"
                "You do not have to determine if the patient is actually eligible; instead please just evaluate whether it is reasonable "
                "for the trial to be considered further by the patient's oncologist.\n"
                "Biomarker criteria have to be considered carefully. Some trials have biomarker requirements that are not assessed until "
                "formal trial screening. A trial may therefore sometimes be a reasonable consideration for a patient even if a required "
                "biomarker is not known to be present in the patient.\n"
                "However, if a required biomarker is known to be absent, or can be assumed to be absent based on other information, the trial "
                "is not a reasonable consideration. For example, if a trial for lung cancer requires an EGFR mutation, documentation that there "
                "is no EGFR mutation indicates the trial is not a reasonable consideration. Similarly, documentation of a KRAS mutation in the "
                "patient indicates the trial is not a reasonable consideration, since, as you know, KRAS and EGFR driver mutations in lung cancer "
                "are mutually exclusive.\n"
                "Many trials describe required washout periods for prior treatments for eligibility. For example, the eligibility criteria might state "
                "that patients may not have received radiation or chemotherapy in the last 14 days or 30 days. It is CRITICAL that you IGNORE these "
                "eligibility criteria when considering prior treatment requirements. Assume that patients could wait for the washout period to enroll. "
                "Also CRITICAL: Ignore your knowledge of today's current date. Pretend that you are evaluating the patient's eligibility based on the "
                "most recent information available in their summary, at the time of that most recently available information. "
                "Do not provide ethical judgments or comment on resource constraints with respect whether the trial is a reasonable clinical "
                "consideration; just evaluate whether it is, given the available information.\n"
                'Reason step by step, then answer the question "Is this trial a reasonable consideration for this patient?" with a one-word '
                '"Yes!" or "No!" answer.\n'
                "Make sure to include the exclamation point in your final one-word answer."
            )}
        ]

        prompt = tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False
        )
        prompts.append(prompt)

    from vllm import SamplingParams  # import locally inside function to keep worker footprint tidy

    responses = llm_model.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            top_k=1,
            max_tokens=5000,
            repetition_penalty=1.2,
            # You can add stop_token_ids if you want to trim reasoning
        )
    )

    response_texts = [x.outputs[0].text for x in responses]

    eligibility_results = []
    for txt in response_texts:
        # look near the end for one of the exact tokens
        tail = txt[-10:].upper()
        if "YES!" in tail:
            eligibility_results.append(1.0)
        elif "NO!" in tail:
            eligibility_results.append(0.0)
        else:
            # fallback: simple heuristic
            eligibility_results.append(1.0 if "YES" in tail else 0.0)
            
    return responses, response_texts, eligibility_results


# ---------- Worker ----------

def worker_process(worker_id: int,
                   shard_path: str,
                   out_dir: str,
                   gpu_group: List[str],
                   model: str,
                   download_dir: str,
                   tp_size: int,
                   max_model_len: int,
                   gpu_memory_utilization: float,
                   prompt_batch_size: int):
    """
    One process = one vLLM kernel pinned to a GPU group.
    Reads its shard parquet, runs inference in batches, writes parquet shards.
    RESUME CAPABILITY: Skips batches where output parquet already exists.
    """
    try:
        # Pin this worker to its GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_group))
        #os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        #os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        shards_dir = os.path.join(out_dir, "shards")
        os.makedirs(shards_dir, exist_ok=True)

        # Load data
        df = pd.read_parquet(shard_path)
        df["_orig_order"] = np.arange(len(df), dtype=np.int64)
        df = df[~df["patient_summary"].isnull()].copy()

        total = len(df)
        if total == 0:
            print(f"[Worker {worker_id}] Shard is empty after filtering; nothing to do.")
            return

        # First pass: check which batches need processing
        batches_to_process = []
        batches_already_done = []
        batch_id = 0
        
        for start in range(0, total, prompt_batch_size):
            end = min(start + prompt_batch_size, total)
            out_name = f"worker{worker_id}_batch{batch_id}_{start}_{end}.parquet"
            out_path = os.path.join(shards_dir, out_name)
            
            if os.path.exists(out_path):
                batches_already_done.append((batch_id, start, end, out_name))
            else:
                batches_to_process.append((batch_id, start, end, out_name))
            
            batch_id += 1

        print(f"[Worker {worker_id}] Total batches: {batch_id}")
        print(f"[Worker {worker_id}] Already completed: {len(batches_already_done)}")
        print(f"[Worker {worker_id}] Need to process: {len(batches_to_process)}")

        if not batches_to_process:
            print(f"[Worker {worker_id}] All batches already complete. Nothing to do.")
            return

        # Only initialize vLLM if we have work to do
        from vllm import LLM

        print(f"[Worker {worker_id}] Initializing vLLM on GPUs {os.environ['CUDA_VISIBLE_DEVICES']} | tp={tp_size}")
        llm = LLM(
            model=model,
            tensor_parallel_size=tp_size,
            download_dir=download_dir,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len
        )

        # Process only the batches that need work
        for batch_id, start, end, out_name in batches_to_process:
            batch = df.iloc[start:end].copy()

            print(f"[Worker {worker_id}] Processing batch {batch_id}: rows {start}-{end} ({end-start} rows)")

            # Run model
            _, resp_texts, elig = ask_about_trials_loosely(
                batch["patient_summary"].astype(str).tolist(),
                batch["this_space"].astype(str).tolist(),
                llm_model=llm
            )

            batch["trialcheck_llm_response"] = resp_texts
            batch["eligibility_result"] = elig

            out_path = os.path.join(shards_dir, out_name)
            batch.to_parquet(out_path, index=False)

            print(f"[Worker {worker_id}] Wrote {out_name} ({end-start} rows; {end}/{total})")

        print(f"[Worker {worker_id}] Done. Processed {len(batches_to_process)} new batches.")

    except Exception as e:
        # Don't crash silently
        import traceback
        print(f"[Worker {worker_id}] ERROR: {e}\n{traceback.format_exc()}")


# ---------- Utilities ----------

def parse_gpu_groups(gpu_str: str, gpus_per_kernel: int) -> List[List[str]]:
    """Turn '0,1,2,3' into [['0','1'],['2','3']] for gpus_per_kernel=2."""
    gpu_list = [g.strip() for g in gpu_str.split(",") if g.strip() != ""]
    if len(gpu_list) < gpus_per_kernel:
        raise ValueError(f"Not enough GPUs ({len(gpu_list)}) for gpus_per_kernel={gpus_per_kernel}")

    n_kernels = len(gpu_list) // gpus_per_kernel
    if n_kernels == 0:
        raise ValueError("gpus_per_kernel is larger than the number of provided GPUs.")
    if len(gpu_list) % gpus_per_kernel != 0:
        print(f"[WARN] Number of GPUs ({len(gpu_list)}) is not divisible by gpus_per_kernel ({gpus_per_kernel}). "
              f"Ignoring the last {len(gpu_list) % gpus_per_kernel} GPU(s).")

    usable = gpu_list[: n_kernels * gpus_per_kernel]
    groups = [usable[i:i+gpus_per_kernel] for i in range(0, len(usable), gpus_per_kernel)]
    return groups


def shard_input_across_kernels(df: pd.DataFrame, n_kernels: int) -> List[pd.DataFrame]:
    """Evenly split rows across kernels."""
    splits = np.array_split(df, n_kernels)
    return [s.reset_index(drop=True) for s in splits]


def finalize(out_dir: str, final_output: str):
    """Concat all worker parquet shards into one file; sort by pseudo_mrn if present."""
    shards_dir = os.path.join(out_dir, "shards")
    if not os.path.isdir(shards_dir):
        print("[Finalize] No shards directory found; skipping.")
        return
    files = sorted([os.path.join(shards_dir, f) for f in os.listdir(shards_dir) if f.endswith(".parquet")])
    if not files:
        print("[Finalize] No shard parquets found; skipping.")
        return

    print(f"[Finalize] Found {len(files)} shard parquet files to combine.")
    parts = [pd.read_parquet(f) for f in files]
    out = pd.concat(parts, axis=0, ignore_index=True)
    
    # Guarantee original input order
    if "_orig_order" in out.columns:
        out = out.sort_values(by="_orig_order", kind="stable")
        out = out.drop(columns=["_orig_order"]).reset_index(drop=True)

    out_path = os.path.join(out_dir, final_output)
    os.makedirs(out_dir, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[Finalize] Wrote combined file: {out_path} ({len(out)} rows)")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Parallel LLM trial reasonableness checker (vLLM per-kernel) with resume capability.")
    parser.add_argument("--input_parquet", required=True, help="Path to input parquet with columns: patient_summary, this_space, (optional) pseudo_mrn, ...")
    parser.add_argument("--out_dir", required=True, help="Output directory; shards go to {out_dir}/shards")
    parser.add_argument("--final_output", default="", help="If provided, write a combined parquet at the end with this filename (inside out_dir).")

    parser.add_argument("--gpus", required=True, help='Comma-separated GPU ids, e.g., "0,1,2,3"')
    parser.add_argument("--gpus_per_kernel", type=int, required=True, help="Number of GPUs to assign to each kernel (tensor_parallel_size).")

    parser.add_argument("--prompt_batch_size", type=int, default=2000, help="Rows per generation batch inside each kernel.")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="HF model id for vLLM.")
    parser.add_argument("--download_dir", default="", help="vLLM/HF download cache dir.")
    parser.add_argument("--max_model_len", type=int, default=30000, help="vLLM max_model_len.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92, help="vLLM gpu_memory_utilization.")
    args = parser.parse_args()

    # Read input once; filter; then shard to per-worker parquet to avoid N× re-reads.
    df = pd.read_parquet(args.input_parquet)
    df = df[~df["patient_summary"].isnull()].reset_index(drop=True)

    if len(df) == 0:
        raise SystemExit("Input becomes empty after filtering null patient_summary. Nothing to do.")

    df['this_space'] = df['this_space'].str.replace(r'^\s*\d+\.', '', regex=True)



    groups = parse_gpu_groups(args.gpus, args.gpus_per_kernel)
    n_kernels = len(groups)
    print(f"[Main] Found {len(df)} rows; launching {n_kernels} kernels "
          f"({args.gpus_per_kernel} GPU(s) each) -> total GPUs used: {n_kernels * args.gpus_per_kernel}")

    # Prepare per-kernel input shards as parquet to minimize memory duplication.
    work_dir = os.path.join(args.out_dir, "work_shards")
    os.makedirs(work_dir, exist_ok=True)
    input_splits = shard_input_across_kernels(df, n_kernels)

    shard_paths = []
    for k, split in enumerate(input_splits):
        shard_path = os.path.join(work_dir, f"worker{k}_input.parquet")
        split.to_parquet(shard_path, index=False)
        shard_paths.append(shard_path)

    # Spawn workers (use spawn to be CUDA-safe)
    mp.set_start_method("spawn", force=True)
    procs = []
    for k in range(n_kernels):
        p = mp.Process(
            target=worker_process,
            args=(
                k,
                shard_paths[k],
                args.out_dir,
                groups[k],
                args.model,
                args.download_dir,
                args.gpus_per_kernel,
                args.max_model_len,
                args.gpu_memory_utilization,
                args.prompt_batch_size
            ),
            daemon=False
        )
        p.start()
        procs.append(p)

    # Join
    for p in procs:
        p.join()

    # Optional finalization
    if args.final_output:
        finalize(args.out_dir, args.final_output)

    print("[Main] All done.")


if __name__ == "__main__":
    main()