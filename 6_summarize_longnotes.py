#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel long-history patient summarization with one vLLM instance per GPU-group.

Examples
--------
# Auto-group sequentially: GPUs 0,1,2,3 into pairs (2 per kernel => 2 workers)
python 6_summarize_longnotes.py \
  --input_parquet useful_trial_enrollments_with_longnotes.parquet \
  --output_parquet patient_summaries_and_their_trials.parquet \
  --model openai/gpt-oss-120b \
  --download_dir ../meta_ai \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 2 \
  --max_model_len 120000 \
  --prompt_batch_size 2000

# Explicit groups (overrides --gpus_per_kernel): two workers on (0,1) and (2,3)
python summarize_long_histories_parallel.py \
  --input_parquet useful_trial_enrollments_with_longnotes.parquet \
  --output_parquet patient_summaries.parquet \
  --model openai/gpt-oss-120b \
  --download_dir /data1/ken/meta/2024/meta_ai \
  --gpu_ids 0,1;2,3 \
  --max_model_len 120000 \
  --prompt_batch_size 64
"""

import argparse
import math
import os
import re
import sys
import warnings
from typing import List, Dict, Tuple
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# -------------------------
# Utilities
# -------------------------

def parse_gpu_groups(gpu_ids_arg: str, gpus_per_kernel: int) -> List[List[str]]:
    """
    Parse GPU groupings. Accepts either:
    - "0,1,2,3" with gpus_per_kernel=2 => [["0","1"],["2","3"]]
    - "0,1;2,3" => [["0","1"],["2","3"]] (explicit groups; gpus_per_kernel ignored)
    """
    if ";" in gpu_ids_arg:
        groups = []
        for grp in gpu_ids_arg.split(";"):
            grp = grp.strip()
            if not grp:
                continue
            groups.append([g.strip() for g in grp.split(",") if g.strip() != ""])
        return groups

    flat = [g.strip() for g in gpu_ids_arg.split(",") if g.strip() != ""]
    if gpus_per_kernel <= 0:
        raise ValueError("--gpus_per_kernel must be >= 1 when GPU groups are not explicit.")
    if len(flat) % gpus_per_kernel != 0:
        raise ValueError(
            f"Number of GPUs ({len(flat)}) not divisible by --gpus_per_kernel ({gpus_per_kernel}). "
            f"Either adjust or pass explicit groups like '0,1;2,3'."
        )
    groups = []
    for i in range(0, len(flat), gpus_per_kernel):
        groups.append(flat[i:i+gpus_per_kernel])
    return groups


def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]


def build_prompt_texts(tokenizer, patient_texts: List[str], max_model_len: int, margin_tokens: int = 5000) -> List[str]:
    """
    For each patient_text:
      - Truncate to fit (max_model_len - margin) tokens by keeping head & tail halves.
      - Wrap with system+user messages and apply chat template to produce prompt strings.
    """
    threshold = max(1024, max_model_len - margin_tokens)
    prompts = []

    for patient_text in patient_texts:
        toks = tokenizer(patient_text, add_special_tokens=False).input_ids
        if len(toks) > threshold:
            half = threshold // 2
            first_part = toks[:half]
            last_part = toks[-half:]
            patient_text = tokenizer.decode(first_part) + " ... " + tokenizer.decode(last_part)

        messages = [{'role':'system', 'content': 'Reasoning: high'}, 
                    {'role':'user', 'content': """You are an experienced clinical oncology history summarization bot.
Your job is to construct a summary of the cancer history for a patient based on an excerpt of the patient's electronic health record. The text in the excerpt is provided in chronological order. Each paragraph in the excerpt represents a summary of a clinical document written on the date indicated in the paragraph.     
Document the patient's most recent age; sex; cancer type/primary site (eg breast cancer, lung cancer, etc); histology (eg adenocarcinoma, squamous carcinoma, etc); current extent (localized, advanced, metastatic, etc); biomarkers (genomic results, protein expression, etc); and treatment history (surgery, radiation, chemotherapy/targeted therapy/immunotherapy, etc, including start and stop dates and best response if known).
Do not consider localized basal cell or squamous carcinomas of the skin, or colon polyps, to be cancers for your purposes.
Do not include the patient's name, but do include relevant dates whenever documented, including dates of diagnosis and start/stop dates of each treatment.
If a patient has a history of more than one cancer, document the cancers one at a time.
CRITICAL: Format your response as free text ONLY. Do NOT output markdown, Unicode, or tables.
Also document any history of conditions that might meet "boilerplate" exclusion criteria, including uncontrolled brain metastases, lack of measurable disease, congestive heart failure, pneumonitis, renal dysfunction, liver dysfunction, and HIV or hepatitis infection. For each of these, present the evidence from the history that the patient has a history of such a condition, including dates.
Clearly separate the "boilerplate" section by labeling it "Boilerplate: " before describing any such conditions.
Here is an example of the desired output format:

Age: 70
Sex: Male
Cancer type: Lung cancer
Histology: Adenocarcinoma
Current extent: Metastatic
Biomarkers: PD-L1 75%, KRAS G12C mutant
Treatment history: 
# 1/5/2020-2/5/2021: carboplatin/pemetrexed/pembrolizumab
# 1/2021: Palliative radiation to progressive spinal metastases
# 3/2021-present: docetaxel
Boilerplate:
No evidence of common boilerplate exclusion criteria
""" + "The excerpt for you to summarize is:\n" + patient_text + """\nNow, write your summary. Do not add preceding text before the abstraction, and do not add notes or commentary afterwards. This will not be used for clinical care, so do not write any disclaimers or cautionary notes."""}

                     ]
        
        prompt = tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False
        )
        prompts.append(prompt)

    return prompts


def postprocess_outputs(raw_texts: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Split off 'assistantfinal' if present; then split based on the *line* containing 'Boilerplate:'.
    
    Returns:
      - full_response_texts (original raw outputs)
      - summary_texts (text occurring before the line containing the marker)
      - boilerplate_texts (text occurring after the line containing the marker)
    """
    reasoning_marker = "assistantfinal"
    boilerplate_marker = "Boilerplate:"

    # 1. Handle Reasoning Marker (Unchanged)
    cleaned = []
    for t in raw_texts:
        if reasoning_marker in t:
            cleaned.append(t.split(reasoning_marker, 1)[-1])
        else:
            cleaned.append(t)

    summaries, boilers = [], []

    # 2. Handle Boilerplate Marker (Updated to split by line)
    for t in cleaned:
        # splitlines(keepends=True) ensures we preserve original formatting/newlines
        lines = t.splitlines(keepends=True)
        
        marker_line_index = -1
        
        # Find which line contains the marker
        for i, line in enumerate(lines):
            if boilerplate_marker in line:
                marker_line_index = i
                break
        
        if marker_line_index != -1:
            # Join lines before the marker line
            pre = "".join(lines[:marker_line_index])
            # Join lines after the marker line (skipping the marker line itself)
            post = "".join(lines[marker_line_index + 1:])
            
            summaries.append(pre)
            boilers.append(post)
        else:
            # If marker not found, the full text goes to both
            summaries.append(t)
            boilers.append(t)

    return raw_texts, summaries, boilers


# -------------------------
# Worker
# -------------------------

def worker_summarize(
    worker_id: int,
    gpu_group: List[str],
    records: List[Tuple[int, str]],
    model_name: str,
    download_dir: str,
    max_model_len: int,
    prompt_batch_size: int,
    temperature: float,
    top_k: int,
    max_tokens: int,
    repetition_penalty: float,
    gpu_memory_utilization: float,
) -> List[Tuple[int, str, str, str]]:
    """
    Each worker:
      - Sets CUDA_VISIBLE_DEVICES to its GPU group
      - Instantiates a vLLM LLM with tensor_parallel_size=len(gpu_group)
      - Summarizes its subset (batched), returns list of tuples:
        (row_id, full_output_text, summary_text, boilerplate_text)
    """
    # Per-process env (must be set before vLLM import/instantiate)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_group)
    #os.environ["VLLM_ATTENTION_BACKEND"] = os.environ.get("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    #os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "0")

    # Import here to ensure env is applied in spawned process
    from vllm import LLM, SamplingParams

    tp_size = max(1, len(gpu_group))
    print(f"[worker{worker_id}] Starting on GPUs {gpu_group} (tp={tp_size}), {len(records)} examples.")

    llama = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        download_dir=download_dir,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len
    )
    tokenizer = llama.get_tokenizer()
    sampling = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )

    out: List[Tuple[int, str, str, str]] = []
    # Keep stable order within worker
    row_ids = [rid for (rid, _) in records]
    texts   = [txt for (_, txt) in records]

    for batch_idx, batch_slice in enumerate(chunk_list(list(range(len(texts))), prompt_batch_size)):
        batch_texts = [texts[i] for i in batch_slice]
        batch_rids  = [row_ids[i] for i in batch_slice]

        prompts = build_prompt_texts(tokenizer, batch_texts, max_model_len=max_model_len, margin_tokens=5000)
        responses = llama.generate(prompts, sampling)
        raw_outputs = [r.outputs[0].text for r in responses]

        full_texts, summaries, boilers = postprocess_outputs(raw_outputs)

        for rid, full, summ, boil in zip(batch_rids, full_texts, summaries, boilers):
            out.append((rid, full, summ, boil))

        print(f"[worker{worker_id}] Completed batch {batch_idx+1}/{math.ceil(len(texts)/prompt_batch_size)} "
              f"({len(batch_slice)} records).")

    print(f"[worker{worker_id}] Done.")
    return out


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser("Parallel patient history summarization with per-group vLLM instances.")
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--output_parquet", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--download_dir", required=True)
    ap.add_argument("--gpu_ids", required=True,
                    help="Either comma list (e.g., 0,1,2,3) used with --gpus_per_kernel, "
                         "or explicit groups (e.g., 0,1;2,3).")
    ap.add_argument("--gpus_per_kernel", type=int, default=1,
                    help="Grouping size when --gpu_ids is a flat list. Ignored if groups are explicit with ';'.")
    ap.add_argument("--max_model_len", type=int, default=120000)
    ap.add_argument("--prompt_batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=7500)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.93)
    args = ap.parse_args()

    # Safer CUDA multiprocessing
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # If already set, that's fine.
        pass

    # Read input
    df = pd.read_parquet(args.input_parquet)
    if "patient_long_text" not in df.columns:
        raise ValueError("Input parquet must contain a 'patient_long_text' column.")

    # Filter empties
    df = df[(df["patient_long_text"].notnull()) & (df["patient_long_text"] != "")]
    df = df.copy()
    df["__row_id__"] = df.index.astype(int)

    # Build work distribution
    groups = parse_gpu_groups(args.gpu_ids, args.gpus_per_kernel)
    num_workers = len(groups)
    if num_workers == 0:
        raise ValueError("No GPU groups parsed from --gpu_ids.")

    # Evenly shard rows across workers
    records = list(zip(df["__row_id__"].tolist(), df["patient_long_text"].tolist()))
    shards: List[List[Tuple[int, str]]] = [[] for _ in range(num_workers)]
    for i, rec in enumerate(records):
        shards[i % num_workers].append(rec)

    print(f"Prepared {len(records)} examples across {num_workers} worker(s).")

    # Launch workers
    futures = []
    results: List[Tuple[int, str, str, str]] = []

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for wid, (gpu_group, shard) in enumerate(zip(groups, shards)):
            futures.append(
                ex.submit(
                    worker_summarize,
                    wid, gpu_group, shard,
                    args.model, args.download_dir,
                    args.max_model_len, args.prompt_batch_size,
                    args.temperature, args.top_k, args.max_tokens, args.repetition_penalty,
                    args.gpu_memory_utilization
                )
            )

        for fut in as_completed(futures):
            res = fut.result()  # will raise if worker failed
            results.extend(res)

    # Assemble outputs
    # results: list of (row_id, full_output_text, summary_text, boilerplate_text)
    out_map: Dict[int, Tuple[str, str, str]] = {rid: (full, summ, boil) for rid, full, summ, boil in results}

    df["patient_summary_reasoning_and_output"] = df["__row_id__"].map(lambda r: out_map[r][0])
    df["patient_summary"] = df["__row_id__"].map(lambda r: out_map[r][1])
    df["patient_boilerplate_text"] = df["__row_id__"].map(lambda r: out_map[r][2])

    # Drop helper
    df = df.drop(columns=["__row_id__"])

    # Save
    df.to_parquet(args.output_parquet, index=False)
    print(f"Wrote {args.output_parquet} with {len(df)} rows.")


if __name__ == "__main__":
    # Quiet some tokenizer warnings if they pop up
    warnings.filterwarnings("ignore")
    main()
