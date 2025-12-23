#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel LLM Tagger with per-kernel vLLM instances (RESUMABLE).

Usage (example):
python 3b_tag_synthetic_patient_notes.py \
  --positive_parquet ./synthetic_notes/synthetic_notes.parquet \
  --negative_parquet ./synthetic_negative_notes/synthetic_notes.parquet \
  --model openai/gpt-oss-120b \
  --download_dir ../meta_ai \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 2 \
  --chunk_size 10000 \
  --prompt_batch_size 5000 \
  --max_model_len 20000 \
  --note_sample_n 300000 \
  --final_output tagged_chunks_enrolled_pt_reports_with_boilerplate.parquet

You can also run with a single parquet containing a 'text' column:
python 3a_tag_synthetic_patient_notes.py \
  --input_parquet ./synthetic_notes/synthetic_notes.parquet \
  --model openai/gpt-oss-120b \
  --download_dir ../meta_ai \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus_per_kernel 2 \
  --chunk_size 10000 \
  --prompt_batch_size 5000 \
  --max_model_len 30000 \
  --note_sample_n 160000 \
  --final_output tagged_chunks_enrolled_pt_reports_with_boilerplate_positive.parquet
"""

import os
import re
import gc
import json
import math
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd


# ========================== Prompt / tagging logic ==========================

def build_messages_from_text(text: str) -> List[dict]:
    temp_patient = re.sub(r"[\n\r]", " ", text.strip())
    temp_patient = re.sub(r"\s+", " ", temp_patient)
    sentences = "<excerpt break>" + re.sub(r"\. ", "<excerpt break>", temp_patient) + "<excerpt break>"

    system_msg = {
        "role": "system",
        "content": """You are an oncology clinical note data extraction bot.
Your job is to review a list of excerpts from a clinical document and extract the excerpts relevant to a list of questions.
Reasoning: high
"""
    }

    user_msg = {
        "role": "user",
        "content": (
            "The list of excerpts, separated by <excerpt break>, is: " + sentences +
            """Now, list the excerpts relevant to any of the following questions.
Format your answer as JSON, tagging each excerpt that is relevant to at least one question with each tag to which it is relevant.
Here is the list of questions:
How old is the patient? (Tag: age)
What is the patient's sex? (Tag: sex)
What type of cancer (primary site and histology) does the patient have? (Tag: cancer_type )
What was the stage at diagnosis? (Tag: stage_at_diagnosis)
What treatments (including surgery, radiation, or systemic therapy) has the patient received? (Tag: treatment)
How widespread is the cancer currently? (Tag: cancer_burden)
What is the prognosis, prognostic score, or risk category? (Tag: prognosis_and_risk)
Is there response to therapy or progressive disease? (Tag: cancer_status)
Is the patient experiencing an adverse event of treatment? (Tag: adverse_event)
What biomarkers, such as protein expression and genetic mutations/alterations, does the patient's tumor have? (Tag: biomarker)
What comorbidities, or diseases other than cancer, does the patient have? (Tag: comorbidity)
Are there uncontrolled brain metastases? (Tag: uncontrolled_brain_met)
Is there measurable disease, meaning a tumor at least 1 cm across or lymph node at least 1.5 cm in short axis dimension? (Tag: measurable_disease)
Is there progressive (worsening) disease? (Tag: progressive_disease)
Is there a history of pneumonitis? (Tag: pneumonitis)
Is there a history of colitis? (Tag: colitis)
Is there a history of hepatitis or HIV? (Tag: hepatitis_or_hiv)
Is the patient anemic, with hemoglobin under 10? (Tag: anemia)
Is there a reduced renal function/creatinine clearance, with estimated GFR < 60? (Tag: renal_dysfunction)
Is there liver dysfunction, with elevated bilirubin, AST, or ALT? (Tag: liver_dysfunction)
Is there a history of heart failure? (Tag: heart_failure)
Does the patient have a poor performance status and/or ECOG performance status of 2 or more? (Tag: poor_ps)
What adverse side effects of treatment has the patient had? (Tag: adverse_event)
Here is an example of the output format:
[{"excerpt": "80M with metastatic lung adenocarcinoma.", "tags": ["age", "sex", "cancer_type", "cancer_burden"]},
 {"excerpt": "The tumor was HER2 positive.", "tags": ["biomarker"]},
 {"excerpt": "Imaging demonstrated new bilateral lung infiltrates.", "tags": ["pneumonitis", "adverse_event"]},
 {"excerpt": "LV ejection fraction was 35%.", "tags": ["heart_failure"]}
]
Do not include excerpts that are not relevant to the questions. 
Do not abbreviate or alter excerpts that you do include; copy them verbatim from the prompt.
Do not add disclaimers or introductory text.
If there are no excerpts relevant to the above questions, just output blank JSON {} .
"""
        )
    }
    return [system_msg, user_msg]


def tag_chunks_batch(patient_texts: List[str],
                     model,
                     tokenizer,
                     sampling_params,
                     truncate_tokens: int = 20000,
                     reasoning_marker: str = "assistantfinal") -> Tuple[List[str], List[str]]:
    # Prepare chat prompts with truncation by token count
    long_messages = []
    prompts = []
    for txt in patient_texts:
        msgs = build_messages_from_text(txt)
        prompts.append(msgs)
        long_messages.append(msgs[1]["content"])

    # token-level truncation
    tokenized = tokenizer(long_messages, add_special_tokens=False)
    truncated_texts = tokenizer.batch_decode([ids[-truncate_tokens:] for ids in tokenized.input_ids])

    rendered_prompts = []
    for i, msgs in enumerate(prompts):
        msgs[1]["content"] = truncated_texts[i]
        rendered = tokenizer.apply_chat_template(
            conversation=msgs, add_generation_prompt=True, tokenize=False
        )
        rendered_prompts.append(rendered)

    # vLLM inference
    responses = model.generate(rendered_prompts, sampling_params)
    response_texts = [x.outputs[0].text for x in responses]

    # Extract final (post-reasoning) portion
    finals = []
    for r in response_texts:
        if reasoning_marker in r:
            finals.append(r.split(reasoning_marker, 1)[-1])
        else:
            finals.append(r)
    return response_texts, finals


# ====================== Utilities: JSON, atomic writes ======================

def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".__tmp__")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)

def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".__tmp__")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


# ================== Worker (one vLLM kernel per GPU group) ==================

def worker_run(worker_id: int,
               gpu_ids: List[int],
               shards: List[Path],
               out_dir: Path,
               model: str,
               download_dir: str,
               gpus_per_kernel: int,
               gpu_memory_utilization: float,
               max_model_len: int,
               prompt_batch_size: int,
               truncate_tokens: int,
               reasoning_marker: str):
    """
    Each worker sets its own CUDA_VISIBLE_DEVICES, launches a vLLM LLM, and processes its assigned shards.
    """
    if not shards:
        print(f"[worker {worker_id}] No shards assigned; skipping kernel launch.", flush=True)
        return

    # Isolate GPUs for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Import inside worker to avoid CUDA init in parent
    from vllm import LLM, SamplingParams

    print(f"[worker {worker_id}] using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']} for {len(shards)} shard(s)", flush=True)

    # Launch per-worker vLLM instance
    llm = LLM(
        model=model,
        tensor_parallel_size=gpus_per_kernel,
        download_dir=download_dir if download_dir else None,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.0,
        top_k=1,
        max_tokens=10000,
        repetition_penalty=1.2,
    )

    for shard_path in shards:
        out_path = out_dir / f"out_{shard_path.stem}.parquet"
        if out_path.exists():
            print(f"[worker {worker_id}] Skip already processed {shard_path.name}", flush=True)
            continue

        try:
            df = pd.read_parquet(shard_path)
            assert "text" in df.columns, f"'text' column missing in {shard_path}"

            texts = df["text"].tolist()
            n = len(texts)
            all_reason = []
            all_final = []

            for i in range(0, n, prompt_batch_size):
                batch = texts[i:i + prompt_batch_size]
                reason, final = tag_chunks_batch(
                    batch, llm, tokenizer, sampling_params,
                    truncate_tokens=truncate_tokens,
                    reasoning_marker=reasoning_marker
                )
                all_reason.extend(reason)
                all_final.extend(final)

            df["tagger_reasoning_and_output"] = all_reason
            df["tagger_llm_output"] = all_final
            df["worker_id"] = worker_id
            df["gpu_ids"] = ",".join(map(str, gpu_ids))

            atomic_write_parquet(df, out_path)
            print(f"[worker {worker_id}] Wrote {out_path.name} ({len(df)} rows)", flush=True)

            del df, texts, all_reason, all_final
            gc.collect()
        except Exception as e:
            errlog = out_dir / "failed_shards.txt"
            with open(errlog, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{shard_path.name}\t{repr(e)}\n")
            print(f"[worker {worker_id}] ERROR {shard_path.name}: {repr(e)}", flush=True)


# ===================== Sharding & orchestration (resumable) =====================

def group_gpus(gpu_list: List[int], gpus_per_kernel: int) -> List[List[int]]:
    groups = []
    for i in range(0, len(gpu_list), gpus_per_kernel):
        g = gpu_list[i:i + gpus_per_kernel]
        if len(g) == gpus_per_kernel:
            groups.append(g)
    return groups


def load_inputs(args) -> pd.DataFrame:
    """Load full input DataFrame; sampling happens later in ensure_shards()."""
    dfs = []

    if args.input_parquet:
        df = pd.read_parquet(args.input_parquet)
        if "text" not in df.columns and "synthetic_note" in df.columns:
            df = df.rename(columns={"synthetic_note": "text"})
        dfs.append(df)

    if args.positive_parquet:
        dfp = pd.read_parquet(args.positive_parquet)
        if "text" not in dfp.columns and "synthetic_note" in dfp.columns:
            dfp = dfp.rename(columns={"synthetic_note": "text"})
        dfs.append(dfp)

    if args.negative_parquet:
        dfn = pd.read_parquet(args.negative_parquet)
        if "pseudo_mrn" in dfn.columns:
            # Preserve original behavior
            dfn["pseudo_mrn"] = dfn["pseudo_mrn"] + 100000
        if "text" not in dfn.columns and "synthetic_note" in dfn.columns:
            dfn = dfn.rename(columns={"synthetic_note": "text"})
        dfs.append(dfn)

    if not dfs:
        raise ValueError("No inputs provided. Use --input_parquet OR --positive_parquet/--negative_parquet.")

    all_reports = pd.concat(dfs, ignore_index=True)

    if "text" not in all_reports.columns:
        raise ValueError("Input data must contain a 'text' column (or 'synthetic_note' to be renamed).")

    return all_reports


def ensure_shards(all_reports: pd.DataFrame,
                  shard_dir: Path,
                  chunk_size: int,
                  sample_n: int,
                  shuffle_seed: int) -> List[Path]:
    """
    Create (or reuse) shards deterministically, with a persisted manifest.
    Returns the list of shard file paths (creating any missing ones).
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = shard_dir / "manifest.json"

    if manifest_path.exists():
        manifest = read_json(manifest_path)
        print(f"Found manifest with {manifest.get('selected_count')} rows; reusing shard plan.", flush=True)

        # Rebuild the sampled subset using recorded indices
        sel_idx = manifest["selected_indices"]
        sampled = all_reports.iloc[sel_idx].reset_index(drop=True)

        # Recreate any missing shard files
        shards = []
        for rec in manifest["shards"]:
            p = shard_dir / rec["path"]
            if not p.exists():
                start, end = rec["start"], rec["end"]
                shard = sampled.iloc[start:end].copy()
                atomic_write_parquet(shard, p)
                print(f"Recreated missing shard {p.name} ({len(shard)} rows).", flush=True)
            shards.append(p)
        return shards

    # First run: create manifest + shards
    n_total = len(all_reports)
    n_sel = min(sample_n, n_total) if sample_n > 0 else n_total
    sampled = all_reports.sample(n=n_sel, random_state=shuffle_seed).reset_index(drop=True)

    shards: List[Path] = []
    shard_recs = []
    for start in range(0, len(sampled), chunk_size):
        end = min(start + chunk_size, len(sampled))
        shard = sampled.iloc[start:end].copy()
        shard_path = shard_dir / f"shard_{start:09d}_{end:09d}.parquet"
        atomic_write_parquet(shard, shard_path)
        shards.append(shard_path)
        shard_recs.append({"path": shard_path.name, "start": start, "end": end, "row_count": len(shard)})

    manifest = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selected_count": int(len(sampled)),
        "chunk_size": int(chunk_size),
        "shuffle_seed": int(shuffle_seed),
        # Record the original positions of sampled rows relative to the concatenated input
        "selected_indices": sampled.index.tolist(),  # indices after reset; not directly useful
        # We need indices relative to the original all_reports; so recompute:
    }

    # Correct selected_indices to refer to original all_reports positions:
    # We can obtain this by doing a merge on a temporary row_id.
    # (Since we reset_index above, we need to re-sample without reset to keep original indices.)
    # Recompute deterministically:
    selected_indices = (
        all_reports.sample(n=n_sel, random_state=shuffle_seed).index.tolist()
        if n_sel < n_total else list(range(n_total))
    )
    manifest["selected_indices"] = selected_indices
    manifest["shards"] = shard_recs
    write_json(manifest_path, manifest)

    # Save a simple file list too (human-friendly)
    with open(shard_dir / "manifest_files.txt", "w") as f:
        for p in shards:
            f.write(p.name + "\n")

    print(f"Created {len(shards)} shards in {shard_dir} (sampled {n_sel}/{n_total}).", flush=True)
    return shards


def main():
    parser = argparse.ArgumentParser("Parallel clinical note tagger with per-kernel vLLM (resumable)")
    io = parser.add_argument_group("I/O")
    io.add_argument("--input_parquet", type=str, default=None,
                    help="Single parquet with a 'text' column (or 'synthetic_note').")
    io.add_argument("--positive_parquet", type=str, default=None,
                    help="Parquet like synthetic_notes.parquet (will rename 'synthetic_note' -> 'text' if present).")
    io.add_argument("--negative_parquet", type=str, default=None,
                    help="Parquet like synthetic_negative_notes.parquet (adds 100000 to 'pseudo_mrn' if present).")
    io.add_argument("--out_dir", type=str, default="tagging_out",
                    help="Directory for shards and outputs.")
    io.add_argument("--final_output", type=str,
                    default="tagged_chunks_enrolled_pt_reports_with_boilerplate.parquet",
                    help="Final merged parquet path (written in out_dir).")
    io.add_argument("--note_sample_n", type=int, default=200000, help="Number of notes to sample and tag. 0 = use all.")

    perf = parser.add_argument_group("Performance / Parallelism")
    perf.add_argument("--gpus", type=str, default="0,1",
                      help="Comma-separated GPU ids to use across all kernels, e.g., '0,1,2,3'.")
    perf.add_argument("--gpus_per_kernel", type=int, default=2,
                      help="How many GPUs each vLLM kernel uses (tensor_parallel_size).")
    perf.add_argument("--chunk_size", type=int, default=10000,
                      help="Rows per shard file.")
    perf.add_argument("--prompt_batch_size", type=int, default=32,
                      help="Prompts per vLLM generate() call.")
    perf.add_argument("--shuffle_seed", type=int, default=42,
                      help="Deterministic shuffling for load balance and manifest reproducibility.")

    modelg = parser.add_argument_group("Model / vLLM")
    modelg.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    modelg.add_argument("--download_dir", type=str, default=None)
    modelg.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    modelg.add_argument("--max_model_len", type=int, default=30000)
    modelg.add_argument("--truncate_tokens", type=int, default=20000)
    modelg.add_argument("--reasoning_marker", type=str, default="assistantfinal")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    shard_dir = out_dir / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs (full), then ensure shards deterministically (manifest-backed)
    all_reports = load_inputs(args)
    print(f"Loaded {len(all_reports)} rows.", flush=True)
    shards = ensure_shards(
        all_reports=all_reports,
        shard_dir=shard_dir,
        chunk_size=args.chunk_size,
        sample_n=args.note_sample_n,
        shuffle_seed=args.shuffle_seed
    )

    # Determine unfinished shards (no 'out_*.parquet' yet)
    unfinished = [p for p in shards if not (out_dir / f"out_{p.stem}.parquet").exists()]
    finished = len(shards) - len(unfinished)
    print(f"Found {len(shards)} total shard(s): {finished} finished, {len(unfinished)} unfinished.", flush=True)

    if not unfinished:
        print("Nothing to do: all shards already processed.", flush=True)
    else:
        # GPU grouping -> one kernel per group
        gpu_list = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
        groups = group_gpus(gpu_list, args.gpus_per_kernel)
        if not groups:
            raise ValueError("No valid GPU groups. Ensure you provided enough GPUs for gpus_per_kernel.")
        if len(groups) * args.gpus_per_kernel < len(gpu_list):
            print(f"Warning: leftover GPUs ignored: {gpu_list[len(groups)*args.gpus_per_kernel:]}", flush=True)

        num_workers = len(groups)
        print(f"Launching {num_workers} kernel(s): {groups}", flush=True)

        # Assign ONLY unfinished shards by worker stride to balance time
        per_worker_shards = [[] for _ in range(num_workers)]
        for i, spath in enumerate(unfinished):
            per_worker_shards[i % num_workers].append(spath)

        # Spawn workers (CUDA requires 'spawn')
        mp.set_start_method("spawn", force=True)

        procs = []
        for wid, gpu_group in enumerate(groups):
            p = mp.Process(
                target=worker_run,
                args=(
                    wid,
                    gpu_group,
                    per_worker_shards[wid],
                    out_dir,
                    args.model,
                    args.download_dir,
                    args.gpus_per_kernel,
                    args.gpu_memory_utilization,
                    args.max_model_len,
                    args.prompt_batch_size,
                    args.truncate_tokens,
                    args.reasoning_marker,
                ),
                daemon=False
            )
            p.start()
            procs.append(p)

        # Join
        for p in procs:
            p.join()

    # Merge whatever outputs exist now
    out_paths = sorted(out_dir.glob("out_shard_*.parquet"))
    if not out_paths:
        print("No shard outputs found; check failed_shards.txt for errors.", flush=True)
        return
    dfs = [pd.read_parquet(p) for p in out_paths]
    final_df = pd.concat(dfs, ignore_index=True)
    final_path = out_dir / args.final_output
    atomic_write_parquet(final_df, final_path)
    print(f"Wrote final merged file: {final_path} ({len(final_df)} rows).", flush=True)


if __name__ == "__main__":
    main()
