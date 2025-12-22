#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel trial-space extraction with vLLM.

Usage examples
--------------
# One instance per GPU (model fits on a single GPU):
python 0b_create_trial_spaces.py \
  --input ctgov_trials.csv \
  --gpus 2,3 \
  --gpus-per-instance 1 \
  --download-dir /data1/ken/meta/2024/meta_ai


Outputs
-------
- trials_with_spaces.csv
- trial_space_lineitems.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from uuid import uuid4

# vLLM imports happen inside worker after setting CUDA_VISIBLE_DEVICES


PROMPT_HEADER = (
    "You are an expert clinical oncologist with an encyclopedic knowledge of cancer and its treatments.\n"
    "Your job is to review a clinical trial document and extract a list of structured clinical spaces that are eligible for that trial.\n"
    "A clinical space is defined as a unique combination of patient age range, sex (if any sex criteria), cancer primary site, histology, which treatments a patient must have received, "
    "which treatments a patient must not have received, cancer burden (eg presence of metastatic disease; this also includes cancer type-specific prognostic scores, risk indices, or categories), tumor biomarkers (such as "
    "germline or somatic gene mutations or alterations, or protein expression on tumor), that a patient must have or must not have to "
    "be eligible for the trial. \n"
    "With respect to sex criteria: For cancers originating in organs only present in one sex, you must assume the sex criteria even if not stated explicitly.\n"
    "For example, a trial space for uterine, ovarian, vulvar, vaginal, or fallopian tube cancer must be assumed to be for female patients.\n"
    "Similarly, a trial space for testicular, penile, or prostate cancer must be assumed to be for male patients.\n"
    "For all other cancer types (including breast cancer), you shoulud assume the trial is open to both sexes unless the clinical trial document states otherwise.\n"
    "Trials often specify that a particular treatment is excluded only if it was given within a short period of time, for example 14 days, "
    "one month, etc , prior to trial start. This is called a washout period. Do not include this type of time-specific treatment washout "
    "eligibility criteria in your output at all.\n"
    "Some trials have only one space, while others have several. Do not output a space that contains multiple cancer types and/or histologies. "
    "Instead, generate separate spaces for each cancer type/histology combination.\n"
    "CRITICAL: Each trial space must contain all information necessary to define that space on its own. It may not refer to other previously "
    "defined spaces for the same trial, since for later use, the spaces will be extracted and separated from each other. YOU MAY NOT include "
    "text describing a given space that refers to a previous space; eg, \"Same as above\"-style output is not allowed!\n"
    "For biomarkers, if the trial specifies whether the biomarker will be assessed during screening, note that.\n"
    "Spell out cancer types; do not abbreviate them. For example, write \"non-small cell lung cancer\" rather than \"NSCLC\".\n"
    "Structure your output like this, as a list of spaces, with spaces separated by newlines, as below. STRICTLY adhere to the formatting.\n"
    "1. Age range allowed: <age_range_allowed>. Sex allowed: <sex_allowed>. Cancer type allowed: <cancer_type_allowed>. Histology allowed: <histology_allowed>. Cancer burden allowed: <cancer_burden_allowed>. Prior treatment required: <prior_treatments_requred>. Prior treatment excluded: <prior_treatments_excluded>. Biomarkers required: <biomarkers_required>. Biomarkers excluded: <biomarkers_excluded>. \n"
    "2. Cancer type allowed: <cancer_type_allowed>, etc.\n"
    "If a concept is not relevant, such as if there are no prior treatents required, simply output NA for that concept.\n"
    "CRITICAL: Anytime you provide a list for a particular concept, you must be completely clear on whether \"or\" versus \"and\" logic applies "
    "to the list. For example, do not output \"EGFR L858R mutant, TP53 mutant\"; if both are required, output \"EGFR L858R mutant and TP53 mutant\". "
    "As another example, do not output \"ER+, PR+\"; if the patient can have either an ER or a PR positive tumor, output \"ER+ or PR+\".\n"
    "NEVER put a newline within a single trial space.\n"
    "After you output the trial spaces, output a newline, then the text \"Boilerplate exclusions:\" VERBATIM, then another newline.\n"
    "Then, list exclusion criteria described in the trial text that are unrelated to the trial space definitions. Such exclusions tend to be common "
    "to clinical trials in general.\n"
    "Common boilerplate exclusion criteria include a history of pneumonitis, heart failure, renal dysfunction, liver dysfunction, uncontrolled brain "
    "metastases, HIV or hepatitis, and poor performance status.\n"
    "ALWAYS output plain text only. NEVER output unicode, Markdown, or tables.\n"
)

PROMPT_SUFFIX = (
    "Now, generate your list of the trial space(s), followed by any boilerplate exclusions, formatted as above.\n"
    "Do not provide any introductory, explanatory, concluding, or disclaimer text.\n"
    "Reminder: Treatment history is an important component of trial space definitions, but treatment history \"washout\" requirements that are "
    "described as applying only in a given period of time prior to trial treatment MUST BE IGNORED.\n"
    "CRITICAL: A given trial space MUST NEVER refer to another previously defined space. You must NEVER output text like \"same as #1\" or "
    "\"same criteria as above.\" Instead, you MUST REPEAT all relevant criteria for each new space SO THAT IT STANDS ON ITS OWN. A user who later "
    "looks at the text for one space will not have access to text for other spaces, and so output like \"Same criteria as #1...\" renders a space useless!"
)

REASONING_MARKER = "assistantfinal"   # adjust if your model uses a different marker
BOILERPLATE_MARKER = "Boilerplate"


def build_messages(trial_text: str):
    """Chat messages for a single trial."""
    return [
        {"role": "system", "content": "Reasoning: high."},
        {
            "role": "user",
            "content": (
                PROMPT_HEADER
                + "Here is a clinical trial document:\n"
                + str(trial_text)
                + "\n"
                + PROMPT_SUFFIX
            ),
        },
    ]


def postprocess_outputs(raw_texts):
    """
    Split out (1) full raw text (reasoning+final), (2) final only,
    (3) space_text (before boilerplate line), and (4) boilerplate text.
    
    The boilerplate split is performed on the ENTIRE line containing 
    the marker, removing that line from both resultant parts.
    """
    no_reasoning = []
    space_only = []
    boiler_only = []

    for t in raw_texts:
        # 1. Handle Reasoning Split (Same as original)
        final_t = t.split(REASONING_MARKER, 1)[-1] if REASONING_MARKER in t else t

        # 2. Handle Boilerplate Split (Line-based)
        if BOILERPLATE_MARKER in final_t:
            # Split into lines, keeping newlines to preserve formatting
            lines = final_t.splitlines(keepends=True)
            
            # Find the index of the line containing the marker
            split_idx = -1
            for i, line in enumerate(lines):
                if BOILERPLATE_MARKER in line:
                    split_idx = i
                    break
            
            if split_idx != -1:
                # Join lines before the marker line
                space_part = "".join(lines[:split_idx])
                # Join lines after the marker line
                boiler_part = "".join(lines[split_idx+1:])
                
                space_only.append(space_part)
                boiler_only.append(boiler_part)
            else:
                # Fallback if marker was in string but not found in line iteration (unlikely edge case)
                space_only.append(final_t)
                boiler_only.append(final_t)
        else:
            space_only.append(final_t)
            boiler_only.append(final_t)

        no_reasoning.append(final_t)

    return no_reasoning, space_only, boiler_only


def worker_process(
    shard_in_path: str,
    shard_out_path: str,
    gpu_group: list,
    model: str,
    download_dir: str,
    max_model_len: int,
    gpu_mem_util: float,
    temperature: float,
    top_k: int,
    max_tokens: int,
    repetition_penalty: float,
    batch_size: int,
):
    """
    Worker: loads its own vLLM on the specified GPU group (via CUDA_VISIBLE_DEVICES),
    processes its shard, and writes the shard output to `shard_out_path` (parquet).
    """
    # Bind this process to its GPU set
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_group)

    # Import vLLM after the environment is set
    from vllm import LLM, SamplingParams

    tp_size = len(gpu_group)

    # Load shard
    shard_df = pd.read_parquet(shard_in_path)

    # Start model
    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        download_dir=download_dir,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
    )
    tokenizer = llm.get_tokenizer()

    # Build prompts
    trial_texts = shard_df["trial_text"].astype(str).tolist()
    prompts = [
        tokenizer.apply_chat_template(
            conversation=build_messages(t),
            add_generation_prompt=True,
            tokenize=False,
        )
        for t in trial_texts
    ]

    # Generate in batches
    raw_texts = []
    sampling = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        # If needed, you can add stop_token_ids here:
        # stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    )

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        results = llm.generate(batch, sampling)
        for r in results:
            raw_texts.append(r.outputs[0].text)

    # Post-process
    no_reasoning, space_only, boiler_only = postprocess_outputs(raw_texts)

    # Attach outputs
    out_df = shard_df.copy()
    out_df["space_reasoning_and_output"] = raw_texts
    out_df["space_output_no_reasoning"] = no_reasoning
    out_df["space_text"] = space_only
    out_df["trial_boilerplate_text"] = boiler_only

    # Persist this shard
    out_df.to_parquet(shard_out_path, index=False)


def split_into_groups(items, group_size):
    """Split a list into fixed-size groups; drop any remainder smaller than group_size."""
    groups = []
    for i in range(0, len(items), group_size):
        g = items[i : i + group_size]
        if len(g) == group_size:
            groups.append(g)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV of trials (ctgov_cancer_trials_9-2025.csv)")
    parser.add_argument("--output-trials", default="trials_with_spaces.csv")
    parser.add_argument("--output-spaces", default="trial_space_lineitems.csv")
    parser.add_argument("--work-dir", default="trial_space_shards", help="Directory for shard inputs/outputs")
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU IDs, e.g. 0,1,2,3")
    parser.add_argument("--gpus-per-instance", type=int, default=1, help="Tensor-parallel GPUs per instance (set >1 for very large models)")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--download-dir", default="../meta_ai")
    parser.add_argument("--max-model-len", type=int, default=10000)
    parser.add_argument("--gpu-mem-util", type=float, default=0.94)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=7500)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--batch-size", type=int, default=1000, help="Prompts per generate() call per instance")
    parser.add_argument("--seed", type=int, default=42, help="Seed for split assignment")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    # Load and pre-process trials
    trials = pd.read_csv(args.input)
    # Deterministic split assignment
    rng = np.random.default_rng(args.seed)
    trials = trials.copy()
    trials["split"] = rng.choice(["train", "val", "test"], size=len(trials), p=[0.8, 0.1, 0.1])

    # Deduplicate by NCT ID like your original
    trials = trials.groupby("nct_id").first().reset_index()

    # We only generate from trial_text, but keep the rest for merging later
    # Keep the original column set so we can write the same outputs
    base_cols = trials.columns.tolist()

    # Determine GPU groups / instances
    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    if len(gpu_list) == 0:
        raise ValueError("No GPUs provided.")
    if args.gpus_per_instance < 1:
        raise ValueError("--gpus-per-instance must be >= 1")

    gpu_groups = split_into_groups(gpu_list, args.gpus_per_instance)
    if len(gpu_groups) == 0:
        raise ValueError(
            f"Not enough GPUs ({len(gpu_list)}) to form at least one instance with --gpus-per-instance={args.gpus_per_instance}"
        )

    n_instances = len(gpu_groups)
    shards = np.array_split(trials, n_instances)

    # Persist shard inputs and schedule workers
    shard_in_paths = []
    shard_out_paths = []
    for i, shard_df in enumerate(shards):
        in_path = os.path.join(args.work_dir, f"input_shard_{i:03d}.parquet")
        out_path = os.path.join(args.work_dir, f"output_shard_{i:03d}.parquet")
        # Ensure column order preserved
        shard_df = shard_df[base_cols]
        shard_df.to_parquet(in_path, index=False)
        shard_in_paths.append(in_path)
        shard_out_paths.append(out_path)

    # Launch one process per instance
    procs = []
    ctx = mp.get_context("spawn")
    for i, gpu_group in enumerate(gpu_groups):
        p = ctx.Process(
            target=worker_process,
            kwargs=dict(
                shard_in_path=shard_in_paths[i],
                shard_out_path=shard_out_paths[i],
                gpu_group=gpu_group,
                model=args.model,
                download_dir=args.download_dir,
                max_model_len=args.max_model_len,
                gpu_mem_util=args.gpu_mem_util,
                temperature=args.temperature,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                repetition_penalty=args.repetition_penalty,
                batch_size=args.batch_size,
            ),
            name=f"vllm_worker_{i}",
            daemon=False,
        )
        p.start()
        procs.append(p)

    # Join all
    exit_codes = []
    for p in procs:
        p.join()
        exit_codes.append(p.exitcode)

    if any(ec != 0 for ec in exit_codes):
        bad = [i for i, ec in enumerate(exit_codes) if ec != 0]
        raise RuntimeError(f"One or more workers failed: {bad}")

    # Merge shard outputs
    out_parts = [pd.read_parquet(pth) for pth in shard_out_paths]
    trials_with_spaces = pd.concat(out_parts, axis=0, ignore_index=True)

    # Keep and order columns as in your original final selection
    final_cols = [
        "nct_id",
        "split",
        "title",
        "brief_summary",
        "eligibility_criteria",
        "trial_text",
        "space_reasoning_and_output",
        "space_output_no_reasoning",
        "space_text",
        "trial_boilerplate_text",
    ]
    # Some inputs may not have every column (defensive handling)
    final_cols = [c for c in final_cols if c in trials_with_spaces.columns]
    trials_with_spaces = trials_with_spaces[final_cols]

    # Write trials_with_spaces.csv
    trials_with_spaces.to_csv(args.output_trials, index=False)

    # Build line-items dataframe
    tws = trials_with_spaces.copy()
    tws["space_text"] = tws["space_text"].astype(str)

    frames = []
    for i in range(tws.shape[0]):
        row = tws.iloc[[i]]  # 1-row DataFrame
        cohorts = pd.Series(str(row.iloc[0]["space_text"]).split("\n"))
        cohorts = cohorts[~(cohorts.isna() | (cohorts == "\n") | (cohorts == ""))].reset_index(drop=True)
        if len(cohorts) == 0:
            continue
        frame = pd.DataFrame(np.repeat(row.values, len(cohorts), axis=0), columns=row.columns)
        frame["this_space"] = cohorts.values

        # quality check - require all space components
        search_list = ["Age", "Sex", "Cancer type", "Histology", "Cancer burden", "Prior treatment", "Prior treatment", "Biomarkers required", "Biomarkers excluded"]
        frame = frame[frame.this_space.apply(lambda x: all(term in x for term in search_list))]
        
        # 1-based space numbering within this trial
        frame["space_number"] = np.arange(1, frame.shape[0] + 1)
        frames.append(frame)

    if frames:
        cohort_level_trials = pd.concat(frames, axis=0, ignore_index=True)
        # Keep only lines that look like numbered spaces ("1.", "2.", ...)
        keep_mask = cohort_level_trials["this_space"].str.strip().str[0].isin(list("123456789"))
        cohort_level_trials = cohort_level_trials[keep_mask]
        # Remove numbers from space names now
        cohort_level_trials["this_space"] = cohort_level_trials['this_space'].str.replace(r'^\s*\d+\.', '', regex=True).str.strip()   

        # Order columns similar to your original (dropping the accidental duplicate nct_id)
        ordered_cols = [
            "nct_id",
            "split",
            "title",
            "eligibility_criteria",
            "trial_text",
            "space_reasoning_and_output",
            "space_output_no_reasoning",
            "space_text",
            "trial_boilerplate_text",
            "this_space",
            "space_number",
        ]
        ordered_cols = [c for c in ordered_cols if c in cohort_level_trials.columns]
        cohort_level_trials = cohort_level_trials[ordered_cols]
        cohort_level_trials.to_csv(args.output_spaces, index=False)
    else:
        # Still emit an empty file with headers if no frames
        pd.DataFrame(
            columns=[
                "nct_id",
                "split",
                "title",
                "eligibility_criteria",
                "trial_text",
                "space_reasoning_and_output",
                "space_output_no_reasoning",
                "space_text",
                "trial_boilerplate_text",
                "this_space",
                "space_number",
            ]
        ).to_csv(args.output_spaces, index=False)


if __name__ == "__main__":
    main()
