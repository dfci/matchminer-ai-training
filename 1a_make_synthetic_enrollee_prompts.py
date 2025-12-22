#!/usr/bin/env python3
import os
import sys
import math
import json
import glob
import shutil
import random
import argparse
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import set_start_method

# -----------------------------
# Prompts / formatting
# -----------------------------
SYSTEM_PROMPT = (
    "I'm a medical oncologist and data scientist. Your job is to help me create synthetic clinical data "
    "for cancer research. Reasoning: high."
)

USER_PROMPT_TEMPLATE_PREFIX = (
    "Imagine the longitudinal clinical history for 5 patients with cancer who enrolled on a clinical trial "
    "with the following eligibility criteria:\n"
)

USER_PROMPT_TEMPLATE_SUFFIX = (
    "\n For each patient, generate a list of events that might have occurred along the disease trajectory "
    "before the point of trial enrollment.\n"
    "Use everything you know about cancer and clinical oncology.\n"
    "Types of events might include a diagnosis as recorded in a cancer registry; initiation of a systemic therapy; "
    "a surgery; a radiation treatment; an adverse event; a clinical progress note; an imaging report; a pathology report; "
    "and a next-generation sequencing (NGS) report. For progress notes, imaging reports, pathology reports, and NGS reports, "
    "include findings documented in the report in the description of the event. \n"
    "NGS reports should be very detailed; they should include both any key actionable alterations if present, and "
    "comutations/fusions/copy number alterations; most reports should describe alterations in many genes, even though only some "
    "of those will be clinically relevant. \n"
    "CRITICAL: Genomic findings should make sense based on known mutation and comutation patterns. For example, remember that EGFR mutant lung cancers almost never have KRAS co-mutations. \n"
    "There should be one event per line of text in your output, and each event should be formatted as a sentence. \n"
    "Most patients will have many events along their disease trajectories (20-30). \n"
    "To ensure diversity in the generated data, vary patient age, gender, name, stage at diagnosis, biomarkers, treatment approaches, "
    "and disease course (e.g., stable disease, progression, remission, recurrence).\n"
    "Tag each event with an event type at the beginning of the line. Acceptable event types include <demographics>, <diagnosis>, "
    "<systemic>, <surgery>, <radiation>, <adverse_event>, <clinical_note>, <imaging_report>, <pathology_report>, and <ngs_report>.\n"
    "Each event should correspond only to one point in time, and each report should correspond only to one report that could have been written at that time.\n"
    "Diagnosis events should include TNM stage, summary stage, site description and code, histology description and code, and all relevant site-specific data elements "
    "that a cancer registrar would annotate.\n"
    "Imaging report events must describe only one radiographic study and should specify the type of study. Imaging report events must also indicate whether cancer was present "
    "on the scan; if so, whether it was responding, progressing, or neither; and what metastatic sites were involved.\n"
    "Oncologist note events must indicate whether cancer was present at the time; and if so, whether it was responding, progressing, or neither. \n"
    "NGS report events should indicate the diagnosis, specimen site, and genomic findings.\n"
    "Here is an example of what your output should look like. This is hypothetical, just to illustrate the formatting. Don't use this text in your output.\n"
    "Do adhere closely to the formatting.\n"
    "(Beginning of example)\n"
    "<demographics> The patient is a male named John Smith.\n"
    "<diagnosis>At age 70 years, the patient had a diagnosis of stage 1E (Best AJCC) LYMPHOMA, MALIG, DIFFUSE, NOS, (MALIGNANT, PRIMARY) of the BASAL GANGLIA. Relevant biomarkers included B Symptoms: 0: No B symptoms (asymptomatic) ; Classified as A by physician when asymptomatic. Other relevant diagnostic information included Confirmation: POSITIVE HISTOLOGY; Tumor Size: 15 mm.\n"
    "<pathology_report>At age 70, the patient had a pathology result of type ANATOMIC PATHOLOGY.\n"
    "<imaging_report>At age 70, the patient had a CT HEAD, which showed no cancer.\n"
    "<imaging_report>At age 70, the patient had a CT HEAD, which showed no cancer.\n"
    "<imaging_report>At age 70, the patient had a NM PET CT SCALP TO TOES, which showed no cancer.\n"
    "<imaging_report>At age 70, the patient had a MRI LUMBAR SPINE, which showed no cancer.\n"
    "<imaging_report>At age 70, the patient had a MRI THORACIC SPINE, which showed no cancer.\n"
    "<imaging_report>At age 70, the patient had a MRI CERVICAL SPINE, which showed no cancer.\n"
    "<systemic>At age 70 years, the patient received curative-intent methotrexate/temozolomide/rituximab.\n"
    "<pathology_report>At age 70, the patient had a pathology result of type CLINICAL ONCOPANEL.\n"
    "<clinical_note>At age 70, the patient had an oncologist office assessment, which showed cancer.\n"
    "<clinical_note>At age 70, the patient had an oncologist office assessment, which showed cancer. There was response to therapy.\n"
    "<clinical_note>At age 70, the patient had an oncologist office assessment, which showed cancer.\n"
    "<ngs_report>At age 70 years, the patient had next generation sequencing performed for diffuse large b-cell lymphoma based on a specimen obtained from a unspecified site (cns/brain), which showed a CD79B p.Y196F mutation, a CDKN2A loss, a CDKN2A p.W110* mutation, a ETV6 p.X11_splice mutation, a MYD88 p.L265P mutation, and a ERBB2 gain.\n"
    "<imaging_report>At age 71, the patient had a MRI BRAIN, which showed cancer. There was response to therapy.\n"
    "<clinical_note>At age 71, the patient had an oncologist office assessment, which showed cancer. There was response to therapy.\n"
    "<clinical_note>At age 71, the patient had an oncologist office assessment, which showed cancer. There was progression of disease.\n"
    "<imaging_report>At age 71, the patient had a MRI ABDOMEN, which showed no cancer.\n"
    "(End of example)\n\n"
    "Now, generate your output for the imagined patients who enrolled on the trial. \n"
    "CRITICAL: Do not mention screening or enrollment on the trial itself. That would contaminate the synthetic data and render them useless, since we will be using the synthetic data to derive clinical history models to match patients to clinical trials on which they have not yet enrolled.\n"
    "Separate the outputs for individual patients using the tag <new_patient>, which should go on its own line."
)

SECOND_PRIMARY_INSTRUCTION = (
    "In this particular group of patients, each patient have had a second primary prior unrelated cancer history before "
    "the cancer relevant to the trial. Incorporate events for the prior cancer as well as the current cancer into your output."
)

REASONING_MARKER = "assistantfinal"  # adjust if your model uses a different delimiter


def build_messages(criteria_text: str, rng: random.Random):
    content = USER_PROMPT_TEMPLATE_PREFIX + criteria_text + USER_PROMPT_TEMPLATE_SUFFIX
    # ~10% chance to include second-primary instruction
    if rng.randrange(10) == 5:
        content = content + "\n" + SECOND_PRIMARY_INSTRUCTION + "\n"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def process_outputs(request_outputs, reasoning_marker=REASONING_MARKER):
    full_responses, final_outputs = [], []
    for ro in request_outputs:
        text = ro.outputs[0].text
        full_responses.append(text)
        if reasoning_marker and reasoning_marker in text:
            final_outputs.append(text.split(reasoning_marker, 1)[-1])
        else:
            final_outputs.append(text)
    return full_responses, final_outputs


# -----------------------------
# Args / sharding
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Resumable synthetic clinical history generator per trial space.")
    p.add_argument("--input-csv", default="sample_trial_space_lineitems.csv", help="Input CSV with trial spaces.")
    p.add_argument("--output-csv", default="trial_spaces_with_positive_prompts.csv", help="Final output CSV path.")
    p.add_argument("--output-dir", default=None, help="Directory for shard outputs and logs. Default: <output_csv_basename>_shards/")
    p.add_argument("--gpu-groups", default="0|1|2|3|4|5|6|7",
                   help='GPU groups string. Examples: "0,1,2,3|4,5,6,7" (two workers, TP=4 each) or "0,1,2,3,4,5,6,7" (one worker, TP=8).')
    p.add_argument("--model-name", default="openai/gpt-oss-120b")
    p.add_argument("--download-dir", default="../meta_ai")
    p.add_argument("--gpu-mem-util", type=float, default=0.95)
    p.add_argument("--max-model-len", type=int, default=15000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard-size", type=int, default=1000, help="Number of rows per shard (contiguous).")
    p.add_argument("--batch-size", type=int, default=1000, help="Prompts per llm.generate() call inside a shard.")
    p.add_argument("--assemble-partial", action="store_true",
                   help="If not all shards are done, write a partial combined CSV (<output_csv>.partial).")
    return p.parse_args()


def make_gpu_groups(gpu_groups_arg: str):
    parts = [grp.strip() for grp in gpu_groups_arg.split("|")]
    parts = [p for p in parts if p]
    return parts


def ensure_dirs(base_dir: Path):
    (base_dir / "shards").mkdir(parents=True, exist_ok=True)
    (base_dir / "logs").mkdir(parents=True, exist_ok=True)


def shard_id_for_index(idx: int, shard_size: int) -> int:
    return idx // shard_size


def shard_bounds(shard_id: int, shard_size: int, n_rows: int):
    start = shard_id * shard_size
    end = min(n_rows, (shard_id + 1) * shard_size)
    return start, end


# -----------------------------
# Worker
# -----------------------------
def worker_process_shards(
    worker_id: int,
    gpu_group: str,
    shard_payloads: list,        # list of tuples: (shard_id, [records])
    model_name: str,
    download_dir: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    base_seed: int,
    batch_size: int,
    out_dir: str,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_group
    # Lazy import post CUDA visibility
    from vllm import LLM, SamplingParams

    local_devices = [d for d in gpu_group.split(",") if d.strip() != ""]
    tp = max(1, len(local_devices))

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp,
        download_dir=download_dir,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )

    completed = []
    for shard_id, recs in shard_payloads:
        shard_path = Path(out_dir) / "shards" / f"shard_{shard_id:06d}.csv"
        tmp_path = shard_path.with_suffix(".csv.part")
        log_path = Path(out_dir) / "logs" / f"shard_{shard_id:06d}.log"

        # Double-check skip inside worker too (in case of race)
        if shard_path.exists():
            completed.append(shard_id)
            continue

        try:
            # Build prompts
            order_keys, prompts = [], []
            for rec in recs:
                # deterministic per-row RNG
                ridx = int(rec["criteria_index"])
                messages = build_messages(rec["criteria"], rng=random.Random(base_seed + ridx))
                prompt = tokenizer.apply_chat_template(
                    conversation=messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                order_keys.append(ridx)
                prompts.append(prompt)

            # Generate in mini-batches to control memory footprint
            full_responses_all, final_outputs_all = [], []
            for i in range(0, len(prompts), batch_size):
                sub_prompts = prompts[i:i+batch_size]
                request_outputs = llm.generate(sub_prompts, sampling)
                fr, fo = process_outputs(request_outputs)
                full_responses_all.extend(fr)
                final_outputs_all.extend(fo)

            # Align and write shard file atomically
            shard_df = pd.DataFrame({
                "criteria_index": order_keys,
                "synthetic_patient_prompt_generation_reasoning": full_responses_all,
                "synthetic_patient_prompts": final_outputs_all,
            }).sort_values("criteria_index").reset_index(drop=True)

            shard_df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, shard_path)

            with open(log_path, "w") as f:
                f.write(json.dumps({
                    "worker_id": worker_id,
                    "gpu_group": gpu_group,
                    "shard_id": shard_id,
                    "n_rows": len(recs),
                }, indent=2))

            completed.append(shard_id)
            print(f"[worker {worker_id}] wrote shard {shard_id} -> {shard_path}", flush=True)

        except Exception as e:
            # Capture error and continue; shard will remain missing and be retried on resume
            err_path = Path(out_dir) / "logs" / f"shard_{shard_id:06d}.error.txt"
            with open(err_path, "w") as f:
                f.write("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            print(f"[worker {worker_id}] ERROR shard {shard_id}: {e}", file=sys.stderr, flush=True)

    return completed


# -----------------------------
# Assembly
# -----------------------------
def assemble_if_ready(spaces: pd.DataFrame, out_dir: Path, output_csv: Path, assemble_partial: bool):
    shard_files = sorted((out_dir / "shards").glob("shard_*.csv"))
    if not shard_files:
        print("No shard files present yet; skipping assembly.")
        return False

    shards_df = pd.concat([pd.read_csv(p) for p in shard_files], ignore_index=True)
    n_expected = len(spaces)
    n_have = len(shards_df)

    if n_have == n_expected:
        # Map back to original order by criteria_index
        merged = spaces.merge(
            shards_df[["criteria_index", "synthetic_patient_prompt_generation_reasoning", "synthetic_patient_prompts"]],
            on="criteria_index",
            how="left",
            validate="one_to_one",
        )
        # Ensure final order by original row order (criteria_index was set equal to original index)
        merged = merged.sort_values("criteria_index").reset_index(drop=True)

        # (Optional) drop helper columns like 'criteria' before writing
        if "criteria" in merged.columns:
            merged = merged.drop(columns=["criteria"])

        tmp_out = output_csv.with_suffix(output_csv.suffix + ".part")
        merged.to_csv(tmp_out, index=False)
        os.replace(tmp_out, output_csv)
        print(f"✅ Final output complete: {output_csv} ({len(merged)} rows).")
        return True
    else:
        remaining = n_expected - n_have
        print(f"Partial progress: {n_have}/{n_expected} rows available; {remaining} remaining.")
        if assemble_partial:
            # Write a partial combined CSV aligned to input order, leaving missing rows absent
            merged_partial = spaces.merge(
                shards_df[["criteria_index", "synthetic_patient_prompt_generation_reasoning", "synthetic_patient_prompts"]],
                on="criteria_index",
                how="left",
                validate="one_to_one",
            ).sort_values("criteria_index").reset_index(drop=True)
            partial_path = output_csv.with_suffix(output_csv.suffix + ".partial")
            tmp_partial = partial_path.with_suffix(partial_path.suffix + ".part")
            merged_partial.to_csv(tmp_partial, index=False)
            os.replace(tmp_partial, partial_path)
            print(f"✳️ Wrote partial output: {partial_path} ({n_have} filled rows).")
        return False


# -----------------------------
# Main
# -----------------------------
def main():
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    rng = np.random.default_rng(args.seed)

    output_csv = Path(args.output_csv)
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"{output_csv.stem}_shards")
    ensure_dirs(out_dir)

    # Load and prep input
    spaces = pd.read_csv(args.input_csv)
    spaces["this_space"] = spaces['this_space'].str.replace(r'^\s*\d+\.', '', regex=True).str.strip()
    spaces["trial_boilerplate_text"] = spaces["trial_boilerplate_text"].fillna("")
    spaces["criteria"] = spaces["this_space"] + "\nNo history of:\n" + spaces["trial_boilerplate_text"]
    spaces["criteria_index"] = spaces.index.astype(int)

    n_rows = len(spaces)
    shard_size = max(1, args.shard_size)
    total_shards = math.ceil(n_rows / shard_size)
    print(f"Total rows: {n_rows} | shard_size={shard_size} -> total_shards={total_shards}")

    # Quick exit if final already assembled (and valid)
    if output_csv.exists():
        print(f"Final output already exists: {output_csv}")
        return

    # Build payloads per shard (contiguous blocks)
    all_shard_ids = list(range(total_shards))
    shard_dir = out_dir / "shards"
    done_shard_ids = sorted([int(p.stem.split("_")[-1]) for p in shard_dir.glob("shard_*.csv")])

    missing_shard_ids = [sid for sid in all_shard_ids if sid not in done_shard_ids]
    if not missing_shard_ids:
        # Nothing to do; try to assemble final (or verify)
        assembled = assemble_if_ready(spaces, out_dir, output_csv, args.assemble_partial)
        if assembled:
            return
        else:
            # If somehow mismatch, fall through to regeneration of any missing rows (shouldn't happen)
            pass

    # Prepare shard records
    def build_shard_records(shard_id: int):
        s, e = shard_bounds(shard_id, shard_size, n_rows)
        block = spaces.iloc[s:e][["criteria_index", "criteria"]]
        return [{"criteria_index": int(ci), "criteria": c} for ci, c in zip(block["criteria_index"], block["criteria"])]

    # Distribute missing shards to workers round-robin
    gpu_groups = make_gpu_groups(args.gpu_groups)
    n_workers = max(1, len(gpu_groups))
    worker_payloads = [[] for _ in range(n_workers)]
    for i, sid in enumerate(missing_shard_ids):
        worker_payloads[i % n_workers].append((sid, build_shard_records(sid)))

    # Launch workers
    futures = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for wid, (grp, payload) in enumerate(zip(gpu_groups, worker_payloads)):
            if not payload:
                continue
            futures.append(
                ex.submit(
                    worker_process_shards,
                    wid,
                    grp,
                    payload,
                    args.model_name,
                    args.download_dir,
                    args.gpu_mem_util,
                    args.max_model_len,
                    args.temperature,
                    args.top_p,
                    args.max_tokens,
                    args.repetition_penalty,
                    args.seed,
                    args.batch_size,
                    str(out_dir),
                )
            )

        for fut in as_completed(futures):
            try:
                completed = fut.result()
                print(f"Worker completed shards: {completed}")
            except Exception as e:
                print(f"Worker crashed: {e}", file=sys.stderr)

    # Assemble if everything is present (or write partial if requested)
    assemble_if_ready(spaces, out_dir, output_csv, args.assemble_partial)


if __name__ == "__main__":
    main()
