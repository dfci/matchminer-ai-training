#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel synthetic note generation with one vLLM instance per GPU,
plus optional drug-name perturbation (generic -> brand/abbrev) per note.

Usage example:
python 2_make_synthetic_notes_sharded.py \
  --input_csv trial_spaces_with_positive_prompts.csv \
  --out_dir ./synthetic_notes \
  --model openai/gpt-oss-120b \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --download_dir ../meta_ai \
  --max_model_len 10000 --max_new_tokens 5000 \
  --batch_size 1000 --tp 1 --temperature 0.75 --top_p 0.5 \
  --perturb_prob 0.3

Notes:
- If your model cannot fit on a single GPU, set --tp > 1 and supply GPU ids in
  contiguous groups per worker (e.g., --gpu_ids "0,1;2,3" for 2 workers, each tp=2).
- The script persists the intermediate `promptframe.parquet` and writes
  per-worker shard files to `out_dir/shards/rows{START}-{END}.parquet` (deterministic),
  then composes `out_dir/synthetic_notes.parquet` at the end.
- Safe resume: existing shard files are detected and skipped by default.
"""

import os
import re
import uuid
import math
import argparse
import glob
import multiprocessing as mp
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np

# ------------------------------
# Data preprocessing (unchanged logic, packaged)
# ------------------------------

def process_clinical_notes_from_df(input_frame: pd.DataFrame) -> pd.DataFrame:
    all_patients_data = []
    unique_spaces = input_frame.space_index.unique()

    for i in unique_spaces:
        df = input_frame[input_frame.space_index == i]
        for _, row in df.iterrows():
            if not isinstance(row.get('synthetic_patient_prompts', None), str):
                continue
            patient_histories = row['synthetic_patient_prompts'].split('<new_patient>')
            for history in patient_histories:
                history = history.strip()
                if not history:
                    continue
                patient_id = f"patient_{uuid.uuid4()}"
                patient_events = []
                pattern = re.compile(r'<(\w+)>(.*?)(?=<|\Z)', re.DOTALL)
                matches = pattern.findall(history)
                for event_type, event_text in matches:
                    patient_events.append({
                        'patient_id': patient_id,
                        'space_index': i,
                        'event_type': event_type.strip(),
                        'event_text': event_text.strip().replace('\n', ' ')
                    })
                if patient_events:
                    all_patients_data.extend(patient_events)

    if not all_patients_data:
        print("No patient data found or processed.")
        return pd.DataFrame()

    final_df = pd.DataFrame(all_patients_data)
    unique_ids = final_df['patient_id'].unique()
    id_to_mrn_map = {patient_id: i for i, patient_id in enumerate(unique_ids, 1)}
    final_df['pseudo_mrn'] = final_df['patient_id'].map(id_to_mrn_map)
    final_df = final_df[['pseudo_mrn', 'space_index', 'patient_id', 'event_type', 'event_text']]
    return final_df


def create_masked_dataset(patient_df: pd.DataFrame) -> pd.DataFrame:
    event_types_to_keep = ['clinical_note', 'imaging_report', 'pathology_report', 'ngs_report']
    filtered_df = patient_df[patient_df['event_type'].isin(event_types_to_keep)].copy()
    full_histories = patient_df.groupby('patient_id')['event_text'].apply(list).to_dict()

    def generate_masked_string(row: pd.Series) -> str:
        patient_id = row['patient_id']
        current_event_text = row['event_text']
        current_event_type = row['event_type']
        patient_history_list = full_histories.get(patient_id, [])
        masked_history_list = []
        is_masked = False
        for event_text in patient_history_list:
            if event_text == current_event_text and not is_masked:
                masked_event = (
                    "<BEGIN EVENT CORRESPONDING TO SYNTHETIC NOTE> " +
                    f"<{current_event_type}>{event_text}" +
                    " <END EVENT CORRESPONDING TO SYNTHETIC NOTE>"
                )
                masked_history_list.append(masked_event)
                is_masked = True
            else:
                masked_history_list.append(event_text)
        return "\n".join(masked_history_list)

    filtered_df['masked_text'] = filtered_df.apply(generate_masked_string, axis=1)
    return filtered_df


# ------------------------------
# NEW: Drug-perturbation utilities
# ------------------------------

# NEW: A practical starter map (generic -> [brand/abbrev choices]).
# Brand pairs grounded from NCI Drug Dictionary / NCI “About Cancer Treatment Drugs” pages.
DEFAULT_DRUG_MAP: Dict[str, List[str]] = {
    # PD-(L)1 & CTLA-4
    "pembrolizumab": ["Keytruda", "pembro"],
    "nivolumab": ["Opdivo", "nivo"],
    "ipilimumab": ["Yervoy", "ipi"],
    "atezolizumab": ["Tecentriq", "atezo"],
    "durvalumab": ["Imfinzi", "durva"],
    "cemiplimab": ["Libtayo", "cemi"],

    # Platinums & taxanes
    "carboplatin": ["Paraplatin", "carbo"],
    "cisplatin": ["Platinol", "cis"],
    "oxaliplatin": ["Eloxatin", "oxali"],
    "paclitaxel": ["Taxol", "pacli", "PTX"],
    "docetaxel": ["Taxotere", "doce"],
    "nab-paclitaxel": ["Abraxane", "nab-pac"],

    # Antimetabolites
    "capecitabine": ["Xeloda", "cape"],
    "fluorouracil": ["5-FU", "Adrucil", "5FU"],
    "5-fluorouracil": ["5-FU", "Adrucil", "5FU"],
    "gemcitabine": ["Gemzar", "gem"],
    "pemetrexed": ["Alimta", "peme", "pem"],
    "methotrexate": ["MTX", "Trexall"],

    # Anthracyclines & others
    "doxorubicin": ["Adriamycin", "doxo"],
    "epirubicin": ["Ellence", "epi"],
    "cyclophosphamide": ["Cytoxan", "CTX", "cyclo"],
    "etoposide": ["VP-16", "eto"],
    "irinotecan": ["Camptosar", "iri"],
    "topotecan": ["Hycamtin", "topo"],

    # HER2 axis
    "trastuzumab": ["Herceptin", "trast"],
    "pertuzumab": ["Perjeta", "pertu"],
    "ado-trastuzumab emtansine": ["Kadcyla", "T-DM1"],
    "trastuzumab emtansine": ["Kadcyla", "T-DM1"],
    "trastuzumab deruxtecan": ["Enhertu", "T-DXd"],
    "tucatinib": ["Tukysa", "tuca"],
    "lapatinib": ["Tykerb", "lapa"],

    # CDK4/6
    "palbociclib": ["Ibrance", "palbo"],
    "ribociclib": ["Kisqali", "ribo"],
    "abemaciclib": ["Verzenio", "abema"],

    # PARP
    "olaparib": ["Lynparza", "ola"],
    "niraparib": ["Zejula", "nira"],
    "rucaparib": ["Rubraca", "ruca"],
    "talazoparib": ["Talzenna", "tala"],

    # VEGF axis
    "bevacizumab": ["Avastin", "bev"],
    "ramucirumab": ["Cyramza", "ramu"],

    # EGFR, ALK, etc.
    "osimertinib": ["Tagrisso", "osi"],
    "erlotinib": ["Tarceva", "erlo"],
    "gefitinib": ["Iressa", "gefi"],
    "afatinib": ["Gilotrif", "afat"],
    "dacomitinib": ["Vizimpro", "daco"],
    "alectinib": ["Alecensa", "alec"],
    "ceritinib": ["Zykadia", "ceri"],
    "crizotinib": ["Xalkori", "crizo"],
    "lorlatinib": ["Lorbrena", "lorla"],

    # BRAF/MEK
    "dabrafenib": ["Tafinlar", "dabra"],
    "trametinib": ["Mekinist", "tram"],
    "vemurafenib": ["Zelboraf", "vem"],
    "encorafenib": ["Braftovi", "enco"],
    "binimetinib": ["Mektovi", "bini"],

    # Multi-TKIs, etc.
    "lenvatinib": ["Lenvima", "lenva"],
    "sorafenib": ["Nexavar", "sora"],
    "regorafenib": ["Stivarga", "rego"],
    "pazopanib": ["Votrient", "pazo"],
    "sunitinib": ["Sutent", "suni"],

    # Antibodies (other)
    "rituximab": ["Rituxan", "ritux"],
    "cetuximab": ["Erbitux", "cetux"],
    "panitumumab": ["Vectibix", "pani"],

    # GU agents
    "enzalutamide": ["Xtandi", "enza"],
    "abiraterone": ["Zytiga", "abi"],
    "apalutamide": ["Erleada", "apa"],
    "leuprolide": ["Lupron", "leup"],
    "degarelix": ["Firmagon", "dega"],
    "relugolix": ["Orgovyx", "relu"],

    # mTOR, alkylators, myeloma, etc.
    "everolimus": ["Afinitor", "evero"],
    "sirolimus": ["Rapamune", "siro"],
    "temozolomide": ["Temodar", "TMZ"],
    "bortezomib": ["Velcade", "bortez"],
    "carfilzomib": ["Kyprolis", "carfil"],
    "ixazomib": ["Ninlaro", "ixa"],
    "daratumumab": ["Darzalex", "dara"],
    "obinutuzumab": ["Gazyva", "obinu"],
}

# NEW: Build compiled regex patterns sorted by key length (avoid partial overlaps).
def _compile_replacement_patterns(drug_map: Dict[str, List[str]]):
    items = sorted(drug_map.items(), key=lambda kv: len(kv[0]), reverse=True)
    compiled = []
    for generic, alts in items:
        # word-boundary match; case-insensitive; escape generic
        pat = re.compile(rf"\b{re.escape(generic)}\b", flags=re.IGNORECASE)
        compiled.append((pat, alts))
    return compiled

# NEW: Load from CSV (optional). CSV with columns: generic, alternatives (pipe-separated).
def load_drug_map(csv_path: str | None) -> Dict[str, List[str]]:
    if not csv_path:
        return DEFAULT_DRUG_MAP
    df = pd.read_csv(csv_path)
    out: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        g = str(r["generic"]).strip()
        alts = [a.strip() for a in str(r["alternatives"]).split("|") if str(a).strip()]
        if g and alts:
            out[g] = alts
    return out

# NEW: Apply all replacements to a text using compiled patterns and RNG
def replace_generics_with_alternatives(text: str, patterns, rng: np.random.Generator) -> str:
    for pat, alts in patterns:
        # Replace every match with a random alternative for that generic
        text = pat.sub(lambda m: rng.choice(alts), text)
    return text


# ------------------------------
# Worker: one process per (GPU group), one vLLM instance inside
# ------------------------------

def build_prompts(masked_texts: List[str], tokenizer, max_ctx_tokens: int = 40000) -> List[str]:
    prompts = []
    for masked_text in masked_texts:
        # Trim very long histories by taking head+tail
        patient_text_tokens = tokenizer(masked_text, add_special_tokens=False).input_ids
        if len(patient_text_tokens) > max_ctx_tokens:
            first_part = patient_text_tokens[: max_ctx_tokens // 2]
            last_part = patient_text_tokens[- max_ctx_tokens // 2 :]
            masked_text = tokenizer.decode(first_part) + " ... " + tokenizer.decode(last_part)

        messages = [{'role':'system', 'content': "Reasoning: high"},
                    {'role':'user', 'content': """You are a brilliant synthetic clinical document generation bot with encyclopedic knowledge about cancer and its treatment. 
You will be given a semi-structured list of events from a patient's clinical history, with each event on its own line of text.
The events are sorted in chronological order.
One of these events will be surrounded by the tags <BEGIN EVENT CORRESPONDING TO SYNTHETIC NOTE> and <END EVENT CORRESPONDING TO SYNTHETIC NOTE>.
Your job is to create a synthetic clinical document corresponding to the event denotated by those tags.
The synthetic document should be a pathology report, an imaging report, or a clinical progress note, as directed by the text within the tags.
Incorporate everything you know about the patient's history, and about cancer generally, to synthesize the document.
Don't directly incorporate information about future events as if they have already occurred, but you can use your knowledge of the future to inform what the synthetic document might have contained at the time it was written.
CRITICAL: Ignore your knowledge of today's date. Do not add dates to the synthetic notes. These will be added later and programatically.
The document should be extremely detailed so it is as realistic as possible. Pathology reports and imaging reports should be about one page long. Clinical progress notes should be about two pages long. Clinical progress notes should be written as a real oncologist would write them; this should include often using common brand names (eg Herceptin, Keytruda, Taxol) and sometimes using generic drug names (eg trastuzumab, pembrolizumab, paclitaxel), and sometimes using abbreviations (eg pembro instead of pembrolizumab, cape instead of capecitabine).
For pathology reports, sections should include specimen ID, date of procedure, type of specimen, diagnostic findings, any ancillary studies, and a description of gross pathology if relevant. Pathology reports should NOT include recommendations about management, since these are not part of real pathology reports.
For imaging reports, sections should include scan type, Findings (broken down by organs imaged by the study), and Impression. 
For clinical notes, sections should include chief complaint, history of present illness, review of systems, physical exam, lab results, imaging results, and assessment/plan. If it is the first clinical note in a given department, it is a consult note, in which case it should also include past medical history, social history, family history, allergies, and medications, all of which should come between review of systems and physical exam.
CRITICAL: Pathology reports and imaging reports should not make treatment or monitoring recommendations.
Within clinical notes and pathology reports, if you do not have any information about key biomarkers explicitly provided, you should imagine what they might be based on cancer type, history, and prior treatments. However, these must be consistent with realistic biological patterns. For example, as you know, EGFR mutant lung cancers almost never have concomittant driver mutations in KRAS, BRAF, etc.
CRITICAL: Do not invent treatments that are not included in the semi-structured list of events.
Within clinical notes, sometimes patients should have adverse events of therapy and/or comorbidities described that are consistent with their clinical trajectories.
Do not include any disclaimers or notes about the fact that the document is synthetic; this is all for research purposes only.
Here is the list of events:\n""" + masked_text + """\nNow, generate the synthetic document corresponding to the notated event."""}]
        prompt = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)
    return prompts


def _find_existing_shard(shards_dir: str, global_lo: int, global_hi: int) -> str:
    deterministic = os.path.join(shards_dir, f"rows{global_lo}-{global_hi}.parquet")
    if os.path.exists(deterministic):
        return deterministic
    legacy_pattern = os.path.join(shards_dir, f"*rows{global_lo}-{global_hi}*.parquet")
    matches = glob.glob(legacy_pattern)
    return matches[0] if matches else ""


def _write_parquet_atomic(df: pd.DataFrame, out_path: str):
    tmp_path = out_path + ".tmp"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)


def run_worker(
    worker_id: int,
    gpu_spec: str,
    parquet_path: str,
    row_id_start: int,
    row_id_end: int,
    out_dir: str,
    model: str,
    download_dir: str,
    tp: int,
    max_model_len: int,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    gpu_mem_util: float,
    overwrite_existing: bool,
    # NEW args for perturbation:
    perturb_prob: float,
    drug_map_csv: str | None,
    perturb_seed: int,
):
    """
    gpu_spec:
      - For tp=1: a single GPU id string, e.g., "3"
      - For tp>1: semicolon-joined group like "0,1" (already grouped by caller)
    """
    # --- GPU scoping BEFORE importing vllm ---
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_spec
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Import here to avoid CUDA re-init issues
    from vllm import LLM, SamplingParams

    # Load my shard
    df = pd.read_parquet(parquet_path)
    shard = df[(df['row_id'] >= row_id_start) & (df['row_id'] < row_id_end)].copy()
    shard = shard.reset_index(drop=True)
    if shard.empty:
        print(f"[worker {worker_id}] No rows for range {row_id_start}:{row_id_end}")
        return

    # Output folder for this worker's parts
    shard_out_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_out_dir, exist_ok=True)

    # Determine planned parts
    n = shard.shape[0]
    num_parts = math.ceil(n / batch_size)

    # Pre-scan to count how many to skip
    to_skip = 0
    for part_idx in range(num_parts):
        rel_lo = part_idx * batch_size
        rel_hi = min((part_idx + 1) * batch_size, n)
        global_lo = row_id_start + rel_lo
        global_hi = row_id_start + rel_hi
        existing = _find_existing_shard(shard_out_dir, global_lo, global_hi)
        if existing and not overwrite_existing:
            to_skip += 1
    print(f"[worker {worker_id}] Planned parts: {num_parts}; skipping {to_skip} existing; generating {num_parts - to_skip}.")

    # Init vLLM lazily
    llm = None
    tokenizer = None
    sampling = None
    reasoning_marker = "assistantfinal"  # adjust if your model uses a different separator

    # NEW: prepare perturbation machinery once per worker
    do_perturb = perturb_prob and perturb_prob > 0.0
    rng = np.random.default_rng((perturb_seed ^ (worker_id + 1) ^ (row_id_start + 1234567)) & 0xFFFFFFFF)
    drug_map = load_drug_map(drug_map_csv) if do_perturb else {}
    repl_patterns = _compile_replacement_patterns(drug_map) if do_perturb else []

    for part_idx in range(num_parts):
        rel_lo = part_idx * batch_size
        rel_hi = min((part_idx + 1) * batch_size, n)
        global_lo = row_id_start + rel_lo
        global_hi = row_id_start + rel_hi

        out_path = os.path.join(shards_dir := shard_out_dir, f"rows{global_lo}-{global_hi}.parquet")

        # Resume: skip if it already exists
        existing_path = _find_existing_shard(shards_dir, global_lo, global_hi)
        if existing_path and not overwrite_existing:
            print(f"[worker {worker_id}] SKIP existing shard for rows {global_lo}-{global_hi}: {os.path.basename(existing_path)}")
            continue

        batch = shard.iloc[rel_lo:rel_hi].copy()

        # Initialize vLLM when we actually need to generate
        if llm is None:
            print(f"[worker {worker_id}] Starting vLLM on GPU(S) {gpu_spec} with tp={tp}...")
            llm = LLM(
                model=model,
                tensor_parallel_size=tp,
                download_dir=download_dir,
                gpu_memory_utilization=gpu_mem_util,
                max_model_len=max_model_len,
            )
            tokenizer = llm.get_tokenizer()
            sampling = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
            )

        prompts = build_prompts(batch['masked_text'].tolist(), tokenizer)
        responses = llm.generate(prompts, sampling)

        # Parse outputs
        all_full = []
        all_final = []
        for out in responses:
            txt = out.outputs[0].text
            all_full.append(txt)
            if reasoning_marker in txt:
                all_final.append(txt.split(reasoning_marker, 1)[-1])
            else:
                all_final.append(txt)

        batch['synth_note_reasoning_and_note'] = all_full
        batch['synthetic_note'] = all_final

        # NEW: Per-note perturbation (generic -> brand/abbrev) with probability 'perturb_prob'
        if do_perturb:
            def maybe_perturb(t: str) -> str:
                if rng.random() < float(perturb_prob):
                    return replace_generics_with_alternatives(t, repl_patterns, rng)
                return t

            # Apply to the final note text; optionally to the full reasoning+note as well
            batch['synthetic_note'] = [maybe_perturb(t) for t in batch['synthetic_note']]
            batch['synth_note_reasoning_and_note'] = [maybe_perturb(t) for t in batch['synth_note_reasoning_and_note']]

        # Persist this part atomically
        _write_parquet_atomic(batch, out_path)
        print(f"[worker {worker_id}] Wrote {out_path} (rows {global_lo}-{global_hi})")

    print(f"[worker {worker_id}] Done.")


# ------------------------------
# Helpers for GPU grouping and sharding
# ------------------------------

def parse_gpu_ids(gpu_ids_arg: str, tp: int) -> List[str]:
    gpu_ids_arg = gpu_ids_arg.strip()
    if ";" in gpu_ids_arg:
        groups = [g.strip() for g in gpu_ids_arg.split(";") if g.strip()]
        for g in groups:
            ids = [x.strip() for x in g.split(",") if x.strip()]
            if len(ids) != tp:
                raise ValueError(f"Group '{g}' does not have {tp} GPUs required for tp={tp}.")
        return groups

    ids = [x.strip() for x in gpu_ids_arg.split(",") if x.strip()]
    if tp == 1:
        return ids
    if len(ids) % tp != 0:
        raise ValueError(f"{len(ids)} GPUs not divisible by tp={tp}. Provide explicit groups with ';' or adjust --tp.")
    groups = []
    for i in range(0, len(ids), tp):
        groups.append(",".join(ids[i:i+tp]))
    return groups


def even_ranges(n_rows: int, n_shards: int) -> List[Tuple[int, int]]:
    base = n_rows // n_shards
    rem = n_rows % n_shards
    ranges = []
    start = 0
    for k in range(n_shards):
        extra = 1 if k < rem else 0
        end = start + base + extra
        ranges.append((start, end))
        start = end
    return ranges


# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser("Parallel synthetic clinical note generation")
    ap.add_argument("--input_csv", type=str, required=True,
                    help="CSV with trial spaces and 'prompt_llm_answer' column")
    ap.add_argument("--out_dir", type=str, default="./synthetic_notes")
    ap.add_argument("--promptframe_path", type=str, default=None,
                    help="Optional: if provided and exists, reuse this parquet instead of recomputing")
    ap.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    ap.add_argument("--download_dir", type=str, default="./vllm_cache")
    ap.add_argument("--gpu_ids", type=str, required=True,
                    help='Comma-separated GPU ids (e.g., "0,1,2") or grouped for tp>1 ("0,1;2,3")')
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size per worker")
    ap.add_argument("--batch_size", type=int, default=8, help="prompts per vLLM.generate() call")
    ap.add_argument("--max_model_len", type=int, default=20000)
    ap.add_argument("--max_new_tokens", type=int, default=15000)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top_p", type=float, default=0.2)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--gpu_mem_util", type=float, default=0.94)
    ap.add_argument("--compose_only", action="store_true",
                    help="Skip generation and only compose existing shard files")
    ap.add_argument("--overwrite_existing", action="store_true",
                    help="Force regeneration even if a shard file already exists")

    # NEW: perturbation controls
    ap.add_argument("--perturb_prob", type=float, default=0.0,
                    help="Probability (0-1) to perturb each generated note by replacing generic drugs with brand/abbrev")
    ap.add_argument("--drug_map_csv", type=str, default=None,
                    help="Optional CSV with columns: generic,alternatives (pipe-separated). If omitted, a built-in map is used.")
    ap.add_argument("--perturb_seed", type=int, default=42,
                    help="Seed for perturbation RNG to make replacements reproducible")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    shards_dir = os.path.join(args.out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    # 1) Build or reuse promptframe parquet
    promptframe_parquet = args.promptframe_path or os.path.join(args.out_dir, "promptframe.parquet")
    if args.compose_only and not os.path.exists(promptframe_parquet):
        raise FileNotFoundError("compose_only is set but promptframe parquet is missing.")

    if (not args.compose_only) and (not os.path.exists(promptframe_parquet)):
        print("[main] Loading input CSV and building promptframe...")
        temp = pd.read_csv(args.input_csv)
        #temp['space_index'] = temp.index # no - should already be in pre-built dataframe
        allevents = process_clinical_notes_from_df(temp)
        promptframe = create_masked_dataset(allevents).copy()
        promptframe = promptframe.reset_index(drop=True)
        promptframe['row_id'] = np.arange(promptframe.shape[0])
        promptframe.to_parquet(promptframe_parquet, index=False)
        print(f"[main] Wrote {promptframe_parquet} with {promptframe.shape[0]} rows.")
    else:
        print(f"[main] Reusing {promptframe_parquet}")

    # 2) If only composing, skip straight to combining
    if args.compose_only:
        compose_outputs(args.out_dir)
        return

    # 3) Load row count and shard across workers
    meta = pd.read_parquet(promptframe_parquet, columns=['row_id'])
    n_rows = meta.shape[0]
    gpu_specs = parse_gpu_ids(args.gpu_ids, args.tp)
    n_workers = len(gpu_specs)
    ranges = even_ranges(n_rows, n_workers)
    print(f"[main] Total rows: {n_rows}; workers: {n_workers}; ranges: {ranges}")

    # 4) Spawn workers with 'spawn' context (safe for CUDA)
    ctx = mp.get_context("spawn")
    procs = []
    for wid, (gpu_spec, (lo, hi)) in enumerate(zip(gpu_specs, ranges)):
        p = ctx.Process(
            target=run_worker,
            args=(
                wid, gpu_spec,
                promptframe_parquet, lo, hi,
                args.out_dir, args.model, args.download_dir,
                args.tp, args.max_model_len, args.max_new_tokens, args.batch_size,
                args.temperature, args.top_p, args.repetition_penalty, args.gpu_mem_util,
                args.overwrite_existing,
                # NEW: pass perturbation controls
                args.perturb_prob, args.drug_map_csv, args.perturb_seed,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"A worker exited with code {p.exitcode}.")

    # 5) Compose shards to a single parquet
    compose_outputs(args.out_dir)
    print("[main] All done.")


def compose_outputs(out_dir: str):
    shards_dir = os.path.join(out_dir, "shards")
    if not os.path.isdir(shards_dir):
        print(f"[compose] No shards directory at {shards_dir}.")
        return
    files = sorted([f for f in os.listdir(shards_dir) if f.endswith(".parquet")])
    if not files:
        print(f"[compose] No parquet shards found in {shards_dir}.")
        return

    out_path = os.path.join(out_dir, "synthetic_notes.parquet")
    print(f"[compose] Combining {len(files)} shard files -> {out_path}")

    frames = []
    for i, fname in enumerate(files):
        fpath = os.path.join(shards_dir, fname)
        try:
            df = pd.read_parquet(fpath)
        except Exception as e:
            print(f"[compose] WARNING: failed to read {fpath}: {e}. Skipping.")
            continue
        frames.append(df)

    if not frames:
        print("[compose] No readable shards.")
        return

    output = pd.concat(frames, ignore_index=True)

    # De-duplicate on row_id in case a shard was regenerated; keep last (most recent)
    if 'row_id' in output.columns:
        before = output.shape[0]
        output = (output.sort_values('row_id')
                         .drop_duplicates(subset=['row_id'], keep='last')
                         .reset_index(drop=True))
        after = output.shape[0]
        if after < before:
            print(f"[compose] Dropped {before - after} duplicate rows by row_id.")

    # Atomic write
    tmp_path = out_path + ".tmp"
    output.to_parquet(tmp_path)
    os.replace(tmp_path, out_path)

    print(f"[compose] Wrote {out_path}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
