#!/usr/bin/env python3
"""
Orchestration script for the SOC (standard of care) clinical trial matching evaluation pipeline.

This script runs the full pipeline in the correct order, managing GPU resources
and allowing for resumability.

Pipeline stages:
0. prepare: Prepare data (uses prepare_data.py)
1. summarize: Summarize patient EHR (uses 6_summarize_patients.py)
2. spacify: Create/copy trial spaces
3. retrieval: Patient-centric + trial-centric retrieval (trialspace embedding)
4. llm_checks: Eligibility + boilerplate LLM checks via API (GPT)
5. oncoreasoning: OncoReasoning LLM inference via vLLM (trial check + boilerplate)
6. aggregation: Consolidate results (for trialspace retrieval)
7. baseline: Baseline evaluation using Qwen3 embedding - includes retrieval,
             eligibility checks, and aggregation (no boilerplate checks)
8. evaluation: Run model evaluations and generate PDF reports

Usage:
    # Run full pipeline with 4 GPUs
    python run_pipeline.py --gpus 0,1,2,3

    # Run only specific stages
    python run_pipeline.py --gpus 0,1 --stages retrieval,llm_checks

    # Run only baseline evaluation
    python run_pipeline.py --gpus 0,1 --stages baseline

    # Resume from a specific stage
    python run_pipeline.py --gpus 0,1,2,3 --start-stage llm_checks

    # Dry run to see commands
    python run_pipeline.py --gpus 0,1,2,3 --dry-run
"""

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

# Repo root for default paths
REPO_ROOT = Path(__file__).resolve().parents[2]  # matchminer-ai-training
SCRIPTS_DIR = Path(__file__).resolve().parent  # eval_phi_soc
ENROLLMENTS_SCRIPTS_DIR = REPO_ROOT / "real_eval_code/eval_phi_enrollments"
DATA_DIR = REPO_ROOT.parent / "data/phi/soc"
GOLD_LLM = 'openai/gpt-oss-120b'
BASELINE_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
ONCOREASONING_MODEL = "ksg-dfci/OncoReasoning-3B-1225"


STAGES = [
    "prepare",        # Data preparation
    "summarize",      # Patient summarization
    "spacify",        # Trial space creation
    "retrieval",      # Patient-centric + trial-centric retrieval (trialspace embedding)
    "llm_checks",     # Eligibility + boilerplate checks (for trialspace retrieval)
    "oncoreasoning",  # OncoReasoning LLM inference (trial check + boilerplate)
    "aggregation",    # Result consolidation (for trialspace retrieval)
    "baseline",       # Baseline: retrieval (Qwen3 embedding) + eligibility checks + aggregation
    "evaluation"      # Model evaluation and PDF report generation
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the SOC clinical trial matching evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline with 4 GPUs
    python run_pipeline.py --gpus 0,1,2,3

    # Run only retrieval and LLM checks
    python run_pipeline.py --gpus 0,1,2,3 --stages retrieval,llm_checks

    # Resume from LLM checks stage
    python run_pipeline.py --gpus 0,1,2,3 --start-stage llm_checks
        """
    )
    parser.add_argument("--gpus", type=str, required=True,
                        help="Comma-separated GPU IDs (e.g., '0,1,2,3')")
    parser.add_argument("--stages", type=str, default=None,
                        help=f"Comma-separated stages to run: {','.join(STAGES)}")
    parser.add_argument("--start-stage", type=str, default=None,
                        choices=STAGES,
                        help="Start from this stage (runs this and all subsequent)")
    parser.add_argument("--summarize-script", type=str,
                        default=str(REPO_ROOT / "6_summarize_patients.py"),
                        help="Path to patient summarization script")
    parser.add_argument("--input-notes", type=str,
                        default=str(DATA_DIR / "note_level_dataset.parquet"),
                        help="Input parquet with patient notes (for summarization)")
    parser.add_argument("--download-dir", type=str,
                        default="/data1/ken/meta/2024/meta_ai",
                        help="Download directory for model weights")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    # Arguments for prepare_data.py
    parser.add_argument("--structured-folder", type=str,
                        default="/data1/ken/pan_dfci_2024/structured_data/",
                        help="Path to structured data folder containing TREATMENT_PLAN.txt")
    parser.add_argument("--split-path", type=str,
                        default="/data1/ken/pan_dfci_2024/derived_data/split_5-2024.csv",
                        help="Path to train/val/test split CSV")
    parser.add_argument("--derived-data-path", type=str,
                        default="/data1/ken/pan_dfci_2024/derived_data",
                        help="Path to derived data directory with parquet files")
    parser.add_argument("--start-date", type=str, default="2016-01-01",
                        help="Start date for treatment filtering")
    parser.add_argument("--end-date", type=str, default="2023-01-01",
                        help="End date for treatment filtering")
    parser.add_argument("--days-buffer", type=int, default=5,
                        help="Number of days after treatment start to include reports")
    parser.add_argument("--split-filter", type=str, default=None,
                        help="Filter to specific split (train, validation, test)")
    # Trial selection parameters
    parser.add_argument("--n-trials", type=int, default=500,
                        help="Number of unique trials to sample for SOC evaluation (default: 500)")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for trial sampling (default: 42)")
    # Arguments for evaluation stage
    parser.add_argument("--trial-checker-model", type=str,
                        default=str(REPO_ROOT.parent / "models/trialchecker"),
                        help="Path to trial checker model for evaluation")
    parser.add_argument("--boilerplate-checker-model", type=str,
                        default=str(REPO_ROOT.parent / "models/boilerplatechecker"),
                        help="Path to boilerplate checker model for evaluation")
    parser.add_argument("--oncoreasoning-model", type=str,
                        default=ONCOREASONING_MODEL,
                        help="OncoReasoning model for vLLM inference (HuggingFace ID or path)")
    parser.add_argument("--oncoreasoning-n-samples", type=int, default=50,
                        help="Number of samples per prompt for OncoReasoning inference")
    parser.add_argument("--eval-output-dir", type=str,
                        default=None,
                        help="Output directory for evaluation PDFs (default: DATA_DIR/evaluation)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run of stages even if outputs already exist")
    return parser.parse_args()


def check_outputs_exist(output_paths: List[Path], description: str) -> bool:
    """Check if all output paths exist. Returns True if all exist."""
    missing = [p for p in output_paths if not p.exists()]
    if not missing:
        print(f"  [SKIP] {description}: All outputs already exist")
        return True
    return False


def check_llm_check_shards_complete(
    output_dir: Path, input_file: Path, mode: str, check_type: str, batch_size: int = 2000
) -> bool:
    """
    Check if all shards are complete for LLM check scripts (check_eligibility.py, check_boilerplate.py).

    These scripts save shards as: {mode}_{check_type}_through_{i}.csv
    where i is the last row index processed. The task is complete when the final
    shard exists with i = total_rows - 1.

    Args:
        output_dir: Directory containing the shard files
        input_file: Input CSV file to count total rows
        mode: "patient_centric" or "trial_centric"
        check_type: "eligibility" or "boilerplate"
        batch_size: Batch size used by the script (default 2000)

    Returns:
        True if all shards are complete, False otherwise
    """
    if not output_dir.exists():
        return False

    # Count total rows in input file
    try:
        df = pd.read_csv(input_file)
        total_rows = len(df)
    except Exception as e:
        print(f"  Warning: Could not read input file {input_file}: {e}")
        return False

    if total_rows == 0:
        return True  # No rows to process

    # The final shard should have through_{total_rows - 1}
    final_idx = total_rows - 1
    final_shard_pattern = f"{mode}_{check_type}_through_{final_idx}.csv"
    final_shard_path = output_dir / final_shard_pattern

    if final_shard_path.exists():
        # Verify the file is valid (not empty/corrupted)
        try:
            test_df = pd.read_csv(final_shard_path, nrows=1)
            if len(test_df) > 0:
                return True
        except Exception:
            pass

    return False


def check_oncoreasoning_shards_complete(
    temp_dir: Path, input_file: Path, batch_size: int = 2000
) -> bool:
    """
    Check if all shards are complete for OncoReasoning vLLM scripts.

    These scripts save shards as: shard_{batch_start:08d}_{batch_end:08d}.csv
    The task is complete when all expected shards exist.

    Note: OncoReasoning scripts may deduplicate rows before processing,
    so we check both the expected shards based on original rows AND
    the presence of the original mapping file which indicates completion.

    Args:
        temp_dir: Directory containing the shard files
        input_file: Input CSV file to count total rows
        batch_size: Batch size used by the script (default 2000)

    Returns:
        True if all shards are complete, False otherwise
    """
    if not temp_dir.exists():
        return False

    # Check if original mapping file exists (indicates processing started)
    mapping_file = temp_dir / "_original_mapping.csv"
    if not mapping_file.exists():
        return False

    # Count deduplicated rows from the mapping or by reading input
    try:
        df = pd.read_csv(input_file)
        # Deduplicate the same way the scripts do
        if "patient_boilerplate_text" in df.columns and "trial_boilerplate_text" in df.columns:
            # Boilerplate script deduplication
            df["_patient_bp_str"] = df["patient_boilerplate_text"].astype(str)
            df["_trial_bp_str"] = df["trial_boilerplate_text"].astype(str)
            df_dedup = df.drop_duplicates(subset=["_patient_bp_str", "_trial_bp_str"])
            total_rows = len(df_dedup)
        elif "patient_summary" in df.columns and "this_space" in df.columns:
            # Trial check script deduplication
            df["_patient_summary_str"] = df["patient_summary"].astype(str)
            df["_trial_space_str"] = df["this_space"].astype(str)
            df_dedup = df.drop_duplicates(subset=["_patient_summary_str", "_trial_space_str"])
            total_rows = len(df_dedup)
        else:
            total_rows = len(df)
    except Exception as e:
        print(f"  Warning: Could not read input file {input_file}: {e}")
        return False

    if total_rows == 0:
        return True  # No rows to process

    # Calculate expected shards
    expected_shards = []
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        expected_shards.append((batch_start, batch_end))

    # Check if all expected shards exist
    for batch_start, batch_end in expected_shards:
        shard_filename = f"shard_{batch_start:08d}_{batch_end:08d}.csv"
        shard_path = temp_dir / shard_filename
        if not shard_path.exists():
            return False
        # Verify file is valid
        try:
            test_df = pd.read_csv(shard_path, nrows=1)
            if len(test_df) == 0:
                return False
        except Exception:
            return False

    return True


def run_command(cmd: List[str], description: str, dry_run: bool = False,
                env: Optional[Dict] = None, cwd: Optional[str] = None) -> int:
    """Run a command and return exit code."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("[DRY RUN - not executing]")
        return 0

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    result = subprocess.run(cmd, env=full_env, cwd=cwd)
    return result.returncode


def run_parallel_commands(commands: List[Dict], max_workers: int, dry_run: bool = False) -> List[int]:
    """Run multiple commands in parallel using subprocess."""
    if dry_run:
        for cmd_info in commands:
            print(f"\n[DRY RUN] Would run: {cmd_info['description']}")
            print(f"  Command: {' '.join(cmd_info['cmd'])}")
        return [0] * len(commands)

    results = []
    processes = []

    for cmd_info in commands:
        print(f"\nStarting: {cmd_info['description']}")
        env = os.environ.copy()
        if 'env' in cmd_info:
            env.update(cmd_info['env'])

        proc = subprocess.Popen(
            cmd_info['cmd'],
            env=env,
            cwd=cmd_info.get('cwd'),
            stdout=subprocess.PIPE if not cmd_info.get('show_output', True) else None,
            stderr=subprocess.PIPE if not cmd_info.get('show_output', True) else None
        )
        processes.append((proc, cmd_info['description']))

    # Wait for all to complete
    for proc, desc in processes:
        returncode = proc.wait()
        results.append(returncode)
        if returncode != 0:
            print(f"WARNING: {desc} exited with code {returncode}")
        else:
            print(f"Completed: {desc}")

    return results


def get_stages_to_run(args) -> List[str]:
    """Determine which stages to run based on arguments."""
    if args.stages:
        return [s.strip() for s in args.stages.split(",")]
    elif args.start_stage:
        start_idx = STAGES.index(args.start_stage)
        return STAGES[start_idx:]
    else:
        return STAGES


def main():
    args = parse_args()

    gpu_list = [g.strip() for g in args.gpus.split(",")]
    num_gpus = len(gpu_list)

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"SOC Pipeline Configuration:")
    print(f"  Repo root: {REPO_ROOT}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Scripts directory: {SCRIPTS_DIR}")
    print(f"  Enrollments scripts directory: {ENROLLMENTS_SCRIPTS_DIR}")
    print(f"  Available GPUs: {gpu_list} ({num_gpus} total)")
    print(f"  Dry run: {args.dry_run}")

    stages_to_run = get_stages_to_run(args)
    print(f"  Stages to run: {stages_to_run}")

    # Track failures
    failures = []

    # Stage 0: Prepare data
    if "prepare" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 0: PREPARE SOC DATA")
        print("="*70)

        prepare_outputs = [
            DATA_DIR / "processed_soc_treatments.csv",
            DATA_DIR / "note_level_dataset.parquet",
        ]
        if not args.force and check_outputs_exist(prepare_outputs, "SOC data preparation"):
            pass  # Skip
        else:
            cmd = [
                "python", "prepare_data.py",
                "--structured-folder", args.structured_folder,
                "--split-path", args.split_path,
                "--derived-data-path", args.derived_data_path,
                "--output-dir", str(DATA_DIR),
                "--start-date", args.start_date,
                "--end-date", args.end_date,
                "--days-buffer", str(args.days_buffer),
            ]

            if args.split_filter:
                cmd.extend(["--split-filter", args.split_filter])

            ret = run_command(cmd, "SOC data preparation", args.dry_run)
            if ret != 0:
                failures.append("prepare")

    # Stage 1: Summarize patients
    if "summarize" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 1: SUMMARIZE PATIENTS")
        print("="*70)

        summarize_outputs = [
            DATA_DIR / "patient_summaries.parquet",
        ]
        if not args.force and check_outputs_exist(summarize_outputs, "Patient summarization"):
            pass  # Skip
        else:
            summary_gpus = ",".join(gpu_list[:min(2, num_gpus)])

            cmd = [
                "python", args.summarize_script,
                "--input_parquet", args.input_notes,
                "--output_parquet", str(DATA_DIR / "patient_summaries_full.parquet"),
                "--patient_summaries_parquet", str(DATA_DIR / "patient_summaries.parquet"),
                "--shard_dir", str(DATA_DIR / "summary_shards"),
                "--download_dir", args.download_dir,
                "--gpu_ids", summary_gpus,
                "--patient_id_col", "pseudo_mrn",
                "--text_col", "text",
                "--gpus_per_kernel", "1",
            ]

            ret = run_command(cmd, "Patient summarization", args.dry_run)
            if ret != 0:
                failures.append("summarize")

    # Stage 2: Select trial spaces (sample from v20_public_data excluding training trials)
    if "spacify" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 2: SELECT TRIAL SPACES")
        print("="*70)

        spacify_outputs = [
            DATA_DIR / "trial_space_lineitems.csv",
        ]
        if not args.force and check_outputs_exist(spacify_outputs, "Trial space selection"):
            pass  # Skip
        else:
            # For SOC, we select a sample of trials from v20_public_data that were
            # NOT used in training. This uses prepare_trials.py which implements
            # the logic from 1_pull_trials.ipynb.
            cmd = [
                "python", "prepare_trials.py",
                "--all-trials", str(REPO_ROOT.parent / "data/no_phi/trial_space_lineitems.csv"),
                "--training-trials", str(REPO_ROOT.parent / "data/no_phi/sample_trial_space_lineitems.csv"),
                "--output-file", str(DATA_DIR / "trial_space_lineitems.csv"),
                "--n-trials", str(args.n_trials),
                "--random-seed", str(args.random_seed),
            ]

            ret = run_command(cmd, "Trial space selection", args.dry_run)
            if ret != 0:
                failures.append("spacify")

    # Stage 3: Retrieval (patient-centric and trial-centric can run in parallel)
    if "retrieval" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 3: PATIENT-TRIAL RETRIEVAL")
        print("="*70)

        retrieval_outputs = [
            DATA_DIR / "patient_centric_candidates.csv",
            DATA_DIR / "trial_centric_candidates.csv",
        ]
        if not args.force and check_outputs_exist(retrieval_outputs, "Patient-trial retrieval"):
            pass  # Skip
        else:
            # Common retrieval arguments
            retrieval_common_args = [
                "--patient-summaries", str(DATA_DIR / "patient_summaries.parquet"),
                "--trial-spaces", str(DATA_DIR / "trial_space_lineitems.csv"),
                "--embedding-model", str(REPO_ROOT.parent / "models/trialspace"),
            ]

            if num_gpus >= 2:
                # Run patient-centric and trial-centric in parallel
                commands = [
                    {
                        'cmd': [
                            "python", str(ENROLLMENTS_SCRIPTS_DIR / "patient_centric_retrieval.py"),
                            "--gpu", gpu_list[0],
                            "--output-file", str(DATA_DIR / "patient_centric_candidates.csv"),
                            "--shard-dir", str(DATA_DIR / "shards_patient_centric"),
                        ] + retrieval_common_args,
                        'description': "Patient-centric retrieval",
                        'show_output': True,
                    },
                    {
                        'cmd': [
                            "python", str(ENROLLMENTS_SCRIPTS_DIR / "trial_centric_retrieval.py"),
                            "--gpu", gpu_list[1],
                            "--output-file", str(DATA_DIR / "trial_centric_candidates.csv"),
                            "--shard-dir", str(DATA_DIR / "shards_trial_centric"),
                        ] + retrieval_common_args,
                        'description': "Trial-centric retrieval",
                        'show_output': True,
                    }
                ]
                results = run_parallel_commands(commands, 2, args.dry_run)
                if any(r != 0 for r in results):
                    failures.append("retrieval")
            else:
                # Run sequentially with single GPU
                for script_name, desc, output_file, shard_dir in [
                    ("patient_centric_retrieval.py", "Patient-centric retrieval",
                     "patient_centric_candidates.csv", "shards_patient_centric"),
                    ("trial_centric_retrieval.py", "Trial-centric retrieval",
                     "trial_centric_candidates.csv", "shards_trial_centric"),
                ]:
                    cmd = [
                        "python", str(ENROLLMENTS_SCRIPTS_DIR / script_name),
                        "--gpu", gpu_list[0],
                        "--output-file", str(DATA_DIR / output_file),
                        "--shard-dir", str(DATA_DIR / shard_dir),
                    ] + retrieval_common_args
                    ret = run_command(cmd, desc, args.dry_run)
                    if ret != 0:
                        failures.append("retrieval")

    # Stage 4: LLM checks (eligibility + boilerplate for both directions)
    if "llm_checks" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 4: LLM ELIGIBILITY/BOILERPLATE CHECKS")
        print("="*70)

        # Define all check tasks with input/output paths and shard checking info
        check_tasks = [
            {
                'script': "check_eligibility.py",
                'args': ["--mode", "patient_centric"],
                'input': str(DATA_DIR / "patient_centric_candidates.csv"),
                'output_dir': str(DATA_DIR / "patient_centric_eligibility_checks"),
                'model': GOLD_LLM,
                'description': "Patient-centric eligibility check",
                'mode': "patient_centric",
                'check_type': "eligibility",
            },
            {
                'script': "check_boilerplate.py",
                'args': ["--mode", "patient_centric"],
                'input': str(DATA_DIR / "patient_centric_candidates.csv"),
                'output_dir': str(DATA_DIR / "patient_centric_boilerplate_checks"),
                'model': GOLD_LLM,
                'description': "Patient-centric boilerplate check",
                'mode': "patient_centric",
                'check_type': "boilerplate",
            },
            {
                'script': "check_eligibility.py",
                'args': ["--mode", "trial_centric"],
                'input': str(DATA_DIR / "trial_centric_candidates.csv"),
                'output_dir': str(DATA_DIR / "trial_centric_eligibility_checks"),
                'model': GOLD_LLM,
                'description': "Trial-centric eligibility check",
                'mode': "trial_centric",
                'check_type': "eligibility",
            },
            {
                'script': "check_boilerplate.py",
                'args': ["--mode", "trial_centric"],
                'input': str(DATA_DIR / "trial_centric_candidates.csv"),
                'output_dir': str(DATA_DIR / "trial_centric_boilerplate_checks"),
                'model': GOLD_LLM,
                'description': "Trial-centric boilerplate check",
                'mode': "trial_centric",
                'check_type': "boilerplate",
            },
        ]

        # Filter out tasks that are already complete (check shards)
        tasks_to_run = []
        for task in check_tasks:
            if args.force:
                tasks_to_run.append(task)
            elif check_llm_check_shards_complete(
                Path(task['output_dir']),
                Path(task['input']),
                task['mode'],
                task['check_type']
            ):
                print(f"  [SKIP] {task['description']}: All shards complete")
            else:
                tasks_to_run.append(task)

        if not tasks_to_run:
            print("  [SKIP] LLM checks: All sub-tasks already complete")
        else:
            # Distribute remaining tasks across available GPUs
            commands = []
            for i, task in enumerate(tasks_to_run):
                gpu_idx = i % num_gpus
                cmd = [
                    "python", str(ENROLLMENTS_SCRIPTS_DIR / task['script']),
                    "--gpu", gpu_list[gpu_idx],
                    "--download-dir", args.download_dir,
                    "--input", task['input'],
                    "--output-dir", task['output_dir'],
                    "--model", task['model'],
                ] + task['args']

                commands.append({
                    'cmd': cmd,
                    'description': task['description'],
                    'show_output': True,
                })

            # Run as many in parallel as we have GPUs
            batch_size = min(num_gpus, len(commands))
            for batch_start in range(0, len(commands), batch_size):
                batch = commands[batch_start:batch_start + batch_size]
                print(f"\nRunning batch of {len(batch)} LLM checks in parallel...")
                results = run_parallel_commands(batch, len(batch), args.dry_run)
                if any(r != 0 for r in results):
                    failures.append("llm_checks")

    # Stage 5: OncoReasoning LLM inference (trial check + boilerplate via vLLM)
    if "oncoreasoning" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 5: ONCOREASONING LLM INFERENCE (vLLM)")
        print("="*70)

        gpu_str = ",".join(gpu_list)

        # Define all OncoReasoning tasks
        oncoreasoning_tasks = [
            {
                'script': "vllm_parallel_trialcheck.py",
                'input': str(DATA_DIR / "patient_centric_candidates.csv"),
                'output': str(DATA_DIR / "oncoreasoning_trialcheck_patient_centric.csv"),
                'temp_dir': str(DATA_DIR / "shards_oncoreasoning_trialcheck_patient_centric"),
                'description': "OncoReasoning trial check (patient-centric)",
            },
            {
                'script': "vllm_parallel_trialcheck.py",
                'input': str(DATA_DIR / "trial_centric_candidates.csv"),
                'output': str(DATA_DIR / "oncoreasoning_trialcheck_trial_centric.csv"),
                'temp_dir': str(DATA_DIR / "shards_oncoreasoning_trialcheck_trial_centric"),
                'description': "OncoReasoning trial check (trial-centric)",
            },
            {
                'script': "vllm_parallel_boilerplate.py",
                'input': str(DATA_DIR / "patient_centric_candidates.csv"),
                'output': str(DATA_DIR / "oncoreasoning_boilerplate_patient_centric.csv"),
                'temp_dir': str(DATA_DIR / "shards_oncoreasoning_boilerplate_patient_centric"),
                'description': "OncoReasoning boilerplate check (patient-centric)",
            },
            {
                'script': "vllm_parallel_boilerplate.py",
                'input': str(DATA_DIR / "trial_centric_candidates.csv"),
                'output': str(DATA_DIR / "oncoreasoning_boilerplate_trial_centric.csv"),
                'temp_dir': str(DATA_DIR / "shards_oncoreasoning_boilerplate_trial_centric"),
                'description': "OncoReasoning boilerplate check (trial-centric)",
            },
        ]

        # Filter out tasks that are already complete
        tasks_to_run = []
        for task in oncoreasoning_tasks:
            output_path = Path(task['output'])
            if args.force:
                tasks_to_run.append(task)
            elif output_path.exists():
                # Final merged output exists
                print(f"  [SKIP] {task['description']}: Output file exists")
            elif check_oncoreasoning_shards_complete(
                Path(task['temp_dir']),
                Path(task['input'])
            ):
                # All shards complete but not merged - will merge on next run
                print(f"  [SKIP] {task['description']}: All shards complete (will merge)")
            else:
                tasks_to_run.append(task)

        if not tasks_to_run:
            print("  [SKIP] OncoReasoning inference: All sub-tasks already complete")
        else:
            for task in tasks_to_run:
                cmd = [
                    "python", str(ENROLLMENTS_SCRIPTS_DIR / task['script']),
                    "--model", args.oncoreasoning_model,
                    "--input-file", task['input'],
                    "--output-file", task['output'],
                    "--temp-dir", task['temp_dir'],
                    "--gpus", gpu_str,
                    "--n-samples", str(args.oncoreasoning_n_samples),
                ]
                ret = run_command(cmd, task['description'], args.dry_run)
                if ret != 0:
                    failures.append("oncoreasoning")

    # Stage 6: Aggregation (CPU only)
    if "aggregation" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 6: AGGREGATE RESULTS")
        print("="*70)

        aggregation_outputs = [
            DATA_DIR / "consolidated_eligibility_patient_centric.csv",
            DATA_DIR / "consolidated_boilerplate_patient_centric.csv",
            DATA_DIR / "consolidated_eligibility_trial_centric.csv",
            DATA_DIR / "consolidated_boilerplate_trial_centric.csv",
        ]
        if not args.force and check_outputs_exist(aggregation_outputs, "Result aggregation"):
            pass  # Skip
        else:
            aggregation_tasks = [
                ("eligibility", "patient_centric",
                 str(DATA_DIR / "patient_centric_eligibility_checks"),
                 str(DATA_DIR / "consolidated_eligibility_patient_centric.csv")),
                ("boilerplate", "patient_centric",
                 str(DATA_DIR / "patient_centric_boilerplate_checks"),
                 str(DATA_DIR / "consolidated_boilerplate_patient_centric.csv")),
                ("eligibility", "trial_centric",
                 str(DATA_DIR / "trial_centric_eligibility_checks"),
                 str(DATA_DIR / "consolidated_eligibility_trial_centric.csv")),
                ("boilerplate", "trial_centric",
                 str(DATA_DIR / "trial_centric_boilerplate_checks"),
                 str(DATA_DIR / "consolidated_boilerplate_trial_centric.csv")),
            ]

            for mode, direction, primary_dir, output_file in aggregation_tasks:
                cmd = [
                    "python", str(ENROLLMENTS_SCRIPTS_DIR / "aggregate_results.py"),
                    "--mode", mode,
                    "--direction", direction,
                    "--primary-dir", primary_dir,
                    "--output-file", output_file,
                ]

                desc = f"Aggregate {mode} results for {direction}"
                ret = run_command(cmd, desc, args.dry_run)
                if ret != 0:
                    failures.append("aggregation")

    # Stage 7: Baseline evaluation (uses Qwen3 embedding, eligibility checks only - no boilerplate)
    if "baseline" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 7: BASELINE EVALUATION (Qwen3 embedding + eligibility checks)")
        print("="*70)

        baseline_outputs = [
            DATA_DIR / "baseline_consolidated_eligibility_patient_centric.csv",
            DATA_DIR / "baseline_consolidated_eligibility_trial_centric.csv",
        ]
        if not args.force and check_outputs_exist(baseline_outputs, "Baseline evaluation"):
            pass  # Skip
        else:
            # Step 1: Baseline retrieval with Qwen3 embedding
            print("\n--- Baseline Retrieval ---")
            baseline_retrieval_common_args = [
                "--patient-summaries", str(DATA_DIR / "patient_summaries.parquet"),
                "--trial-spaces", str(DATA_DIR / "trial_space_lineitems.csv"),
                "--embedding-model", BASELINE_EMBEDDING_MODEL,
            ]

            if num_gpus >= 2:
                # Run patient-centric and trial-centric in parallel
                commands = [
                    {
                        'cmd': [
                            "python", str(ENROLLMENTS_SCRIPTS_DIR / "patient_centric_retrieval.py"),
                            "--gpu", gpu_list[0],
                            "--output-file", str(DATA_DIR / "baseline_patient_centric_candidates.csv"),
                            "--shard-dir", str(DATA_DIR / "shards_baseline_patient_centric"),
                        ] + baseline_retrieval_common_args,
                        'description': "Baseline patient-centric retrieval (Qwen3)",
                        'show_output': True,
                    },
                    {
                        'cmd': [
                            "python", str(ENROLLMENTS_SCRIPTS_DIR / "trial_centric_retrieval.py"),
                            "--gpu", gpu_list[1],
                            "--output-file", str(DATA_DIR / "baseline_trial_centric_candidates.csv"),
                            "--shard-dir", str(DATA_DIR / "shards_baseline_trial_centric"),
                        ] + baseline_retrieval_common_args,
                        'description': "Baseline trial-centric retrieval (Qwen3)",
                        'show_output': True,
                    }
                ]
                results = run_parallel_commands(commands, 2, args.dry_run)
                if any(r != 0 for r in results):
                    failures.append("baseline")
            else:
                # Run sequentially with single GPU
                for script_name, desc, output_file, shard_dir in [
                    ("patient_centric_retrieval.py", "Baseline patient-centric retrieval (Qwen3)",
                     "baseline_patient_centric_candidates.csv", "shards_baseline_patient_centric"),
                    ("trial_centric_retrieval.py", "Baseline trial-centric retrieval (Qwen3)",
                     "baseline_trial_centric_candidates.csv", "shards_baseline_trial_centric"),
                ]:
                    cmd = [
                        "python", str(ENROLLMENTS_SCRIPTS_DIR / script_name),
                        "--gpu", gpu_list[0],
                        "--output-file", str(DATA_DIR / output_file),
                        "--shard-dir", str(DATA_DIR / shard_dir),
                    ] + baseline_retrieval_common_args
                    ret = run_command(cmd, desc, args.dry_run)
                    if ret != 0:
                        failures.append("baseline")

            # Step 2: Baseline eligibility checks (no boilerplate for baseline)
            print("\n--- Baseline Eligibility Checks ---")
            baseline_check_tasks = [
                {
                    'script': "check_eligibility.py",
                    'args': ["--mode", "patient_centric"],
                    'input': str(DATA_DIR / "baseline_patient_centric_candidates.csv"),
                    'output_dir': str(DATA_DIR / "baseline_patient_centric_eligibility_checks"),
                    'model': GOLD_LLM,
                    'description': "Baseline patient-centric eligibility check",
                    'mode': "patient_centric",
                    'check_type': "eligibility",
                },
                {
                    'script': "check_eligibility.py",
                    'args': ["--mode", "trial_centric"],
                    'input': str(DATA_DIR / "baseline_trial_centric_candidates.csv"),
                    'output_dir': str(DATA_DIR / "baseline_trial_centric_eligibility_checks"),
                    'model': GOLD_LLM,
                    'description': "Baseline trial-centric eligibility check",
                    'mode': "trial_centric",
                    'check_type': "eligibility",
                },
            ]

            # Filter out tasks that are already complete (check shards)
            baseline_tasks_to_run = []
            for task in baseline_check_tasks:
                if check_llm_check_shards_complete(
                    Path(task['output_dir']),
                    Path(task['input']),
                    task['mode'],
                    task['check_type']
                ):
                    print(f"  [SKIP] {task['description']}: All shards complete")
                else:
                    baseline_tasks_to_run.append(task)

            if baseline_tasks_to_run:
                # Distribute remaining tasks across available GPUs
                commands = []
                for i, task in enumerate(baseline_tasks_to_run):
                    gpu_idx = i % num_gpus
                    cmd = [
                        "python", str(ENROLLMENTS_SCRIPTS_DIR / task['script']),
                        "--gpu", gpu_list[gpu_idx],
                        "--download-dir", args.download_dir,
                        "--input", task['input'],
                        "--output-dir", task['output_dir'],
                        "--model", task['model'],
                    ] + task['args']

                    commands.append({
                        'cmd': cmd,
                        'description': task['description'],
                        'show_output': True,
                    })

                batch_size = min(num_gpus, len(commands))
                for batch_start in range(0, len(commands), batch_size):
                    batch = commands[batch_start:batch_start + batch_size]
                    print(f"\nRunning batch of {len(batch)} baseline eligibility checks in parallel...")
                    results = run_parallel_commands(batch, len(batch), args.dry_run)
                    if any(r != 0 for r in results):
                        failures.append("baseline")
            else:
                print("  [SKIP] Baseline eligibility checks: All sub-tasks already complete")

            # Step 3: Baseline aggregation (eligibility only)
            print("\n--- Baseline Aggregation ---")
            baseline_aggregation_tasks = [
                ("eligibility", "patient_centric",
                 str(DATA_DIR / "baseline_patient_centric_eligibility_checks"),
                 str(DATA_DIR / "baseline_consolidated_eligibility_patient_centric.csv")),
                ("eligibility", "trial_centric",
                 str(DATA_DIR / "baseline_trial_centric_eligibility_checks"),
                 str(DATA_DIR / "baseline_consolidated_eligibility_trial_centric.csv")),
            ]

            for mode, direction, primary_dir, output_file in baseline_aggregation_tasks:
                cmd = [
                    "python", str(ENROLLMENTS_SCRIPTS_DIR / "aggregate_results.py"),
                    "--mode", mode,
                    "--direction", direction,
                    "--primary-dir", primary_dir,
                    "--output-file", output_file,
                ]

                desc = f"Baseline aggregate {mode} results for {direction}"
                ret = run_command(cmd, desc, args.dry_run)
                if ret != 0:
                    failures.append("baseline")

    # Stage 8: Evaluation (generate PDF reports with metrics)
    if "evaluation" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 8: MODEL EVALUATION")
        print("="*70)

        eval_output_dir = Path(args.eval_output_dir) if args.eval_output_dir else DATA_DIR / "evaluation"
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        # Define evaluation tasks for each module (scripts are now in SCRIPTS_DIR)
        eval_tasks = [
            # Qwen3 baseline evaluation (uses baseline_ prefix for data files)
            {
                'script': "eval_baseline.py",
                'args': ["--mode", "patient_centric", "--data-dir", str(DATA_DIR),
                         "--output-dir", str(eval_output_dir / "qwen3-baseline"),
                         "--prefix", "baseline_"],
                'description': "SOC Qwen3 baseline patient-centric evaluation",
            },
            {
                'script': "eval_baseline.py",
                'args': ["--mode", "trial_centric", "--data-dir", str(DATA_DIR),
                         "--output-dir", str(eval_output_dir / "qwen3-baseline"),
                         "--prefix", "baseline_"],
                'description': "SOC Qwen3 baseline trial-centric evaluation",
            },
            # ModernBERT trial checker evaluation
            {
                'script': "eval_modernbert_trial_checker.py",
                'args': ["--mode", "patient_centric", "--data-dir", str(DATA_DIR),
                         "--output-dir", str(eval_output_dir / "modernbert-trial-checker"),
                         "--model-path", args.trial_checker_model, "--gpu", gpu_list[0]],
                'description': "SOC ModernBERT trial checker patient-centric evaluation",
            },
            {
                'script': "eval_modernbert_trial_checker.py",
                'args': ["--mode", "trial_centric", "--data-dir", str(DATA_DIR),
                         "--output-dir", str(eval_output_dir / "modernbert-trial-checker"),
                         "--model-path", args.trial_checker_model, "--gpu", gpu_list[0]],
                'description': "SOC ModernBERT trial checker trial-centric evaluation",
            },
            # ModernBERT boilerplate checker evaluation
            {
                'script': "eval_modernbert_boilerplate_checker.py",
                'args': ["--mode", "patient_centric", "--data-dir", str(DATA_DIR),
                         "--output-dir", str(eval_output_dir / "modernbert-boilerplate-checker"),
                         "--model-path", args.boilerplate_checker_model, "--gpu", gpu_list[0]],
                'description': "SOC ModernBERT boilerplate checker patient-centric evaluation",
            },
            {
                'script': "eval_modernbert_boilerplate_checker.py",
                'args': ["--mode", "trial_centric", "--data-dir", str(DATA_DIR),
                         "--output-dir", str(eval_output_dir / "modernbert-boilerplate-checker"),
                         "--model-path", args.boilerplate_checker_model, "--gpu", gpu_list[0]],
                'description': "SOC ModernBERT boilerplate checker trial-centric evaluation",
            },
            # OncoReasoning LLM trial checker evaluation
            {
                'script': "eval_llm_trial_checker.py",
                'args': ["--mode", "patient_centric", "--data-dir", str(DATA_DIR),
                         "--output-dir", str(eval_output_dir / "oncoreasoning-trial-checker")],
                'description': "SOC OncoReasoning trial checker patient-centric evaluation",
            },
            {
                'script': "eval_llm_trial_checker.py",
                'args': ["--mode", "trial_centric", "--data-dir", str(DATA_DIR),
                         "--output-dir", str(eval_output_dir / "oncoreasoning-trial-checker")],
                'description': "SOC OncoReasoning trial checker trial-centric evaluation",
            },
            # OncoReasoning LLM boilerplate checker evaluation
            {
                'script': "eval_llm_boilerplate_checker.py",
                'args': ["--mode", "patient_centric", "--data-dir", str(DATA_DIR),
                         "--output-dir", str(eval_output_dir / "oncoreasoning-boilerplate-checker")],
                'description': "SOC OncoReasoning boilerplate checker patient-centric evaluation",
            },
            {
                'script': "eval_llm_boilerplate_checker.py",
                'args': ["--mode", "trial_centric", "--data-dir", str(DATA_DIR),
                         "--output-dir", str(eval_output_dir / "oncoreasoning-boilerplate-checker")],
                'description': "SOC OncoReasoning boilerplate checker trial-centric evaluation",
            },
        ]

        for task in eval_tasks:
            cmd = ["python", task['script']] + task['args']
            ret = run_command(cmd, task['description'], args.dry_run)
            if ret != 0:
                print(f"Warning: {task['description']} failed, continuing...")
                # Don't add to failures for evaluation - it's non-critical

        print(f"\nEvaluation reports saved to: {eval_output_dir}")

    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)

    if failures:
        print(f"WARNING: The following stages had failures: {failures}")
        sys.exit(1)
    else:
        print("All stages completed successfully!")
        print(f"\nOutputs saved to: {DATA_DIR}")
        sys.exit(0)


if __name__ == "__main__":
    main()
