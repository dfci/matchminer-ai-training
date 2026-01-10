#!/usr/bin/env python3
"""
Orchestration script for the clinical trial matching evaluation pipeline.

This script runs the full pipeline in the correct order, managing GPU resources
and allowing for resumability.

Pipeline stages:
0. prepare: Prepare data (uses prepare_data.py)
1. summarize: Summarize patient EHR (uses 6_summarize_patients.py)
2. spacify: Create trial spaces
3. retrieval: Patient-centric and trial-centric retrieval (can run in parallel)
4. llm_checks: Eligibility and boilerplate LLM checks (can run in parallel)
5. aggregation: Consolidate results (CPU only)

Usage:
    # Run full pipeline with 4 GPUs
    python run_pipeline.py --gpus 0,1,2,3

    # Run only specific stages
    python run_pipeline.py --gpus 0,1 --stages retrieval,llm_checks

    # Resume from a specific stage
    python run_pipeline.py --gpus 0,1,2,3 --start-stage llm_checks

    # Dry run to see commands
    python run_pipeline.py --gpus 0,1,2,3 --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

# Repo root for default paths
REPO_ROOT = Path(__file__).resolve().parents[3]  # matchminer-ai-training
SCRIPTS_DIR = Path(__file__).resolve()
DATA_DIR = REPO_ROOT.parent / "data/phi/enrollments"
GOLD_LLM = "openai/gpt-oss-120b"

STAGES = [
    "prepare",        # Data preparation
    "summarize",      # Patient summarization
    "spacify",        # Trial space creation
    "retrieval",      # Patient-centric + trial-centric retrieval
    "llm_checks",     # Eligibility + boilerplate checks
    "aggregation"     # Result consolidation
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the clinical trial matching evaluation pipeline",
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
    parser.add_argument("--forken-path", type=str,
                        default=str(DATA_DIR / "forken.json"),
                        help="Path to forken.json file (for data preparation)")
    parser.add_argument("--derived-data-path", type=str,
                        default="/data1/ken/pan_dfci_2024/derived_data",
                        help="Path to derived data directory with parquet files")
    parser.add_argument("--enrollments-path", type=str,
                        default=str(DATA_DIR / "useful_trial_enrollments.csv"),
                        help="Path to useful_trial_enrollments.csv")
    parser.add_argument("--days-buffer", type=int, default=5,
                        help="Number of days after trial start to include reports")
    return parser.parse_args()


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

    print(f"Pipeline Configuration:")
    print(f"  Repo root: {REPO_ROOT}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Scripts directory: {SCRIPTS_DIR}")
    print(f"  Available GPUs: {gpu_list} ({num_gpus} total)")
    print(f"  Dry run: {args.dry_run}")

    stages_to_run = get_stages_to_run(args)
    print(f"  Stages to run: {stages_to_run}")

    # Track failures
    failures = []

    # Stage 0: Prepare data
    if "prepare" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 0: PREPARE DATA")
        print("="*70)

        cmd = [
            "python", "prepare_data.py",
            "--forken-path", args.forken_path,
            "--derived-data-path", args.derived_data_path,
            "--enrollments-path", args.enrollments_path,
            "--output-dir", str(DATA_DIR),
            "--days-buffer", str(args.days_buffer),
        ]

        ret = run_command(cmd, "Data preparation", args.dry_run)
        if ret != 0:
            failures.append("prepare")

    # Stage 1: Summarize patients
    if "summarize" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 1: SUMMARIZE PATIENTS")
        print("="*70)

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

    # Stage 2: Create trial spaces
    if "spacify" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 2: CREATE TRIAL SPACES")
        print("="*70)

        spacify_gpus = ",".join(gpu_list[:min(2, num_gpus)])

        cmd = [
            "python", "spacify_dfci_trials.py",
            "--gpus", spacify_gpus,
            "--gpus-per-instance", "2",
            "--input-file", str(DATA_DIR / "processed_trial_enrollments.csv"),
            "--output-dir", str(DATA_DIR),
            "--download-dir", args.download_dir,
        ]

        ret = run_command(cmd, "Trial space creation", args.dry_run)
        if ret != 0:
            failures.append("spacify")

    # Stage 3: Retrieval (patient-centric and trial-centric can run in parallel)
    if "retrieval" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 3: PATIENT-TRIAL RETRIEVAL")
        print("="*70)

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
                        "python", "patient_centric_retrieval.py",
                        "--gpu", gpu_list[0],
                        "--output-file", str(DATA_DIR / "patient_centric_candidates.csv"),
                        "--shard-dir", str(DATA_DIR / "shards_patient_centric"),
                    ] + retrieval_common_args,
                    'description': "Patient-centric retrieval",
                    'show_output': True,
                },
                {
                    'cmd': [
                        "python", "trial_centric_retrieval.py",
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
                    "python", str(SCRIPTS_DIR / script_name),
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

        # Define all check tasks with input/output paths
        check_tasks = [
            {
                'script': "check_eligibility.py",
                'args': ["--mode", "patient_centric"],
                'input': str(DATA_DIR / "patient_centric_candidates.csv"),
                'output_dir': str(DATA_DIR / "patient_centric_eligibility_checks"),
                'model': GOLD_LLM,
                'description': "Patient-centric eligibility check",
            },
            {
                'script': "check_boilerplate.py",
                'args': ["--mode", "patient_centric"],
                'input': str(DATA_DIR / "patient_centric_candidates.csv"),
                'output_dir': str(DATA_DIR / "patient_centric_boilerplate_checks"),
                'model': GOLD_LLM,
                'description': "Patient-centric boilerplate check",
            },
            {
                'script': "check_eligibility.py",
                'args': ["--mode", "trial_centric"],
                'input': str(DATA_DIR / "trial_centric_candidates.csv"),
                'output_dir': str(DATA_DIR / "trial_centric_eligibility_checks"),
                'model': GOLD_LLM,
                'description': "Trial-centric eligibility check",
            },
            {
                'script': "check_boilerplate.py",
                'args': ["--mode", "trial_centric"],
                'input': str(DATA_DIR / "trial_centric_candidates.csv"),
                'output_dir': str(DATA_DIR / "trial_centric_boilerplate_checks"),
                'model': GOLD_LLM,
                'description': "Trial-centric boilerplate check",
            },
        ]

        # Distribute tasks across available GPUs
        commands = []
        for i, task in enumerate(check_tasks):
            gpu_idx = i % num_gpus
            cmd = [
                "python", task['script'],
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

    # Stage 5: Aggregation (CPU only)
    if "aggregation" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 5: AGGREGATE RESULTS")
        print("="*70)

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
                "python", str(SCRIPTS_DIR / "aggregate_results.py"),
                "--mode", mode,
                "--direction", direction,
                "--primary-dir", primary_dir,
                "--output-file", output_file,
            ]

            desc = f"Aggregate {mode} results for {direction}"
            ret = run_command(cmd, desc, args.dry_run)
            if ret != 0:
                failures.append("aggregation")

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
