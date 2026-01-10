#!/usr/bin/env python3
"""
Orchestration script for the SOC baseline clinical trial matching evaluation pipeline.

This script runs the baseline pipeline using Qwen3-Embedding for retrieval and
openai/gpt-oss-120b for eligibility checking. It assumes that patient summaries
and trial spaces have already been prepared by running scripts/run_pipeline.py.

Pipeline stages:
1. retrieval: Patient-centric and trial-centric retrieval (can run in parallel)
2. llm_checks: Eligibility LLM checks (can run in parallel)
3. aggregation: Consolidate results (CPU only)

Usage:
    # Run full baseline pipeline with 4 GPUs
    python run_baseline_pipeline.py --gpus 0,1,2,3

    # Run only specific stages
    python run_baseline_pipeline.py --gpus 0,1 --stages retrieval,llm_checks

    # Resume from a specific stage
    python run_baseline_pipeline.py --gpus 0,1,2,3 --start-stage llm_checks

    # Dry run to see commands
    python run_baseline_pipeline.py --gpus 0,1,2,3 --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Repo root for default paths
REPO_ROOT = Path(__file__).resolve().parents[3]  # matchminer-ai-training
BASELINE_SCRIPTS_DIR = Path(__file__).parent.resolve()
SCRIPTS_DIR = BASELINE_SCRIPTS_DIR.parent / "scripts"
SOURCE_DATA_DIR = REPO_ROOT.parent / "data/phi/soc"  # Where run_pipeline.py outputs data
BASELINE_DATA_DIR = REPO_ROOT.parent / "data/phi/soc/trialspace-baseline"  # Baseline outputs


STAGES = [
    "retrieval",      # Patient-centric + trial-centric retrieval
    "llm_checks",     # Eligibility checks
    "aggregation"     # Result consolidation
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the SOC baseline clinical trial matching evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline with 4 GPUs
    python run_baseline_pipeline.py --gpus 0,1,2,3

    # Run only retrieval and LLM checks
    python run_baseline_pipeline.py --gpus 0,1,2,3 --stages retrieval,llm_checks

    # Resume from LLM checks stage
    python run_baseline_pipeline.py --gpus 0,1,2,3 --start-stage llm_checks
        """
    )
    parser.add_argument("--gpus", type=str, required=True,
                        help="Comma-separated GPU IDs (e.g., '0,1,2,3')")
    parser.add_argument("--stages", type=str, default=None,
                        help=f"Comma-separated stages to run: {','.join(STAGES)}")
    parser.add_argument("--start-stage", type=str, default=None,
                        choices=STAGES,
                        help="Start from this stage (runs this and all subsequent)")
    parser.add_argument("--download-dir", type=str,
                        default="/data1/ken/meta/2024/meta_ai",
                        help="Download directory for model weights")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    # Baseline-specific arguments
    parser.add_argument("--embedding-model", type=str,
                        default="Qwen/Qwen3-Embedding-0.6B",
                        help="Embedding model for baseline retrieval")
    parser.add_argument("--llm-model", type=str,
                        default="openai/gpt-oss-120b",
                        help="LLM model for baseline eligibility checks")
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

    # Ensure baseline data directory exists
    BASELINE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"SOC Baseline Pipeline Configuration:")
    print(f"  Repo root: {REPO_ROOT}")
    print(f"  Source data directory: {SOURCE_DATA_DIR}")
    print(f"  Baseline data directory: {BASELINE_DATA_DIR}")
    print(f"  Baseline scripts directory: {BASELINE_SCRIPTS_DIR}")
    print(f"  Available GPUs: {gpu_list} ({num_gpus} total)")
    print(f"  Embedding model: {args.embedding_model}")
    print(f"  LLM model: {args.llm_model}")
    print(f"  Dry run: {args.dry_run}")

    stages_to_run = get_stages_to_run(args)
    print(f"  Stages to run: {stages_to_run}")

    # Verify source data exists
    patient_summaries_path = SOURCE_DATA_DIR / "patient_summaries.parquet"
    trial_spaces_path = SOURCE_DATA_DIR / "trial_space_lineitems.csv"

    if not patient_summaries_path.exists():
        print(f"ERROR: Patient summaries not found at {patient_summaries_path}")
        print("Please run scripts/run_pipeline.py first to prepare patient summaries.")
        sys.exit(1)

    if not trial_spaces_path.exists():
        print(f"ERROR: Trial spaces not found at {trial_spaces_path}")
        print("Please run scripts/run_pipeline.py first to prepare trial spaces.")
        sys.exit(1)

    # Track failures
    failures = []

    # Stage 1: Retrieval (patient-centric and trial-centric can run in parallel)
    if "retrieval" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 1: PATIENT-TRIAL RETRIEVAL (Baseline)")
        print("="*70)

        # Common retrieval arguments - use source data, output to baseline dir
        retrieval_common_args = [
            "--patient-summaries", str(patient_summaries_path),
            "--trial-spaces", str(trial_spaces_path),
            "--embedding-model", args.embedding_model,
        ]

        if num_gpus >= 2:
            # Run patient-centric and trial-centric in parallel
            commands = [
                {
                    'cmd': [
                        "python", str(BASELINE_SCRIPTS_DIR / "patient_centric_retrieval.py"),
                        "--gpu", gpu_list[0],
                        "--output-file", str(BASELINE_DATA_DIR / "patient_centric_candidates.csv"),
                        "--shard-dir", str(BASELINE_DATA_DIR / "shards_patient_centric"),
                    ] + retrieval_common_args,
                    'description': "Patient-centric retrieval (baseline)",
                    'show_output': True,
                },
                {
                    'cmd': [
                        "python", str(BASELINE_SCRIPTS_DIR / "trial_centric_retrieval.py"),
                        "--gpu", gpu_list[1],
                        "--output-file", str(BASELINE_DATA_DIR / "trial_centric_candidates.csv"),
                        "--shard-dir", str(BASELINE_DATA_DIR / "shards_trial_centric"),
                    ] + retrieval_common_args,
                    'description': "Trial-centric retrieval (baseline)",
                    'show_output': True,
                }
            ]
            results = run_parallel_commands(commands, 2, args.dry_run)
            if any(r != 0 for r in results):
                failures.append("retrieval")
        else:
            # Run sequentially with single GPU
            for script_name, desc, output_file, shard_dir in [
                ("patient_centric_retrieval.py", "Patient-centric retrieval (baseline)",
                 "patient_centric_candidates.csv", "shards_patient_centric"),
                ("trial_centric_retrieval.py", "Trial-centric retrieval (baseline)",
                 "trial_centric_candidates.csv", "shards_trial_centric"),
            ]:
                cmd = [
                    "python", str(BASELINE_SCRIPTS_DIR / script_name),
                    "--gpu", gpu_list[0],
                    "--output-file", str(BASELINE_DATA_DIR / output_file),
                    "--shard-dir", str(BASELINE_DATA_DIR / shard_dir),
                ] + retrieval_common_args
                ret = run_command(cmd, desc, args.dry_run)
                if ret != 0:
                    failures.append("retrieval")

    # Stage 2: LLM checks (eligibility for both directions)
    if "llm_checks" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 2: LLM ELIGIBILITY CHECKS (Baseline)")
        print("="*70)

        # Define eligibility check tasks (no boilerplate for baseline)
        check_tasks = [
            {
                'script': "check_eligibility.py",
                'args': ["--mode", "patient_centric"],
                'input': str(BASELINE_DATA_DIR / "patient_centric_candidates.csv"),
                'output_dir': str(BASELINE_DATA_DIR / "patient_centric_eligibility_checks"),
                'description': "Patient-centric eligibility check (baseline)",
            },
            {
                'script': "check_eligibility.py",
                'args': ["--mode", "trial_centric"],
                'input': str(BASELINE_DATA_DIR / "trial_centric_candidates.csv"),
                'output_dir': str(BASELINE_DATA_DIR / "trial_centric_eligibility_checks"),
                'description': "Trial-centric eligibility check (baseline)",
            },
        ]

        # Distribute tasks across available GPUs
        commands = []
        for i, task in enumerate(check_tasks):
            gpu_idx = i % num_gpus
            cmd = [
                "python", str(BASELINE_SCRIPTS_DIR / task['script']),
                "--gpu", gpu_list[gpu_idx],
                "--download-dir", args.download_dir,
                "--input", task['input'],
                "--output-dir", task['output_dir'],
                "--model", args.llm_model,
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

    # Stage 3: Aggregation (CPU only)
    if "aggregation" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 3: AGGREGATE RESULTS (Baseline)")
        print("="*70)

        # Only eligibility aggregation for baseline (no boilerplate)
        aggregation_tasks = [
            ("eligibility", "patient_centric",
             str(BASELINE_DATA_DIR / "patient_centric_eligibility_checks"),
             str(BASELINE_DATA_DIR / "consolidated_eligibility_patient_centric.csv")),
            ("eligibility", "trial_centric",
             str(BASELINE_DATA_DIR / "trial_centric_eligibility_checks"),
             str(BASELINE_DATA_DIR / "consolidated_eligibility_trial_centric.csv")),
        ]

        for mode, direction, primary_dir, output_file in aggregation_tasks:
            cmd = [
                "python", str(SCRIPTS_DIR / "aggregate_results.py"),
                "--mode", mode,
                "--direction", direction,
                "--primary-dir", primary_dir,
                "--output-file", output_file,
            ]

            desc = f"Aggregate {mode} results for {direction} (baseline)"
            ret = run_command(cmd, desc, args.dry_run)
            if ret != 0:
                failures.append("aggregation")

    # Summary
    print("\n" + "="*70)
    print("BASELINE PIPELINE COMPLETE")
    print("="*70)

    if failures:
        print(f"WARNING: The following stages had failures: {failures}")
        sys.exit(1)
    else:
        print("All stages completed successfully!")
        print(f"\nOutputs saved to: {BASELINE_DATA_DIR}")
        sys.exit(0)


if __name__ == "__main__":
    main()
