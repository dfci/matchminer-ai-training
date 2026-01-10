#!/usr/bin/env python3
"""
Orchestration script for the trialspace baseline evaluation pipeline.

This script runs the baseline pipeline using Qwen3-Embedding-0.6B for retrieval
and the baseline LLM for eligibility checking.

Pipeline stages:
1. retrieval: Patient-centric and trial-centric retrieval (can run in parallel)
2. llm_checks: Eligibility checks for both directions (can run in parallel)
3. aggregation: Consolidate results (CPU only)

Note: This pipeline does NOT include data preparation or trial space creation.
It assumes those have already been run via the main pipeline in ../scripts/run_pipeline.py.

Usage:
    # Run full baseline pipeline with 2 GPUs
    python run_baseline_pipeline.py --gpus 0,1

    # Run only specific stages
    python run_baseline_pipeline.py --gpus 0,1 --stages retrieval,llm_checks

    # Resume from a specific stage
    python run_baseline_pipeline.py --gpus 0,1 --start-stage llm_checks

    # Dry run to see commands
    python run_baseline_pipeline.py --gpus 0,1 --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Repo root and directory paths
REPO_ROOT = Path(__file__).resolve().parents[4]  # matchminer-ai-training
BASELINE_DIR = Path(__file__).parent.resolve()
SCRIPTS_DIR = BASELINE_DIR.parent / "scripts"
DATA_DIR = REPO_ROOT.parent / "data/phi/enrollments"
OUTPUT_DIR = DATA_DIR / "trialspace_baseline"

STAGES = [
    "retrieval",      # Patient-centric + trial-centric retrieval
    "llm_checks",     # Eligibility checks
    "aggregation"     # Result consolidation
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the trialspace baseline evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full baseline pipeline with 2 GPUs
    python run_baseline_pipeline.py --gpus 0,1

    # Run only retrieval
    python run_baseline_pipeline.py --gpus 0 --stages retrieval

    # Resume from LLM checks stage
    python run_baseline_pipeline.py --gpus 0,1 --start-stage llm_checks
        """
    )
    parser.add_argument("--gpus", type=str, required=True,
                        help="Comma-separated GPU IDs (e.g., '0,1')")
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

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Baseline Pipeline Configuration:")
    print(f"  Repo root: {REPO_ROOT}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Baseline scripts: {BASELINE_DIR}")
    print(f"  Available GPUs: {gpu_list} ({num_gpus} total)")
    print(f"  Dry run: {args.dry_run}")

    stages_to_run = get_stages_to_run(args)
    print(f"  Stages to run: {stages_to_run}")

    # Track failures
    failures = []

    # Stage 1: Retrieval (patient-centric and trial-centric can run in parallel)
    if "retrieval" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 1: PATIENT-TRIAL RETRIEVAL (Qwen3-Embedding-0.6B Baseline)")
        print("="*70)

        # Common retrieval arguments (using source data from main pipeline)
        retrieval_common_args = [
            "--patient-summaries", str(DATA_DIR / "patient_summaries.parquet"),
            "--trial-spaces", str(DATA_DIR / "trial_space_lineitems.csv"),
            "--embedding-model", "Qwen/Qwen3-Embedding-0.6B",
        ]

        if num_gpus >= 2:
            # Run patient-centric and trial-centric in parallel
            commands = [
                {
                    'cmd': [
                        "python", str(BASELINE_DIR / "patient_centric_retrieval.py"),
                        "--gpu", gpu_list[0],
                        "--output-file", str(OUTPUT_DIR / "patient_centric_candidates.csv"),
                        "--shard-dir", str(OUTPUT_DIR / "shards_patient_centric"),
                    ] + retrieval_common_args,
                    'description': "Patient-centric retrieval (baseline)",
                    'show_output': True,
                },
                {
                    'cmd': [
                        "python", str(BASELINE_DIR / "trial_centric_retrieval.py"),
                        "--gpu", gpu_list[1],
                        "--output-file", str(OUTPUT_DIR / "trial_centric_candidates.csv"),
                        "--shard-dir", str(OUTPUT_DIR / "shards_trial_centric"),
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
                    "python", str(BASELINE_DIR / script_name),
                    "--gpu", gpu_list[0],
                    "--output-file", str(OUTPUT_DIR / output_file),
                    "--shard-dir", str(OUTPUT_DIR / shard_dir),
                ] + retrieval_common_args
                ret = run_command(cmd, desc, args.dry_run)
                if ret != 0:
                    failures.append("retrieval")

    # Stage 2: LLM checks (eligibility for both directions)
    if "llm_checks" in stages_to_run:
        print("\n" + "="*70)
        print("STAGE 2: LLM ELIGIBILITY CHECKS (Baseline)")
        print("="*70)

        # Define check tasks with input/output paths
        check_tasks = [
            {
                'script': "check_eligibility.py",
                'args': ["--mode", "patient_centric"],
                'input': str(OUTPUT_DIR / "patient_centric_candidates.csv"),
                'output_dir': str(OUTPUT_DIR / "patient_centric_eligibility_checks"),
                'description': "Patient-centric eligibility check (baseline)",
            },
            {
                'script': "check_eligibility.py",
                'args': ["--mode", "trial_centric"],
                'input': str(OUTPUT_DIR / "trial_centric_candidates.csv"),
                'output_dir': str(OUTPUT_DIR / "trial_centric_eligibility_checks"),
                'description': "Trial-centric eligibility check (baseline)",
            },
        ]

        # Distribute tasks across available GPUs
        commands = []
        for i, task in enumerate(check_tasks):
            gpu_idx = i % num_gpus
            cmd = [
                "python", str(BASELINE_DIR / task['script']),
                "--gpu", gpu_list[gpu_idx],
                "--download-dir", args.download_dir,
                "--input", task['input'],
                "--output-dir", task['output_dir'],
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
        print("STAGE 3: AGGREGATE RESULTS")
        print("="*70)

        aggregation_tasks = [
            ("eligibility", "patient_centric",
             str(OUTPUT_DIR / "patient_centric_eligibility_checks"),
             str(OUTPUT_DIR / "consolidated_eligibility_patient_centric.csv")),
            ("eligibility", "trial_centric",
             str(OUTPUT_DIR / "trial_centric_eligibility_checks"),
             str(OUTPUT_DIR / "consolidated_eligibility_trial_centric.csv")),
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
        print(f"\nOutputs saved to: {OUTPUT_DIR}")
        sys.exit(0)


if __name__ == "__main__":
    main()
