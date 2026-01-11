#!/usr/bin/env python3
"""
LLM-based eligibility checking for patient-trial matches (SOC evaluation).

Two modes:
- "patient_centric": Check if trial spaces are reasonable for patients
- "trial_centric": Check if patients are reasonable for trial spaces

Usage:
    python check_eligibility.py --gpu 0 --mode patient_centric
    python check_eligibility.py --gpu 0 --mode trial_centric
"""

import argparse
import os
import sys
import glob
import pandas as pd
from pathlib import Path

# Repo root for default paths
REPO_ROOT = Path(__file__).resolve().parents[3]  # matchminer-ai-training


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-based eligibility checking")
    parser.add_argument("--gpu", type=str, required=True,
                        help="GPU ID to use (e.g., '0')")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["patient_centric", "trial_centric"],
                        help="Check mode: 'patient_centric' or 'trial_centric'")
    parser.add_argument("--input", type=str, default=None,
                        help="Input CSV file with candidates (default: derived from mode)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for shard files (default: derived from mode)")
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="Batch size for LLM inference")
    parser.add_argument("--model", type=str, default='openai/gpt-oss-120b',
                        help="Model path (default: openai/gpt-oss-120b")
    parser.add_argument("--download-dir", type=str,
                        default="/data1/ken/meta/2024/meta_ai",
                        help="Download directory for model weights (used if model not found locally)")
    parser.add_argument("--gpu-mem-util", type=float, default=0.94,
                        help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=10000,
                        help="Maximum model context length")
    return parser.parse_args()


def ask_about_trials_loosely(patient_summaries, trial_summaries, llama_model):
    """Ask LLM whether trials are reasonable considerations for patients."""
    from vllm import SamplingParams

    tokenizer = llama_model.get_tokenizer()
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
            conversation=messages, add_generation_prompt=True, tokenize=False
        )
        prompts.append(prompt)

    responses = llama_model.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            top_k=1,
            max_tokens=7500,
            repetition_penalty=1.2,
        )
    )

    response_texts = [x.outputs[0].text for x in responses]
    eligibility_results = []

    for response_text in response_texts:
        if ("Yes!" in response_text[-10:]) or ("YES!" in response_text[-10:]):
            eligibility_results.append(1.0)
        else:
            eligibility_results.append(0.0)

    return responses, response_texts, eligibility_results


def get_completed_batches(output_dir):
    """Get the highest completed batch index."""
    pattern = os.path.join(output_dir, "*_through_*.csv")
    files = glob.glob(pattern)
    max_idx = -1
    for f in files:
        try:
            basename = os.path.basename(f)
            idx_str = basename.split("_through_")[1].replace(".csv", "")
            idx = int(idx_str)
            max_idx = max(max_idx, idx)
        except:
            pass
    return max_idx


def main():
    args = parse_args()

    # Set defaults based on mode
    if args.model is None:
        args.model = 'openai/gpt-oss-120b'

    if args.input is None:
        if args.mode == "patient_centric":
            args.input = str(REPO_ROOT.parent / "data/phi_soc/patient_centric_candidates.csv")
        else:
            args.input = str(REPO_ROOT.parent / "data/phi_soc/trial_centric_candidates.csv")

    if args.output_dir is None:
        if args.mode == "patient_centric":
            args.output_dir = str(REPO_ROOT.parent / "data/phi_soc/patient_centric_eligibility_checks")
        else:
            args.output_dir = str(REPO_ROOT.parent / "data/phi_soc/trial_centric_eligibility_checks")

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    from vllm import LLM

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input data
    print(f"Loading input from {args.input}...")
    candidates = pd.read_csv(args.input)
    print(f"Loaded {candidates.shape[0]} candidate pairs")

    # Initialize LLM
    print(f"Initializing LLM from {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        download_dir=args.download_dir,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len
    )

    # Check for resume point
    last_completed = get_completed_batches(str(output_dir))
    start_idx = last_completed + 1 if last_completed >= 0 else 0
    print(f"Starting from index {start_idx}")

    # Process in batches
    batch_list = []
    num_in_batch = 0

    for i in range(start_idx, candidates.shape[0]):
        batch_list.append(candidates.iloc[[i]])
        num_in_batch += 1

        if (num_in_batch == args.batch_size) or (i == (candidates.shape[0] - 1)):
            output = pd.concat(batch_list, axis=0)

            _, output['llama_response'], output['eligibility_result'] = ask_about_trials_loosely(
                output['patient_summary'].astype(str).tolist(),
                output['this_space'].astype(str).tolist(),
                llm
            )

            # Determine output filename based on mode
            if args.mode == "patient_centric":
                output_file = output_dir / f"patient_centric_eligibility_through_{i}.csv"
            else:
                output_file = output_dir / f"trial_centric_eligibility_through_{i}.csv"

            output.to_csv(str(output_file), index=False)
            print(f"Saved batch through {i}")

            num_in_batch = 0
            batch_list = []

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
