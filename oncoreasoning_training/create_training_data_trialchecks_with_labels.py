#!/usr/bin/env python3
"""
Generate trial checks with labels training data.

Reads multiple parquet files from ../../data and produces
trialchecks_with_labels.parquet with formatted prompts for training.
"""

import pandas as pd
from transformers import AutoTokenizer


def trialcheck(frame, tokenizer):
    """Format trial check data into training prompts."""
    prompts = []
    for i in range(frame.shape[0]):
        patient_summary = frame.iloc[i].patient_summary
        trial_summary = frame.iloc[i].this_space
        answer = frame.iloc[i].trialcheck_llm_response

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
            )},
            {'role': 'assistant', 'content': answer}
        ]

        prompt = tokenizer.apply_chat_template(conversation=messages, tokenize=False)
        prompts.append(prompt)

    return prompts


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/llama-3.2-3B-Instruct')

    # Load primary trial spaces data
    print("Loading space_specific_eligibility_checks.parquet...")
    trial_spaces = pd.read_parquet('../../data/no_phi/space_specific_eligibility_checks.parquet')
    trial_spaces = trial_spaces[['patient_summary', 'this_space', 'trialcheck_llm_response', 'eligibility_result']]
    print(f"  Loaded {len(trial_spaces)} records")

    # Load patient cohort rounds (patientcentric checks)
    print("Loading patient cohort rounds...")
    patient_round1 = pd.read_parquet('../../data/no_phi/round1_patientcentric_checks/top_cohorts_checked_round1.parquet').rename(
        columns={'trialcheck_llama_response': 'trialcheck_llm_response'}
    )[['patient_summary', 'this_space', 'trialcheck_llm_response', 'eligibility_result']]
    print(f"  Round 1: {len(patient_round1)} records")

    patient_round2 = pd.read_parquet('../../data/no_phi/round2_patientcentric_checks/top_cohorts_checked_round2.parquet').rename(
        columns={'trialcheck_llama_response': 'trialcheck_llm_response'}
    )[['patient_summary', 'this_space', 'trialcheck_llm_response', 'eligibility_result']]
    print(f"  Round 2: {len(patient_round2)} records")

    patient_round3 = pd.read_parquet('../../data/no_phi/round3_patientcentric_checks/top_cohorts_checked_round3.parquet').rename(
        columns={'trialcheck_llama_response': 'trialcheck_llm_response'}
    )[['patient_summary', 'this_space', 'trialcheck_llm_response', 'eligibility_result']]
    print(f"  Round 3: {len(patient_round3)} records")

    # Load trial patient rounds (trialcentric checks)
    print("Loading trial patient rounds...")
    trial_round1 = pd.read_parquet('../../data/no_phi/round1_trialcentric_checks/top_patients_checked_round1.parquet').rename(
        columns={'trialcheck_llama_response': 'trialcheck_llm_response'}
    )[['patient_summary', 'this_space', 'trialcheck_llm_response', 'eligibility_result']]
    print(f"  Round 1: {len(trial_round1)} records")

    trial_round2 = pd.read_parquet('../../data/no_phi/round2_trialcentric_checks/top_patients_checked_round2.parquet').rename(
        columns={'trialcheck_llama_response': 'trialcheck_llm_response'}
    )[['patient_summary', 'this_space', 'trialcheck_llm_response', 'eligibility_result']]
    print(f"  Round 2: {len(trial_round2)} records")

    trial_round3 = pd.read_parquet('../../data/no_phi/round3_trialcentric_checks/top_patients_checked_round3.parquet').rename(
        columns={'trialcheck_llama_response': 'trialcheck_llm_response'}
    )[['patient_summary', 'this_space', 'trialcheck_llm_response', 'eligibility_result']]
    print(f"  Round 3: {len(trial_round3)} records")

    # Concatenate all data
    print("Concatenating all data...")
    allchecks = pd.concat([
        trial_spaces, patient_round1, patient_round2, patient_round3,
        trial_round1, trial_round2, trial_round3
    ], ignore_index=True)
    print(f"Total records: {len(allchecks)}")

    # Deduplicate by taking first occurrence
    print("Deduplicating by (patient_summary, this_space)...")
    firstchecks = allchecks.groupby(['patient_summary', 'this_space']).first().reset_index()
    print(f"After deduplication: {len(firstchecks)} records")

    print("Generating training prompts...")
    output = trialcheck(firstchecks, tokenizer)

    print("Saving to ../../data/no_phi/oncoreasoning_training_data/trialchecks_with_labels.parquet...")
    pd.DataFrame({
        'text': output,
        'label': firstchecks.eligibility_result
    }).to_parquet('../../data/no_phi/oncoreasoning_training_data/trialchecks_with_labels.parquet')
    print(f"Done! Saved {len(output)} records")


if __name__ == '__main__':
    main()
