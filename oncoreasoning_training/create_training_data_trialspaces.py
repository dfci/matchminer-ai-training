#!/usr/bin/env python3
"""
Generate trial spaces training data.

Reads trial_space_lineitems.csv from ../../data and produces
spacified_trial.parquet with formatted prompts for training.
"""

import pandas as pd
from transformers import AutoTokenizer


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


def spacify_trial(frame, tokenizer):
    """Format trial space data into training prompts."""
    prompts = []
    for i in range(frame.shape[0]):
        trial = frame.iloc[i].trial_text
        answer = frame.iloc[i].space_reasoning_and_output

        messages = [
            {'role': 'system', 'content': """
        Reasoning: high.
        """},
            {'role': 'user', 'content': PROMPT_HEADER + "\n" + trial + "\n" + PROMPT_SUFFIX},
            {'role': 'assistant', 'content': answer}
        ]

        prompt = tokenizer.apply_chat_template(conversation=messages, tokenize=False)
        prompts.append(prompt)

    return prompts


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/llama-3.2-3B-Instruct')

    print("Loading trial spaces data from ../../data/no_phi/trial_space_lineitems.csv...")
    trial_spaces = pd.read_csv("../../data/no_phi/trial_space_lineitems.csv")
    print(f"Loaded {len(trial_spaces)} records")

    print("Generating training prompts...")
    output = spacify_trial(trial_spaces, tokenizer)

    print("Saving to ../../data/no_phi/oncoreasoning_training_data/spacified_trial.parquet...")
    pd.DataFrame(output, columns=['text']).to_parquet('../../data/no_phi/oncoreasoning_training_data/spacified_trial.parquet')
    print(f"Done! Saved {len(output)} records")


if __name__ == '__main__':
    main()
