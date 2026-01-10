#!/usr/bin/env python3
"""
Generate serial patient summarization training data.

Reads patient_serial_summaries.parquet from ../../data and produces
summarized_patient_histories.parquet with formatted prompts for training.

This uses the serial summarization approach: each prompt includes the prior
summary and a new note, and the model outputs an updated summary.
"""

import pandas as pd
from transformers import AutoTokenizer


def build_prompt_text(
    tokenizer,
    prior_summary: str,
    note_date: str,
    note_text: str,
    max_model_len: int = 120000,
    margin_tokens: int = 5000
) -> str:
    """
    Build a single prompt for iterative summarization.
    Truncates note_text if too long, keeping head & tail.
    """
    threshold = max(1024, max_model_len - margin_tokens)

    # Truncate note_text if needed
    toks = tokenizer(note_text, add_special_tokens=False).input_ids
    if len(toks) > threshold:
        half = threshold // 2
        first_part = toks[:half]
        last_part = toks[-half:]
        note_text = tokenizer.decode(first_part) + " ... " + tokenizer.decode(last_part)

    prior_summary_text = prior_summary if prior_summary else "None - this is the first note for this patient"

    user_content = f"""You are an experienced clinical oncology history summarization bot.

You are maintaining a running summary of a patient's cancer history based on their electronic health record.
You will be given:
1. A PRIOR SUMMARY of the patient's history (may be empty for first note)
2. A NEW CLINICAL NOTE to incorporate

Your task:
- Update the summary to incorporate any new relevant information from the new note
- If the new note contains no information that would change the summary, output the prior summary exactly as-is
- The patient may not yet have a cancer diagnosis. If not, state "No cancer diagnosis documented as of [date]" and summarize relevant medical history that might be relevant to a future oncology workup.

Document the patient's most recent age; sex; cancer type/primary site (eg breast cancer, lung cancer, etc); histology (eg adenocarcinoma, squamous carcinoma, etc); current extent (localized, advanced, metastatic, etc); biomarkers (genomic results, protein expression, etc); and treatment history (surgery, radiation, chemotherapy/targeted therapy/immunotherapy, etc, including start and stop dates and best response if known).
Do not consider localized basal cell or squamous carcinomas of the skin, or colon polyps, to be cancers for your purposes.
Do not include the patient's name, but do include relevant dates whenever documented.
If a patient has a history of more than one cancer, document the cancers one at a time.
CRITICAL: Format your response as free text ONLY. Do NOT output markdown, Unicode, or tables.

Also document any history of conditions that might meet "boilerplate" exclusion criteria for clinical trials, including uncontrolled brain metastases, lack of measurable disease, congestive heart failure, pneumonitis, renal dysfunction, liver dysfunction, lack of measurable disease,and HIV or hepatitis infection.
Clearly separate the "boilerplate" section by labeling it "Boilerplate: " before describing any such conditions.

Here is an example of the desired output format:

Age: 70
Sex: Male
Cancer type: Lung cancer
Histology: Adenocarcinoma
Current extent: Metastatic
Biomarkers: PD-L1 75%, KRAS G12C mutant
Treatment history:
# 1/5/2020-2/5/2021: carboplatin/pemetrexed/pembrolizumab
# 1/2021: Palliative radiation to progressive spinal metastases
# 3/2021-present: docetaxel
Boilerplate:
No evidence of common boilerplate exclusion criteria

---
PRIOR SUMMARY:
{prior_summary_text}

NEW NOTE (dated {note_date}):
{note_text}
---
Now, write your updated summary. Do not add preceding text before the abstraction, and do not add commentary afterwards."""

    return user_content


def reconstruct_prior_summaries(df: pd.DataFrame, patient_id_col: str = 'pseudo_mrn', date_col: str = 'date') -> pd.Series:
    """
    Reconstruct prior_summary for each row by looking at the previous note's new_summary.

    For the first note of each patient, returns None.
    For subsequent notes, returns the new_summary from the previous note.
    """
    # Sort by patient and date
    df = df.sort_values([patient_id_col, date_col]).reset_index(drop=True)

    prior_summaries = []
    current_patient = None
    last_summary = None

    for idx, row in df.iterrows():
        pid = row[patient_id_col]

        if pid != current_patient:
            # New patient - no prior summary
            current_patient = pid
            prior_summaries.append(None)
        else:
            # Same patient - use last summary
            prior_summaries.append(last_summary)

        # Update last_summary for next iteration
        last_summary = row['new_summary']

    return pd.Series(prior_summaries, index=df.index)


def create_training_prompts(df: pd.DataFrame, tokenizer) -> list:
    """
    Create training prompts from serial summarization data.
    """
    # Sort and reconstruct prior summaries
    df = df.sort_values(['pseudo_mrn', 'date']).reset_index(drop=True)
    df['prior_summary'] = reconstruct_prior_summaries(df)

    prompts = []
    for i in range(len(df)):
        row = df.iloc[i]

        prior_summary = row['prior_summary']
        note_date = str(row['date']) if pd.notna(row['date']) else "unknown date"
        note_text = row['synthetic_note']

        # Combine reasoning + "assistantfinal" + summary for full assistant response
        reasoning = row['new_summary_reasoning']
        summary = row['new_summary']
        full_response = reasoning + "assistantfinal" + summary

        # Build user content
        user_content = build_prompt_text(tokenizer, prior_summary, note_date, note_text)

        # Build the full message
        messages = [
            {'role': 'system', 'content': 'Reasoning: high'},
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': full_response}
        ]

        prompt = tokenizer.apply_chat_template(conversation=messages, tokenize=False)
        prompts.append(prompt)

    return prompts


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/llama-3.2-3B-Instruct')

    print("Loading patient serial summaries data from ../../data/no_phi/patient_serial_summaries.parquet...")
    serial_summaries = pd.read_parquet('../../data/no_phi/patient_serial_summaries.parquet')
    print(f"Loaded {len(serial_summaries)} records")

    print("Generating training prompts...")
    prompts = create_training_prompts(serial_summaries, tokenizer)

    print("Saving to ../../data/no_phi/oncoreasoning_training_data/summarized_patient_histories.parquet...")
    pd.DataFrame(prompts, columns=['text']).to_parquet('../../data/no_phi/oncoreasoning_training_data/summarized_patient_histories.parquet')
    print(f"Done! Saved {len(prompts)} records")


if __name__ == '__main__':
    main()
