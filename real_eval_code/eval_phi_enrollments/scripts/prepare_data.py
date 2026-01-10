#!/usr/bin/env python3
"""
prepare_data.py

Prepares data for clinical trials matching inference on patients who previously
enrolled on therapeutic trials.

Steps:
1. Parse enrollment data from forken.json
2. Load and concatenate EHR reports (imaging, clinical notes, pathology)
3. Load and process useful_trial_enrollments.csv
4. Merge with consent data and filter by date
5. Create note-level dataset with reports prior to trial start date
"""

import json
import pandas as pd
import argparse
from pathlib import Path


def parse_enrollments(forken_path: str) -> pd.DataFrame:
    """
    Parse enrollment data from forken.json.

    Returns DataFrame with columns: dfci_mrn, protocol_number, trial_start_dt
    """
    with open(forken_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for protocol_number, enrollments in data.items():
        # Normalize to a list of dicts
        if isinstance(enrollments, dict):
            enrollments = [enrollments]
        if not isinstance(enrollments, list):
            continue
        for rec in enrollments:
            if isinstance(rec, dict):
                row = rec.copy()
                row["protocol_number"] = protocol_number
                rows.append(row)

    df = pd.DataFrame(rows)

    # Filter for DFCI consents only
    df = df[~df.studySiteOriginalMrn.isnull()]
    df = df[df['studySiteOriginalMrn'].str.contains('DFCI')]

    # Parse dfci_mrn from studySiteOriginalMrn by removing leading "DFCI"
    if "studySiteOriginalMrn" in df.columns:
        df["dfci_mrn"] = (
            df["studySiteOriginalMrn"]
            .astype("string")
            .str.replace(r"(?i)^\s*DFCI\s*", "", regex=True)
            .str.strip()
        )
    else:
        df["dfci_mrn"] = pd.NA

    df['trial_start_dt'] = pd.to_datetime(df['consentDate'])

    # Return only necessary columns for consent data
    consents = df[['dfci_mrn', 'protocol_number', 'trial_start_dt']].copy()

    return consents


def load_all_reports(derived_data_path: str) -> pd.DataFrame:
    """
    Load and concatenate all EHR reports from parquet files.

    Returns DataFrame sorted by dfci_mrn and date.
    """
    prefix = Path(derived_data_path)

    imaging = pd.read_parquet(prefix / 'all_imaging_reports.parquet')
    medonc = pd.read_parquet(prefix / 'all_clinical_notes.parquet')
    path = pd.read_parquet(prefix / 'all_path_reports.parquet')

    all_reports = pd.concat([imaging, medonc, path], axis=0)
    all_reports = all_reports.sort_values(by=['dfci_mrn', 'date']).reset_index(drop=True)

    return all_reports


def load_and_process_trial_enrollments(
    enrollments_path: str,
    consents: pd.DataFrame
) -> pd.DataFrame:
    """
    Load useful_trial_enrollments.csv and process it.

    Returns processed DataFrame with trial enrollment information.
    """
    useful_trial_enrollments = pd.read_csv(enrollments_path)[
        ['dfci_mrn', 'protocol_number', 'nct_id', 'title',
         'brief_summary', 'detailed_summary', 'eligibility_criteria', 'trial_text']
    ]
    useful_trial_enrollments = useful_trial_enrollments.groupby(
        ['dfci_mrn', 'protocol_number']
    ).first().reset_index()

    # Ensure dfci_mrn types match for merge
    consents = consents.copy()
    consents['dfci_mrn'] = consents['dfci_mrn'].astype(int)

    # Merge with consents
    useful_trial_enrollments = pd.merge(
        useful_trial_enrollments,
        consents,
        on=['dfci_mrn', 'protocol_number']
    )

    # Filter out null values
    useful_trial_enrollments = useful_trial_enrollments[
        ~useful_trial_enrollments.trial_start_dt.isnull()
    ]
    useful_trial_enrollments = useful_trial_enrollments[
        ~useful_trial_enrollments.dfci_mrn.isnull()
    ]

    # Convert trial_start_dt to datetime and filter by date
    useful_trial_enrollments['trial_start_dt'] = pd.to_datetime(
        useful_trial_enrollments.trial_start_dt
    )
    useful_trial_enrollments = useful_trial_enrollments[
        useful_trial_enrollments.trial_start_dt >= pd.to_datetime('2016-01-01')
    ]

    # Create pseudo_mrn for each unique (dfci_mrn, trial_start_dt) combination
    # This handles patients who enroll on multiple trials
    unique_combos = useful_trial_enrollments[['dfci_mrn', 'trial_start_dt']].drop_duplicates()
    unique_combos = unique_combos.reset_index(drop=True)
    unique_combos['pseudo_mrn'] = range(1, len(unique_combos) + 1)
    useful_trial_enrollments = pd.merge(
        useful_trial_enrollments,
        unique_combos,
        on=['dfci_mrn', 'trial_start_dt'],
        how='left'
    )

    return useful_trial_enrollments


def create_note_level_dataset(
    useful_trial_enrollments: pd.DataFrame,
    all_reports: pd.DataFrame,
    days_buffer: int = 5
) -> pd.DataFrame:
    """
    Create a note-level dataset by pulling all reports from before each
    patient's trial start date.

    For each combination of patient and trial_start_dt, pulls all reports
    from earlier than the trial_start_dt (plus optional buffer) and adds
    them to the output dataframe.

    Args:
        useful_trial_enrollments: DataFrame with trial enrollment info
        all_reports: DataFrame with all EHR reports
        days_buffer: Number of days after trial_start_dt to include (default 5)

    Returns:
        DataFrame with note-level data for each enrollment
    """
    # Ensure date column is datetime
    all_reports = all_reports.copy()
    all_reports['date'] = pd.to_datetime(all_reports['date'])

    patient_notes_list = []

    for i in range(useful_trial_enrollments.shape[0]):
        enrollment = useful_trial_enrollments.iloc[[i]]
        dfci_mrn = enrollment.dfci_mrn.iloc[0]
        trial_start = enrollment.trial_start_dt.iloc[0]

        # Get all reports for this patient
        patient_reports = all_reports[all_reports.dfci_mrn == dfci_mrn]

        if patient_reports.shape[0] > 0:
            # Filter to reports before trial start (plus buffer)
            cutoff_date = trial_start + pd.Timedelta(days=days_buffer)
            patient_reports = patient_reports[patient_reports.date < cutoff_date]

            if patient_reports.shape[0] > 0:
                # Add enrollment info to each report
                patient_reports = patient_reports.copy()
                patient_reports['pseudo_mrn'] = enrollment.pseudo_mrn.iloc[0]
                patient_reports['protocol_number'] = enrollment.protocol_number.iloc[0]
                patient_reports['nct_id'] = enrollment.nct_id.iloc[0]
                patient_reports['title'] = enrollment.title.iloc[0]
                patient_reports['brief_summary'] = enrollment.brief_summary.iloc[0]
                patient_reports['detailed_summary'] = enrollment.detailed_summary.iloc[0]
                patient_reports['eligibility_criteria'] = enrollment.eligibility_criteria.iloc[0]
                patient_reports['trial_text'] = enrollment.trial_text.iloc[0]
                patient_reports['trial_start_dt'] = trial_start

                patient_notes_list.append(patient_reports)

        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{useful_trial_enrollments.shape[0]} enrollments")

    if patient_notes_list:
        note_level_dataset = pd.concat(patient_notes_list, axis=0).reset_index(drop=True)
    else:
        note_level_dataset = pd.DataFrame()

    return note_level_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for clinical trials matching inference"
    )
    parser.add_argument(
        "--forken-path",
        type=str,
        default="../../../data/phi/enrollments/forken.json",
        help="Path to forken.json file"
    )
    parser.add_argument(
        "--derived-data-path",
        type=str,
        default="/data1/ken/pan_dfci_2024/derived_data",
        help="Path to derived data directory with parquet files"
    )
    parser.add_argument(
        "--enrollments-path",
        type=str,
        default="../../../data/phi/enrollments/useful_trial_enrollments.csv",
        help="Path to useful_trial_enrollments.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../../data/phi/enrollments/",
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--days-buffer",
        type=int,
        default=5,
        help="Number of days after trial start to include reports"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse enrollments from forken.json
    print("Parsing enrollment data from forken.json...")
    consents = parse_enrollments(args.forken_path)
    print(f"  Found {len(consents)} DFCI enrollments")

    # Save consents for reference
    consents.to_csv(output_dir / 'true_dfci_enrollments.csv', index=False)
    print(f"  Saved to {output_dir / 'true_dfci_enrollments.csv'}")

    # Step 2: Load all reports
    print("\nLoading EHR reports...")
    all_reports = load_all_reports(args.derived_data_path)
    print(f"  Loaded {len(all_reports)} total reports")

    # Step 3: Load and process trial enrollments
    print("\nProcessing trial enrollments...")
    useful_trial_enrollments = load_and_process_trial_enrollments(
        args.enrollments_path,
        consents
    )
    print(f"  Found {len(useful_trial_enrollments)} useful trial enrollments")
    print(f"  From {useful_trial_enrollments.dfci_mrn.nunique()} unique patients")

    # Step 4: Create note-level dataset
    print("\nCreating note-level dataset...")
    note_level_dataset = create_note_level_dataset(
        useful_trial_enrollments,
        all_reports,
        days_buffer=args.days_buffer
    )
    print(f"  Created dataset with {len(note_level_dataset)} note-level records")

    # Save outputs
    useful_trial_enrollments.to_csv(
        output_dir / 'processed_trial_enrollments.csv',
        index=False
    )
    print(f"  Saved enrollments to {output_dir / 'processed_trial_enrollments.csv'}")

    note_level_dataset.to_parquet(
        output_dir / 'note_level_dataset.parquet'
    )
    print(f"  Saved note-level dataset to {output_dir / 'note_level_dataset.parquet'}")

    print("\nData preparation complete!")
    print(f"\nSummary:")
    print(f"  - Total enrollments: {len(useful_trial_enrollments)}")
    print(f"  - Unique patients (dfci_mrn): {useful_trial_enrollments.dfci_mrn.nunique()}")
    print(f"  - Unique patient-trial combinations (pseudo_mrn): {useful_trial_enrollments.pseudo_mrn.nunique()}")
    print(f"  - Note-level records: {len(note_level_dataset)}")


if __name__ == "__main__":
    main()
