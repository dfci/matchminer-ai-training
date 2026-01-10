#!/usr/bin/env python3
"""
prepare_data.py

Prepares data for clinical trials matching inference on patients who started
standard of care (SOC) treatments.

Steps:
1. Load SOC treatment data from TREATMENT_PLAN.txt
2. Filter for standard chemo plans within date range
3. Merge with train/val/test split data
4. Create pseudo_mrn for each unique (dfci_mrn, tplan_start_dt) combination
5. Load and concatenate EHR reports (imaging, clinical notes, pathology)
6. Create note-level dataset with reports prior to treatment start date
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def load_soc_treatments(
    structured_folder: str,
    split_path: str,
    start_date: str = "2016-01-01",
    end_date: str = "2023-01-01"
) -> pd.DataFrame:
    """
    Load and filter SOC treatments from TREATMENT_PLAN.txt.

    Returns DataFrame with SOC treatment information.
    """
    print(f"Loading treatments from {structured_folder}/TREATMENT_PLAN.txt...")
    treatments = pd.read_csv(
        f"{structured_folder}/TREATMENT_PLAN.txt",
        sep="|",
        encoding='latin1',
        low_memory=False
    ).rename(columns={
        'TPLAN_GOAL': 'tplan_goal',
        'PATIENT_ID': 'patient_id',
        'TPLAN_START_DT': 'tplan_start_dt'
    })

    # Convert dates
    treatments['tplan_start_dt'] = pd.to_datetime(treatments.tplan_start_dt)

    # Filter out rows with no treatment plan
    treatments = treatments[
        ~(treatments.STD_CHEMO_PLAN.isnull() & treatments.RESEARCH_CHEMO_PLAN.isnull())
    ]

    # Create plan column and is_trial flag
    treatments['plan'] = np.where(
        treatments.STD_CHEMO_PLAN.isnull(),
        treatments.RESEARCH_CHEMO_PLAN,
        treatments.STD_CHEMO_PLAN
    )
    treatments['is_trial'] = np.where(treatments.STD_CHEMO_PLAN.isnull(), 1, 0)
    treatments['dfci_mrn'] = treatments['DFCI_MRN']
    treatments['tplan_id'] = treatments['TPLAN_ID']
    treatments['dx'] = treatments.TPLAN_ICD_DX_CODES.str[:3]

    # Filter out null goals
    treatments = treatments[~treatments.tplan_goal.isnull()]

    # Add palliative flag
    treatments['is_palliative'] = np.where(
        treatments.tplan_goal.str.contains('PALLIATIVE|CONTROL'),
        1, 0
    )
    treatments['protocol_nbr'] = treatments['RESEARCH_CHEMO_PLAN_NBR']

    # Filter for SOC only
    treatments = treatments[
        treatments.TREATMENT_PLAN_CATEGORY == 'ONCOLOGY STANDARD CHEMO PLAN'
    ]
    print(f"  Found {len(treatments)} SOC treatment records")

    # Filter by date range
    treatments = treatments[
        treatments.tplan_start_dt >= pd.to_datetime(start_date)
    ]
    treatments = treatments[
        treatments.tplan_start_dt <= pd.to_datetime(end_date)
    ]
    print(f"  {len(treatments)} treatments within date range {start_date} to {end_date}")

    # Merge with split data
    print(f"Loading split data from {split_path}...")
    split = pd.read_csv(split_path)
    treatments = pd.merge(split, treatments, on='dfci_mrn')
    print(f"  {len(treatments)} treatments after merging with split")
    print(f"  Split distribution:\n{treatments.split.value_counts().to_string()}")

    # Create trial_start_dt alias for consistency with enrollments code
    treatments['trial_start_dt'] = treatments['tplan_start_dt']

    return treatments


def create_pseudo_mrn(treatments: pd.DataFrame) -> pd.DataFrame:
    """
    Create pseudo_mrn for each unique (dfci_mrn, trial_start_dt) combination.

    This handles patients who start multiple treatments at different times.
    """
    unique_combos = treatments[['dfci_mrn', 'trial_start_dt']].drop_duplicates()
    unique_combos = unique_combos.reset_index(drop=True)
    unique_combos['pseudo_mrn'] = range(1, len(unique_combos) + 1)

    treatments = pd.merge(
        treatments,
        unique_combos,
        on=['dfci_mrn', 'trial_start_dt'],
        how='left'
    )

    print(f"Created {treatments.pseudo_mrn.nunique()} unique pseudo_mrn values")
    return treatments


def load_all_reports(derived_data_path: str) -> pd.DataFrame:
    """
    Load and concatenate all EHR reports from parquet files.

    Returns DataFrame sorted by dfci_mrn and date.
    """
    prefix = Path(derived_data_path)

    print(f"Loading EHR reports from {derived_data_path}...")
    imaging = pd.read_parquet(prefix / 'all_imaging_reports.parquet')
    print(f"  Loaded {len(imaging)} imaging reports")

    medonc = pd.read_parquet(prefix / 'all_clinical_notes.parquet')
    print(f"  Loaded {len(medonc)} clinical notes")

    path = pd.read_parquet(prefix / 'all_path_reports.parquet')
    print(f"  Loaded {len(path)} pathology reports")

    all_reports = pd.concat([imaging, medonc, path], axis=0)
    all_reports = all_reports.sort_values(by=['dfci_mrn', 'date']).reset_index(drop=True)
    print(f"  Total: {len(all_reports)} reports")

    return all_reports


def create_note_level_dataset(
    soc_treatments: pd.DataFrame,
    all_reports: pd.DataFrame,
    days_buffer: int = 5
) -> pd.DataFrame:
    """
    Create a note-level dataset by pulling all reports from before each
    patient's treatment start date.

    For each combination of patient and trial_start_dt, pulls all reports
    from earlier than the trial_start_dt (plus optional buffer) and adds
    them to the output dataframe.

    Args:
        soc_treatments: DataFrame with SOC treatment info
        all_reports: DataFrame with all EHR reports
        days_buffer: Number of days after trial_start_dt to include (default 5)

    Returns:
        DataFrame with note-level data for each treatment
    """
    # Ensure date column is datetime
    all_reports = all_reports.copy()
    all_reports['date'] = pd.to_datetime(all_reports['date'])

    patient_notes_list = []

    for i in range(soc_treatments.shape[0]):
        treatment = soc_treatments.iloc[[i]]
        dfci_mrn = treatment.dfci_mrn.iloc[0]
        trial_start = treatment.trial_start_dt.iloc[0]

        # Get all reports for this patient
        patient_reports = all_reports[all_reports.dfci_mrn == dfci_mrn]

        if patient_reports.shape[0] > 0:
            # Filter to reports before treatment start (plus buffer)
            cutoff_date = trial_start + pd.Timedelta(days=days_buffer)
            patient_reports = patient_reports[patient_reports.date < cutoff_date]

            if patient_reports.shape[0] > 0:
                # Add treatment info to each report
                patient_reports = patient_reports.copy()
                patient_reports['pseudo_mrn'] = treatment.pseudo_mrn.iloc[0]
                patient_reports['tplan_id'] = treatment.tplan_id.iloc[0]
                patient_reports['plan'] = treatment.plan.iloc[0]
                patient_reports['tplan_goal'] = treatment.tplan_goal.iloc[0]
                patient_reports['dx'] = treatment.dx.iloc[0]
                patient_reports['is_palliative'] = treatment.is_palliative.iloc[0]
                patient_reports['trial_start_dt'] = trial_start
                patient_reports['patient_split'] = treatment.split.iloc[0]

                patient_notes_list.append(patient_reports)

        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{soc_treatments.shape[0]} treatments")

    if patient_notes_list:
        note_level_dataset = pd.concat(patient_notes_list, axis=0).reset_index(drop=True)
    else:
        note_level_dataset = pd.DataFrame()

    return note_level_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for SOC treatment matching inference"
    )
    parser.add_argument(
        "--structured-folder",
        type=str,
        default="/data1/ken/pan_dfci_2024/structured_data/",
        help="Path to structured data folder containing TREATMENT_PLAN.txt"
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default="/data1/ken/pan_dfci_2024/derived_data/split_5-2024.csv",
        help="Path to train/val/test split CSV"
    )
    parser.add_argument(
        "--derived-data-path",
        type=str,
        default="/data1/ken/pan_dfci_2024/derived_data",
        help="Path to derived data directory with parquet files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../../data/phi/soc/",
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2016-01-01",
        help="Start date for treatment filtering"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2023-01-01",
        help="End date for treatment filtering"
    )
    parser.add_argument(
        "--days-buffer",
        type=int,
        default=5,
        help="Number of days after treatment start to include reports"
    )
    parser.add_argument(
        "--split-filter",
        type=str,
        default=None,
        help="Filter to specific split (train, validation, test). If None, use all."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load SOC treatments
    print("\n" + "="*60)
    print("STEP 1: Loading SOC treatments")
    print("="*60)
    soc_treatments = load_soc_treatments(
        args.structured_folder,
        args.split_path,
        args.start_date,
        args.end_date
    )

    # Optional: filter by split
    if args.split_filter:
        print(f"\nFiltering to split: {args.split_filter}")
        soc_treatments = soc_treatments[
            soc_treatments.split.str.contains(args.split_filter)
        ].reset_index(drop=True)
        print(f"  {len(soc_treatments)} treatments after split filter")

    # Step 2: Create pseudo_mrn
    print("\n" + "="*60)
    print("STEP 2: Creating pseudo_mrn")
    print("="*60)
    soc_treatments = create_pseudo_mrn(soc_treatments)

    # Save processed treatments
    treatments_path = output_dir / 'processed_soc_treatments.csv'
    soc_treatments.to_csv(treatments_path, index=False)
    print(f"Saved processed treatments to {treatments_path}")

    # Step 3: Load all reports
    print("\n" + "="*60)
    print("STEP 3: Loading EHR reports")
    print("="*60)
    all_reports = load_all_reports(args.derived_data_path)

    # Step 4: Create note-level dataset
    print("\n" + "="*60)
    print("STEP 4: Creating note-level dataset")
    print("="*60)
    note_level_dataset = create_note_level_dataset(
        soc_treatments,
        all_reports,
        days_buffer=args.days_buffer
    )
    print(f"Created dataset with {len(note_level_dataset)} note-level records")

    # Save outputs
    note_level_path = output_dir / 'note_level_dataset.parquet'
    note_level_dataset.to_parquet(note_level_path)
    print(f"Saved note-level dataset to {note_level_path}")

    # Summary
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nSummary:")
    print(f"  - Total SOC treatments: {len(soc_treatments)}")
    print(f"  - Unique patients (dfci_mrn): {soc_treatments.dfci_mrn.nunique()}")
    print(f"  - Unique patient-treatment combinations (pseudo_mrn): {soc_treatments.pseudo_mrn.nunique()}")
    print(f"  - Note-level records: {len(note_level_dataset)}")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
