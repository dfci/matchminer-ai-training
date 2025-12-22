import pandas as pd
import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch
#torch.compile.disable = True
#torch.set_float32_matmul_precision('high')

import transformers

from pathlib import Path
import sys
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


def main():
    
    def concatenate_csvs_in_directory(directory_path_str: str) -> pd.DataFrame | None:
        """
        Reads all CSV files in the specified directory, concatenates them,
        and returns a single pandas DataFrame.
    
        Args:
            directory_path_str: The path to the directory containing the CSV files (as a string).
    
        Returns:
            A single pandas DataFrame containing data from all CSV files,
            or None if the directory is invalid or contains no CSV files.
            Prints error messages if issues occur.
        """
        directory_path = Path(directory_path_str)
    
        # --- Input Validation ---
        if not directory_path.is_dir():
            print(f"Error: Directory not found or is not a valid directory: {directory_path}", file=sys.stderr)
            return None
    
        # --- Find CSV files ---
        # Use glob to find all files ending with .csv (case-insensitive)
        csv_files = list(directory_path.glob('*.csv'))
        if not csv_files:
            # Also check for uppercase CSV extension, though glob is usually case-insensitive on Windows/macOS
            # but might be case-sensitive on Linux depending on filesystem. A more robust check:
            csv_files = list(directory_path.glob('*.[cC][sS][vV]'))
    
        if not csv_files:
            print(f"Warning: No CSV files found in directory: {directory_path}", file=sys.stderr)
            return None
    
        print(f"Found {len(csv_files)} CSV files to process:")
        for f in csv_files:
            print(f" - {f.name}")
    
        # --- Read and collect DataFrames ---
        all_dataframes = []
        errors_found = False
        for csv_file in csv_files:
            try:
                # Add error_bad_lines=False (older pandas) or on_bad_lines='warn'/'skip' (newer pandas)
                # if you suspect some CSVs might be slightly malformed.
                # Add encoding='your_encoding' if files aren't standard UTF-8.
                df = pd.read_csv(csv_file)
                if not df.empty:
                     # Optional: Add a column indicating the source file
                     # df['source_file'] = csv_file.name
                     all_dataframes.append(df)
                else:
                    print(f"Warning: CSV file is empty, skipping: {csv_file.name}", file=sys.stderr)
            except pd.errors.EmptyDataError:
                 print(f"Warning: CSV file is empty, skipping: {csv_file.name}", file=sys.stderr)
            except Exception as e:
                print(f"Error reading {csv_file.name}: {e}", file=sys.stderr)
                errors_found = True # Mark that an error occurred
    
        # --- Concatenate ---
        if not all_dataframes:
            print("Error: No valid dataframes could be read from the CSV files.", file=sys.stderr)
            return None
        if errors_found:
            print("Warning: Some files could not be read. Concatenating the successfully read files.", file=sys.stderr)
    
    
        try:
            # ignore_index=True creates a new continuous index for the combined DataFrame
            # join='outer' keeps all columns from all files (filling missing with NaN)
            # join='inner' would only keep columns present in *all* files
            combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False, join='outer')
            print("\nConcatenation successful!")
            return combined_df
        except Exception as e:
            print(f"Error during concatenation: {e}", file=sys.stderr)
            return None
    
    
    
    boilerplate_checks = concatenate_csvs_in_directory("./boilerplate_checks")
    
    
    
    
    boilerplate_checks.info()
    
    
    
    boilerplate_checks = boilerplate_checks[~boilerplate_checks.patient_summary.isnull()]
    boilerplate_checks = boilerplate_checks[~(boilerplate_checks.patient_summary == "")]
    
    boilerplate_checks.info()
    dataset = boilerplate_checks
    
    dataset.exclusion_result.value_counts()
    
    dataset['exclusion_result'] = dataset.exclusion_result.astype(int)
    
    
    dataset.info()
    
    dataset['boilerplate_pair'] = "Patient history: " + dataset['patient_boilerplate_text'] + "\nTrial exclusions:" + dataset['trial_boilerplate_text']
    
    dataset = dataset[['boilerplate_pair', 'exclusion_result']].rename(columns={'boilerplate_pair':'text','exclusion_result':'label'})
    
    
    train_ds = Dataset.from_pandas(dataset)
    
    
    data_dict = DatasetDict({"train":train_ds})
    
    data_dict
    
    data_dict['train'][0]
    
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    #tokenizer.pad_token = tokenizer.eos_token
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=3072)
    
    tokenized_data = data_dict.map(preprocess_function, batched=True)
    
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        #predictions = np.argmax(predictions, axis=1)
        return auroc.compute(predictions=predictions, references=labels)
    
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "answerdotai/ModernBERT-large", num_labels=2, id2label=id2label, label2id=label2id, reference_compile=False
    )
    
    
    training_args = TrainingArguments(
        output_dir="modernbert_boilerplate_checker_training",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        #evaluation_strategy="epoch",
        save_strategy="epoch",
        #load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        tokenizer=tokenizer,
        data_collator=data_collator
        #compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    
    
    
    trainer.save_model('modernbert-boilerplate-checker')
    
if __name__ == '__main__':
    main()