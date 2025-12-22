import pandas as pd
import numpy as np
import os
import torch
torch.compile.disable = True
torch.set_float32_matmul_precision('high')


def main():
    
    enrollments = pd.read_parquet('./initial_trialcheck_outputs/space_specific_eligibility_checks.parquet')
    enrollments.info()
    
    round1_patient = pd.read_parquet('./round1_patientcentric_checks/top_cohorts_checked_round1.parquet')
    round2_patient = pd.read_parquet('./round2_patientcentric_checks/top_cohorts_checked_round2.parquet')
    round3_patient = pd.read_parquet('./round3_patientcentric_checks/top_cohorts_checked_round3.parquet')
    patient = pd.concat([round1_patient, round2_patient, round3_patient], ignore_index=True, axis=0)
    patient.info()
    
    round1_space = pd.read_parquet('./round1_trialcentric_checks/top_patients_checked_round1.parquet')
    round2_space = pd.read_parquet('./round2_trialcentric_checks/top_patients_checked_round2.parquet')
    round3_space = pd.read_parquet('./round3_trialcentric_checks/top_patients_checked_round3.parquet')
    
    space = pd.concat([round1_space, round2_space, round3_space], axis=0, ignore_index=True)
    space.info()
    
    dataset = pd.concat([enrollments, patient, space], axis=0, ignore_index=True).groupby(['patient_summary','this_space']).first().reset_index()
    dataset = dataset[['split','patient_summary','this_space','eligibility_result']]
    dataset.info()
    
    dataset.eligibility_result.value_counts()
    
    dataset['eligibility_result'] = dataset.eligibility_result.astype(int)
    
    from transformers import AutoTokenizer
    
    dataset.info()
    
    dataset['pt_trial_pair'] = dataset['this_space'] + "\nNow here is the patient summary:" + dataset['patient_summary']
    
    dataset=dataset[dataset.split != 'test']
    dataset = dataset[['pt_trial_pair', 'eligibility_result', 'split']].rename(columns={'pt_trial_pair':'text','eligibility_result':'label'})
    
    from datasets import Dataset, DatasetDict
    
    train_ds = Dataset.from_pandas(dataset)
    #valid_ds = Dataset.from_pandas(dataset[dataset.split.str.contains('valid')])
    
    data_dict = DatasetDict({"train":train_ds})
    
    data_dict
    
    data_dict['train'][0]
    
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    #tokenizer.pad_token = tokenizer.eos_token
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=4096)
    
    tokenized_data = data_dict.map(preprocess_function, batched=True)
    
    from transformers import DataCollatorWithPadding
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        #predictions = np.argmax(predictions, axis=1)
        return auroc.compute(predictions=predictions, references=labels)
    
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    
    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "answerdotai/ModernBERT-large", num_labels=2, id2label=id2label, label2id=label2id, reference_compile=False
    )
    #model.config.pad_token_id = model.config.eos_token_id
    
    
    training_args = TrainingArguments(
        output_dir="modernbert_checker_training",
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
        #eval_dataset=tokenized_data["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator
        #compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    trainer.save_model('modernbert-trial-checker')

if __name__ == "__main__":
    main()
