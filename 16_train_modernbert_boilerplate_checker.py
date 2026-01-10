import pandas as pd
import numpy as np
import os
import argparse
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


def main(checkpoint_dir: str, output_dir: str):
    
    boilerplate_checks = pd.read_parquet("../data/no_phi/boilerplate_checks/final_boilerplate_checks.parquet")
    
    
    
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
        output_dir=checkpoint_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
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
    
    # Resume from checkpoint only if checkpoint directory exists
    resume_from_checkpoint = os.path.isdir(checkpoint_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(checkpoint_dir)
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    
    
    
    trainer.save_model(output_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ModernBERT boilerplate checker model")
    parser.add_argument("--checkpoint_dir", type=str, default="../models/boilerplatechecker_checkpoints",
                        help="Directory to save training checkpoints (default: ../models/boilerplatechecker_checkpoints)")
    parser.add_argument("--output_dir", type=str, default="../models/boilerplatechecker",
                        help="Directory to save final model (default: ../models/boilerplatechecker)")
    args = parser.parse_args()
    main(checkpoint_dir=args.checkpoint_dir, output_dir=args.output_dir)
