import numpy as np
import pandas as pd
import re
import json
import os
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import random
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

summarized_notes = pd.read_parquet('tagged_chunks_enrolled_pt_reports_with_boilerplate.parquet')

summarized_notes = summarized_notes[~summarized_notes.tagger_llm_output.isnull()]

summarized_notes.info()

def generate_rowwise_chunk_labels(original_note, llm_output, valid_tags_list):
    valid_tags_array = np.array(valid_tags_list)
    chunks = re.sub("\n|\r", " ", original_note.strip())
    chunks = re.sub(r'\s+', " ", chunks)
    chunks = "<excerpt break>" + re.sub("\\. ", "<excerpt break>", chunks) + "<excerpt break>"
    chunks = pd.Series(chunks.split("<excerpt break>")).str.strip()
    chunks = chunks[chunks != '']
    chunk_frame = pd.DataFrame({'excerpt':chunks})
    tag_dict = {}
    try:
        json_output = pd.DataFrame.from_records(json.loads(llm_output))
        json_output['tags'] = json_output['tags'].astype(str).str.strip("[|]")

        chunk_frame = pd.merge(chunk_frame, json_output, on='excerpt', how='left')
        chunk_frame['is_tagged'] = np.where(chunk_frame.tags.isnull(), 0, 1)
        chunk_frame['tags'] = np.where(chunk_frame.tags.isnull(), "", chunk_frame.tags)
        chunk_frame['good_json'] = 1
        for tag in valid_tags_array[valid_tags_array != 'is_tagged'].tolist():
            chunk_frame[tag] = np.where(chunk_frame.tags.str.contains(tag), 1, 0)
    except:
        chunk_frame['tags'] = ""
        chunk_frame['is_tagged'] = 0
        chunk_frame['good_json'] = 0
        for tag in valid_tags_array[valid_tags_array != 'is_tagged'].tolist():
            chunk_frame[tag] = 0
    return chunk_frame

valid_tags_list = ['is_tagged']

n = 700
temp = generate_rowwise_chunk_labels(summarized_notes.text.iloc[n], summarized_notes.tagger_llm_output.iloc[n], valid_tags_list)

# Split train and validation data based on 'split' column
train_summarized_notes = summarized_notes[summarized_notes.split.str.contains('train', case=False, na=False)]
val_summarized_notes = summarized_notes[summarized_notes.split.str.contains('val', case=False, na=False)]

print(f"Training examples: {len(train_summarized_notes)}")
print(f"Validation examples: {len(val_summarized_notes)}")

# Process training data
train_outputs = []
for i in range(train_summarized_notes.shape[0]):
    out = generate_rowwise_chunk_labels(train_summarized_notes.text.iloc[i], 
                                        train_summarized_notes.tagger_llm_output.iloc[i], 
                                        valid_tags_list)
    try:
        if out['good_json'].iloc[0] == 1:
            train_outputs.append(out)
    except:
        pass

# Process validation data
val_outputs = []
for i in range(val_summarized_notes.shape[0]):
    out = generate_rowwise_chunk_labels(val_summarized_notes.text.iloc[i], 
                                        val_summarized_notes.tagger_llm_output.iloc[i], 
                                        valid_tags_list)
    try:
        if out['good_json'].iloc[0] == 1:
            val_outputs.append(out)
    except:
        pass

print(f"Valid training chunks: {len(train_outputs)}")
print(f"Valid validation chunks: {len(val_outputs)}")

train_excerpts = pd.concat(train_outputs, axis=0)
val_excerpts = pd.concat(val_outputs, axis=0)

train_excerpts.info()
val_excerpts.info()

print(f"\nTraining label distribution:\n{train_excerpts.is_tagged.value_counts()}")
print(f"\nValidation label distribution:\n{val_excerpts.is_tagged.value_counts()}")

train_excerpts = train_excerpts[~train_excerpts.excerpt.isnull()]
train_excerpts = train_excerpts.rename(columns={'excerpt':'text', 'is_tagged':'label'})

val_excerpts = val_excerpts[~val_excerpts.excerpt.isnull()]
val_excerpts = val_excerpts.rename(columns={'excerpt':'text', 'is_tagged':'label'})

from datasets import Dataset, DatasetDict

train_ds = Dataset.from_pandas(train_excerpts[['text','label']])
val_ds = Dataset.from_pandas(val_excerpts[['text','label']])

data_dict = DatasetDict({"train": train_ds, "validation": val_ds})

print(data_dict)
print(data_dict['train'][0])

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./onc_bert_tiny")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_data = data_dict.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Simple compute_metrics for validation during training
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1)[:, 1].numpy()
    auroc = roc_auc_score(labels, probs)
    return {"auroc": auroc}

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "./onc_bert_tiny", num_labels=2, id2label=id2label, label2id=label2id,
)

training_args = TrainingArguments(
    output_dir="tinybert_tagger_training",
    learning_rate=5e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    eval_strategy="epoch",  # Evaluate after each epoch
    save_strategy="epoch",
    lr_scheduler_type="linear",
    optim="adamw_torch_fused",
    load_best_model_at_end=False,
    #metric_for_best_model="auroc",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('auto-tiny-bert-tagger-with-pretraining')


