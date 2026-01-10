

import pandas as pd
import numpy as np
import torch


import os
import torch
from datasets import Dataset, load_dataset
#from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType

dataset = Dataset.load_from_disk('../../data/no_phi/oncoreasoning_training_data/tokenized_training_data.dataset/')


lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    use_rslora=True,
    init_lora_weights="gaussian",
    modules_to_save=["lm_head"],
)




repo_id = "meta-llama/Llama-3.2-3B-Instruct"


model = AutoModelForCausalLM.from_pretrained(
   repo_id, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

sft_config = SFTConfig(
    ## GROUP 1: Memory usage
    # These arguments will squeeze the most out of your GPU's RAM
    # Checkpointing
    gradient_checkpointing=True,    # this saves a LOT of memory
    # Set this to avoid exceptions in newer versions of PyTorch
    gradient_checkpointing_kwargs={'use_reentrant': False}, 
    # Gradient Accumulation / Batch size
    # Actual batch (for updating) is same (1x) as micro-batch size
    gradient_accumulation_steps=1,  
    # The initial (micro) batch size to start off with
    per_device_train_batch_size=4, 
    bf16=True,
    # If batch size would cause OOM, halves its size until it works
    auto_find_batch_size=False,
    save_total_limit=2,
    #save_safetensors=False,

    ## GROUP 2: Dataset-related
    max_length=tokenizer.max_len_single_sentence,
    # Dataset
    # packing a dataset means no padding is needed
    packing=False,

    ## GROUP 3: These are typical training parameters
    num_train_epochs=1,
    learning_rate=5e-6,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio = 0.10,
    #save_strategy='epoch',
    save_steps=2000,
    #evaluation_strategy='no',
    # Optimizer
    # 8-bit Adam optimizer - doesn't help much if you're using LoRA!
    optim='adamw_torch_fused',       
    dataset_kwargs = {'skip_prepare_dataset':True},
    model_init_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
    lr_scheduler_kwargs={'num_cycles':3},
    ## GROUP 4: Logging parameters
    logging_steps=20,
    #activation_offloading=True,
    use_liger_kernel=True,
    logging_dir='./logs',
    output_dir='../../models/onco_reasoning_3b',
    report_to='none'
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    #peft_config = lora_config,
    train_dataset=dataset,
)

len(dataset)





trainer.train()



